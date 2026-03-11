"""
CRAM-AdaMuon Optimizer
Parameter-type-aware optimization:
    - AdaMuon for all 2D matrix parameters (RSP, SAMG, SLE, FFN, ADR)
    - AdamW  for embeddings, 1D vectors, biases

AdaMuon = Muon (orthogonal updates via Newton-Schulz) + Adam (element-wise variance)
This combination gives:
    - Better loss curves than AdamW (empirically ~2x compute efficient)
    - Stable training at large scale
    - muP-compatible (hyperparameters transfer across model sizes)

Reference: Muon optimizer + AdaMuon extensions (2024-2025)
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Dict, Tuple, Optional, Iterable
import math


def newton_schulz_orthogonalize(G: torch.Tensor, n_steps: int = 5) -> torch.Tensor:
    """
    Orthogonalize matrix G via Newton-Schulz iteration.
    
    Iteration: X_{i+1} = 1.5*X_i - 0.5*X_i @ X_i.T @ X_i
    Converges to the orthogonal factor of G in ~5 steps.
    
    This is the core of Muon — replacing Adam's element-wise sqrt
    with a matrix-wise orthogonalization that respects the geometry
    of the weight space.
    
    Args:
        G: [m, n] gradient matrix
        n_steps: Newton-Schulz iterations (5 is sufficient)
    Returns:
        U: [m, n] near-orthogonal matrix, ||U||_F ≈ sqrt(min(m,n))
    """
    assert G.ndim == 2
    m, n = G.shape
    
    # Normalize for numerical stability
    X = G / (G.norm() + 1e-7)
    
    # Handle non-square: work with smaller dimension
    if m > n:
        # X is tall: work with X.T @ X (n×n)
        for _ in range(n_steps):
            A = X.T @ X                          # [n, n]
            X = X @ (1.5 * torch.eye(n, device=G.device, dtype=G.dtype) - 0.5 * A)
    else:
        # X is wide or square: work with X @ X.T (m×m)  
        for _ in range(n_steps):
            A = X @ X.T                          # [m, m]
            X = (1.5 * torch.eye(m, device=G.device, dtype=G.dtype) - 0.5 * A) @ X
    
    # Scale to match original gradient magnitude
    X = X * math.sqrt(min(m, n))
    return X


class AdaMuon(Optimizer):
    """
    AdaMuon: Orthogonal updates + element-wise variance adaptation.
    
    For 2D matrix parameters W ∈ R^{m×n}:
    1. Compute momentum with sign stabilization
    2. Orthogonalize via Newton-Schulz (5 iters)
    3. Track element-wise variance of orthogonal updates
    4. Scale by 1/sqrt(variance) — RMS normalization
    5. Apply weight update with weight decay
    
    Key advantage over AdamW:
    - Updates are orthogonal → each step moves in a new direction
    - No redundant update directions (AdamW can waste steps)
    - ~2x compute efficiency empirically
    
    Args:
        params: 2D matrix parameters only
        lr: learning rate (default: 0.02 from muP)
        beta1: momentum coefficient
        beta2: variance tracking coefficient  
        eps: numerical stability
        weight_decay: L2 regularization
        ns_steps: Newton-Schulz iterations
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta1: float = 0.95,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2,
            eps=eps, weight_decay=weight_decay, ns_steps=ns_steps
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                assert p.ndim == 2, f"AdaMuon expects 2D params, got shape {p.shape}"
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaMuon does not support sparse gradients")
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)
                    state['variance'] = torch.zeros_like(p)
                
                state['step'] += 1
                t = state['step']
                m = state['momentum']
                v = state['variance']
                
                # Step 1: Momentum update (with sign stabilization)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Step 2: Orthogonalize momentum via Newton-Schulz
                # This is the key Muon step — project gradient to orthogonal space
                m_orth = newton_schulz_orthogonalize(m, n_steps=ns_steps)
                
                # Step 3: Track variance of orthogonal updates
                v.mul_(beta2).addcmul_(m_orth, m_orth, value=1 - beta2)
                
                # Step 4: Bias correction
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                m_hat = m / bc1
                v_hat = v / bc2
                
                # Step 5: Re-orthogonalize bias-corrected momentum
                m_hat_orth = newton_schulz_orthogonalize(m_hat, n_steps=ns_steps)
                
                # Step 6: Adaptive scaling
                update = m_hat_orth / (v_hat.sqrt() + eps)
                
                # Step 7: RMS alignment — match AdamW update magnitude
                rms_scale = update.norm() / (p.norm() + eps)
                if rms_scale > 0:
                    update = update / rms_scale * lr
                else:
                    update = update * lr
                
                # Step 8: Weight update with weight decay
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-1)
        
        return loss


def get_param_groups(model: nn.Module, config) -> List[Dict]:
    """
    Split model parameters into AdaMuon (2D matrices) and AdamW (everything else).
    
    AdaMuon group: RSP, ADR, SAMG, SLE, FFN weight matrices
    AdamW group:   Embeddings, norms, biases, 1D vectors, SAMG graph buffers
    
    Returns list of param groups suitable for CRAMOptimizer.
    """
    adamuon_params = []
    adamw_params = []
    
    adamuon_names = []
    adamw_names = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 2D matrices in hidden layers → AdaMuon
        if (param.ndim == 2 and 
            'embedding' not in name and 
            'norm' not in name):
            adamuon_params.append(param)
            adamuon_names.append(name)
        else:
            # Embeddings, norms, biases, 1D params → AdamW
            adamw_params.append(param)
            adamw_names.append(name)
    
    return adamuon_params, adamw_params, adamuon_names, adamw_names


class CRAMOptimizer:
    """
    Combined optimizer for CRAM:
    - AdaMuon for 2D matrix parameters
    - AdamW for everything else
    
    Exposes a unified .step() interface.
    """
    
    def __init__(self, model: nn.Module, config):
        adamuon_params, adamw_params, adamuon_names, adamw_names = \
            get_param_groups(model, config)
        
        print(f"  AdaMuon params: {len(adamuon_params)} tensors")
        print(f"  AdamW   params: {len(adamw_params)} tensors")
        
        self.adamuon = AdaMuon(
            adamuon_params,
            lr=config.lr_peak,
            beta1=0.95,
            beta2=0.999,
            eps=1e-8,
            weight_decay=config.weight_decay,
            ns_steps=5,
        ) if adamuon_params else None
        
        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=config.lr_adamw,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
        ) if adamw_params else None
        
        self._step_count = 0
    
    def step(self):
        if self.adamuon:
            self.adamuon.step()
        if self.adamw:
            self.adamw.step()
        self._step_count += 1
    
    def zero_grad(self):
        if self.adamuon:
            self.adamuon.zero_grad()
        if self.adamw:
            self.adamw.zero_grad()
    
    def set_lr(self, lr_muon: float, lr_adamw: float):
        """Update learning rates (called by scheduler)."""
        if self.adamuon:
            for g in self.adamuon.param_groups:
                g['lr'] = lr_muon
        if self.adamw:
            for g in self.adamw.param_groups:
                g['lr'] = lr_adamw
    
    def state_dict(self):
        return {
            'adamuon': self.adamuon.state_dict() if self.adamuon else None,
            'adamw': self.adamw.state_dict() if self.adamw else None,
            'step': self._step_count,
        }
    
    def load_state_dict(self, state):
        if self.adamuon and state['adamuon']:
            self.adamuon.load_state_dict(state['adamuon'])
        if self.adamw and state['adamw']:
            self.adamw.load_state_dict(state['adamw'])
        self._step_count = state.get('step', 0)


class WSACScheduler:
    """
    Warmup-Stable-Anneal-Cooldown learning rate schedule.
    
    Phase 1 — WARMUP   (0 → warmup_steps):      linear ramp to peak
    Phase 2 — STABLE   (warmup → anneal_start):  constant at peak
    Phase 3 — ANNEAL   (anneal → cooldown_start): 0.3x decay (NOT to zero)
    Phase 4 — COOLDOWN (cooldown → total):        cosine to 0
    
    Key insight: keeping LR at 0.3x peak during high-quality data phase
    preserves curriculum benefit (Muon 2025 finding).
    """
    
    def __init__(
        self,
        optimizer: CRAMOptimizer,
        total_steps: int,
        warmup_frac: float = 0.01,    # 1% warmup
        anneal_start_frac: float = 0.85,
        cooldown_start_frac: float = 0.95,
        min_lr_ratio: float = 0.3,    # Anneal to 30% of peak (not zero)
        peak_lr_muon: float = 0.02,
        peak_lr_adamw: float = 3e-4,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_frac)
        self.anneal_start = int(total_steps * anneal_start_frac)
        self.cooldown_start = int(total_steps * cooldown_start_frac)
        self.min_lr_ratio = min_lr_ratio
        self.peak_muon = peak_lr_muon
        self.peak_adamw = peak_lr_adamw
        self._step = 0
    
    def get_lr_multiplier(self, step: int) -> float:
        """Get LR multiplier ∈ [min_lr_ratio, 1.0] for current step."""
        if step < self.warmup_steps:
            # Phase 1: Linear warmup
            return step / max(self.warmup_steps, 1)
        
        elif step < self.anneal_start:
            # Phase 2: Stable (constant peak)
            return 1.0
        
        elif step < self.cooldown_start:
            # Phase 3: Moderate anneal (to 30% peak — NOT zero)
            progress = (step - self.anneal_start) / (self.cooldown_start - self.anneal_start)
            return 1.0 - (1.0 - self.min_lr_ratio) * progress
        
        else:
            # Phase 4: Cosine cooldown from min_lr_ratio → 0
            progress = (step - self.cooldown_start) / (self.total_steps - self.cooldown_start)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr_ratio * cosine
    
    def step(self):
        """Advance scheduler one step and update optimizer LR."""
        mult = self.get_lr_multiplier(self._step)
        self.optimizer.set_lr(
            lr_muon=self.peak_muon * mult,
            lr_adamw=self.peak_adamw * mult,
        )
        self._step += 1
        return mult
    
    def get_current_phase(self) -> str:
        s = self._step
        if s < self.warmup_steps:
            return "WARMUP"
        elif s < self.anneal_start:
            return "STABLE"
        elif s < self.cooldown_start:
            return "ANNEAL"
        else:
            return "COOLDOWN"
