"""
ADR: Adaptive Depth Router
Routes each token to exactly the right type and amount of compute.

4D difficulty decomposition:
    d_depth:  syntactic complexity → deep RSP
    d_width:  semantic ambiguity   → wide MoE experts
    d_memory: long-range recall    → SAMG access
    d_reason: logical reasoning    → SLE inference engine

5 routing paths:
    FAST:   2-layer RSP only (easy tokens: punctuation, function words)
    DEEP:   Full L-layer RSP (complex syntax, long dependencies)
    WIDE:   MoE FFN experts (domain-specific, rare vocabulary)
    MEMORY: RSP + SAMG (coreference, long-range recall)
    REASON: RSP + SLE (math, logic, causal reasoning)

Key properties:
    - Soft routing (no irrecoverable token drops)
    - Uncertainty-aware (Bayesian gate with annealed temperature)
    - Budget-constrained (explicit FLOP target)
    - Gradient surgery for conflicting path gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

from cram.configs.cram_config import CRAMConfig


# Path indices
PATH_FAST   = 0
PATH_DEEP   = 1
PATH_WIDE   = 2
PATH_MEMORY = 3
PATH_REASON = 4
N_PATHS = 5
PATH_NAMES = ["fast", "deep", "wide", "memory", "reason"]

# Approximate relative FLOP costs per path (normalized to fast=1.0)
PATH_FLOP_COSTS = torch.tensor([1.0, 3.5, 4.0, 2.5, 5.0])


class CheapDifficultyEstimator(nn.Module):
    """
    CDE: Cheap Difficulty Estimator
    Estimates 4D token difficulty before spending expensive compute.

    Architecture: 2-layer MLP with bottleneck
    Input:  x ∈ R^d
    Output: d_t ∈ R^4  (difficulty per axis)

    Cost: ~1.5% of full RSP layer (r = d/8)
    """

    def __init__(self, d_model: int, r_dim: int):
        super().__init__()
        self.d_model = d_model
        self.r_dim = r_dim

        # Two-layer MLP with bottleneck
        self.mlp1 = nn.Linear(d_model, r_dim, bias=True)
        self.mlp2 = nn.Linear(r_dim, 4, bias=True)    # 4 difficulty dimensions

        # Initialize for ~uniform difficulty at start
        nn.init.normal_(self.mlp1.weight, std=0.01)
        nn.init.zeros_(self.mlp1.bias)
        nn.init.normal_(self.mlp2.weight, std=0.01)
        nn.init.zeros_(self.mlp2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            d: [B, L, 4] — difficulty scores in (0, 1) per dimension
        """
        e = F.gelu(self.mlp1(x))      # [B, L, r]
        d = torch.sigmoid(self.mlp2(e))  # [B, L, 4] — bounded in (0,1)
        return d


class UncertaintyAwareGate(nn.Module):
    """
    Bayesian routing gate with annealed temperature.

    During training: samples from N(mu, diag(sigma^2)) for exploration
    During inference: uses mean (deterministic, optimal)

    This solves the dead router problem — all paths receive gradient signal
    during training because uncertainty exploration visits all paths.
    """

    def __init__(self, d_model: int, n_paths: int = N_PATHS):
        super().__init__()
        self.n_paths = n_paths

        # Uncertainty estimation from input variance
        self.W_sigma = nn.Linear(d_model, 4, bias=False)
        nn.init.normal_(self.W_sigma.weight, std=0.01)

    def forward(
        self,
        difficulty: torch.Tensor,    # [B, L, 4]
        x: torch.Tensor,             # [B, L, d_model] for uncertainty estimate
        temperature: float = 1.0,
        training: bool = True
    ) -> torch.Tensor:
        """
        Returns:
            P: [B, L, N_PATHS] — routing probabilities summing to 1
        """
        B, L, _ = difficulty.shape

        # Uncertainty estimate (heteroscedastic)
        sigma = torch.abs(x - x.mean(dim=-1, keepdim=True))[:, :, :4]
        sigma = torch.sigmoid(self.W_sigma(x)) * 0.5    # [B, L, 4]

        # Sample or use mean
        if training and self.training:
            noise = torch.randn_like(difficulty)
            d_tilde = difficulty + sigma * noise          # [B, L, 4]
        else:
            d_tilde = difficulty                          # deterministic

        # Convert 4D difficulty to 5-way routing probabilities
        # P_fast = residual mass (1 - max difficulty)
        # P_path = corresponding difficulty component
        max_diff = d_tilde.max(dim=-1, keepdim=True).values  # [B, L, 1]
        p_fast = (1.0 - max_diff).clamp(0.05, 0.95)          # [B, L, 1]

        # Scale remaining 4 paths by their difficulty scores
        p_others = d_tilde * (1.0 - p_fast)                   # [B, L, 4]

        # Combine and normalize
        p_all = torch.cat([p_fast, p_others], dim=-1)          # [B, L, 5]

        # Apply temperature and softmax
        p_all = F.softmax(p_all / temperature, dim=-1)

        # Enforce minimum probability (no dead paths)
        epsilon = 0.02  # 2% minimum per path
        p_all = p_all * (1 - epsilon * N_PATHS) + epsilon
        p_all = p_all / p_all.sum(dim=-1, keepdim=True)       # renormalize

        return p_all


class ADR(nn.Module):
    """
    Adaptive Depth Router.
    Orchestrates multi-path execution and soft blending.

    Does NOT contain the actual path computations (those are in CRAMBlock).
    ADR is responsible for:
    1. Estimating difficulty (CDE)
    2. Computing routing probabilities (UncertaintyAwareGate)
    3. Blending path outputs (soft weighted sum)
    4. Computing routing losses (budget, balance, calibration)
    """

    def __init__(self, config: CRAMConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        # Cheap difficulty estimator
        self.cde = CheapDifficultyEstimator(config.d_model, config.adr_r_dim)

        # Uncertainty-aware routing gate
        self.gate = UncertaintyAwareGate(config.d_model, N_PATHS)

        # Current routing temperature (annealed externally)
        self.temperature = config.adr_temp_init

        # FLOP costs per path (for budget loss)
        self.register_buffer("flop_costs", PATH_FLOP_COSTS)

    def estimate_difficulty(self, x: torch.Tensor) -> torch.Tensor:
        """Cheap 4D difficulty estimate. [B, L, 4]"""
        return self.cde(x)

    def route(
        self,
        x: torch.Tensor,              # [B, L, D]
        difficulty: torch.Tensor      # [B, L, 4]
    ) -> torch.Tensor:
        """
        Compute routing probabilities. [B, L, N_PATHS]
        """
        return self.gate(difficulty, x, self.temperature, self.training)

    def blend(
        self,
        path_outputs: Dict[str, torch.Tensor],   # {path_name: [B, L, D]}
        routing_probs: torch.Tensor               # [B, L, N_PATHS]
    ) -> torch.Tensor:
        """
        Soft weighted blend of path outputs.

        output = sum_p P_p * y_p

        This is FULLY DIFFERENTIABLE — gradients flow to all paths
        weighted by their routing probability. No straight-through needed.
        No irrecoverable token drops.
        """
        device = routing_probs.device
        B, L, _ = routing_probs.shape

        result = torch.zeros(B, L, self.d_model, device=device,
                             dtype=routing_probs.dtype)

        path_map = {
            "fast":   PATH_FAST,
            "deep":   PATH_DEEP,
            "wide":   PATH_WIDE,
            "memory": PATH_MEMORY,
            "reason": PATH_REASON,
        }

        for name, output in path_outputs.items():
            if name in path_map:
                idx = path_map[name]
                p = routing_probs[:, :, idx:idx+1]   # [B, L, 1]
                result = result + p * output

        return result

    def compute_routing_losses(
        self,
        routing_probs: torch.Tensor,         # [B, L, N_PATHS]
        difficulty: torch.Tensor,             # [B, L, 4]
        task_loss: Optional[torch.Tensor] = None  # [B, L] per-token loss
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all ADR auxiliary losses.

        L_budget:  FLOP utilization should not exceed F_target
        L_balance: All paths should be used roughly equally
        L_calib:   Difficulty scores should predict actual hardness
        """
        losses = {}

        # ── Budget loss ────────────────────────────────────────────
        # Expected FLOPs = sum_p P_p * cost_p
        expected_flops = (routing_probs * self.flop_costs.to(routing_probs.device)).sum(-1)  # [B, L]
        avg_flops = expected_flops.mean()
        f_target = self.config.adr_budget_target * self.flop_costs.max()
        losses["budget"] = F.relu(avg_flops - f_target).pow(2)

        # ── Balance loss ───────────────────────────────────────────
        # Each path should be used ~1/N_PATHS of the time
        avg_path_probs = routing_probs.mean(dim=[0, 1])      # [N_PATHS]
        target_prob = 1.0 / N_PATHS
        losses["balance"] = (avg_path_probs - target_prob).pow(2).sum()

        # ── Calibration loss ───────────────────────────────────────
        # If we have per-token task loss, use it as difficulty target
        if task_loss is not None:
            # Normalize task loss to [0, 1] range
            max_loss = task_loss.detach().max() + 1e-8
            normalized_loss = task_loss.detach() / max_loss   # [B, L]

            # difficulty[:,:,0] should predict depth difficulty
            # Best proxy: just use total difficulty vs normalized loss
            pred_difficulty = difficulty.max(dim=-1).values   # [B, L]
            losses["calib"] = F.mse_loss(pred_difficulty, normalized_loss)
        else:
            losses["calib"] = torch.tensor(0.0, device=routing_probs.device)

        return losses

    def get_routing_stats(self, routing_probs: torch.Tensor) -> Dict:
        """Return routing statistics for monitoring."""
        avg = routing_probs.mean(dim=[0, 1]).detach().cpu()
        stats = {}
        for i, name in enumerate(PATH_NAMES):
            stats[f"path_{name}_prob"] = avg[i].item()
        stats["routing_entropy"] = (
            -(routing_probs * (routing_probs + 1e-8).log()).sum(-1).mean().item()
        )
        return stats


class GradientSurgery:
    """
    Path-Wise Gradient Surgery.
    Resolves conflicting gradients between routing paths.

    For each pair of paths (A, B):
        if g_A · g_B < 0:  [conflicting]
            g_A -= (g_A · g_B / ||g_B||^2) * g_B   [project out conflict]
            g_B -= (g_B · g_A / ||g_A||^2) * g_A

    Applied before gradient accumulation.
    O(paths^2) overhead — negligible with 5 paths.
    """

    @staticmethod
    def apply(gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply gradient surgery to path gradients.

        Args:
            gradients: {path_name: flattened_gradient_tensor}
        Returns:
            corrected_gradients: {path_name: corrected_gradient}
        """
        path_names = list(gradients.keys())
        g = {name: grad.clone() for name, grad in gradients.items()}

        for i in range(len(path_names)):
            for j in range(i + 1, len(path_names)):
                name_a = path_names[i]
                name_b = path_names[j]
                g_a = g[name_a]
                g_b = g[name_b]

                dot = (g_a * g_b).sum()

                if dot < 0:  # conflicting directions
                    # Project g_a to remove component in g_b direction
                    g[name_a] = g_a - (dot / (g_b.norm().pow(2) + 1e-8)) * g_b
                    g[name_b] = g_b - (dot / (g_a.norm().pow(2) + 1e-8)) * g_a

        return g
