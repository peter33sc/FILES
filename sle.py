"""
SLE: Structured Logic Engine
Internal reasoning module — replaces chain-of-thought with latent inference.

Architecture:
    PE  — Proposition Encoder: decomposes h into typed logical propositions
    LS  — Latent Scratchpad: working memory for intermediate derivations
    IE  — Inference Engine: forward + backward chaining
    CG  — Convergence Gate: halts when reasoning converges

Proposition types:
    FACT (0):       Known information
    RULE (1):       If-then inference rules
    GOAL (2):       What we're trying to derive
    CONSTRAINT (3): Must-satisfy conditions
    UNK (4):        Unknown — IE will try to derive

Key properties:
    - No chain-of-thought tokens needed
    - Latent reasoning: O(I * K * d) per hard token
    - Fully differentiable via soft firing thresholds
    - Adaptive depth: I_max controlled by ADR difficulty score
    - Self-supervised via consistency regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional

from cram.configs.cram_config import CRAMConfig


# Proposition type indices
FACT = 0
RULE = 1
GOAL = 2
CONSTRAINT = 3
UNK = 4
N_TYPES = 5
TYPE_NAMES = ["FACT", "RULE", "GOAL", "CONSTRAINT", "UNK"]


class PropositionEncoder(nn.Module):
    """
    PE: Proposition Encoder
    Decomposes h_t into K typed logical proposition slots.

    Each slot contains:
        content:    d/K dimensional vector (what this proposition says)
        type:       soft distribution over {FACT, RULE, GOAL, CONSTRAINT, UNK}
        confidence: scalar in (0, 1) (how certain we are)

    All operations differentiable → gradients flow back to RSP+SAMG.
    """

    def __init__(self, config: CRAMConfig):
        super().__init__()
        self.K = config.sle_n_props
        self.d_model = config.d_model
        self.d_prop = config.d_model // config.sle_n_props  # per-prop dim

        # Content projections (K independent projections)
        self.W_content = nn.Linear(config.d_model, config.d_model, bias=False)

        # Type classification (K * N_TYPES outputs)
        self.W_type = nn.Linear(config.d_model, self.K * N_TYPES, bias=True)
        nn.init.zeros_(self.W_type.bias)

        # Confidence estimation (K outputs)
        self.W_conf = nn.Linear(config.d_model, self.K, bias=True)
        nn.init.constant_(self.W_conf.bias, 0.5)    # start at 50% confidence

    def forward(
        self, h: torch.Tensor    # [B, L, D] or [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            content:    [B, (L,) K, d_prop]
            types:      [B, (L,) K, N_TYPES]  — soft type distribution
            confidence: [B, (L,) K]            — in (0, 1)
        """
        squeeze = h.dim() == 2
        if squeeze:
            h = h.unsqueeze(1)    # [B, 1, D]

        B, L, D = h.shape

        # Content: project and split into K slots
        content_flat = self.W_content(h)                      # [B, L, D]
        content = content_flat.view(B, L, self.K, self.d_prop)  # [B, L, K, d/K]

        # Type: soft distribution over N_TYPES
        type_logits = self.W_type(h)                          # [B, L, K*N_TYPES]
        type_logits = type_logits.view(B, L, self.K, N_TYPES)
        types = F.softmax(type_logits, dim=-1)                # [B, L, K, N_TYPES]

        # Confidence: in (0, 1)
        confidence = torch.sigmoid(self.W_conf(h))            # [B, L, K]

        if squeeze:
            content = content.squeeze(1)     # [B, K, d/K]
            types = types.squeeze(1)         # [B, K, N_TYPES]
            confidence = confidence.squeeze(1)  # [B, K]

        return content, types, confidence


class LatentScratchpad(nn.Module):
    """
    LS: Latent Scratchpad
    Fixed-size slot-based working memory for reasoning.

    M_s slots, each d_prop dimensional.
    Soft attention-based addressing (differentiable).
    Alternating store/operate schedule (Module 4 theory).
    """

    def __init__(self, config: CRAMConfig):
        super().__init__()
        self.M_s = config.sle_scratchpad_slots
        self.d_prop = config.d_model // config.sle_n_props

        # Learned write gate (how much to overwrite vs blend)
        self.W_gamma = nn.Linear(self.d_prop * 2, self.d_prop, bias=True)
        nn.init.zeros_(self.W_gamma.bias)

        # Read projection (goal-conditioned)
        self.W_read = nn.Linear(self.d_prop * 2, self.M_s, bias=False)

    def init_slots(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize empty scratchpad. [B, M_s, d_prop]"""
        return torch.zeros(batch_size, self.M_s, self.d_prop, device=device, dtype=dtype)

    def soft_write(
        self,
        S: torch.Tensor,           # [B, M_s, d_prop]
        candidate: torch.Tensor,   # [B, d_prop]
        conf: float = 1.0
    ) -> torch.Tensor:
        """
        Soft attention-based write to scratchpad.
        Write to most similar slot, blend with learned gate.
        """
        B = S.shape[0]
        cand_exp = candidate.unsqueeze(1)                     # [B, 1, d_prop]

        # Attention over slots (which slot to write to)
        sim = F.cosine_similarity(cand_exp, S, dim=-1)       # [B, M_s]
        alpha = F.softmax(sim * 4.0, dim=-1)                  # [B, M_s] — sharply peaked

        # Learned blend gate
        best_slot_idx = alpha.argmax(dim=-1)                  # [B]
        best_slots = S[torch.arange(B), best_slot_idx]        # [B, d_prop]
        gamma_input = torch.cat([best_slots, candidate], dim=-1)  # [B, 2*d_prop]
        gamma = torch.sigmoid(self.W_gamma(gamma_input))      # [B, d_prop]

        # Soft update: weighted across all slots
        new_content = gamma * candidate + (1 - gamma) * best_slots  # [B, d_prop]
        alpha_exp = alpha.unsqueeze(-1)                        # [B, M_s, 1]
        S_new = S + conf * alpha_exp * (new_content.unsqueeze(1) - S)

        return S_new

    def read(
        self,
        S: torch.Tensor,           # [B, M_s, d_prop]
        goal: torch.Tensor         # [B, d_prop]  — goal-conditioned readout
    ) -> torch.Tensor:
        """Goal-conditioned weighted readout. Returns [B, d_prop]"""
        B = S.shape[0]
        goal_exp = goal.unsqueeze(1)                           # [B, 1, d_prop]
        sim = F.cosine_similarity(goal_exp, S, dim=-1)        # [B, M_s]
        beta = F.softmax(sim, dim=-1)                          # [B, M_s]
        out = (beta.unsqueeze(-1) * S).sum(1)                 # [B, d_prop]
        return out

    def frobenius_change(self, S_old: torch.Tensor, S_new: torch.Tensor) -> torch.Tensor:
        """
        Normalized Frobenius norm change for convergence detection.
        delta = ||S_new - S_old||_F / (||S_old||_F + eps)
        """
        diff = (S_new - S_old).pow(2).sum(dim=[-1, -2])      # [B]
        norm = S_old.pow(2).sum(dim=[-1, -2]) + 1e-8         # [B]
        return (diff / norm).sqrt()                            # [B]


class InferenceEngine(nn.Module):
    """
    IE: Inference Engine
    Neural forward + backward chaining over proposition slots.

    Forward chaining: FACT + RULE → new FACT (modus ponens)
    Backward chaining: GOAL → subgoals via relevant RULES

    Soft firing thresholds make everything differentiable.
    """

    def __init__(self, config: CRAMConfig):
        super().__init__()
        self.K = config.sle_n_props
        self.d_prop = config.d_model // config.sle_n_props

        # Rule antecedent / consequent projections
        self.W_ante = nn.Linear(self.d_prop, self.d_prop, bias=False)
        self.W_cons = nn.Linear(self.d_prop * 2, self.d_prop, bias=False)

        # Subgoal generation (backward chaining)
        self.W_subgoal = nn.Linear(self.d_prop * 2, self.d_prop, bias=False)

        # Firing threshold (annealed: starts high → conservative)
        self.log_fire_threshold = nn.Parameter(
            torch.tensor(math.log(config.sle_fire_threshold))
        )

    @property
    def fire_threshold(self):
        return torch.exp(self.log_fire_threshold)

    def forward_chain(
        self,
        content: torch.Tensor,    # [B, K, d_prop]
        types: torch.Tensor,      # [B, K, N_TYPES]
        conf: torch.Tensor,       # [B, K]
        S: torch.Tensor,          # [B, M_s, d_prop] scratchpad
        scratchpad: LatentScratchpad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward chaining: apply rules to known facts.
        Produces new derived facts, writes to scratchpad.

        Returns:
            S_new: updated scratchpad
            derived_conf: [B] confidence of new derivation
        """
        B, K, dp = content.shape

        # Extract FACT and RULE propositions (soft masking)
        p_fact = types[:, :, FACT]    # [B, K] — probability of being FACT
        p_rule = types[:, :, RULE]    # [B, K] — probability of being RULE

        best_derived_conf = torch.zeros(B, device=content.device, dtype=content.dtype)
        S_new = S

        for k_rule in range(K):
            rule_weight = p_rule[:, k_rule]    # [B]
            rule_content = content[:, k_rule]  # [B, dp]

            # Compute what this rule needs (antecedent)
            ante = self.W_ante(rule_content)   # [B, dp]

            # Match antecedent against all FACT slots
            ante_exp = ante.unsqueeze(1)       # [B, 1, dp]
            match_scores = F.cosine_similarity(
                ante_exp, content, dim=-1
            )  # [B, K]
            match_scores = match_scores * p_fact  # weight by FACT probability

            # Soft firing: rule fires if antecedent well-matched
            match_strength = (match_scores * conf).sum(-1)  # [B]
            fire = torch.sigmoid(
                (match_strength - self.fire_threshold) / 0.5
            )  # [B] — soft firing in (0,1)
            fire = fire * rule_weight            # scale by rule confidence

            # Derive consequence
            matched_content = (match_scores.unsqueeze(-1) * content).sum(1)  # [B, dp]
            new_fact = self.W_cons(
                torch.cat([rule_content, matched_content], dim=-1)
            )  # [B, dp]

            derived_conf = fire * conf[:, k_rule]

            # Write to scratchpad (weighted by firing strength)
            for b in range(B):
                if fire[b].item() > 0.1:    # only write if meaningfully firing
                    S_new = scratchpad.soft_write(
                        S_new, new_fact[b:b+1].squeeze(0),
                        conf=derived_conf[b].item()
                    )
                    # Update S_new properly (expand back to batch)

            best_derived_conf = torch.maximum(best_derived_conf, derived_conf)

        return S_new, best_derived_conf

    def backward_chain(
        self,
        content: torch.Tensor,    # [B, K, d_prop]
        types: torch.Tensor,      # [B, K, N_TYPES]
        conf: torch.Tensor,       # [B, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward chaining: decompose goals into subgoals.

        Returns:
            subgoals: [B, d_prop] — generated subgoal vector
            goal_conf: [B] — confidence of goal support
        """
        B, K, dp = content.shape

        p_goal = types[:, :, GOAL]    # [B, K]
        p_rule = types[:, :, RULE]    # [B, K]

        # Find most confident goal
        goal_idx = (p_goal * conf).argmax(dim=-1)  # [B]
        goal_content = content[torch.arange(B), goal_idx]  # [B, dp]
        goal_conf_vals = conf[torch.arange(B), goal_idx]   # [B]

        # Find rules relevant to goal (by similarity to consequent space)
        rule_rel = F.cosine_similarity(
            goal_content.unsqueeze(1), content, dim=-1
        )  # [B, K]
        rule_rel = rule_rel * p_rule * conf            # [B, K]

        # Generate subgoal from most relevant rule
        best_rule_idx = rule_rel.argmax(dim=-1)        # [B]
        best_rule = content[torch.arange(B), best_rule_idx]  # [B, dp]

        subgoal = self.W_subgoal(
            torch.cat([goal_content, best_rule], dim=-1)
        )  # [B, dp]

        # Goal support confidence
        goal_support = (rule_rel * conf).sum(-1) / (K + 1e-8)  # [B]

        return subgoal, goal_support


class ConvergenceGate(nn.Module):
    """
    CG: Convergence Gate
    Decides when SLE reasoning has converged — enables early exit.

    Three signals:
    1. delta_i: scratchpad change (Frobenius norm)
    2. conf_goal: best goal confidence
    3. i/I_max: budget pressure

    halt_score = sigmoid(W_h @ [delta; conf_goal; i/I_max])
    HALT if halt_score > 0.9
    """

    def __init__(self, config: CRAMConfig):
        super().__init__()
        self.halt_threshold = config.sle_halt_threshold

        # Halt predictor
        self.W_halt = nn.Linear(3, 1, bias=True)
        # Init bias=-3 so halt_score ≈ 0.05 initially (run full iterations)
        nn.init.constant_(self.W_halt.bias, -3.0)
        nn.init.normal_(self.W_halt.weight, std=0.1)

    def forward(
        self,
        delta: torch.Tensor,       # [B] scratchpad change
        conf_goal: torch.Tensor,   # [B] goal confidence
        step_frac: float           # i / I_max
    ) -> torch.Tensor:
        """
        Returns:
            halt: [B] — halt score in (0, 1)
        """
        step_frac_t = torch.full_like(delta, step_frac)
        gate_input = torch.stack([delta, conf_goal, step_frac_t], dim=-1)  # [B, 3]
        halt = torch.sigmoid(self.W_halt(gate_input)).squeeze(-1)  # [B]
        return halt

    def should_halt(self, halt_scores: torch.Tensor) -> bool:
        """Return True if majority of batch has converged."""
        return (halt_scores > self.halt_threshold).float().mean().item() > 0.5


class SLE(nn.Module):
    """
    Structured Logic Engine — full assembly.

    Applied only to tokens routed to the REASON path by ADR.
    Runs I iterations of alternating store/operate.
    Exits early via convergence gate.
    """

    def __init__(self, config: CRAMConfig):
        super().__init__()
        self.config = config
        self.K = config.sle_n_props
        self.d_prop = config.d_model // config.sle_n_props
        self.I_max = config.sle_max_iters

        # Four subsystems
        self.pe = PropositionEncoder(config)
        self.ls = LatentScratchpad(config)
        self.ie = InferenceEngine(config)
        self.cg = ConvergenceGate(config)

        # Output aggregation
        self.W_out = nn.Linear(
            self.d_prop + config.d_model,   # [S_out; P_summary; h_in]
            config.d_model,
            bias=False
        )
        nn.init.normal_(self.W_out.weight, std=0.02 / math.sqrt(config.n_layers))

        # Confidence projection (for ADR feedback)
        self.W_conf_out = nn.Linear(self.K, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,                      # [B, L, D]
        n_iters: Optional[int] = None         # override I_max (for test-time scaling)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full SLE forward pass.

        Process each position independently (SLE is position-wise for hard tokens).

        Returns:
            y:         [B, L, D] — reasoning-enriched representation
            conf_out:  [B, L]    — output confidence (fed back to ADR)
        """
        B, L, D = h.shape
        I = n_iters or self.I_max
        device, dtype = h.device, h.dtype

        # Process each position (in practice: only P_reason > epsilon positions)
        outputs = []
        confs = []

        for pos in range(L):
            h_pos = h[:, pos, :]    # [B, D]
            y_pos, conf_pos = self._process_token(h_pos, I, device, dtype)
            outputs.append(y_pos)
            confs.append(conf_pos)

        y = torch.stack(outputs, dim=1)     # [B, L, D]
        conf = torch.stack(confs, dim=1)    # [B, L]

        return y, conf

    def _process_token(
        self,
        h: torch.Tensor,       # [B, D]
        I: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single token position through full SLE."""
        B, D = h.shape

        # ── Step 1: Proposition Encoding ──────────────────────────
        content, types, conf = self.pe(h)   # [B,K,d/K], [B,K,5], [B,K]

        # ── Step 2: Initialize Scratchpad ─────────────────────────
        S = self.ls.init_slots(B, device, dtype)  # [B, M_s, d_prop]

        # Goal content (for convergence checking and readout)
        p_goal = types[:, :, GOAL]
        goal_idx = (p_goal * conf).argmax(dim=-1)
        goal_content = content[torch.arange(B), goal_idx]    # [B, d_prop]

        best_conf_goal = torch.zeros(B, device=device, dtype=dtype)

        # ── Step 3: Alternating Store/Operate Iterations ──────────
        for i in range(I):
            S_prev = S.clone()

            if i % 2 == 0:
                # STORE step: forward chain → write new facts
                S, derived_conf = self.ie.forward_chain(
                    content, types, conf, S, self.ls
                )
                best_conf_goal = torch.maximum(best_conf_goal, derived_conf)
            else:
                # OPERATE step: backward chain → generate subgoals
                subgoal, goal_support = self.ie.backward_chain(
                    content, types, conf
                )
                # Write subgoal to scratchpad
                for b in range(B):
                    S = self.ls.soft_write(S, subgoal[b], conf=goal_support[b].item())
                best_conf_goal = torch.maximum(best_conf_goal, goal_support)

            # ── Convergence check ──────────────────────────────────
            delta = self.ls.frobenius_change(S_prev, S)    # [B]
            halt = self.cg(delta, best_conf_goal, (i+1) / I)
            if self.cg.should_halt(halt) and i >= 1:
                break

        # ── Step 4: Output Aggregation ────────────────────────────
        # Goal-conditioned scratchpad readout
        S_out = self.ls.read(S, goal_content)              # [B, d_prop]

        # Proposition summary (confidence-weighted content)
        p_summary = (
            conf.unsqueeze(-1) * content.view(B, self.K, self.d_prop)
        ).sum(1)  # [B, d_prop] — average prop
        # Reduce to single prop dim for concatenation
        p_summary_red = p_summary[:, :self.d_prop]         # [B, d_prop]

        # Fuse: [S_out; h] → d_model
        fused = torch.cat([S_out, h], dim=-1)              # [B, d_prop + D]
        y = self.W_out(fused)                              # [B, D]

        # Output confidence (for ADR feedback)
        conf_out = torch.sigmoid(
            self.W_conf_out(conf)
        ).squeeze(-1)                                       # [B]

        return y, conf_out

    def compute_consistency_loss(
        self,
        content: torch.Tensor,   # [B, K, d_prop]
        types: torch.Tensor,     # [B, K, N_TYPES]
        conf: torch.Tensor,      # [B, K]
        S: torch.Tensor          # [B, M_s, d_prop]
    ) -> torch.Tensor:
        """
        L_consist = ||FC(S_final) - S_final||^2

        The final scratchpad should be closed under forward chaining.
        Running one more step should produce no change.
        This forces genuine logical closure.
        """
        S_after, _ = self.ie.forward_chain(content, types, conf, S, self.ls)
        return F.mse_loss(S_after, S.detach())
