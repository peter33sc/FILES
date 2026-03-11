"""
CRAM Model: Full Architecture Assembly
Assembles RSP + SAMG + ADR + SLE + FFN into complete CRAM blocks,
then stacks them into the full language model.

Architecture per block:
    1. ADR:     estimate difficulty, compute routing probs
    2. RSP:     O(n) dual-timescale recurrence (fast + deep paths)
    3. SAMG:    memory read/write (memory path)
    4. SLE:     structured reasoning (reason path)
    5. FFN:     SwiGLU expansion (always active)

Global properties:
    - SAMG is shared across ALL layers (unified memory)
    - RSP state threads through all layers per sequence
    - Tied input/output embeddings
    - DB-RMSNorm at each module boundary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List

from cram.configs.cram_config import CRAMConfig
from cram.core.rsp import MultiHeadRSP
from cram.core.samg import SAMG
from cram.core.adr import ADR, PATH_NAMES, N_PATHS
from cram.core.sle import SLE
from cram.core.normalization import DBRMSNorm, RMSNorm, SwiGLU


class CRAMBlock(nn.Module):
    """
    Single CRAM block.

    Contains: ADR + RSP (fast/deep) + SAMG (memory path) + SLE (reason path) + FFN

    SAMG is passed in (shared globally).
    RSP state (f, s) is passed in and returned for threading.
    """

    def __init__(self, config: CRAMConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model

        # ── ADR (Adaptive Depth Router) ───────────────────────────
        self.adr = ADR(config)

        # ── RSP (Resonant State Propagation) ──────────────────────
        # Fast path: shallow 1-layer RSP
        self.rsp_fast = MultiHeadRSP(config)
        # Deep path: same RSP but we use multiple applications
        # (In full implementation, deep = more layers; here single layer shared)
        self.rsp_deep = MultiHeadRSP(config)

        # ── SLE (Structured Logic Engine) — reason path ───────────
        self.sle = SLE(config)

        # ── FFN (SwiGLU) — always active ─────────────────────────
        d_ffn = int(config.d_model * config.ffn_multiplier)
        # Make divisible by 8 for hardware efficiency
        d_ffn = (d_ffn + 7) // 8 * 8
        self.ffn = SwiGLU(config.d_model, d_ffn, config.dropout)

        # ── DB-RMSNorm for each module ────────────────────────────
        self.norm_rsp = DBRMSNorm(config.d_model, config.norm_eps, config.residual_beta_init)
        self.norm_mem = DBRMSNorm(config.d_model, config.norm_eps, config.residual_beta_init)
        self.norm_sle = DBRMSNorm(config.d_model, config.norm_eps, config.residual_beta_init)
        self.norm_ffn = DBRMSNorm(config.d_model, config.norm_eps, config.residual_beta_init)

        # ── Wide path: sparse MoE FFN ─────────────────────────────
        # Simplified: multiple independent FFN experts
        n_experts = 4
        self.moe_experts = nn.ModuleList([
            SwiGLU(config.d_model, d_ffn // 2, config.dropout)
            for _ in range(n_experts)
        ])
        self.moe_router = nn.Linear(config.d_model, n_experts, bias=False)

    def wide_path(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse MoE: route to top-2 experts."""
        B, L, D = x.shape
        router_logits = self.moe_router(x)               # [B, L, n_experts]
        router_probs = F.softmax(router_logits, dim=-1)  # [B, L, n_experts]
        # Top-2 experts
        top2_probs, top2_idx = router_probs.topk(2, dim=-1)  # [B, L, 2]
        top2_probs = top2_probs / top2_probs.sum(-1, keepdim=True)  # renormalize

        output = torch.zeros_like(x)
        for e_idx, expert in enumerate(self.moe_experts):
            mask = (top2_idx == e_idx).any(-1)           # [B, L]
            if mask.any():
                e_out = expert(x)                         # [B, L, D]
                # Weight by router probability for this expert
                e_prob = router_probs[:, :, e_idx:e_idx+1]  # [B, L, 1]
                output = output + e_prob * e_out
        return output

    def forward(
        self,
        x: torch.Tensor,                                    # [B, L, D]
        samg: SAMG,                                         # global memory graph
        f_prev: Optional[torch.Tensor] = None,              # [B, H, d_head] fast state
        s_prev: Optional[torch.Tensor] = None,              # [B, H, d_head] slow state
        position_ids: Optional[torch.Tensor] = None,        # [B, L]
        sle_n_iters: Optional[int] = None,                  # override for test-time scaling
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Full CRAM block forward pass.

        Returns:
            output:  [B, L, D]
            f_last:  [B, H, d_head]
            s_last:  [B, H, d_head]
            aux:     dict of auxiliary outputs (losses, stats)
        """
        B, L, D = x.shape
        aux = {}

        # ── Stage 1: Difficulty Routing ────────────────────────────
        x_normed = self.norm_rsp.pre_norm(x)
        difficulty = self.adr.estimate_difficulty(x_normed)       # [B, L, 4]
        routing_probs = self.adr.route(x_normed, difficulty)      # [B, L, N_PATHS]
        aux["routing_probs"] = routing_probs
        aux["difficulty"] = difficulty

        # ── Stage 2: RSP (fast + deep paths) ──────────────────────
        # Fast path
        h_fast, f_last_fast, s_last_fast = self.rsp_fast(
            x_normed, f_prev, s_prev, position_ids
        )
        # Deep path (separate RSP head — learns different dynamics)
        h_deep, f_last_deep, s_last_deep = self.rsp_deep(
            x_normed, f_prev, s_prev, position_ids
        )
        # Wide path (MoE experts)
        h_wide = self.wide_path(x_normed)

        # Blend RSP outputs by routing probs (fast + deep + wide)
        p_fast  = routing_probs[:, :, 0:1]    # [B, L, 1]
        p_deep  = routing_probs[:, :, 1:2]
        p_wide  = routing_probs[:, :, 2:3]
        p_mem   = routing_probs[:, :, 3:4]
        p_rsn   = routing_probs[:, :, 4:5]

        h_rsp = p_fast * h_fast + p_deep * h_deep + p_wide * h_wide
        h_rsp = h_rsp + (p_mem + p_rsn) * h_fast  # memory+reason also use fast base

        # Residual connection
        h_rsp = self.norm_rsp.post_norm_residual(x, h_rsp)

        # Weighted final RSP state (routing-weighted combination)
        f_last = p_fast[:, -1, :].mean(-1, keepdim=True).unsqueeze(-1) * f_last_fast \
               + p_deep[:, -1, :].mean(-1, keepdim=True).unsqueeze(-1) * f_last_deep
        s_last = p_fast[:, -1, :].mean(-1, keepdim=True).unsqueeze(-1) * s_last_fast \
               + p_deep[:, -1, :].mean(-1, keepdim=True).unsqueeze(-1) * s_last_deep

        # ── Stage 3: SAMG Memory (memory path) ────────────────────
        h_mem_path, surprise_gate = samg(h_rsp, do_write=self.training)
        # Apply only to memory-routed fraction
        h_after_mem = h_rsp + p_mem * (h_mem_path - h_rsp)
        aux["surprise_gate"] = surprise_gate
        aux["samg_stats"] = samg.get_graph_stats()

        # ── Stage 4: SLE Reasoning (reason path) ──────────────────
        h_sle, sle_conf = self.sle(h_after_mem, n_iters=sle_n_iters)
        h_after_sle = h_after_mem + p_rsn * self.norm_sle.beta * (
            self.norm_sle.norm_out(h_sle - h_after_mem)
        )
        aux["sle_conf"] = sle_conf

        # ── Stage 5: FFN (always active) ──────────────────────────
        h_ffn_out = self.norm_ffn.forward(h_after_sle, self.ffn)

        # ── ADR auxiliary losses ───────────────────────────────────
        adr_losses = self.adr.compute_routing_losses(routing_probs, difficulty)
        aux["adr_losses"] = adr_losses
        aux["routing_stats"] = self.adr.get_routing_stats(routing_probs)

        return h_ffn_out, f_last, s_last, aux


class CRAMModel(nn.Module):
    """
    Full CRAM Language Model.

    Stacks L CRAMBlocks with:
    - Token embedding (tied with output projection)
    - Global SAMG (shared across all layers)
    - RSP state threading across layers
    - Final RMSNorm + output projection
    """

    def __init__(self, config: CRAMConfig):
        super().__init__()
        self.config = config

        # ── Token Embedding ───────────────────────────────────────
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.token_embedding.weight, std=1.0)  # muP: no 1/√d scaling

        # ── Global SAMG (shared across all layers) ────────────────
        self.samg = SAMG(config)

        # ── CRAM Blocks ───────────────────────────────────────────
        self.blocks = nn.ModuleList([
            CRAMBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # ── Final Normalization ────────────────────────────────────
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        # ── Output Projection (tied to embedding) ─────────────────
        # logits = h_L @ W_embed.T  — weight tying saves ~V*d parameters
        # e.g. 32K vocab * 512 dim = 16M params saved
        self.output_proj = None  # handled via tied weights in forward

        # Track number of parameters
        self._n_params = sum(p.numel() for p in self.parameters())

    def get_num_params(self) -> int:
        return self._n_params

    def forward(
        self,
        input_ids: torch.Tensor,                    # [B, L]
        labels: Optional[torch.Tensor] = None,       # [B, L] for computing loss
        f_states: Optional[List[torch.Tensor]] = None,  # per-layer fast states
        s_states: Optional[List[torch.Tensor]] = None,  # per-layer slow states
        sle_n_iters: Optional[int] = None,           # test-time scaling dial
    ) -> Dict:
        """
        Full forward pass.

        Returns dict containing:
            logits:     [B, L, V]
            loss:       scalar (if labels provided)
            aux_losses: dict of auxiliary losses
            states:     (f_states, s_states) for next chunk
            stats:      monitoring statistics
        """
        B, L = input_ids.shape
        device = input_ids.device

        # ── Token Embedding ───────────────────────────────────────
        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        # Scale by √d (muP parameterization)

        # ── Position IDs ──────────────────────────────────────────
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        # ── Initialize RSP States ─────────────────────────────────
        if f_states is None:
            f_states = [None] * self.config.n_layers
        if s_states is None:
            s_states = [None] * self.config.n_layers

        # ── Forward Through CRAM Blocks ───────────────────────────
        new_f_states = []
        new_s_states = []
        all_aux = []

        for i, block in enumerate(self.blocks):
            x, f_last, s_last, aux = block(
                x,
                samg=self.samg,
                f_prev=f_states[i],
                s_prev=s_states[i],
                position_ids=position_ids,
                sle_n_iters=sle_n_iters,
            )
            new_f_states.append(f_last)
            new_s_states.append(s_last)
            all_aux.append(aux)

        # ── Final Normalization ────────────────────────────────────
        x = self.final_norm(x)

        # ── Output Projection (tied weights) ─────────────────────
        logits = F.linear(x, self.token_embedding.weight)  # [B, L, V]

        # ── Compute Loss ──────────────────────────────────────────
        output = {"logits": logits, "states": (new_f_states, new_s_states)}

        if labels is not None:
            # Standard next-token prediction loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean"
            )
            output["loss"] = task_loss

            # ── Aggregate Auxiliary Losses ─────────────────────────
            aux_losses = self._aggregate_aux_losses(all_aux, task_loss)
            output["aux_losses"] = aux_losses

            # Total loss (weighted sum — lambdas set by curriculum stage)
            cfg = self.config
            total_loss = task_loss
            total_loss = total_loss + cfg.lambda_budget  * aux_losses.get("budget", 0)
            total_loss = total_loss + cfg.lambda_balance * aux_losses.get("balance", 0)
            total_loss = total_loss + cfg.lambda_calib   * aux_losses.get("calib", 0)
            output["total_loss"] = total_loss

        # ── Stats ─────────────────────────────────────────────────
        output["stats"] = self._aggregate_stats(all_aux)

        return output

    def _aggregate_aux_losses(
        self, all_aux: List[Dict], task_loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Average auxiliary losses across layers."""
        aggregated = {}
        for key in ["budget", "balance", "calib"]:
            values = []
            for aux in all_aux:
                if "adr_losses" in aux and key in aux["adr_losses"]:
                    v = aux["adr_losses"][key]
                    if isinstance(v, torch.Tensor):
                        values.append(v)
            if values:
                aggregated[key] = torch.stack(values).mean()
        return aggregated

    def _aggregate_stats(self, all_aux: List[Dict]) -> Dict:
        """Aggregate monitoring statistics across layers."""
        stats = {}
        # Average routing probs across layers
        route_keys = [f"path_{name}_prob" for name in PATH_NAMES]
        for key in route_keys:
            vals = [aux["routing_stats"][key] for aux in all_aux
                   if "routing_stats" in aux and key in aux["routing_stats"]]
            if vals:
                stats[f"avg_{key}"] = sum(vals) / len(vals)

        # SAMG stats from last layer
        if all_aux and "samg_stats" in all_aux[-1]:
            stats["samg"] = all_aux[-1]["samg_stats"]

        return stats

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,         # [B, L]
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        sle_n_iters: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        O(d) per new token — constant cost regardless of context length.
        (vs transformer's O(n*d) growing cost)
        """
        B, L = input_ids.shape
        generated = input_ids.clone()

        f_states = None
        s_states = None

        # Process initial context
        out = self.forward(generated, f_states=f_states, s_states=s_states,
                          sle_n_iters=sle_n_iters)
        f_states, s_states = out["states"]

        for _ in range(max_new_tokens):
            # Only pass the last token (with full RSP state carrying context)
            last_token = generated[:, -1:]    # [B, 1]

            out = self.forward(
                last_token,
                f_states=f_states,
                s_states=s_states,
                sle_n_iters=sle_n_iters,
            )
            f_states, s_states = out["states"]

            logits = out["logits"][:, -1, :]  # [B, V]

            # Temperature + top-k sampling
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                topk_vals = torch.topk(logits, top_k).values[:, -1:]
                logits[logits < topk_vals] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> Dict[str, int]:
        """Detailed parameter count by module."""
        counts = {}
        counts["embedding"] = self.token_embedding.weight.numel()
        counts["samg"] = sum(p.numel() for p in self.samg.parameters())

        per_block_rsp = sum(p.numel() for p in self.blocks[0].rsp_fast.parameters())
        per_block_rsp += sum(p.numel() for p in self.blocks[0].rsp_deep.parameters())
        per_block_adr = sum(p.numel() for p in self.blocks[0].adr.parameters())
        per_block_sle = sum(p.numel() for p in self.blocks[0].sle.parameters())
        per_block_ffn = sum(p.numel() for p in self.blocks[0].ffn.parameters())

        counts["rsp_total"] = per_block_rsp * self.config.n_layers
        counts["adr_total"] = per_block_adr * self.config.n_layers
        counts["sle_total"] = per_block_sle * self.config.n_layers
        counts["ffn_total"] = per_block_ffn * self.config.n_layers
        counts["total"] = sum(counts.values())
        return counts
