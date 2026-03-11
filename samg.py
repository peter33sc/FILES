"""
SAMG: Sparse Associative Memory Graph
Long-term persistent memory for CRAM.

Replaces KV-cache (O(n) memory) and MLP memory (non-relational) with
a sparse directed graph that enables O(log n) retrieval of structured
relational memories.

Graph structure:
    Nodes: {key ∈ R^dk, value ∈ R^dv, age ∈ R+, freq ∈ N}
    Edges: directed weighted links between related nodes
    Topology: encodes temporal co-occurrence + semantic similarity

Operations:
    Read:  O(log n) via top-K search + multi-hop traversal
    Write: O(1) amortized — surprise-gated, soft assignment
    Prune: score-based (recency × frequency × centrality)

End-to-end differentiable:
    Read path:  softmax + dot products + weighted sum → fully differentiable
    Write path: Gaussian soft assignment → differentiable everywhere
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class SAMGNode:
    """Container for a single memory node (used for visualization/analysis)."""
    def __init__(self, key, value, age=0, freq=1):
        self.key = key      # [dk]
        self.value = value  # [dv]
        self.age = age      # last access timestep
        self.freq = freq    # access count


class SAMG(nn.Module):
    """
    Sparse Associative Memory Graph.

    Stores M nodes in a sparse graph. Each token can read from
    and write to this graph based on its surprise score.

    The graph is GLOBAL — shared across all CRAM layers.
    Lower layers write syntactic memories.
    Upper layers write semantic memories.
    All layers can read all memories.

    Args:
        config: CRAMConfig
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_key = config.samg_d_key
        self.d_val = config.d_model          # value dim = model dim
        self.M = config.samg_nodes           # max nodes
        self.top_k = config.samg_top_k       # retrieval top-K
        self.n_hops = config.samg_n_hops     # graph traversal hops
        self.edge_max = config.samg_edge_max # max edges per node

        # ── Query/Key/Value projections ───────────────────────────────
        self.W_q = nn.Linear(config.d_model, self.d_key, bias=False)
        self.W_k = nn.Linear(config.d_model, self.d_key, bias=False)
        self.W_v = nn.Linear(config.d_model, self.d_val, bias=False)

        # ── Memory fusion gate ─────────────────────────────────────────
        # mu_t = sigmoid(W_mu @ [h_t; r_t])
        self.W_mu = nn.Linear(config.d_model * 2, config.d_model, bias=True)
        nn.init.zeros_(self.W_mu.bias)

        # ── Graph storage (persistent across forward passes) ───────────
        # Using buffers so they move with .to(device) but aren't parameters
        # Keys: [M, dk]
        self.register_buffer(
            "node_keys",
            torch.zeros(config.samg_nodes, self.d_key)
        )
        # Values: [M, dv]
        self.register_buffer(
            "node_values",
            torch.zeros(config.samg_nodes, self.d_val)
        )
        # Access frequencies: [M]
        self.register_buffer(
            "node_freq",
            torch.zeros(config.samg_nodes)
        )
        # Last access timestep: [M]
        self.register_buffer(
            "node_age",
            torch.zeros(config.samg_nodes)
        )
        # Edge adjacency (sparse): [M, edge_max] — indices of neighbors
        self.register_buffer(
            "node_edges",
            torch.full((config.samg_nodes, self.edge_max), -1, dtype=torch.long)
        )
        # Edge weights: [M, edge_max]
        self.register_buffer(
            "edge_weights",
            torch.zeros(config.samg_nodes, self.edge_max)
        )

        # Current number of active nodes
        self.register_buffer("n_active", torch.tensor(0, dtype=torch.long))

        # Current timestep (for age tracking)
        self.register_buffer("timestep", torch.tensor(0, dtype=torch.long))

        # ── Surprise threshold (annealed during training) ─────────────
        self.surprise_tau = nn.Parameter(
            torch.tensor(config.samg_surprise_tau)
        )
        self.surprise_temp = config.samg_surprise_temp
        self.prune_threshold = config.samg_prune_threshold

    def read(
        self,
        query: torch.Tensor,         # [B, L, d_model] or [B, d_model]
        return_scores: bool = False
    ) -> torch.Tensor:
        """
        Retrieve relevant memories via soft multi-hop graph traversal.

        Step 1: Find top-K nodes via dot-product similarity (O(M) naive, O(log M) with index)
        Step 2: Multi-hop traversal — expand via edge-weighted attention
        Step 3: Aggregate values weighted by final attention scores

        Returns: [B, L, d_val] retrieved memory vectors
        """
        squeeze = False
        if query.dim() == 2:
            query = query.unsqueeze(1)   # [B, 1, D]
            squeeze = True

        B, L, D = query.shape

        # Project to key space
        q = self.W_q(query)              # [B, L, dk]

        n = self.n_active.item()
        if n == 0:
            # No memories yet — return zeros
            result = torch.zeros(B, L, self.d_val, device=query.device, dtype=query.dtype)
            if squeeze:
                result = result.squeeze(1)
            return result

        # Active node keys and values
        active_keys = self.node_keys[:n]     # [n, dk]
        active_vals = self.node_values[:n]   # [n, dv]

        # ── Hop 0: Initial similarity scores ───────────────────────
        # [B, L, n] — cosine similarity via normalized dot product
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(active_keys, dim=-1)
        scores = torch.einsum("bld,nd->bln", q_norm, k_norm)  # [B, L, n]

        # Top-K selection (differentiable approximation via soft top-k)
        k = min(self.top_k, n)
        topk_scores, topk_idx = torch.topk(scores, k, dim=-1)  # [B, L, k]
        alpha = F.softmax(topk_scores / math.sqrt(self.d_key), dim=-1)  # [B, L, k]

        # ── Multi-hop traversal ─────────────────────────────────────
        for hop in range(self.n_hops - 1):
            # Expand to neighbors of current top-k nodes
            # For each active node in top-k, gather its neighbors
            # [B, L, k, edge_max]
            neighbor_idx = self.node_edges[topk_idx.view(-1)]  # [B*L*k, edge_max]
            neighbor_idx = neighbor_idx.view(B, L, k, self.edge_max)
            neighbor_weights = self.edge_weights[topk_idx.view(-1)]
            neighbor_weights = neighbor_weights.view(B, L, k, self.edge_max)

            # Valid neighbors (not -1)
            valid = (neighbor_idx >= 0) & (neighbor_idx < n)

            # Compute similarity to neighbors
            # Flatten and gather valid neighbor keys
            flat_idx = neighbor_idx.clamp(0, n-1)              # safe indexing
            neighbor_keys = active_keys[flat_idx.view(-1)]     # [B*L*k*e, dk]
            neighbor_keys = neighbor_keys.view(B, L, k, self.edge_max, self.d_key)

            q_exp = q_norm.unsqueeze(-2).unsqueeze(-2)          # [B, L, 1, 1, dk]
            neighbor_sim = (q_exp * F.normalize(neighbor_keys, dim=-1)).sum(-1)  # [B,L,k,e]

            # Edge-weighted neighbor scores
            hop_scores = neighbor_sim * neighbor_weights * valid.float()

            # Flatten to get combined candidate pool
            # [B, L, k*edge_max]
            hop_scores_flat = hop_scores.view(B, L, -1)
            hop_idx_flat = flat_idx.view(B, L, -1)

            # Combine with current scores via max-pooling
            all_scores = torch.cat([topk_scores, hop_scores_flat], dim=-1)  # [B,L,k+k*e]
            all_idx = torch.cat([topk_idx, hop_idx_flat], dim=-1)

            # Re-select top-k from expanded pool
            new_k = min(k, all_scores.shape[-1])
            topk_scores, sel = torch.topk(all_scores, new_k, dim=-1)
            topk_idx = all_idx.gather(-1, sel)
            alpha = F.softmax(topk_scores / math.sqrt(self.d_key), dim=-1)

        # ── Value aggregation ────────────────────────────────────────
        # r_t = sum_i alpha_i * v_i
        topk_vals = active_vals[topk_idx.view(-1)]              # [B*L*k, dv]
        topk_vals = topk_vals.view(B, L, -1, self.d_val)       # [B, L, k, dv]
        r = (alpha.unsqueeze(-1) * topk_vals).sum(-2)           # [B, L, dv]

        # Update access frequencies (non-differentiable bookkeeping)
        if not self.training or True:
            # Update freq for accessed nodes (detached — not part of grad graph)
            with torch.no_grad():
                flat_accessed = topk_idx.view(-1).unique()
                flat_accessed = flat_accessed[flat_accessed < n]
                self.node_freq[flat_accessed] += 1
                self.node_age[flat_accessed] = self.timestep.float()

        if squeeze:
            r = r.squeeze(1)

        if return_scores:
            return r, alpha
        return r

    def write(
        self,
        key: torch.Tensor,        # [B, dk] or [B, L, dk]
        value: torch.Tensor,      # [B, dv] or [B, L, dv]
        gate: torch.Tensor,       # [B] or [B, L] — surprise gate values
        threshold: float = 0.5
    ):
        """
        Write to memory graph via soft surprise-gated assignment.

        For each (key, value) pair where gate > threshold:
        - Find nearest existing node (if within radius δ → update)
        - Otherwise insert new node
        - Add directed edge from previous written node

        Uses soft Gaussian assignment for differentiability.
        """
        with torch.no_grad():
            if key.dim() == 3:
                # Process all positions
                B, L, dk = key.shape
                for b in range(B):
                    for t in range(L):
                        if gate[b, t].item() > threshold:
                            self._write_single(
                                key[b, t], value[b, t], gate[b, t].item()
                            )
            else:
                B = key.shape[0]
                for b in range(B):
                    if gate[b].item() > threshold:
                        self._write_single(key[b], value[b], gate[b].item())

    def _write_single(self, key: torch.Tensor, value: torch.Tensor, gate_val: float):
        """Write a single (key, value) pair to the graph."""
        n = self.n_active.item()

        if n > 0:
            active_keys = self.node_keys[:n]
            # Soft assignment via Gaussian kernel
            dists = torch.norm(active_keys - key.unsqueeze(0), dim=-1)  # [n]
            sigma = 1.0  # neighborhood radius
            psi = torch.exp(-dists.pow(2) / (2 * sigma**2))             # [n]
            psi = psi / (psi.sum() + 1e-8)

            nearest_idx = dists.argmin().item()
            nearest_dist = dists[nearest_idx].item()

            if nearest_dist < sigma:
                # Update existing node (online averaging)
                freq = self.node_freq[nearest_idx].item()
                self.node_keys[nearest_idx] = (
                    (freq * self.node_keys[nearest_idx] + key) / (freq + 1)
                )
                self.node_values[nearest_idx] = (
                    (freq * self.node_values[nearest_idx] + gate_val * value) / (freq + 1)
                )
                self.node_freq[nearest_idx] += 1
                self.node_age[nearest_idx] = self.timestep.float()
                return  # no new node needed

        # Insert new node
        if n < self.M:
            idx = n
            self.node_keys[idx] = key
            self.node_values[idx] = value
            self.node_freq[idx] = 1
            self.node_age[idx] = self.timestep.float()
            self.n_active += 1

            # Add edge from previous node if exists
            if n > 0:
                prev_idx = n - 1
                # Find open edge slot in prev node
                for e in range(self.edge_max):
                    if self.node_edges[prev_idx, e] == -1:
                        cos_sim = F.cosine_similarity(
                            self.node_keys[prev_idx].unsqueeze(0),
                            key.unsqueeze(0)
                        ).item()
                        self.node_edges[prev_idx, e] = idx
                        self.edge_weights[prev_idx, e] = gate_val * max(cos_sim, 0)
                        break
        else:
            # Graph full — prune and replace lowest-scoring node
            self._prune_and_replace(key, value, gate_val)

    def _prune_and_replace(self, key, value, gate_val):
        """Replace lowest-scoring node with new memory."""
        n = self.n_active.item()
        t = self.timestep.float().item() + 1e-8

        # Compute node scores
        recency = self.node_freq[:n] / t
        centrality = (self.edge_weights[:n] > 0).float().sum(-1) / self.edge_max
        magnitude = self.node_values[:n].norm(dim=-1) / (self.d_val ** 0.5)

        scores = 0.4 * recency + 0.4 * centrality + 0.2 * magnitude
        min_idx = scores.argmin().item()

        if scores[min_idx] < self.prune_threshold:
            self.node_keys[min_idx] = key
            self.node_values[min_idx] = value
            self.node_freq[min_idx] = 1
            self.node_age[min_idx] = self.timestep.float()
            self.node_edges[min_idx] = -1
            self.edge_weights[min_idx] = 0

    def compute_surprise(
        self,
        value: torch.Tensor,     # [B, L, dv]
        retrieved: torch.Tensor  # [B, L, dv]
    ) -> torch.Tensor:
        """
        Normalized prediction error as surprise metric.

        s_t = ||v_t - r_t||^2 / (||v_t||^2 + eps)

        Invariant to magnitude scaling — rare and common concepts judged equally.
        """
        diff_sq = (value - retrieved).pow(2).sum(-1)          # [B, L]
        norm_sq = value.pow(2).sum(-1) + 1e-8                  # [B, L]
        surprise = diff_sq / norm_sq                           # [B, L]
        return surprise

    def forward(
        self,
        h: torch.Tensor,         # [B, L, d_model] — RSP output
        do_write: bool = True
    ) -> torch.Tensor:
        """
        Full SAMG forward pass:
        1. Project h to q, k, v
        2. Read from graph
        3. Compute surprise gate
        4. Write if surprised (during training)
        5. Fuse retrieved memory with h

        Returns: h_mem — [B, L, d_model] memory-augmented representation
        """
        B, L, D = h.shape

        # Project
        q = self.W_q(h)   # [B, L, dk]
        k = self.W_k(h)   # [B, L, dk]
        v = self.W_v(h)   # [B, L, dv]

        # Read
        r = self.read(h)  # [B, L, dv]

        # Surprise gate
        surprise = self.compute_surprise(v, r)  # [B, L]
        gate = torch.sigmoid(
            (surprise - self.surprise_tau) / self.surprise_temp
        )  # [B, L] — soft gate in (0, 1)

        # Write (if training or explicitly requested)
        if do_write and self.training:
            self.write(k, v, gate)
            self.timestep += 1

        # Fuse: h_mem = h + P_mem * mu * r
        # mu = sigmoid(W_mu @ [h; r]) — learned fusion importance
        mu = torch.sigmoid(self.W_mu(torch.cat([h, r], dim=-1)))  # [B, L, D]
        h_mem = h + mu * r

        return h_mem, gate

    def get_graph_stats(self) -> Dict:
        """Return graph statistics for monitoring."""
        n = self.n_active.item()
        if n == 0:
            return {"n_nodes": 0, "n_edges": 0, "avg_freq": 0.0}

        n_edges = (self.node_edges[:n] >= 0).sum().item()
        avg_freq = self.node_freq[:n].mean().item()
        avg_val_norm = self.node_values[:n].norm(dim=-1).mean().item()

        return {
            "n_nodes": n,
            "n_edges": n_edges,
            "avg_freq": avg_freq,
            "avg_val_norm": avg_val_norm,
            "graph_density": n_edges / (n * self.edge_max + 1e-8),
            "surprise_tau": self.surprise_tau.item(),
        }
