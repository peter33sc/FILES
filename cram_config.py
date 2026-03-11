"""
CRAM Configuration
All hyperparameters for every module in one place.
Supports the full model family: CRAM-0.4B to CRAM-70B
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CRAMConfig:
    # ── Model dimensions ──────────────────────────────────────────────
    d_model: int = 512              # Hidden dimension
    n_layers: int = 16             # Number of CRAM blocks
    n_heads: int = 8               # RSP multi-head count
    d_head: int = 64               # Head dimension (d_model / n_heads)
    vocab_size: int = 32000        # Vocabulary size
    max_seq_len: int = 8192        # Maximum sequence length for training

    # ── RSP (Resonant State Propagation) ──────────────────────────────
    rsp_fast_dim: int = 512        # Fast state dimension (= d_model)
    rsp_slow_dim: int = 512        # Slow state dimension (= d_model)
    rsp_dt_min: float = 0.001      # Min value for Λ (prevents pure passthrough)
    rsp_dt_max: float = 0.1        # Max value for Λ (prevents zero forgetting)
    rsp_init_bias_rho: float = 2.0 # Initial resonance gate bias (trust fast state)
    rsp_init_bias_alpha: float = -2.0  # Initial slow rate bias (slow is slow)

    # ── RTE (Resonant Temporal Embedding) ────────────────────────────
    rte_base: int = 500_000        # Frequency base (extended RoPE)
    rte_phase_scale: float = 1.0   # Scale for input-dependent phase

    # ── SAMG (Sparse Associative Memory Graph) ────────────────────────
    samg_nodes: int = 8192         # Max graph nodes M
    samg_d_key: int = 128          # Key dimension dk
    samg_d_val: int = 512          # Value dimension dv (= d_model)
    samg_top_k: int = 16           # Top-K nodes for retrieval
    samg_n_hops: int = 2           # Graph traversal hops
    samg_surprise_tau: float = 0.9 # Initial surprise threshold (annealed)
    samg_surprise_temp: float = 0.1 # Surprise gate temperature
    samg_prune_threshold: float = 0.01  # Prune nodes below this score
    samg_edge_max: int = 8         # Max edges per node

    # ── ADR (Adaptive Depth Router) ───────────────────────────────────
    adr_r_dim: int = 64            # CDE projection dim (d_model / 8)
    adr_n_paths: int = 5           # [fast, deep, wide, memory, reason]
    adr_temp_init: float = 2.0     # Initial routing temperature
    adr_temp_final: float = 0.5    # Final routing temperature
    adr_budget_target: float = 0.35 # Target FLOP fraction
    adr_epsilon: float = 0.05      # Min path probability (no dead paths)

    # ── SLE (Structured Logic Engine) ────────────────────────────────
    sle_n_props: int = 8           # Number of proposition slots K
    sle_n_types: int = 5           # Proposition types: FACT/RULE/GOAL/CONSTRAINT/UNK
    sle_scratchpad_slots: int = 16 # Working memory slots M_s
    sle_max_iters: int = 4         # Maximum IE iterations I_max
    sle_fire_threshold: float = 2.0 # Initial rule firing threshold
    sle_halt_threshold: float = 0.9 # Convergence gate halt threshold

    # ── FFN (SwiGLU Feed-Forward) ─────────────────────────────────────
    ffn_multiplier: float = 8/3    # SwiGLU expansion ratio
    ffn_dropout: float = 0.0       # Dropout (0 for large scale training)

    # ── Normalization ─────────────────────────────────────────────────
    norm_eps: float = 1e-6         # RMSNorm epsilon
    residual_beta_init: float = 0.1 # Initial residual scale β

    # ── Training ──────────────────────────────────────────────────────
    dropout: float = 0.0
    weight_decay: float = 0.1
    lr_peak: float = 2e-2          # AdaMuon peak LR
    lr_adamw: float = 3e-4         # AdamW LR for embeddings/1D params
    warmup_tokens: int = 2_000_000_000  # 2B tokens warmup
    grad_clip: float = 1.0

    # ── Loss weights (staged — start at 0, activated by curriculum) ───
    lambda_budget: float = 0.0
    lambda_balance: float = 0.0
    lambda_calib: float = 0.0
    lambda_consist: float = 0.0
    lambda_conf: float = 0.0
    lambda_graph: float = 0.0

    # ── Device ────────────────────────────────────────────────────────
    device: str = "cuda"
    dtype: str = "bfloat16"        # bfloat16 for training stability

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        self.d_head = self.d_model // self.n_heads
        self.adr_r_dim = self.d_model // 8
        self.rsp_fast_dim = self.d_model
        self.rsp_slow_dim = self.d_model
        self.samg_d_val = self.d_model

    @classmethod
    def cram_400m(cls):
        return cls(d_model=512, n_layers=16, n_heads=8, samg_nodes=8192)

    @classmethod
    def cram_1b(cls):
        return cls(d_model=768, n_layers=20, n_heads=12, samg_nodes=16384)

    @classmethod
    def cram_3b(cls):
        return cls(d_model=1024, n_layers=28, n_heads=16, samg_nodes=32768)

    @classmethod
    def cram_7b(cls):
        return cls(d_model=2048, n_layers=32, n_heads=16, samg_nodes=65536,
                   d_head=128)

    @classmethod
    def cram_debug(cls):
        """Tiny model for fast unit testing"""
        return cls(
            d_model=64, n_layers=2, n_heads=4, vocab_size=256,
            max_seq_len=128, samg_nodes=64, sle_n_props=4,
            sle_scratchpad_slots=4, sle_max_iters=2,
            adr_r_dim=8, ffn_multiplier=2.0
        )
