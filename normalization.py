"""
DB-RMSNorm: Dual-Boundary RMSNorm
Normalizes both input AND output of each CRAM module.
Combines Pre-LN stability with Post-LN bounded activations.

Math:
    x_in  = RMSNorm(x)                  [input boundary]
    y_raw = Module(x_in)
    y_out = RMSNorm(y_raw) * alpha       [output boundary, learned scale]
    output = x + beta * y_out            [residual, beta initialized small]
"""

import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Faster than LayerNorm — no mean subtraction, no bias.
    Used in LLaMA, Mistral, Gemma.

    RMSNorm(x) = x / RMS(x) * gamma
    RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.gamma

    def extra_repr(self):
        return f"d_model={self.gamma.shape[0]}, eps={self.eps}"


class DBRMSNorm(nn.Module):
    """
    Dual-Boundary RMSNorm for CRAM blocks.

    Wraps any module with:
        1. Input normalization  (Pre-LN benefit: stable gradients)
        2. Output normalization (Post-LN benefit: bounded activations)
        3. Learned residual scale beta (initialized small = near-identity start)

    Usage:
        norm = DBRMSNorm(d_model, module)
        output = norm(x)   # handles everything internally
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        beta_init: float = 0.1
    ):
        super().__init__()
        self.norm_in = RMSNorm(d_model, eps)
        self.norm_out = RMSNorm(d_model, eps)

        # Per-dimension learned residual scale
        # Init small → each block starts as near-identity transformation
        # Network learns to amplify only when useful
        self.beta = nn.Parameter(torch.full((d_model,), beta_init))

    def pre_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply input boundary normalization."""
        return self.norm_in(x)

    def post_norm_residual(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply output boundary normalization and add residual.
        x = original input, y = module output
        Returns: x + beta * RMSNorm(y)
        """
        y_normed = self.norm_out(y)
        return x + self.beta * y_normed

    def forward(self, x: torch.Tensor, module_fn) -> torch.Tensor:
        """
        Full DB-RMSNorm pass.
        module_fn: callable that takes normalized x and returns y
        """
        x_normed = self.pre_norm(x)
        y = module_fn(x_normed)
        return self.post_norm_residual(x, y)


class ResonantActivation(nn.Module):
    """
    Resonant Activation (RA) — custom activation for RSP layers.

    phi_RA(x) = x * sigmoid(x) * (1 + 0.1 * sin(pi * x))

    Properties:
    - Swish-like (x * sigmoid(x)) for smooth, non-saturating base
    - Harmonic term (sin(pi*x)) prevents flat gradient regions
    - 0.1 coefficient keeps it near-swish for stability
    - Always nonzero gradient for finite x

    Gradient:
    phi'(x) = sigmoid(x)(1 + x(1-sigmoid(x))) * (1 + 0.1*sin(pi*x))
             + x*sigmoid(x) * 0.1*pi*cos(pi*x)
    """

    def __init__(self, harmonic_scale: float = 0.1):
        super().__init__()
        self.harmonic_scale = harmonic_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = x * torch.sigmoid(x)
        harmonic = 1.0 + self.harmonic_scale * torch.sin(math.pi * x)
        return swish * harmonic

    def extra_repr(self):
        return f"harmonic_scale={self.harmonic_scale}"


class SwiGLU(nn.Module):
    """
    SwiGLU activation for FFN layers.
    Used in LLaMA, PaLM, Gemma — empirically ~0.5 PPL better than ReLU FFN.

    SwiGLU(x, W1, Wg, W2) = (GeLU(x @ W1) * (x @ Wg)) @ W2

    Implements as full FFN block with SwiGLU gating.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        # Two parallel projections (gate + transform)
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)   # transform
        self.wg = nn.Linear(d_model, d_ffn, bias=False)   # gate
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)   # output
        self.dropout = nn.Dropout(dropout)

        # Initialize output projection small (residual scaling)
        nn.init.normal_(self.w2.weight, std=0.02 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        gate = torch.nn.functional.gelu(self.w1(x))
        transform = self.wg(x)
        hidden = gate * transform          # element-wise gating
        hidden = self.dropout(hidden)
        return self.w2(hidden)
