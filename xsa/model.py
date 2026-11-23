"""nanoGPT-derived pre-norm GPT; see references/nanogpt_LICENSE."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    head_dim: int = 64
    seq_len: int = 512
    attention_type: str = "sa"
    dropout: float = 0.0
    bias: bool = False
    rope_base: float = 10000.0
    projection_eps: float = 1e-12
    projection_precision: str = "float32"
    backend: str = "sdpa"
    alignment_tau: float | None = None

    def __post_init__(self):
        if self.attention_type not in ("sa", "xsa", "alignment_cap"):
            raise ValueError("attention_type must be sa, xsa or alignment_cap")
        if self.attention_type == "alignment_cap":
            if self.alignment_tau is None or not 0 <= self.alignment_tau < 1:
                raise ValueError("alignment_tau must satisfy 0 <= tau < 1")
            if self.projection_precision != "float32":
                raise ValueError("alignment_cap requires float32 projection reductions")
        elif self.alignment_tau is not None:
            raise ValueError("alignment_tau is only valid for alignment_cap")
        if self.head_dim % 2 or min(self.head_dim, self.n_heads, self.d_model, self.n_layers, self.seq_len) <= 0:
            raise ValueError("positive dimensions and even head_dim required")
        if self.projection_precision not in ("float32", "native") or self.projection_eps <= 0:
            raise ValueError("invalid projection precision/epsilon")
        if self.backend not in ("sdpa", "reference"):
            raise ValueError("unknown attention backend")


def project(y, v, eps=1e-12, precision="float32"):
    """Project each aggregate off its own value; preserve float64 references."""
    dtype = y.dtype
    if precision == "float32" and dtype in (torch.float16, torch.bfloat16):
        y, v = y.float(), v.float()
    v_hat = F.normalize(v, dim=-1, eps=eps)
    return (y - (y * v_hat).sum(dim=-1, keepdim=True) * v_hat).to(dtype)


def cap_alignment(y, v, tau, eps=1e-12):
    """Direction A; cosine bound applies when ||v|| >= eps and output is nonzero.

    Match XSA's clamped normalization below eps, including its tau=0 limit.
    Zero v leaves y unchanged. Float64 is retained for numerical references.
    """
    if tau is None or not 0 <= tau < 1:
        raise ValueError("tau must satisfy 0 <= tau < 1")
    if tau == 0:
        return project(y, v, eps, "float32")
    dtype = y.dtype
    if dtype in (torch.float16, torch.bfloat16):
        y, v = y.float(), v.float()
    u = F.normalize(v, dim=-1, eps=eps)
    c = (y * u).sum(dim=-1, keepdim=True)
    r = y - c * u
    b = (tau / math.sqrt(1 - tau * tau)) * r.norm(dim=-1, keepdim=True)
    return (r + c.clamp(min=-b, max=b) * u).to(dtype)


def correct_aggregate(y, v, kind, eps=1e-12, precision="float32", tau=None):
    if kind == "xsa":
        return project(y, v, eps, precision)
    if kind == "alignment_cap":
        return cap_alignment(y, v, tau, eps)
    if kind != "sa":
        raise ValueError(kind)
    return y


def rope(x, offset=0, base=10000.0):
    """Full-head RoPE using adjacent even/odd pairs, Q and K only."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inv = base ** (-torch.arange(0, x.size(-1), 2, device=x.device, dtype=dtype) / x.size(-1))
    angle = torch.arange(offset, offset + x.size(-2), device=x.device, dtype=dtype)[:, None] * inv
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def attention(q, k, v, kind="sa", backend="sdpa", dropout=0.0, eps=1e-12, precision="float32", tau=None):
    if backend == "sdpa":
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout)
    elif backend == "reference":
        mask = torch.ones(q.size(-2), k.size(-2), device=q.device, dtype=torch.bool).tril()
        weights = ((q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))).masked_fill(~mask, -float("inf")).softmax(-1)
        y = F.dropout(weights, dropout, training=dropout > 0) @ v
    else:
        raise ValueError(backend)
    return correct_aggregate(y, v, kind, eps, precision, tau)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.n_heads * cfg.head_dim
        self.c_attn = nn.Linear(cfg.d_model, 3 * self.width, bias=cfg.bias)
        self.c_proj = nn.Linear(self.width, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.diagnostic_sink = None

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = [a.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
                   for a in self.c_attn(x).split(self.width, dim=-1)]
        q, k = rope(q, base=self.cfg.rope_base), rope(k, base=self.cfg.rope_base)
        y = attention(q, k, v, "sa", self.cfg.backend, self.cfg.dropout if self.training else 0.0)
        z = correct_aggregate(y, v, self.cfg.attention_type, self.cfg.projection_eps,
                              self.cfg.projection_precision, self.cfg.alignment_tau)
        if self.diagnostic_sink is not None:
            self.diagnostic_sink(q.detach(), k.detach(), v.detach(), y.detach(), z.detach())
        return self.resid_dropout(self.c_proj(z.transpose(1, 2).contiguous().view(b, t, self.width)))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, eps=1e-5, bias=cfg.bias)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model, eps=1e-5, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.ln_emb = nn.LayerNorm(cfg.d_model, eps=1e-5, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=1e-5, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.size(1) > self.cfg.seq_len:
            raise ValueError("sequence exceeds configured context")
        x = self.drop(self.ln_emb(self.wte(tokens)))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        embedding = self.wte.weight.numel()
        return {"total_unique": total, "token_embedding_tied_head": embedding,
                "non_embedding": total - embedding}
