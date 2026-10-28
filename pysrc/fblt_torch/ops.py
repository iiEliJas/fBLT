import torch
import torch.nn.functional as F


def rmsnorm(x, weight, eps=1e-6):
    # x: (..., D)
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * weight


def rope_precompute(max_seq_len, head_dim, theta, device, dtype):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    pos = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(pos, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x: (S, H, D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


def swiglu(gate, up):
    return F.silu(gate) * up
