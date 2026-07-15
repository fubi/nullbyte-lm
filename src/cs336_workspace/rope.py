import torch

def build_rope_cache(block_size: int, head_dim: int, base: float = 10000.0):
    """
    Precompute cos/sin rotation values for every (position, dimension-pair) combo.
    Returns cos, sin each of shape (block_size, head_dim // 2).
    """
    assert head_dim % 2 == 0, "RoPE requires an even head dimension"

    # theta_i for i = 0 .. head_dim/2 - 1
    i = torch.arange(0, head_dim, 2).float()  # [0, 2, 4, ..., head_dim-2]
    theta = base ** (-i / head_dim)           # shape: (head_dim/2,)

    positions = torch.arange(block_size).float()  # [0, 1, ..., block_size-1]

    # outer product: angle for every (position, freq) pair
    angles = torch.outer(positions, theta)  # shape: (block_size, head_dim/2)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to x.
    x shape:   (batch, n_head, seq_len, head_dim)
    cos, sin:  (seq_len, head_dim // 2) - sliced from the precomputed cache
    """
    # split the last dim into even/odd pairs: x1 = x[...,0::2], x2 = x[...,1::2]
    x1 = x[..., 0::2]  # shape: (batch, n_head, seq_len, head_dim/2)
    x2 = x[..., 1::2]

    # broadcast cos/sin across batch and n_head dims
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # the 2D rotation, applied to every pair at once
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    # interleave x1_rot and x2_rot back into the original even/odd positions
    out = torch.empty_like(x)
    out[..., 0::2] = x1_rot
    out[..., 1::2] = x2_rot
    return out