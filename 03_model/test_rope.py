import torch
from cs336_workspace.rope import build_rope_cache, apply_rope

def test_position_zero_is_unrotated():
    cos, sin = build_rope_cache(block_size=10, head_dim=8)
    x = torch.randn(1, 1, 10, 8)  # (batch, n_head, seq_len, head_dim)
    out = apply_rope(x, cos, sin)
    # position 0's rotation angle is always 0 -> output should equal input exactly
    assert torch.allclose(out[:, :, 0, :], x[:, :, 0, :], atol=1e-6)

def test_rope_preserves_shape():
    cos, sin = build_rope_cache(block_size=256, head_dim=64)
    x = torch.randn(4, 6, 256, 64)  # matches our real batch/head config
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape

def test_rope_preserves_vector_norm():
    # rotation should never change a vector's length - a strong correctness check,
    # since any bug that leaks magnitude would break this
    cos, sin = build_rope_cache(block_size=16, head_dim=8)
    x = torch.randn(2, 3, 16, 8)
    out = apply_rope(x, cos, sin)
    # compare norm pair-by-pair (each 2D rotation preserves its own pair's norm,
    # so checking the full vector norm per position is a valid aggregate check)
    norm_in = x.norm(dim=-1)
    norm_out = out.norm(dim=-1)
    assert torch.allclose(norm_in, norm_out, atol=1e-4)

def test_relative_position_property():
    # the dot product between a rotated query/key pair should depend only on
    # their POSITION DIFFERENCE, not their absolute positions - this is RoPE's
    # entire reason for existing, worth verifying directly
    cos, sin = build_rope_cache(block_size=20, head_dim=16)
    torch.manual_seed(0)
    q_base = torch.randn(1, 1, 1, 16)
    k_base = torch.randn(1, 1, 1, 16)

    # same q, k content, but placed at two different (pos_q, pos_k) pairs
    # with the SAME relative distance (5)
    q1 = apply_rope(q_base, cos[3:4], sin[3:4])    # position 3
    k1 = apply_rope(k_base, cos[8:9], sin[8:9])    # position 8, distance 5
    dot1 = (q1 * k1).sum()

    q2 = apply_rope(q_base, cos[10:11], sin[10:11])  # position 10
    k2 = apply_rope(k_base, cos[15:16], sin[15:16])  # position 15, distance 5
    dot2 = (q2 * k2).sum()

    assert torch.allclose(dot1, dot2, atol=1e-4)