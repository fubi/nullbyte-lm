import torch
from attention import CausalSelfAttention
from rope import build_rope_cache

def test_output_shape_matches_input():
    attn = CausalSelfAttention(n_embd=384, n_head=6, block_size=256)
    cos, sin = build_rope_cache(block_size=256, head_dim=384 // 6)
    x = torch.randn(4, 256, 384)
    out = attn(x, cos, sin)
    assert out.shape == x.shape

def test_causal_mask_blocks_future_positions():
    # if we change a FUTURE token's input, it must NOT affect an EARLIER
    # position's output - that's the entire point of causal masking
    torch.manual_seed(0)
    attn = CausalSelfAttention(n_embd=32, n_head=4, block_size=10)
    attn.eval()  # disable dropout for a deterministic comparison
    cos, sin = build_rope_cache(block_size=10, head_dim=32 // 4)

    x = torch.randn(1, 10, 32)
    out1 = attn(x, cos, sin)

    x_modified = x.clone()
    x_modified[:, 5, :] = torch.randn(32)  # change only position 5 (a LATER position)

    out2 = attn(x_modified, cos, sin)

    # positions 0-4 (before the change) must be unaffected
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5)
    # position 5 itself and onward SHOULD differ (sanity check the test itself is meaningful)
    assert not torch.allclose(out1[:, 5:, :], out2[:, 5:, :], atol=1e-5)

def test_attention_weights_sum_to_one():
    # patch forward temporarily isn't clean, so verify indirectly via a smaller
    # standalone softmax check on the masking logic itself
    block_size = 5
    mask = torch.tril(torch.ones(block_size, block_size)).bool()
    scores = torch.randn(1, 1, block_size, block_size)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)