import torch
from cs336_workspace.block import TransformerBlock
from cs336_workspace.rope import build_rope_cache

def test_output_shape_matches_input():
    block = TransformerBlock(n_embd=384, n_head=6, block_size=256)
    cos, sin = build_rope_cache(block_size=256, head_dim=384 // 6)
    x = torch.randn(4, 256, 384)
    out = block(x, cos, sin)
    assert out.shape == x.shape

def test_residual_connection_present():
    # if we zero out attention AND ffn output entirely, the block's output
    # should equal the input exactly (pure residual passthrough) - a direct
    # check that the "x + ..." structure is really there, not accidentally
    # replaced with a non-residual assignment
    block = TransformerBlock(n_embd=16, n_head=2, block_size=8, dropout=0.0)
    block.eval()
    cos, sin = build_rope_cache(block_size=8, head_dim=16 // 2)

    # zero out every weight in attn and ffn so their outputs are exactly zero
    with torch.no_grad():
        for p in block.attn.parameters():
            p.zero_()
        for p in block.ffn.parameters():
            p.zero_()

    x = torch.randn(1, 8, 16)
    out = block(x, cos, sin)
    assert torch.allclose(out, x, atol=1e-5)

def test_gradients_flow_to_all_components():
    # a real end-to-end check that nothing is accidentally detached from
    # the computation graph (a common from-scratch bug: forgetting a
    # connection, silently stopping gradient flow to some parameters)
    block = TransformerBlock(n_embd=32, n_head=4, block_size=16)
    cos, sin = build_rope_cache(block_size=16, head_dim=32 // 4)
    x = torch.randn(2, 16, 32, requires_grad=True)

    out = block(x, cos, sin)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"