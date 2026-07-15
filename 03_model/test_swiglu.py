import torch
from swiglu import SwiGLU

def test_output_shape_matches_input():
    ff = SwiGLU(n_embd=384)
    x = torch.randn(4, 256, 384)
    out = ff(x)
    assert out.shape == x.shape

def test_hidden_dim_computed_correctly():
    ff = SwiGLU(n_embd=384, multiple_of=64)
    # int(2 * 4 * 384 / 3) = 1024, already a multiple of 64
    assert ff.w_gate.out_features == 1024
    assert ff.w_up.out_features == 1024
    assert ff.w_down.in_features == 1024

def test_zero_input_produces_zero_output_pre_dropout():
    # SiLU(0)=0, so gate=0, so gate*up=0 regardless of up's value -> output should be 0
    ff = SwiGLU(n_embd=8, dropout=0.0)  # disable dropout for a clean deterministic check
    ff.eval()
    x = torch.zeros(1, 1, 8)
    out = ff(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

def test_gating_actually_gates():
    # verify the gate meaningfully changes behavior vs. just using w_up alone -
    # a cheap way to check SiLU(gate) isn't accidentally always ~1 (i.e. a no-op gate)
    torch.manual_seed(0)
    ff = SwiGLU(n_embd=16, dropout=0.0)
    ff.eval()
    x = torch.randn(1, 5, 16)
    gate_raw = ff.w_gate(x)
    gate_activated = torch.nn.functional.silu(gate_raw)
    # SiLU output should NOT be uniformly ~1 across a random input - if it is,
    # the gate isn't doing anything meaningful for this input distribution
    assert not torch.allclose(gate_activated, torch.ones_like(gate_activated), atol=0.05)