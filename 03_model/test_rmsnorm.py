import torch
from cs336_workspace.rmsnorm import RMSNorm

def test_rmsnorm_output_has_unit_rms_at_init():
    norm = RMSNorm(dim=8)
    x = torch.randn(2, 5, 8)
    out = norm(x)
    rms_out = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=1e-4)

def test_rmsnorm_preserves_shape():
    norm = RMSNorm(dim=384)
    x = torch.randn(4, 256, 384)
    out = norm(x)
    assert out.shape == x.shape

def test_rmsnorm_zero_input_does_not_nan():
    norm = RMSNorm(dim=8)
    x = torch.zeros(1, 1, 8)
    out = norm(x)
    assert not torch.isnan(out).any()