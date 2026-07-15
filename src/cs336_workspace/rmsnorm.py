import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # learned scale, init to 1 (identity at start)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim) - works for any leading dims (batch, sequence, etc.)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma