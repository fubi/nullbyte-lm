import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1, multiple_of: int = 64):
        super().__init__()
        # standard scaling: 2/3 of the usual 4x expansion, since we have 3 matrices not 2
        hidden_dim = int(2 * (4 * n_embd) / 3)
        # round up to a multiple of `multiple_of` for clean tensor shapes
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        out = self.w_down(gate * up)
        return self.dropout(out)