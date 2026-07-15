import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import apply_rope

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # combined qkv projection (one matmul instead of three - standard efficiency trick)
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        # causal mask: lower-triangular, precomputed once, reused every forward pass
        mask = torch.tril(torch.ones(block_size, block_size)).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, n_embd

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)  # each (B, T, C)

        # reshape into heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # apply RoPE to q and k (NOT v)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # scaled dot-product attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_head, T, T)

        # apply causal mask: positions that can't be attended to get -inf
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v  # (B, n_head, T, head_dim)

        # merge heads back: (B, n_head, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)