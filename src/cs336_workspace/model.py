import torch
import torch.nn as nn
from rmsnorm import RMSNorm
from rope import build_rope_cache
from block import TransformerBlock

class TinyStoriesLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 7170,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        block_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        head_dim = n_embd // n_head

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        self.norm_final = RMSNorm(n_embd)
        self.output_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying: output head reuses the SAME weight matrix as the input
        # embedding, rather than learning a separate one
        self.output_head.weight = self.token_emb.weight

        # RoPE cache is precomputed once, registered as a buffer so it moves
        # with the model (.to("mps")) but is never trained
        cos, sin = build_rope_cache(block_size, head_dim)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # standard GPT-style init: small random normal for linear/embedding weights
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        # idx shape: (B, T) - integer token ids
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"

        x = self.token_emb(idx)  # (B, T, n_embd)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)

        x = self.norm_final(x)
        logits = self.output_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # flatten batch+seq dims for cross_entropy: (B*T, vocab_size) vs (B*T,)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss