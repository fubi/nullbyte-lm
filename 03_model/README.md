# Model — TinyStories Transformer

A ~13.5M parameter, Llama-style decoder-only Transformer, hand-built in
PyTorch: RoPE, RMSNorm, SwiGLU, no biases, weight-tied output head.

The implementation itself lives in the installable package at
`../src/cs336_workspace/` (imported as `from cs336_workspace.X import Y`
from anywhere in the project). This folder holds the test suite and the
architecture documentation.

## Quickstart

```python
from cs336_workspace.model import TinyStoriesLM
import torch

model = TinyStoriesLM(
    vocab_size=7170, n_layer=6, n_head=6, n_embd=384, block_size=256
)

checkpoint = torch.load("../04_training/checkpoints/checkpoint_best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Architecture at a glance

```
Token Embedding (7,170 → 384)
        │
        ▼
┌─────────────────────────┐
│  × 6 Transformer Blocks    │
│  x = x + Attention(          │
│        RMSNorm(x), RoPE)      │
│  x = x + SwiGLU(RMSNorm(x))    │
└─────────────────────────┘
        │
        ▼
RMSNorm → Output head (weight-tied) → logits (7,170)
```

| Setting | Value |
|---|---|
| Layers / heads / dim | 6 / 6 / 384 |
| Parameters | 13,375,104 |
| Positional encoding | RoPE |
| Normalization | RMSNorm, pre-norm |
| Feedforward | SwiGLU |
| Bias terms | None |
| Weight tying | Yes |
| Best val loss (trained) | 1.6038 (perplexity ≈ 4.97) |

## Files in this module

| File | Role |
|---|---|
| `../src/cs336_workspace/model.py` | **Core implementation.** `TinyStoriesLM` — the full model. |
| `../src/cs336_workspace/block.py` | `TransformerBlock` — one pre-norm residual block. |
| `../src/cs336_workspace/attention.py` | `CausalSelfAttention` — multi-head causal self-attention with RoPE. |
| `../src/cs336_workspace/swiglu.py` | `SwiGLU` — the gated feedforward block. |
| `../src/cs336_workspace/rmsnorm.py` | `RMSNorm` — the normalization layer. |
| `../src/cs336_workspace/rope.py` | `build_rope_cache` / `apply_rope` — rotary position embedding math. |
| `test_rmsnorm.py` `test_rope.py` `test_attention.py` `test_swiglu.py` `test_block.py` `test_model.py` | Test suite — one file per component. |
| `README.md` | This file. |
| `TRANSFORMER_DEEPDIVE.md` | Full technical walkthrough: design rationale, math, training config, real results. |

## Testing

```bash
pytest test_rmsnorm.py test_rope.py test_attention.py test_swiglu.py test_block.py test_model.py -v
```

## Full documentation

For the reasoning behind every architecture choice, the math for each
component, and the untrained-model sanity check that validated the
architecture before any real training — see
**[TRANSFORMER_DEEPDIVE.md](./TRANSFORMER_DEEPDIVE.md)**.