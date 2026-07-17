┌────────────────────────────────────────────────────┐
│                  GENERATION CONFIG                     │
├────────────────────────────────────────────────────┤
│  Sampling:        temperature + top-p (nucleus)          │
│  Seeding:          unconditional (from <|endoftext|>)      │
│                    or prompted (user text)                  │
│  max_new_tokens:   250                                        │
│  temperature:      1.0                                         │
│  top_p:            0.95                                         │
│  Context handling: sliding window (last block_size=256 tokens) │
│  Stop condition:    early stop if <|endoftext|> is generated     │
└────────────────────────────────────────────────────┘

# Inference — Text Generation

Sampling actual text from the trained TinyStories Transformer.

## Quickstart

```bash
python generate.py
```

Loads `checkpoint_best.pt`, generates one unconditional story and one
prompted continuation, prints both.

## Files in this module

| File | Role |
|---|---|
| `generate.py` | `generate()` — the sampling loop, plus `top_p_sample()` (nucleus sampling core). |
| `README.md` | This file. |
| `INFERENCE_DEEPDIVE.md` | Sampling strategy rationale, real generated examples with analysis. |

## Full documentation

See **[INFERENCE_DEEPDIVE.md](./INFERENCE_DEEPDIVE.md)** for why top-p
sampling was chosen, the math behind it, and an honest breakdown of what
the trained model gets right and wrong.