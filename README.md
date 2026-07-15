# Language Modeling From Scratch

A complete language model training pipeline built from scratch in Python:
byte-level BPE tokenizer, a hand-written Llama-style Transformer (RoPE,
RMSNorm, SwiGLU), a data pipeline, and a training loop — every component
implemented and understood, not imported from a library.

Trained end-to-end on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

## Project stages

| Stage | What's there | Deep dive |
|---|---|---|
| [`01_tokenizer/`](./01_tokenizer) | Byte-level BPE tokenizer, trained on Shakespeare + TinyStories | [TOKENIZER_DEEPDIVE.md](./01_tokenizer/TOKENIZER_DEEPDIVE.md) |
| [`02_data_pipeline/`](./02_data_pipeline) | Corpus tokenization, binary storage, batch sampling | [DATA_PIPELINE_DEEPDIVE.md](./02_data_pipeline/DATA_PIPELINE_DEEPDIVE.md) |
| [`03_model/`](./03_model) | The Transformer architecture, fully unit-tested | [TRANSFORMER_DEEPDIVE.md](./03_model/TRANSFORMER_DEEPDIVE.md) |
| [`04_training/`](./04_training) | Training loop, checkpoints, real training results | *(see TRANSFORMER_DEEPDIVE.md, §4-5)* |

## Result so far

A ~13.5M parameter Transformer, trained from scratch on TinyStories:

- **Final validation loss:** 1.6038 (perplexity ≈ 4.97)
- **Training time:** ~51 minutes on an Apple M4 Max (MPS)
- **No overfitting observed** — train/val loss stayed within 0.05 of each other throughout

## Setup

```bash
pip install -e ".[dev]"
```

## Package structure

The reusable core (tokenizer + model components) lives in `src/cs336_workspace/`
as a proper installable package — imported as `from cs336_workspace.X import Y`
from anywhere in the project. Each numbered stage folder contains that stage's
tests, one-off training/experiment scripts, and documentation; the package
itself is the single source of truth for the actual implementation.

## Status

Architecture and training are complete and verified (see deep-dive docs for
full test coverage and real measured results). Next: inference/generation —
sampling actual text from the trained model.