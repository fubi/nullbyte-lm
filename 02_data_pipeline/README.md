# Data Pipeline

Turns raw TinyStories text into the binary token files the model trains on.

## Quickstart

```python
# see scripts/tokenize_corpus.py for the full working version
from cs336_workspace.tokenizer import Tokenizer
import numpy as np

tok = Tokenizer.load("../01_tokenizer/artifacts/tinystories_tok_compact")
text = open("../data/TinyStories-train.txt", encoding="utf-8").read(150_000_000)
ids = tok.encode(text)

split = int(len(ids) * 0.95)
np.array(ids[:split], dtype=np.uint16).tofile("../data/train.bin")
np.array(ids[split:], dtype=np.uint16).tofile("../data/val.bin")
```

## Pipeline

```
raw text (TinyStories-train.txt)
       │
       ▼
Tokenizer.encode()          — from 01_tokenizer
       │
       ▼
flat array of token ids
       │
       ▼
95/5 split → train.bin / val.bin   (uint16, raw binary, memmap-able)
       │
       ▼
get_batch() — random-offset sampling at training time (in cs336_workspace.batching)
```

## Files in this module

| File | Role |
|---|---|
| `scripts/sample_1mb.py` | Carves a small slice of raw text for fast tokenizer-training iteration. |
| `scripts/train_tinystories_tokenizer.py` | Trains the TinyStories tokenizer (see `01_tokenizer/`). |
| `scripts/compact_tokenizer.py` | One-off fix: re-saves a tokenizer with its `vocab_size` trimmed to match the merges it actually learned (see deep-dive §2). |
| `scripts/tokenize_corpus.py` | The real pipeline: encodes the full training slice, splits 95/5, writes `train.bin`/`val.bin`. |
| `../src/cs336_workspace/batching.py` | `get_batch()` — random-offset batch sampler, used at training time (lives in the package, not this folder, since it's imported by the training loop). |
| `README.md` | This file. |
| `DATA_PIPELINE_DEEPDIVE.md` | Full design rationale, real run numbers, and the vocab-size-vs-corpus-diversity finding. |

## Testing

No dedicated test file for this stage — it's exercised end-to-end via the
tokenizer's tests plus the batching shape/shift-by-one checks that were
verified manually during development (see deep-dive).

## Full documentation

For every design decision (why TinyStories, why 150MB, why uint16, why
random-offset batching, why 95/5) and the real measured results from
encoding 150MB of text — see **[DATA_PIPELINE_DEEPDIVE.md](./DATA_PIPELINE_DEEPDIVE.md)**.