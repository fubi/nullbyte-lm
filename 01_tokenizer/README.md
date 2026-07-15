# Byte-Level BPE Tokenizer

A GPT-2-style byte-level BPE tokenizer, built from scratch in pure Python.

## Quickstart

```bash
pip install regex pytest
```

```python
from tokenizer import Tokenizer

tok = Tokenizer(vocab_size=10000)
tok.train([open("input.txt", encoding="utf-8").read()])
tok.save("my_tokenizer")

tok = Tokenizer.load("my_tokenizer")
ids = tok.encode("Some text<|endoftext|>Another document")
text = tok.decode(ids)
```

## Files in this module

## Files in this module

| File | Role |
|---|---|
| `tokenizer.py` | **Core module.** The `Tokenizer` class and all supporting functions. This is the only file other code should import from. |
| `test_tokenizer.py` | Test suite. Run with `pytest test_tokenizer.py -v`. |
| `README.md` | This file — quickstart + orientation. |
| `TOKENIZER_DEEPDIVE.md` | Full technical walkthrough: math, function-by-function explanation, training results. |
| `10k-training.py`, `compare.py`, `create_sampler.py`, etc. | **Experimental/scratch scripts**, not part of the module. Used to train specific tokenizer checkpoints and inspect results during development. Not imported by `tokenizer.py` or the tests — safe to ignore, modify, or delete without affecting the core module. |

## Testing

```bash
pytest test_tokenizer.py -v
```

## Full documentation

For the complete architecture breakdown, line-by-line math explanation of
every function, the special-token design, and real training results/
benchmarks — see **[TOKENIZER_DEEPDIVE.md](./TOKENIZER_DEEPDIVE.md)**.