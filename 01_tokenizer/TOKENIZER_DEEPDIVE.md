# Byte-Level BPE Tokenizer — From Scratch

A GPT-2-style byte-level Byte-Pair Encoding (BPE) tokenizer, implemented from
scratch in pure Python. No external tokenizer libraries — only the `regex`
package for Unicode-aware pattern matching.

## Table of Contents
1. [Concept overview](#1-concept-overview)
2. [Installation](#2-installation)
3. [Architecture](#3-architecture)
4. [Function-by-function walkthrough](#4-function-by-function-walkthrough)
5. [The `Tokenizer` class](#5-the-tokenizer-class)
6. [Special token handling](#6-special-token-handling)
7. [Testing](#7-testing)
8. [Training results (real runs)](#8-training-results-real-runs)
9. [Usage](#9-usage)
10. [Known limitations](#10-known-limitations)

---

## 1. Concept overview

**What BPE does:** starts with the smallest possible units (raw bytes,
0–255) and iteratively merges the *most frequent adjacent pair* into a new
single unit, repeating until a target vocabulary size is reached. Over many
iterations this discovers common substrings — first letter pairs (`th`,
`he`), then whole words (`the`, `and`), then in a large enough vocab, common
multi-word phrases.

**Why byte-level, not character-level:** operating on raw UTF-8 bytes
(0–255) rather than Unicode characters means the tokenizer can represent
*any* input — any language, emoji, malformed text — using only 256 possible
"letters" (bytes) as the base alphabet. There is no possible input that
produces an "unknown token" error, because every string reduces to bytes.

**Why pre-tokenization:** before BPE ever counts a byte-pair, raw text is
split into word-like chunks using a regex (letters, digits, punctuation, and
whitespace are kept in separate chunks). This stops BPE from learning merges
that span across word boundaries (e.g. merging the end of one word with the
start of the next), which would pollute the vocabulary with non-generalizing
tokens.

---

## 2. Installation

```bash
pip install regex pytest
```
or, if using the project's `pyproject.toml`:
```bash
pip install -e ".[dev]"
```

`regex` (not stdlib `re`) is required because pretokenization needs
`\p{L}` / `\p{N}` Unicode category matching, which stdlib `re` doesn't
support.

---

## 3. Architecture

```
raw text
   │
   ▼
┌─────────────────┐
│  Pre-tokenizer   │  regex split → word-like string chunks
└─────────────────┘
   │
   ▼
┌─────────────────┐
│  Byte encoder    │  each chunk → list of ints 0–255 (UTF-8 bytes)
└─────────────────┘
   │
   ▼
┌─────────────────┐
│  BPE Trainer     │  iteratively merge most-frequent adjacent pair
│                  │  until vocab_size reached
└─────────────────┘
   │
   ▼
┌─────────────────┐         ┌─────────────────┐
│    Encoder       │ ◄─────► │    Decoder       │
│ (apply merges,   │         │ (ids → bytes →   │
│  text → ids)     │         │  UTF-8 string)   │
└─────────────────┘         └─────────────────┘
   │
   ▼
┌─────────────────┐
│  Serialization    │  save/load merges + vocab to disk
└─────────────────┘
```

---

## 4. Function-by-function walkthrough

### `pretokenize(text: str) -> list[str]`

```python
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(text: str) -> list[str]:
    return re.findall(GPT2_SPLIT_PATTERN, text)
```

The regex tries alternatives **left to right**, first match wins, applied
repeatedly across the string:

| Pattern piece | Matches | Example |
|---|---|---|
| `'s\|'t\|'re\|'ve\|'m\|'ll\|'d` | Common English contraction suffixes | `"don't"` → `"don"`, `"'t"` |
| ` ?\p{L}+` | A run of letters (any script), optional leading space | `" world"`, `"Hello"` |
| ` ?\p{N}+` | A run of digits, optional leading space | `" 42"` |
| ` ?[^\s\p{L}\p{N}]+` | A run of symbols/punctuation (not space/letter/digit) | `"!"`, `"('"` |
| `\s+(?!\S)` | Trailing whitespace not followed by non-space | end-of-string spaces |
| `\s+` | Any remaining whitespace | catch-all |

The **optional leading space** on the letter/digit patterns is intentional:
`"dog"` and `" dog"` become *different* chunks (`"dog"` vs `" dog"`). This
lets the model later learn different embeddings for "word at the start of a
sentence" vs "word after a space" — the same design GPT-2 uses.

### `chunk_to_byte_ids(chunk: str) -> list[int]`

```python
def chunk_to_byte_ids(chunk: str) -> list[int]:
    return list(chunk.encode("utf-8"))
```

Converts a text chunk to its raw UTF-8 byte sequence, as a list of ints
0–255. ASCII characters are 1 byte each; other Unicode characters take
2–4 bytes (e.g. `'é'` → `[195, 169]`, 2 bytes).

### `get_stats(chunks: list[list[int]]) -> dict[tuple[int, int], int]`

```python
def get_stats(chunks: list[list[int]]) -> dict[tuple[int, int], int]:
    counts = {}
    for ids in chunks:
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts
```

**The math:** for a sequence `ids = [a, b, c, d]`, `zip(ids, ids[1:])`
produces the adjacent pairs `(a,b), (b,c), (c,d)`. This is counted across
*every* chunk, and counts are summed — this is a global frequency count of
every adjacent-pair "bigram" across the whole corpus. Cost: O(total tokens
across all chunks) per call.

### `merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]`

```python
def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
```

Greedy **left-to-right, non-overlapping** replacement. Given `ids=[1,1,1]`
and `pair=(1,1)`: at `i=0` the pair matches → append `new_id`, jump `i` by
2 (past both consumed elements) → `i=2`. At `i=2`, only one element left, no
pair possible → append it unchanged. Result: `[new_id, 1]`, **not**
`[new_id, new_id]` — you cannot form two overlapping pairs from 3 elements.
This greedy-scan behavior is the reference definition that the rest of the
tokenizer must stay consistent with.

### `Tokenizer._train_merges` (the core training loop)

```python
for i in range(num_merges):
    stats = get_stats(chunks)
    if not stats:
        break
    pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
    new_id = NUM_BASE_BYTES + i
    chunks = [merge(ids, pair, new_id) for ids in chunks]
    merges[pair] = new_id
```

Each iteration:
1. Recount all adjacent-pair frequencies across the whole (partially merged)
   corpus.
2. Pick the single most frequent pair. **Tie-break rule:**
   `key=lambda p: (stats[p], -p[0], -p[1])` — Python's `max` compares tuples
   element by element, so ties on `stats[p]` (the count) fall through to
   comparing `-p[0]`, then `-p[1]`. Negating flips "largest wins" into
   "numerically smallest pair wins," giving a fully deterministic,
   reproducible result for a given corpus (no dependence on dict iteration
   order).
3. Assign it the next available id (`256 + i`, since ids 0–255 are reserved
   for raw bytes).
4. Replace every occurrence across every chunk.
5. Record the merge.

This stops early if no pairs remain (fully-merged corpus) even if the vocab
budget isn't exhausted — a **valid, expected outcome for small/repetitive
corpora**, not a bug.

**Complexity:** this is the *naive* version — O(merges × corpus size),
since `get_stats` rescans the entire corpus every iteration. Deliberately
chosen over an efficient incremental-count version for learning clarity;
real runs on ~1MB of text at ~5,000–10,000 merges take on the order of
10–20 minutes in pure Python (see [§8](#8-training-results-real-runs) for
actual measured numbers).

### `Tokenizer._encode_chunk` (applying learned merges to new text)

```python
def _encode_chunk(self, ids: list[int]) -> list[int]:
    ids = list(ids)
    while len(ids) >= 2:
        stats = get_stats([ids])
        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
        if pair not in self.merges:
            break
        ids = merge(ids, pair, self.merges[pair])
    return ids
```

Merges must be replayed in the **exact order they were learned**, because
later merges build on earlier ones (e.g. training might first learn
`(a,b)→X`, then `(X,c)→Y` — applying `(X,c)` before `(a,b)` exists would be
meaningless). `self.merges[pair]` stores the id each pair was merged *into*,
and since ids were assigned sequentially during training (`256, 257, 258…`),
a pair's id **is** its training-order rank. `min(..., key=...)` finds
whichever mergeable pair in the current sequence has the *lowest* id — i.e.
was learned *earliest** — and applies that one first, repeating until no
known pair remains.

### `build_vocab` / `decode`

```python
def _build_vocab(self) -> None:
    vocab = {idx: bytes([idx]) for idx in range(NUM_BASE_BYTES)}
    for (p0, p1), idx in self.merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    vocab[self.eot_id] = EOT_TOKEN.encode("utf-8")
    self.vocab = vocab
```

Recursively builds each merged token's actual byte sequence: a merged
token's bytes = its two parent tokens' bytes concatenated. Base ids 0–255
map to single bytes; everything above is built up from there.

```python
def decode(self, ids: list[int]) -> str:
    raw_bytes = b"".join(self.vocab[i] for i in ids)
    return raw_bytes.decode("utf-8", errors="replace")
```

Looks up each id's byte sequence, concatenates, and decodes as UTF-8.
`errors="replace"` swaps in `�` for any byte sequence that isn't valid
UTF-8 on its own (this can happen because BPE operates on raw bytes with no
awareness of character boundaries — a token may represent only *half* of a
multi-byte character).

---

## 5. The `Tokenizer` class

Wraps all of the above into stateful methods:

| Method | Purpose |
|---|---|
| `Tokenizer(vocab_size)` | Construct with a target vocab size (must be > 257) |
| `.train(texts, progress_every, verbose)` | Learn merges from a list of training documents |
| `.encode(text)` | Text → list of token ids |
| `.decode(ids)` | List of token ids → text |
| `.save(path_prefix)` | Write `.merges.txt` + `.vocab.json` |
| `Tokenizer.load(path_prefix)` | Reconstruct a trained tokenizer from disk |

`vocab_size` breaks down as: `256` base byte ids + `1` reserved for
`<|endoftext|>` + `N` learned merges. So `num_merges = vocab_size - 257`.

---

## 6. Special token handling

`<|endoftext|>` is reserved at `id = vocab_size - 1` and is treated as
**atomic** — never split, never merged, never built out of byte-pairs.

```python
pieces = stdlib_re.split(f"({stdlib_re.escape(EOT_TOKEN)})", text)
```

The capturing group in `re.split` keeps the delimiter itself in the output
list. So `"cat<|endoftext|>dog"` splits into
`["cat", "<|endoftext|>", "dog"]`; the middle piece is special-cased to the
reserved id directly, and `pretokenize`/BPE never sees it. This guarantees
bytes from text before and after the marker can never become adjacent in
the same chunk, so no merge can ever cross that boundary.

---

## 7. Testing

Run the full suite:
```bash
pytest test_tokenizer.py -v
```

Coverage includes: pretokenizer regex behavior (including the
leading-space distinction), `get_stats`/`merge` correctness on edge cases
(overlapping pairs, no-match cases), trainer determinism (tie-breaking
gives identical results across runs on the same input), full encode/decode
round-trips (including non-ASCII text with zero learned merges), the
special-token atomicity guarantee, and save/load correctness (including
rejecting genuinely corrupt files while correctly allowing early-stopped
training runs that used fewer merges than the budget allowed).

**Two real bugs caught by this suite during development**, both worth
remembering as design lessons:
1. `load()` originally required an *exact* match between the declared
   `vocab_size` and the number of merge lines in the file — but training
   can legitimately stop early on small/repetitive corpora. Fixed to
   `actual_merge_lines <= expected_merge_lines` (over-budget is corruption;
   under-budget is valid early stopping).
2. The corruption-detection test itself was then found encoding the *old,
   wrong* invariant (`!=` instead of `>`) — a reminder that when a
   validation rule changes, tests asserting the old rule need to be
   re-examined, not just left passing on stale assumptions.

---

## 8. Training results (real runs)

Two real end-to-end training runs, both on Tiny Shakespeare text, on an
Apple M4 Max (naive trainer, single-threaded CPU-bound — MPS/GPU is not
used by the tokenizer itself, only by model training later):

| Run | Corpus size | vocab_size | Merges learned | Wall time |
|---|---|---|---|---|
| A | 100,000 chars (slice) | 5,000 | 4,743 (full budget) | 33.2s |
| B | ~1,000,000 chars (full file) | 5,000 | 4,743 (full budget) | 486.8s (~8.1 min) |
| C | ~1,000,000 chars (full file) | 10,000 | 9,743 (full budget) | 1073.0s (~17.9 min) |

**Compression (chars/token) on a fixed 1000-char held-out slice:**
- Run B (vocab 5,000): 307 tokens → **3.26 chars/token**
- Run C (vocab 10,000): 295 tokens → **3.39 chars/token**

**Key finding:** doubling vocab_size measurably improved compression, but
the *composition* of learned tokens changed more interestingly than the
raw number suggests. At vocab_size=5,000 on the full multi-play corpus, the
longest learned tokens were dominated by proper nouns/character names
(`Northumberland`, `BOLINGBROKE`, `Buckingham`) — the limited budget forced
BPE to spend slots on corpus-specific names just to keep up with their high
frequency. At vocab_size=10,000, general formal-English vocabulary started
appearing alongside the names (`notwithstanding`, `circumstances`,
`proclamation`) — more budget meant BPE no longer had to choose between
"generalizable English" and "this corpus's proper nouns."

**Practical implication:** vocab_size needs to scale with corpus
*diversity*, not just corpus size. A small vocab on a diverse, multi-play
corpus hits a real ceiling that a small vocab on a single-play corpus
doesn't — this is a concrete, hands-on version of why production tokenizers
(GPT-4 ~100k, Llama ~32k) use much larger vocabularies than this toy
project.

---

## 9. Usage

```python
from tokenizer import Tokenizer

# train
tok = Tokenizer(vocab_size=10000)
tok.train([open("input.txt", encoding="utf-8").read()], progress_every=200)
tok.save("my_tokenizer")

# load + use later
tok = Tokenizer.load("my_tokenizer")
ids = tok.encode("Some new text<|endoftext|>Another document")
text = tok.decode(ids)
```

---

## 10. Known limitations

- **Naive O(merges × corpus_size) trainer.** Deliberate design choice for
  learning clarity over speed — see [§4](#tokenizer_train_merges-the-core-training-loop).
  Fine at this project's scale (~10-20 min for ~1MB / ~10k vocab); would
  need an incremental-count + reverse-index approach to scale to
  production-sized corpora in reasonable time.
- **Tie-breaking is arbitrary but deterministic.** Ties in merge frequency
  are broken by numerically smallest pair id — a reasonable, reproducible
  choice, but not the only valid one; different tokenizers may break ties
  differently and produce a different (equally valid) merge sequence.
- **`decode()` raises `TokenizerError` on unknown ids**, by design — this
  is treated as a signal of a real bug (mismatched tokenizer version,
  corrupted file) rather than something to silently paper over.