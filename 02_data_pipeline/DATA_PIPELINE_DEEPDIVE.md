# Data Pipeline — Deep Dive

Covers turning raw TinyStories text into the `train.bin`/`val.bin` files
the model trains on, and the batch-sampling function used during training.

## 1. Design decisions & rationale

| Decision | Choice | Why |
|---|---|---|
| Corpus | TinyStories | Restricted vocabulary lets a small hand-written model produce coherent output (see `01_tokenizer/TOKENIZER_DEEPDIVE.md` for the full reasoning and the empirical vocab-size finding below) |
| Training slice size | 150MB | "Medium" tier — real training signal without the encode/train time of the full ~2.2GB file |
| Dtype on disk | `uint16` | Fits `vocab_size=7,170` comfortably (max 65,535); half the disk/bandwidth cost of `uint32` |
| File format | Raw binary + numpy memmap | Minimal overhead, no header parsing needed, nanoGPT-style convention |
| Block size | 256 tokens | Matches TinyStories' natural short-story length; keeps attention compute manageable for a hand-written (non-fused-kernel) implementation |
| Train/val split | 95/5 | Training data is the scarcer resource at this scale; 5% is still a statistically stable validation signal |
| Batch sampling | Random offsets, with replacement | Simpler and less bug-prone than sequential epoch tracking; standard nanoGPT-style approach; unevenness washes out over the thousands of steps a real run takes |

## 2. An important empirical finding: vocab_size vs. corpus diversity

The tokenizer was originally targeted at `vocab_size=32,000` to mirror
production tokenizers. Training on a 1MB TinyStories slice **stopped
early at 6,913 merges** — the corpus ran out of distinct multi-byte
chunks to merge entirely, well short of the 32,000 target.

**Why:** BPE never merges across pretokenizer chunk (word) boundaries —
each distinct word can only contribute new merges until it's fully
collapsed into one token, after which repeating it further contributes
nothing new. TinyStories is deliberately written with a restricted,
repetitive vocabulary (by design, for exactly the small-model-friendly
reason we chose it), so it exhausts its "learnable" merges much sooner
than a more lexically diverse corpus (Shakespeare, at the same 1MB size,
used its full budget without running dry).

A 5MB retry was attempted but abandoned in favor of accepting the result:
extrapolating from measured tokenizer-training timing (10x more Shakespeare
text cost ~14.7x more time — clearly superlinear, not linear), reaching
close to 32,000 merges would likely have required ~20MB of text and many
hours of naive-trainer time, for a gain that doesn't really serve the
project's goals (TinyStories' restricted vocabulary is a *feature*, not a
gap to fill).

**Resolution:** the tokenizer was compacted — re-saved with
`vocab_size = 256 (base bytes) + 6,913 (real merges) + 1 (EOT) = 7,170` —
eliminating a ~24,830-id dead zone that would otherwise have wasted
embedding-table rows in the model with zero benefit. This is the
`tinystories_tok_compact` artifact used throughout the rest of the
pipeline. See `01_tokenizer/TOKENIZER_DEEPDIVE.md` for the tokenizer-side
detail.

## 3. Encoding the training corpus

**Boundary safety:** cutting a corpus slice at an arbitrary byte offset
risks landing mid-story or mid-character. The real pipeline reads slightly
more than the target size, then trims back to the last complete
`<|endoftext|>` boundary before the target — guaranteeing every saved
document is complete.

**Split point:** done on the *token* array, after encoding — not on raw
text — so `train.bin`/`val.bin` are already in final on-disk form with no
further processing needed.

## 4. Real run results

Encoding 150MB of TinyStories text with the compacted tokenizer:

| Metric | Value |
|---|---|
| Input size | 149,999,811 characters (150.0MB), cut cleanly on a story boundary |
| Output | 37,018,642 tokens |
| Compression | **4.05 chars/token** |
| Encoding time | 112.0s |
| Train split | 35,167,709 tokens → `train.bin`, 70.3MB |
| Val split | 1,850,933 tokens → `val.bin`, 3.7MB |

**4.05 chars/token is notably better than either Shakespeare tokenizer**
achieved (3.26–3.39, at larger vocab sizes of 5,000/10,000) — direct
confirmation that TinyStories' restricted, repetitive vocabulary compresses
efficiently even with a comparatively small 7,170-token vocab, since there's
little long-tail rare vocabulary left uncaptured.

**Encoding speed (112s for 150M chars, ~1.34M chars/sec) scaled roughly
linearly** — unlike tokenizer *training*, which rescans the whole corpus
every merge round, `encode()` processes each pretokenized chunk
independently, so cost grows with corpus size, not superlinearly.

## 5. Batching (`get_batch`)

```python
def get_batch(split, batch_size, block_size, device):
    data = np.memmap(filename, dtype=np.uint16, mode="r")
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i+block_size] for i in starts])
    y = np.stack([data[i+1:i+block_size+1] for i in starts])
    return torch.from_numpy(x.astype(np.int64)).to(device), \
           torch.from_numpy(y.astype(np.int64)).to(device)
```

`np.memmap` avoids loading the full file into RAM — only the slices
actually touched are read from disk. `y` is `x` shifted by one position:
at every index, `y[i]` is the correct next token after `x[i]`, which is
the standard next-token-prediction training target. Cast from `uint16`
(disk-efficient storage) to `int64` happens here, since PyTorch's
embedding/cross-entropy layers expect `long`, not `uint16` — the smaller
dtype was only ever for on-disk compactness.

**Verified directly** (not just assumed): a real batch was pulled and the
shift relationship checked element-by-element —
`y[0][:-1] == x[0][1:]` held exactly, confirming the target construction
is correct before any model training was attempted on top of it.

## 6. Known limitations

- **Batch sampling is with-replacement, not epoch-tracked** — no
  guarantee every token is seen exactly once per "epoch"; acceptable at
  the training-step scale this project uses (see `04_training` for
  actual step counts), per the design tradeoff above.
- **150MB was a deliberate middle-ground choice**, not a tuned optimum —
  a larger slice would give more training signal at the cost of
  proportionally longer encode time and disk space.
- **`get_batch`'s file paths are currently hardcoded** to look for
  `train.bin`/`val.bin` in a fixed relative location — a natural next
  cleanup would be to parameterize the data directory rather than
  hardcode it.