# TinyStories Transformer — From Scratch

A ~13.5M parameter, Llama-style decoder-only Transformer, hand-written in
PyTorch, trained on TinyStories using the byte-level BPE tokenizer built
earlier in this project. See `01_tokenizer/TOKENIZER_DEEPDIVE.md` for the
tokenizer; this document covers the model architecture and training loop.

## Table of Contents
1. [Design decisions & rationale](#1-design-decisions--rationale)
2. [Architecture overview](#2-architecture-overview)
3. [Component-by-component math](#3-component-by-component-math)
4. [Training loop](#4-training-loop)
5. [Real training results](#5-real-training-results)
6. [Testing](#6-testing)
7. [Known limitations & open questions](#7-known-limitations--open-questions)

---

## 1. Design decisions & rationale

Every architectural choice below was made deliberately, not defaulted to.

| Decision | Choice | Why |
|---|---|---|
| Model size | 6 layers, 6 heads, 384-dim (~13.5M params) | "Small" tier, matches original TinyStories paper's smallest working models; fits comfortably on a single M4 Max |
| Corpus | TinyStories | Restricted vocabulary lets a *small*, hand-written model produce *coherent* output — richer corpora (Wikipedia, web text) need much bigger models to avoid producing incoherent text |
| Positional encoding | RoPE | Encodes position via rotation rather than addition; attention scores depend on *relative* distance, not absolute position — better generalization than learned/absolute embeddings |
| Normalization | RMSNorm, pre-norm placement | Simpler math than LayerNorm (no mean-centering, no bias param), pairs naturally with RoPE (both are magnitude/rotation-focused, not centering-focused); pre-norm keeps the residual stream gradient path unobstructed, more stable at depth |
| Feedforward | SwiGLU | Gated activation, modest quality improvement over plain GELU at the cost of a third weight matrix; completes the "modern architecture" stack alongside RoPE + RMSNorm |
| Linear layer bias | None | Llama-style; fewer params, negligible quality cost |
| Weight tying | Yes | Input token embedding and output projection share one weight matrix — fewer params, standard practice |
| Dropout | 0.1 | Regularization, given TinyStories' repetitive style raises overfitting risk faster than a diverse corpus would |
| Optimizer | AdamW | Per-parameter adaptive learning rates handle Transformers' widely varying gradient scales across parameter types; the "W" (decoupled weight decay) fixes a real bug in plain Adam's weight decay interacting badly with per-parameter rescaling |

---

## 2. Architecture overview

```
Input token ids (B, T)
      │
      ▼
Token Embedding (vocab_size=7170 → 384-dim)   [weight-tied with output head]
      │
      ▼
┌─────────────────────────────┐
│  Transformer Block  × 6       │
│  ┌───────────────────────┐   │
│  │ x = x + Attention(       │   │
│  │       RMSNorm(x), RoPE)  │   │   <- pre-norm residual
│  │ x = x + SwiGLU(           │   │
│  │       RMSNorm(x))         │   │   <- pre-norm residual
│  └───────────────────────┘   │
└─────────────────────────────┘
      │
      ▼
RMSNorm (final)
      │
      ▼
Output head (384-dim → vocab_size=7170)   [= token embedding weight, transposed]
      │
      ▼
Logits (B, T, vocab_size) → cross-entropy loss against shifted targets
```

**Parameter count:** 13,375,104 (measured directly from the trained model,
confirmed to match the ~13.5M design estimate).

---

## 3. Component-by-component math

### RMSNorm

```
RMSNorm(x) = (x / RMS(x)) * gamma,   RMS(x) = sqrt(mean(x²) + eps)
```

Rescales a vector to unit RMS (no mean-centering, unlike LayerNorm), then
applies a learned per-dimension scale `gamma` (initialized to all-ones, so
the layer starts as pure normalization and *learns* how much to deviate).
`eps` (1e-6) prevents divide-by-zero on an all-zero input.

### RoPE (Rotary Position Embeddings)

Splits each `head_dim`-sized query/key vector into `head_dim/2` consecutive
pairs. For a token at position `pos`, pair `i` is rotated by angle
`θ_i × pos`, where `θ_i = base^(-2i/head_dim)` (base=10,000) — different
pairs rotate at different "clock speeds," fast for low `i`, slow for high
`i`, mirroring the frequency spread used in sinusoidal position encodings.

```
x'_i = x_i·cos(φ) - y_i·sin(φ)
y'_i = x_i·sin(φ) + y_i·cos(φ)      where φ = θ_i × pos
```

Applied to queries and keys only, never values — RoPE encodes position for
the *matching* mechanism, not the content being retrieved. Two properties
verified directly by tests: rotation at position 0 is always the identity
(`angle = θ × 0 = 0` for every frequency), and rotation never changes a
vector's norm (a property of all rotations). The most important verified
property: the dot product between a rotated query and rotated key depends
only on their **relative** distance, not their absolute positions — this
is RoPE's entire reason for existing, and it was checked directly rather
than assumed.

### Causal Self-Attention

For one head: `Q = xW_q`, `K = xW_k`, `V = xW_v` → apply RoPE to `Q`, `K` →
`scores = QK^T / sqrt(head_dim)` → mask out any `(query, key)` pair where
`key_position > query_position` by setting those scores to `-inf` → softmax
→ weighted sum of `V`. Multi-head: split `n_embd` into `n_head` chunks of
`head_dim = n_embd / n_head`, run the above per head in parallel, concatenate,
project back down. The `1/sqrt(head_dim)` scaling prevents large dot
products from pushing softmax into a near-one-hot regime as dimension grows.
The causal-masking property was verified directly: perturbing a *future*
token's input and confirming *earlier* positions' outputs are mathematically
unaffected — not just trusting the mask logic looks right.

### SwiGLU Feedforward

```
SwiGLU(x) = (SiLU(xW_gate) * (xW_up)) @ W_down,   SiLU(z) = z·sigmoid(z)
```

Three weight matrices (gate, up, down) instead of a plain MLP's two. The
gate signal (via SiLU, roughly 0-1 per dimension) controls how much of the
"content" signal (`xW_up`) passes through — a more expressive mechanism
than a fixed nonlinearity applied uniformly. To keep parameter count
comparable to a standard `4×n_embd` GELU MLP despite the extra matrix,
hidden dim is scaled to `≈ (2/3) × 4 × n_embd`, rounded to a multiple of 64.
For `n_embd=384`: `hidden_dim = 1024`.

### Transformer Block (pre-norm residual)

```
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

Two independent RMSNorm instances (attention gets its own, feedforward
gets its own). The residual (`x + ...`) structure gives gradients an
unobstructed path backward through the whole stack — verified directly by
zeroing all attention/FFN weights and confirming the block's output then
equals its input exactly (pure passthrough), and by checking every single
parameter in the block receives a non-`None` gradient after a backward pass.

### Full Model

Token embedding → dropout → 6× Transformer Block → final RMSNorm → output
head (weight-tied to the token embedding — literally the same tensor
object, not a copy, so gradients to one automatically update the other).
Weight init: `std=0.02` normal, the standard GPT-2-style scale that keeps
early activations from exploding through a deep stack.

**Critical sanity check performed before any real training:** a freshly
initialized (untrained) model's loss should sit near `ln(vocab_size)` —
what you'd get from guessing uniformly at random across the vocabulary.
Measured: `ln(7170) ≈ 8.878`; actual measured loss at step 1 of real
training was `8.9235` — confirms the architecture has no hidden bugs
(no causal-mask leak, no broken init, no scaling error) before investing
any real training time.

---

## 4. Training loop

**Config:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW, betas=(0.9, 0.95), weight_decay=0.1 |
| Peak LR | 3e-4 |
| LR schedule | Linear warmup (320 steps) → cosine decay to `0.1 × peak` |
| Batch size | 64 |
| Block size | 256 |
| Grad clip | max-norm 1.0 |
| Target steps | 6,400 (~3 epochs over the 35.2M-token train split) |
| Eval interval | every 200 steps, averaged over 20 batches per split |
| Checkpointing | best-val-loss (overwrite on improvement) + periodic snapshots every 1,000 steps, all retained |

**Why AdamW:** Transformers' gradient magnitudes vary widely across
parameter types (embeddings vs. attention projections vs. feedforward
weights); Adam's per-parameter adaptive learning rate (tracking both
gradient mean and squared-gradient mean) handles this far better than
plain SGD's single global rate. The "W" decouples weight decay from the
gradient-based update, fixing a documented issue where vanilla Adam's
per-parameter rescaling was inadvertently also rescaling (and weakening)
weight decay's regularizing effect.

**Why warmup + cosine decay:** starting at a very low LR and ramping up
avoids destabilizing the randomly-initialized weights with large updates
before the model has any sense of gradient direction; cosine decay then
smoothly reduces the LR over the rest of training, letting the model
settle into a minimum rather than continuing to bounce around at a high
learning rate.

**Why gradient clipping:** caps the norm of the full gradient vector at
1.0 before the optimizer step, preventing any single anomalous batch from
causing a destructively large weight update — a standard stabilizer for
deep Transformer training.

---

## 5. Real training results

Full 6,400-step run, TinyStories, M4 Max (MPS backend):

| Step | Train loss | Val loss | LR |
|---|---|---|---|
| 1 | 8.9235 | 8.9218 | 9.37e-07 |
| 201 | 4.0719 | 4.0716 | 1.88e-04 |
| 1,001 | 2.2083 | 2.2372 | 2.92e-04 |
| 2,001 | 1.8434 | 1.8820 | 2.52e-04 |
| 4,001 | 1.6207 | 1.6797 | 1.21e-04 |
| 6,400 | 1.5549 | 1.6038 | 3.00e-05 |

**Wall-clock time:** 3,042.6s (~50.7 minutes) for the full run.
Throughput settled around **2.1-2.7 steps/sec**, translating to roughly
**~140,000-175,000 tokens/sec** processed (`batch_size × block_size ×
steps/sec`).

**Final perplexity:** `exp(1.6038) ≈ 4.97` — on held-out validation text,
the model's effective uncertainty is choosing among roughly 5 similarly-
likely next tokens on average.

**Train/val gap stayed small throughout** (final: `1.5549` vs `1.6038`,
a gap of `0.049`) — direct evidence the model is learning generalizable
patterns rather than memorizing the training set. No sign of overfitting
at any point in the run.

**Val loss was still decreasing at the final step** (`1.6079 → 1.6038`
in the last 200 steps) — no plateau reached, suggesting the model likely
had headroom for further improvement with more training steps, not yet at
its ceiling for this size/data combination.

**Checkpoints produced:** `checkpoint_best.pt` (val_loss 1.6038, step 6400)
plus periodic snapshots at steps 1000/2000/3000/4000/5000/6000 (~1GB total
on disk), enabling inspection or resumption from any recorded point in
training.

---

## 6. Testing

Every component was unit-tested *before* being composed into the next:

- **RMSNorm**: unit-RMS output verified at init, shape preservation,
  no NaN on zero input.
- **RoPE**: position-0 identity rotation, vector norm preservation, and
  (most importantly) the relative-position dot-product invariance property
  directly verified — the actual reason RoPE exists, not just a shape check.
- **Causal attention**: shape correctness, valid softmax distributions
  (rows sum to 1), and direct causal-masking verification via perturbation
  (changing a future token provably does not change earlier outputs).
- **SwiGLU**: shape correctness, exact hidden-dim arithmetic verified,
  and a check that the gate is actually gating (not a degenerate no-op).
- **Transformer block**: shape correctness, residual-passthrough property
  (zeroing attention/FFN weights reproduces the input exactly), and full
  gradient-flow verification to every parameter (catches the common
  from-scratch bug of a layer silently disconnected from the graph).
- **Full model**: shape correctness at both toy and real config, real
  weight-tying verification (same tensor object, not just equal values),
  and the untrained-loss-near-`ln(vocab_size)` sanity check.

Run all model-layer tests: `pytest test_rmsnorm.py test_rope.py
test_attention.py test_swiglu.py test_block.py test_model.py -v`

---

## 7. Known limitations & open questions

- **6,400 steps was a design estimate (~3 epochs), not a scientifically
  tuned stopping point.** Val loss was still improving at the final step —
  more training would likely help further; 6,400 was chosen as a
  reasonable first target, not a discovered optimum.
- **No learning-rate sweep or hyperparameter search performed.** `3e-4`
  peak LR, `0.1` dropout, and the warmup/decay shape are all standard
  defaults for small Transformers, not tuned specifically for this model/
  corpus combination.
- **Single training run, no seed variation checked.** Loss curve shape
  and final numbers reflect one run; re-running with a different random
  seed would show some variance, not investigated here.
- **Throughput (~2.1-2.7 steps/sec) is MPS-specific** and reflects a
  hand-written (non-fused-kernel) attention implementation — a
  CUDA GPU or a FlashAttention-style fused kernel would likely be
  substantially faster; this number shouldn't be treated as representative
  of what this architecture "should" achieve on different hardware.
- **Qualitative output quality (does it actually write coherent short
  stories?) has not yet been assessed** — loss/perplexity are proxies,
  not a direct measure of generated text quality. That's the next stage.