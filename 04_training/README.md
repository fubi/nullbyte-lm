# Training

The training loop: AdamW optimizer, warmup + cosine LR schedule, gradient
clipping, periodic evaluation, and checkpointing — run on the model from
`03_model/` using data from `02_data_pipeline/`.

## Quickstart

```bash
python train.py
```

Reads `train.bin`/`val.bin` from `../data/`, trains `TinyStoriesLM`, and
writes checkpoints into `checkpoints/`.

## Config

| Parameter | Value |
|---|---|
| Optimizer | AdamW, betas=(0.9, 0.95), weight_decay=0.1 |
| Peak LR | 3e-4 |
| LR schedule | Linear warmup (320 steps) → cosine decay to `0.1 × peak` |
| Batch size | 64 |
| Grad clip | max-norm 1.0 |
| Target steps | 6,400 (~3 epochs) |
| Eval interval | every 200 steps |
| Checkpointing | best-val-loss + periodic snapshots every 1,000 steps |

## Result

Trained on TinyStories, Apple M4 Max (MPS): **val loss 1.6038**
(perplexity ≈ 4.97) in **~51 minutes**, with train/val loss staying
within 0.05 of each other throughout — no overfitting observed, and val
loss was still improving at the final step.

## Files in this module

| File | Role |
|---|---|
| `train.py` | The training loop. |
| `checkpoints/checkpoint_best.pt` | Best checkpoint by val loss (step 6400, val_loss 1.6038). |
| `checkpoints/checkpoint_step*.pt` | Periodic snapshots (every 1,000 steps), all retained. |
| `README.md` | This file. |

## Full documentation

Full training loop rationale (why AdamW, why warmup+cosine, why grad
clipping) and the complete real-run results table — see
**[../03_model/TRANSFORMER_DEEPDIVE.md](../03_model/TRANSFORMER_DEEPDIVE.md)**,
§4-5.