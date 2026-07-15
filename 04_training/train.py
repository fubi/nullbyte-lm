import time
import math
import torch
from cs336_workspace.model import TinyStoriesLM
from cs336_workspace.batching import get_batch

# ---------- Config (all decisions locked in during design) ----------

VOCAB_SIZE = 7170
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
BLOCK_SIZE = 256
DROPOUT = 0.1

BATCH_SIZE = 64
PEAK_LR = 3e-4
MIN_LR = PEAK_LR * 0.1  # cosine decay floor, standard convention (10% of peak)
WARMUP_STEPS = 320
MAX_STEPS = 6400
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)

EVAL_INTERVAL = 200
EVAL_BATCHES = 20  # how many val batches to average per eval, for a stable estimate
CHECKPOINT_INTERVAL = 1000

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------- Learning rate schedule: warmup + cosine decay ----------

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        # linear warmup: 0 -> PEAK_LR over WARMUP_STEPS
        return PEAK_LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    # cosine decay from PEAK_LR down to MIN_LR over the remaining steps
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return MIN_LR + coeff * (PEAK_LR - MIN_LR)

# ---------- Eval: average loss over several val batches ----------

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        batch_losses = torch.zeros(EVAL_BATCHES)
        for i in range(EVAL_BATCHES):
            x, y = get_batch(split, BATCH_SIZE, BLOCK_SIZE, device=DEVICE)
            _, loss = model(x, y)
            batch_losses[i] = loss.item()
        losses[split] = batch_losses.mean().item()
    model.train()
    return losses

# ---------- Setup ----------

model = TinyStoriesLM(
    vocab_size=VOCAB_SIZE, n_layer=N_LAYER, n_head=N_HEAD,
    n_embd=N_EMBD, block_size=BLOCK_SIZE, dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params:,} parameters")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=PEAK_LR, betas=BETAS, weight_decay=WEIGHT_DECAY
)

best_val_loss = float("inf")

# ---------- Training loop ----------

print(f"Starting training: {MAX_STEPS} steps, batch_size={BATCH_SIZE}, block_size={BLOCK_SIZE}")
start_time = time.time()

for step in range(MAX_STEPS):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    x, y = get_batch("data/train", BATCH_SIZE, BLOCK_SIZE, device=DEVICE)

    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    # ---------- periodic eval ----------
    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - start_time
        steps_done = step + 1
        steps_per_sec = steps_done / elapsed
        eta = (MAX_STEPS - steps_done) / steps_per_sec if steps_per_sec > 0 else float("inf")

        print(f"[{steps_done:5d}/{MAX_STEPS}] "
              f"train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} "
              f"lr={lr:.2e} | elapsed={elapsed:6.1f}s | "
              f"{steps_per_sec:.2f} steps/s | ETA={eta:6.1f}s")

        # best-checkpoint: save whenever val loss improves
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "val_loss": best_val_loss,
            }, "checkpoint_best.pt")
            print(f"  -> new best val_loss, saved checkpoint_best.pt")

    # ---------- periodic snapshot ----------
    if step % CHECKPOINT_INTERVAL == 0 and step > 0:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        }, f"checkpoint_step{step}.pt")
        print(f"  -> periodic snapshot saved: checkpoint_step{step}.pt")

print(f"\nTraining complete in {time.time() - start_time:.1f}s. Best val_loss: {best_val_loss:.4f}")