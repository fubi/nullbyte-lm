import time
import numpy as np
from cs336_workspace.tokenizer import Tokenizer

TARGET_SIZE = 150_000_000  # 150MB
READ_BUFFER = TARGET_SIZE + 1_000_000  # read a bit extra, so we can trim to a clean boundary

print("Reading corpus slice...")
with open("TinyStories-train.txt", "r", encoding="utf-8") as f:
    text = f.read(READ_BUFFER)

# trim back to the last complete story boundary at/before our target size
cutoff = text.rfind("<|endoftext|>", 0, TARGET_SIZE)
if cutoff == -1:
    raise ValueError("No <|endoftext|> found before target size - check the file/format")
text = text[:cutoff + len("<|endoftext|>")]

print(f"Using {len(text)} characters ({len(text)/1_000_000:.1f}MB), cleanly cut on a story boundary")

print("Loading tokenizer...")
tok = Tokenizer.load("tinystories_tok_compact")

print("Encoding... (this touches every character once, should be roughly linear time)")
start = time.time()
ids = tok.encode(text)
elapsed = time.time() - start
print(f"Encoded {len(text)} chars -> {len(ids)} tokens in {elapsed:.1f}s "
      f"({len(text)/len(ids):.2f} chars/token)")

# 95/5 split - contiguous, not shuffled (standard for this kind of split)
split_idx = int(len(ids) * 0.95)
train_ids = ids[:split_idx]
val_ids = ids[split_idx:]
print(f"Train: {len(train_ids)} tokens | Val: {len(val_ids)} tokens")

# save as raw binary, uint16 (fits our vocab_size=7170 comfortably)
train_arr = np.array(train_ids, dtype=np.uint16)
val_arr = np.array(val_ids, dtype=np.uint16)

train_arr.tofile("train.bin")
val_arr.tofile("val.bin")

print(f"Saved train.bin ({train_arr.nbytes / 1_000_000:.1f}MB) "
      f"and val.bin ({val_arr.nbytes / 1_000_000:.1f}MB)")