import numpy as np
import torch

def get_batch(split: str, batch_size: int, block_size: int, device: str = "mps"):
    """
    Sample a random batch of (x, y) sequences from train.bin or val.bin.

    x: input tokens, shape (batch_size, block_size)
    y: target tokens, shape (batch_size, block_size) - same as x shifted by 1
    """
    filename = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(filename, dtype=np.uint16, mode="r")

    max_start = len(data) - block_size - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    x = np.stack([data[i : i + block_size] for i in starts])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in starts])

    x = torch.from_numpy(x.astype(np.int64))
    y = torch.from_numpy(y.astype(np.int64))

    return x.to(device), y.to(device)


if __name__ == "__main__":
    x, y = get_batch("train", batch_size=4, block_size=256, device="cpu")
    print(f"x shape: {x.shape}, dtype: {x.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    print(f"x[0][:10]: {x[0][:10].tolist()}")
    print(f"y[0][:10]: {y[0][:10].tolist()}")
    assert torch.equal(y[0][:-1], x[0][1:])
    print("Shift-by-one relationship verified")