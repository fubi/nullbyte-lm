import time
from cs336_workspace.tokenizer import Tokenizer

with open("tinystories_tokenizer_sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Training on {len(text)} characters, vocab_size=32000")

tok = Tokenizer(vocab_size=32000)

start = time.time()
tok.train([text], progress_every=1000)  # bigger step since we expect ~31,743 merges total
elapsed = time.time() - start

print(f"\nDone: {len(tok.merges)} merges in {elapsed:.1f}s")
tok.save("tinystories_tok_32k")