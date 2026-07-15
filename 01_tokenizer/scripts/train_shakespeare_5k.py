import time
from tokenizer.tokenizer import Tokenizer

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Training on {len(text)} characters")

tok = Tokenizer(vocab_size=5000)

start = time.time()
tok.train([text], progress_every=100)
elapsed = time.time() - start

print(f"\nDone: {len(tok.merges)} merges in {elapsed:.1f}s")
tok.save("shakespeare_full_tok")