from tokenizer.tokenizer import Tokenizer, NUM_BASE_BYTES

tok = Tokenizer.load("shakespeare_full_tok_10k")

merged_tokens = {idx: tok.vocab[idx] for idx in tok.vocab if idx >= NUM_BASE_BYTES}
by_length = sorted(merged_tokens.items(), key=lambda kv: len(kv[1]), reverse=True)

print("Top 20 longest learned tokens:")
for idx, tok_bytes in by_length[:20]:
    print(f"  id={idx:5d}  {tok_bytes!r}")

with open("input.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

test_slice = full_text[500_000:501_000]
test_ids = tok.encode(test_slice)
print(f"\nCompression on same 1000-char slice: {len(test_ids)} tokens "
      f"({len(test_slice)/len(test_ids):.2f} chars/token)")
print(f"(previous vocab_size=5000 result was 307 tokens, 3.26 chars/token)")