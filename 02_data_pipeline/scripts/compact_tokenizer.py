from cs336_workspace.tokenizer import Tokenizer

# load the sparse tokenizer (vocab_size=32000, but only 6913 real merges)
old = Tokenizer.load("tinystories_tok_32k")

# compute the real, compact vocab size: base bytes + actual merges + EOT
compact_vocab_size = 256 + len(old.merges) + 1
print(f"Old vocab_size: {old.vocab_size}, actual merges: {len(old.merges)}")
print(f"Compact vocab_size: {compact_vocab_size}")

# build a fresh Tokenizer at the correct size and transplant the merges in
new_tok = Tokenizer(vocab_size=compact_vocab_size)
new_tok.merges = old.merges          # same merge dict, ids already 256..7168 - unchanged
new_tok._build_vocab()               # rebuild vocab so EOT lands at the new, correct id

print(f"New eot_id: {new_tok.eot_id}")  # should be 7169, not 31999

# sanity check: encode/decode still round-trips correctly after the resize
sample = "Once upon a time, there was a little rabbit."
assert new_tok.decode(new_tok.encode(sample)) == sample
print("Round-trip check passed")

new_tok.save("tinystories_tok_compact")
print("Saved as tinystories_tok_compact.merges.txt / .vocab.json")