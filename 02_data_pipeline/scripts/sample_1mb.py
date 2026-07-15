with open("TinyStories-train.txt", "r", encoding="utf-8") as f:
    text = f.read(1_000_000)  # read only first ~1MB, don't load the whole file into memory

with open("tinystories_tokenizer_sample.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Wrote {len(text)} characters for tokenizer training")