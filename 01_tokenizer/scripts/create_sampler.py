with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()[:100_000]

with open("input_sample.txt", "w", encoding="utf-8") as f:
    f.write(text)