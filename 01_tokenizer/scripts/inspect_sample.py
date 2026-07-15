with open("input_sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

# does it look like play text (character names, "Exeunt", etc.) or prose preamble?
print(text[:500])
print("---")
print(text[50000:50500])  # peek at the middle too