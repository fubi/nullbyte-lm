import pytest
from cs336_workspace.tokenizer import (
    Tokenizer, TokenizerError,
    pretokenize, chunk_to_byte_ids, get_stats, merge,
    NUM_BASE_BYTES,
)

# ---------- Pretokenizer (still module-level, unchanged) ----------

def test_pretokenize_basic_split():
    assert pretokenize("Hello world!") == ["Hello", " world", "!"]

def test_pretokenize_contraction():
    assert pretokenize("don't") == ["don", "'t"]

def test_pretokenize_leading_space_distinguishes_chunks():
    assert pretokenize("dog") == ["dog"]
    assert pretokenize(" dog") == [" dog"]

def test_chunk_to_byte_ids_ascii():
    assert chunk_to_byte_ids("AB") == [65, 66]

def test_chunk_to_byte_ids_multibyte_unicode():
    assert chunk_to_byte_ids("é") == [195, 169]


# ---------- get_stats / merge (still module-level, unchanged) ----------

def test_get_stats_counts_adjacent_pairs():
    stats = get_stats([[1, 2, 3, 1, 2]])
    assert stats[(1, 2)] == 2
    assert stats[(2, 3)] == 1

def test_get_stats_sums_across_chunks():
    stats = get_stats([[1, 2], [1, 2], [3, 4]])
    assert stats[(1, 2)] == 2
    assert stats[(3, 4)] == 1

def test_merge_replaces_non_overlapping_pairs():
    assert merge([1, 1, 1], (1, 1), 99) == [99, 1]

def test_merge_no_match_returns_unchanged():
    assert merge([1, 2, 3], (5, 6), 99) == [1, 2, 3]


# ---------- Tokenizer construction / validation ----------

def test_init_rejects_vocab_size_too_small():
    with pytest.raises(TokenizerError):
        Tokenizer(vocab_size=257)  # <= 256 base + 1 special, no room for merges

def test_init_accepts_minimal_valid_vocab_size():
    tok = Tokenizer(vocab_size=258)  # exactly 1 merge slot
    assert tok.num_merges == 1

def test_train_rejects_empty_input():
    tok = Tokenizer(vocab_size=280)
    with pytest.raises(TokenizerError):
        tok.train([])


# ---------- Training ----------

def test_train_learns_most_frequent_pair_first():
    tok = Tokenizer(vocab_size=258)  # room for exactly 1 merge
    # "ab" appears in every word -> should be the first (and only) merge
    tok.train(["ab ab ab cd ef"])
    assert (97, 98) in tok.merges  # 'a'=97, 'b'=98
    assert tok.merges[(97, 98)] == NUM_BASE_BYTES

def test_train_respects_vocab_size_budget():
    tok = Tokenizer(vocab_size=260)  # room for exactly 3 merges
    tok.train(["abcdefabcdefabcdef"])
    assert len(tok.merges) <= 3

def test_train_tiebreak_is_deterministic():
    tok1 = Tokenizer(vocab_size=257 + 1)
    tok1.train(["abcd"])
    tok2 = Tokenizer(vocab_size=257 + 1)
    tok2.train(["abcd"])
    assert tok1.merges == tok2.merges


# ---------- Encode / Decode ----------

def test_encode_decode_roundtrip_ascii():
    tok = Tokenizer(vocab_size=280)
    tok.train(["the cat sat on the mat"])
    text = "the cat sat"
    assert tok.decode(tok.encode(text)) == text

def test_encode_decode_roundtrip_unicode_no_merges():
    tok = Tokenizer(vocab_size=258)  # trivial vocab, no meaningful merges trained
    text = "Здравейте 世界 🚀"
    assert tok.decode(tok.encode(text)) == text

def test_encode_empty_string():
    tok = Tokenizer(vocab_size=280)
    tok.train(["some text"])
    assert tok.encode("") == []

def test_decode_raises_tokenizer_error_on_unknown_id():
    tok = Tokenizer(vocab_size=280)
    tok.train(["some text"])
    with pytest.raises(TokenizerError):
        tok.decode([999999])  # id far outside any valid range


# ---------- Special token handling ----------

def test_special_token_never_split_by_pretokenizer():
    tok = Tokenizer(vocab_size=280)
    tok.train(["cat dog"])
    ids = tok.encode("cat<|endoftext|>dog")
    assert ids.count(tok.eot_id) == 1

def test_special_token_boundary_blocks_cross_merges():
    tok = Tokenizer(vocab_size=280)
    tok.train(["cat dog catdog"])  # even with "catdog" trained together
    ids = tok.encode("cat<|endoftext|>dog")
    eot_pos = ids.index(tok.eot_id)
    assert tok.decode(ids[:eot_pos]) == "cat"
    assert tok.decode(ids[eot_pos + 1:]) == "dog"


# ---------- Serialization ----------

def test_save_load_roundtrip(tmp_path):
    tok = Tokenizer(vocab_size=280)
    tok.train(["the cat sat on the mat"])
    prefix = str(tmp_path / "test_tok")
    tok.save(prefix)

    loaded = Tokenizer.load(prefix)
    assert loaded.merges == tok.merges
    assert loaded.vocab_size == tok.vocab_size

def test_save_load_preserves_encode_output(tmp_path):
    tok = Tokenizer(vocab_size=280)
    tok.train(["the cat sat on the mat"])
    prefix = str(tmp_path / "test_tok2")
    tok.save(prefix)

    loaded = Tokenizer.load(prefix)
    text = "the cat"
    assert tok.encode(text) == loaded.encode(text)

def test_load_missing_file_raises_tokenizer_error(tmp_path):
    with pytest.raises(TokenizerError):
        Tokenizer.load(str(tmp_path / "does_not_exist"))

def test_load_rejects_header_mismatch(tmp_path):
    prefix = str(tmp_path / "corrupt")
    # header claims vocab_size=258 (allows at most 1 merge line) but file has 2 -
    # more merges than the budget permits is structurally impossible, hence corrupt
    with open(f"{prefix}.merges.txt", "w") as f:
        f.write("258\n97 98\n99 100\n")
    with pytest.raises(TokenizerError):
        Tokenizer.load(prefix)