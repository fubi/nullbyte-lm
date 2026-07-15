"""
A from-scratch byte-level BPE tokenizer, GPT-2 style.

Pipeline: pretokenize (regex) -> byte-encode -> BPE merge training ->
          encode (apply merges) -> decode (ids -> bytes -> str)

Special token <|endoftext|> is reserved at the last vocab id and is never
split or merged - it's carved out of the text before pretokenization runs.
"""

import re as stdlib_re
import json
import time

try:
    import regex as re  # needed for \p{L} / \p{N} unicode category matching
except ImportError as e:
    raise ImportError(
        "This tokenizer requires the 'regex' package for Unicode-aware "
        "pretokenization (stdlib 're' can't match \\p{L}/\\p{N}).\n"
        "Install it with: pip install regex"
    ) from e

__all__ = [
    "Tokenizer",
    "TokenizerError",
    "pretokenize",
    "chunk_to_byte_ids",
    "get_stats",
    "merge",
    "GPT2_SPLIT_PATTERN",
    "EOT_TOKEN",
    "NUM_BASE_BYTES",
]

# ---------- Config ----------

EOT_TOKEN = "<|endoftext|>"
NUM_BASE_BYTES = 256

GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class TokenizerError(Exception):
    """Raised for tokenizer-specific failures (bad config, corrupt vocab, etc.)."""
    pass


# ---------- Stateless helpers (pure functions, no tokenizer state needed) ----------

def pretokenize(text: str) -> list[str]:
    """Split raw text into word-like chunks, GPT-2 style."""
    return re.findall(GPT2_SPLIT_PATTERN, text)


def chunk_to_byte_ids(chunk: str) -> list[int]:
    """Convert a text chunk into a list of raw UTF-8 byte values (0-255)."""
    return list(chunk.encode("utf-8"))


def get_stats(chunks: list[list[int]]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair of ids occurs, across all chunks."""
    counts = {}
    for ids in chunks:
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# ---------- Tokenizer: stateful wrapper around merges/vocab/special tokens ----------

class Tokenizer:
    """
    Byte-level BPE tokenizer.

    Usage:
        tok = Tokenizer(vocab_size=5000)
        tok.train(["some training text", "more text..."])
        ids = tok.encode("hello world<|endoftext|>")
        text = tok.decode(ids)
        tok.save("my_tokenizer")

        tok2 = Tokenizer.load("my_tokenizer")
    """

    def __init__(self, vocab_size: int):
        if vocab_size <= NUM_BASE_BYTES + 1:
            raise TokenizerError(
                f"vocab_size must be > {NUM_BASE_BYTES + 1} "
                f"(256 base bytes + 1 special token), got {vocab_size}"
            )
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - NUM_BASE_BYTES - 1
        self.eot_id = vocab_size - 1
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self._build_vocab()  # base vocab (just bytes + EOT) until train()/load() runs

    # ---------- training ----------

    def train(self, texts: list[str], progress_every: int = 100, verbose: bool = True) -> None:
        """Train BPE merges on a list of raw training documents.

        progress_every: print a progress line every N merges.
        verbose: set False to train silently.
        """
        if not texts:
            raise TokenizerError("train() requires at least one non-empty text")

        chunks = []
        for text in texts:
            for piece in pretokenize(text):
                byte_ids = chunk_to_byte_ids(piece)
                if byte_ids:  # guard against empty chunks
                    chunks.append(byte_ids)

        if not chunks:
            raise TokenizerError("train() found no usable content after pretokenization")

        self.merges = self._train_merges(chunks, self.num_merges, progress_every, verbose)
        self._build_vocab()

    @staticmethod
    def _train_merges(
        chunks: list[list[int]],
        num_merges: int,
        progress_every: int = 100,
        verbose: bool = True,
    ) -> dict[tuple[int, int], int]:
        merges = {}
        chunks = [list(c) for c in chunks]
        start = time.time()

        for i in range(num_merges):
            stats = get_stats(chunks)
            if not stats:
                if verbose:
                    print(f"Stopped early at {i}/{num_merges} merges - corpus fully merged, no pairs left")
                break  # corpus fully merged down, nothing left to pair

            # Tie-break rule: highest frequency wins; among ties, the pair with
            # the numerically smallest ids wins (arbitrary but deterministic -
            # negating flips max()'s "largest wins" into "smallest wins").
            pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))

            new_id = NUM_BASE_BYTES + i
            chunks = [merge(ids, pair, new_id) for ids in chunks]
            merges[pair] = new_id

            if verbose and (i % progress_every == 0 or i == num_merges - 1):
                elapsed = time.time() - start
                done = i + 1
                rate = done / elapsed if elapsed else 0
                eta = (num_merges - done) / rate if rate else float("inf")
                print(f"[{done:5d}/{num_merges}] {100*done/num_merges:5.1f}% | "
                      f"merged {pair} -> {new_id} | elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s")

        return merges

    def _build_vocab(self) -> None:
        vocab = {idx: bytes([idx]) for idx in range(NUM_BASE_BYTES)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        vocab[self.eot_id] = EOT_TOKEN.encode("utf-8")
        self.vocab = vocab

    # ---------- encode / decode ----------

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        ids = list(ids)
        while len(ids) >= 2:
            stats = get_stats([ids])
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids, treating <|endoftext|> as atomic."""
        pieces = stdlib_re.split(f"({stdlib_re.escape(EOT_TOKEN)})", text)

        ids = []
        for piece in pieces:
            if piece == EOT_TOKEN:
                ids.append(self.eot_id)
                continue
            if piece == "":
                continue
            for chunk in pretokenize(piece):
                byte_ids = chunk_to_byte_ids(chunk)
                ids.extend(self._encode_chunk(byte_ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text. Raises TokenizerError on unknown ids."""
        try:
            raw_bytes = b"".join(self.vocab[i] for i in ids)
        except KeyError as e:
            raise TokenizerError(
                f"decode() received id {e.args[0]} which is not in this "
                f"tokenizer's vocab (size {len(self.vocab)}). This usually means "
                f"ids came from a different tokenizer/training run, or the "
                f"vocab file is corrupted/mismatched."
            ) from e
        return raw_bytes.decode("utf-8", errors="replace")

    # ---------- serialization ----------

    def save(self, path_prefix: str) -> None:
        """Save merges (.merges.txt) and a human-readable vocab (.vocab.json)."""
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        with open(f"{path_prefix}.merges.txt", "w", encoding="utf-8") as f:
            f.write(f"{self.vocab_size}\n")
            for (p0, p1), _new_id in ordered:
                f.write(f"{p0} {p1}\n")

        readable = {
            idx: tok.decode("utf-8", errors="replace")
            for idx, tok in self.vocab.items()
        }
        with open(f"{path_prefix}.vocab.json", "w", encoding="utf-8") as f:
            json.dump(readable, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path_prefix: str) -> "Tokenizer":
        """Load a tokenizer previously saved with save()."""
        try:
            with open(f"{path_prefix}.merges.txt", "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError as e:
            raise TokenizerError(f"No merges file found at {path_prefix}.merges.txt") from e

        if not lines:
            raise TokenizerError(f"{path_prefix}.merges.txt is empty")

        try:
            vocab_size = int(lines[0])
        except ValueError as e:
            raise TokenizerError(
                f"Expected vocab_size integer on line 1 of {path_prefix}.merges.txt, "
                f"got {lines[0]!r}"
            ) from e

        expected_merge_lines = vocab_size - NUM_BASE_BYTES - 1
        actual_merge_lines = len(lines) - 1
        if actual_merge_lines > expected_merge_lines:
            raise TokenizerError(
                f"{path_prefix}.merges.txt header says vocab_size={vocab_size} "
                f"(allows at most {expected_merge_lines} merge lines) but file has "
                f"{actual_merge_lines} merge lines. File may be corrupted or "
                f"from a different training run."
            )

        tok = cls(vocab_size=vocab_size)
        merges = {}
        for i, line in enumerate(lines[1:]):
            try:
                p0, p1 = map(int, line.split())
            except ValueError as e:
                raise TokenizerError(f"Malformed merge line {i + 2}: {line!r}") from e
            merges[(p0, p1)] = NUM_BASE_BYTES + i

        tok.merges = merges
        tok._build_vocab()
        return tok