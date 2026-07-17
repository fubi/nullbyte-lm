import torch
import torch.nn.functional as F
from cs336_workspace.model import TinyStoriesLM
from cs336_workspace.tokenizer import Tokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def top_p_sample(logits: torch.Tensor, top_p: float) -> int:
    """
    Nucleus (top-p) sampling: keep the smallest set of highest-probability
    tokens whose cumulative probability just exceeds top_p, zero out
    everything else, renormalize, and sample from what's left.
    """
    probs = F.softmax(logits, dim=-1)  # (vocab_size,)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # keep tokens up to and including the first one that pushes
    # cumulative probability past top_p
    cutoff = cumulative > top_p
    # always keep at least the single highest-probability token,
    # even if it alone exceeds top_p
    cutoff[0] = False

    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()  # renormalize

    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx[sampled_sorted_idx].item()


@torch.no_grad()
def generate(
    model: TinyStoriesLM,
    tok: Tokenizer,
    prompt: str = "",
    max_new_tokens: int = 250,
    temperature: float = 0.7,  # updated from 1.0, based on empirical comparison
    top_p: float = 0.9,          # updated from 0.95
    block_size: int = 256,
) -> str:

    # seed: unconditional starts from EOT, prompted starts from encoded prompt
    if prompt:
        ids = tok.encode(prompt)
    else:
        ids = [tok.eot_id]

    for _ in range(max_new_tokens):
        # sliding window: only feed the model the last block_size tokens,
        # since RoPE/attention were only built for sequences up to that length
        context = ids[-block_size:]
        idx = torch.tensor([context], dtype=torch.long, device=DEVICE)

        logits, _ = model(idx)  # (1, T, vocab_size)
        next_token_logits = logits[0, -1, :] / temperature  # last position's logits only

        next_id = top_p_sample(next_token_logits, top_p)
        ids.append(next_id)

        if next_id == tok.eot_id:
            break  # story ended naturally

    return tok.decode(ids)


if __name__ == "__main__":
    tok = Tokenizer.load("../01_tokenizer/artifacts/tinystories_tok_compact")

    model = TinyStoriesLM(
        vocab_size=7170, n_layer=6, n_head=6, n_embd=384, block_size=256
    ).to(DEVICE)

    checkpoint = torch.load("../04_training/checkpoints/checkpoint_best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from step {checkpoint['step']}, val_loss {checkpoint['val_loss']:.4f}")

    print("\n--- Unconditional generation ---")
    story = generate(model, tok, prompt="", max_new_tokens=250)
    print(story)

    print("\n--- Prompted generation ---")
    story2 = generate(model, tok, prompt="Once upon a time, there was a little dog named", max_new_tokens=250)
    print(story2)