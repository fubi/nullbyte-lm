import torch
from model import TinyStoriesLM

def test_forward_output_shapes():
    model = TinyStoriesLM(vocab_size=100, n_layer=2, n_head=2, n_embd=16, block_size=32)
    idx = torch.randint(0, 100, (4, 32))
    logits, loss = model(idx)
    assert logits.shape == (4, 32, 100)
    assert loss is None  # no targets given

def test_loss_computed_when_targets_given():
    model = TinyStoriesLM(vocab_size=100, n_layer=2, n_head=2, n_embd=16, block_size=32)
    idx = torch.randint(0, 100, (4, 32))
    targets = torch.randint(0, 100, (4, 32))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.item() > 0  # cross-entropy is always positive

def test_weight_tying_is_real():
    model = TinyStoriesLM(vocab_size=100, n_layer=1, n_head=2, n_embd=16, block_size=32)
    assert model.output_head.weight is model.token_emb.weight  # same object, not just equal values

def test_untrained_loss_near_random_baseline():
    # a fresh, untrained model should produce a loss close to ln(vocab_size) -
    # i.e. roughly what you'd get from a uniform random guess over the vocab.
    # this is a strong sanity check that the architecture isn't badly broken.
    vocab_size = 100
    model = TinyStoriesLM(vocab_size=vocab_size, n_layer=2, n_head=2, n_embd=16, block_size=32)
    idx = torch.randint(0, vocab_size, (8, 32))
    targets = torch.randint(0, vocab_size, (8, 32))
    _, loss = model(idx, targets)
    expected_random_loss = torch.log(torch.tensor(float(vocab_size)))
    assert abs(loss.item() - expected_random_loss.item()) < 1.0  # within a reasonable margin

def test_full_config_runs_end_to_end():
    # exact config we locked in for real training
    model = TinyStoriesLM(vocab_size=7170, n_layer=6, n_head=6, n_embd=384, block_size=256)
    idx = torch.randint(0, 7170, (4, 256))
    targets = torch.randint(0, 7170, (4, 256))
    logits, loss = model(idx, targets)
    assert logits.shape == (4, 256, 7170)
    assert loss is not None

def test_param_count_in_expected_range():
    model = TinyStoriesLM(vocab_size=7170, n_layer=6, n_head=6, n_embd=384, block_size=256)
    n_params = sum(p.numel() for p in model.parameters())
    # weight tying means token_emb params aren't double-counted, so this should
    # land close to our ~13.5M estimate from the design stage
    assert 10_000_000 < n_params < 20_000_000