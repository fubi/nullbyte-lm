# Inference / Generation — Deep Dive

Sampling text from the trained ~13.5M parameter TinyStories model
(`checkpoint_best.pt`, step 6400, val_loss 1.6038).

## 1. Design decisions & rationale

| Decision | Choice | Why |
|---|---|---|
| Sampling strategy | Temperature + top-p (nucleus) | More adaptive than top-k: keeps however many tokens are needed to cover `top_p` cumulative probability, rather than a fixed count — narrows sharply when the model is confident, widens when it's uncertain |
| Seeding | Both unconditional (`<|endoftext|>`) and prompted | No reason to pick one — both are useful: unconditional shows the model's unprompted "default" story shape, prompted tests whether it can build on a given premise |
| max_new_tokens | 250 | Near the full `block_size=256`, allowing near-complete stories rather than truncated fragments |
| Context handling | Sliding window (last 256 tokens) | RoPE's cache and attention were only built for sequences up to `block_size` — generation must never feed the model more context than that |
| Stop condition | Early-stop on generated `<|endoftext|>` | Lets the model end a story naturally rather than always running the full token budget |
| Initial params | `temperature=1.0, top_p=0.95` | "More random/creative" starting point, to see the model's raw, unconstrained behavior before tuning |

## 2. The math: top-p (nucleus) sampling

Given the model's raw output logits for the next token:

1. Convert to probabilities via softmax.
2. Sort tokens by probability, descending.
3. Walk down the sorted list, accumulating cumulative probability, until
   it just exceeds `top_p`.
4. Zero out every token *after* that cutoff point, renormalize the
   remaining probabilities so they sum to 1, and sample from that reduced
   set.

**Why this is more adaptive than top-k:** top-k always keeps exactly `k`
candidate tokens, regardless of how confident the model is. Top-p instead
keeps *however many* tokens are needed to cover a target probability mass —
when the model is very confident (one token dominates), the nucleus
shrinks to just 1-2 tokens; when it's genuinely uncertain (probability
spread across many plausible words), the nucleus widens automatically.
This tends to avoid both "too repetitive" (always taking the single most
likely token) and "too incoherent" (sampling from the model's long,
low-probability tail) failure modes better than a fixed-count cutoff.

**Temperature's role:** logits are divided by `temperature` *before*
softmax. Values `<1.0` sharpen the distribution (more confident, more
repetitive); values `>1.0` flatten it (more random, more diverse, more
error-prone). At `temperature=1.0`, this step is a no-op — the model's
raw probabilities are used unmodified, meaning `top_p` was the only
active lever in the first tested configuration.

## 3. Real generated examples (temperature=1.0, top_p=0.95)

**Unconditional:**

> Once upon a time, there was a little squirrel named Sam. Sam loved to
> find nuts and make friends with his family. One day, he saw a big acorn
> in the park. It was brown and had a long tail. Sam was so happy! Sam ran
> to his family and said, "Look, I found a acorn!" His family agreed, and
> they all took out some acorns. They put it on a branch and shared it
> with other animals. Sam was very happy with his reorn, and he thanked
> his family for their help. From that day on, Sam was never scared of the
> loud acorn again. He learned that even the smallest friends can help
> make the right choice. And everyone lived happily ever after.

**Prompted** (seed: *"Once upon a time, there was a little dog named"*):

> ...Max. Max was an honest dog who loved to share his treats with his
> friends. One day, Max saw a big box in the park. He wanted to get the
> treats, but he did not know how. Max asked his friend, a bird named Ben,
> "Can you help me get the treats?" Ben said, "Yes, Max. We can add things
> to the apples. If you give me them, I will be very careful." Max agreed
> and said, "Thank you, Ben." Tim and Ben started to pick many things
> together. They saw apples, bananas, and bunnies. At last, they found a
> big surprise. There was a yummy apple for both of them! They both ate
> and were very happy. From that day on, Max and Ben always shared their
> treats. They always played together and shared their treats. And they
> always shared their treats with each other. And they were never scared
> again.

## 4. Honest analysis

**What's working:**
- Grammar and sentence structure are consistently correct across both
  samples — no broken syntax.
- Genuine story structure: character introduction → small goal/problem →
  resolution → a tacked-on moral, matching TinyStories' typical shape.
- `<|endoftext|>` was generated correctly to end both stories — confirms
  the special-token handling built into the tokenizer is working
  end-to-end through to model behavior, not just at the tokenizer layer.
- Prompted generation correctly builds on the given seed rather than
  ignoring it.

**Real flaws, not glossed over:**
- **Word-formation glitch**: *"his reorn"* — a nonsense blend, should be
  "acorn." A close-but-wrong token choice, typical of a model that's
  learned a word's phonetic/token shape without fully locking in its
  identity.
- **Attribute-binding error**: *"It was brown and had a long tail"*
  describes the *squirrel*, but the sentence structure attaches it to the
  acorn instead. The model pattern-matches plausible sentence shapes
  without reliably tracking which noun each description belongs to.
- **Entity-tracking failure**: *"Tim and Ben started to pick..."* — "Tim"
  was never introduced; the actual characters were Max and Ben. Losing
  track of who's in a scene over ~150-200 tokens is a well-known failure
  mode at this model scale — reliable entity tracking tends to require
  meaningfully more parameters and/or training.
- **Repetition loop**: *"they always shared their treats... they always
  shared their treats with each other"* — a classic small-LM degenerate
  loop, falling into a locally high-probability cycle.
- **Templated endings**: phrases like *"and they lived happily ever
  after"* / *"never scared again"* likely reflect the corpus's own
  repetitive phrasing (thousands of TinyStories use nearly identical
  closing lines) rather than novel generation — genuinely hard to fully
  separate "learned the genre convention" from "memorized a common
  phrase" without more rigorous evaluation than reading two samples.

**Overall assessment:** exactly the profile expected of a correctly-
implemented ~13.5M parameter model trained for ~51 minutes on ~35M
tokens — locally fluent, loosely coherent at the paragraph level, breaks
down on longer-range consistency (entity tracking, repetition avoidance).
This is a *capability ceiling*, not a pipeline bug — consistent with the
training deep-dive's finding that val loss was still improving at the
final step, meaning more training and/or more parameters would plausibly
reduce (not eliminate) these specific failure patterns.

## 5. Next: parameter exploration

Planned follow-up: testing lower `top_p` (narrower, more conservative
sampling) and different `temperature` values, to see whether the
repetition-loop and entity-tracking issues are partially an artifact of
sampling randomness rather than purely a model-capability limit. Results
to be added once run.

## 5.1. Parameter exploration results

**Tested: `temperature=0.7, top_p=0.9`** (tightened from the initial
`temperature=1.0, top_p=0.95`).

**Unconditional:**

> Once upon a time there was a girl called Mia. She was three years old
> and she loved to explore. One day, Mia was walking in the park when she
> saw a big, yellow ball. She was so excited and wanted to play with it.
> Mia asked her mom if she could borrow the ball. Her mom said yes, so Mia
> took the ball and ran home. When she got home, she put the ball back in
> the basket and went to show her mom. Mia's mom said, "Mia, you have to
> be careful with the ball. It is not safe to play with." Mia nodded and
> promised to be careful. She went back to the park and played with the
> ball all day.

**Prompted** (same seed as before):

> ...Spot. Spot was a very adventurous dog. He loved to play outside and
> run around. One day, Spot saw a big ball in the yard. He wanted to play
> with it. Spot tried to reach the ball, but it was too high. He tried and
> tried, but he could not reach it. Then, a big cat named Fluffy came by.
> Fluffy saw Spot trying to get the ball. Fluffy wanted to help Spot.
> Fluffy had an idea. She found a long stick and used it to get the ball
> down. Spot was so happy! He thanked Fluffy and they played together all
> day. From that day on, Spot and Fluffy were best friends.

**Result: a clear qualitative improvement over `temp=1.0/top_p=0.95`.**
No word-formation glitches (no "reorn"-style errors), no repetition
loops, and entity tracking held correctly across both stories — no
unintroduced characters appearing mid-story (contrast with the earlier
"Tim" error). Story 2 in particular shows genuine causal structure: an
obstacle (ball too high), a helper arriving, a concrete solution (fetching
a stick), and a resolution — not just template-filling.

**A prediction this result corrected:** before running this, the
hypothesis was that tightening `temperature`/`top_p` might reduce
nonsense errors while *increasing* repetition risk (since a more
confident, narrower distribution could make the model more likely to
loop on a "safe" continuation). That did not happen here — repetition
actually went away, not worse. Worth stating plainly: **this prediction
was wrong**, at least in this sample. A remaining minor issue: story 1
has a small continuity slip (the ball is put away, then the story
continues as though play resumes at the park) — subtler than the earlier
errors, but a reminder the model still isn't fully consistent.

**Caveat on sample size:** this comparison is two generations per
setting — genuinely suggestive of an improvement, but not a rigorous
claim. A proper comparison would generate many samples per setting and
measure something more systematic (e.g. repetition rate, a simple
entity-consistency check) rather than reading a handful of examples.
That kind of rigorous eval is a natural next step, not done here.

**Tested: `temperature=0.5, top_p=0.8`** (tightened further).

**Result: the improving trend did not continue monotonically.** Literal
repetition loops remained absent (consistent with the previous run), but
*different* coherence errors surfaced:

- **Object confusion** (unconditional sample): the story's stated problem
  ("the swing is too high") gets silently resolved via an unrelated
  action (climbing a tree with a stick) — the model tracks the
  *template* ("problem → stick-based solution") without reliably tracking
  *which* object the problem concerned. Same failure family as the
  earlier "brown, long tail" attribute-binding error from the first test.
- **Entity duplication** (prompted sample): "Tweet" is introduced as *"a
  little bird"*, then the story later refers to *"their friend, Bird"* as
  if a second, separate character — despite Tweet already being the bird.
  Functionally the same failure family as the earlier "Tim" entity error,
  manifesting as duplication rather than invention this time.

**Revised conclusion:** tightening sampling from `1.0/0.95` to `0.7/0.9`
produced a real, visible improvement. Tightening further to `0.5/0.8` did
not continue that improvement — it appears closer to a local optimum than
a monotonic curve, with over-tightening trading one error type
(repetition) for others (object/attribute confusion, entity duplication)
that were likely present underneath all along, just less visible next to
the more obvious repetition/glitch errors at higher randomness. This
reframes the earlier result: sampling parameters can suppress *some*
surface symptoms, but the underlying coherence/entity-tracking
limitations are a property of the trained model itself, not something
sampling tuning can fully resolve.

**Tested: `temperature=0.6, top_p=0.85`** (midpoint between the two prior tests).

**Result: a distinct, not intermediate, failure profile.** Neither
literal repetition nor the object/entity-confusion errors from `0.5/0.8`
reappeared, but two new issues surfaced:

- **Ungrounded plot twist** (unconditional sample): a bug being helped is
  suddenly revealed to "not be a bug at all, but a big, friendly bear" —
  grammatically clean, but with no setup supporting the transformation. A
  different failure category from earlier runs: not a word glitch or
  entity-tracking slip, but an apparent default toward a "surprising happy
  twist" template without narrative grounding.
- **Unresolved story arc** (prompted sample): ends on *"he could not find
  the ball and he could not play with it anymore"* — a genuine break from
  TinyStories' near-universal happy-resolution convention, which every
  other tested sample (including this run's own unconditional story)
  otherwise followed.

**Revised conclusion:** the relationship between sampling tightness and
output quality is **not a single smooth curve** — each tested setting
exposed a distinct kind of weakness (raw repetition at `1.0/0.95`;
attribute-binding and entity errors at `0.5/0.8`; ungrounded twists and
unresolved arcs at `0.6/0.85`) rather than a predictable, monotonically
worsening pattern as parameters tighten. Across all four settings tested,
**`temperature=0.7, top_p=0.9` remains the strongest** — the only
configuration where both generated samples were simultaneously coherent
*and* properly resolved. Recommended as the default going forward.

**Final caveat:** this conclusion rests on 2 samples per setting across 4
settings (8 stories total) — sufficient to observe real, distinct
qualitative patterns, but not a statistically rigorous parameter sweep. A
proper follow-up would generate many samples per setting and score them
against concrete criteria (resolution presence, entity consistency,
repetition rate) rather than manual reading.

**Confirmation run at `temperature=0.7, top_p=0.9`** (after setting it as
the script default): both generated stories were coherent and properly
resolved, with no entity-tracking errors or word-formation glitches.
However, mild issues persisted — a near-duplicate sentence in one story
("Sarah ran to the cloud" repeated with slightly different phrasing) and
an ungrounded narrative twist in the other (a ball unexpectedly revealed
to be magic and talking) — echoing, in reduced form, the repetition and
ungrounded-twist patterns seen at other settings. **Conclusion stands:**
`0.7/0.9` is the best setting found, but not a setting that eliminates
these failure modes — it reduces their frequency/severity rather than
removing them, consistent with these being underlying model-capability
limitations that sampling parameters can only partially mask.