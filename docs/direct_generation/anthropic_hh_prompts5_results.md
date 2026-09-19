# Five prompts for `anthropic_hh_balanced`, one 600-row set each

The toolace study (`analysis/toolace_prompts5_results.md`) repeated on the largest highstakes
split. Five generator prompts written from measurements of the split
(`ANTHROPIC_HH_GENERATOR_PROMPTS.md`), each turned into a 600-row set by deepseek-v4-pro, each
fit once with **no base data** and scored on all four full highstakes eval splits.

## Protocol

Identical to the toolace study, so the two are comparable:
`scripts/generate_prompt_variants.py` (deepseek-v4-pro, temperature 1.0, 20 rows per call,
10 per label, 10 calls in flight, per-row guards for shape, label, ≤1024 gemma tokens,
novelty and leakage against every dev and eval hh user turn), then
`scripts/subsample_curve_concept.py --concept highstakes --no-base --dev-data
dev_samples/highstakes_500 --sizes 600 --draws 1`. Results in
`scripts/highstakes_anthropic_hh_prompts5.csv`. Every set came out 300/300, and **no row was
dropped for leakage in any run**.

One fit per set, so there is no error bar; single-fit movement here runs to about ±0.02.

## Results

| set | prompt | **anthropic_hh** | mean of 4 | toolace | mt | mts | dev |
|---|---|---|---|---|---|---|---|
| replica | 1. faithful replica | **0.976** | 0.858 | 0.762 | 0.801 | 0.895 | 0.860 |
| twin | 2. same conversation, good reply and bad | 0.973 | **0.884** | 0.808 | 0.803 | 0.951 | 0.910 |
| redteam | 4. red-team side + edgy lookalikes | 0.928 | 0.700 | 0.530 | 0.655 | 0.686 | 0.729 |
| long | 5. long conversations and drift | 0.927 | 0.846 | 0.641 | 0.888 | 0.927 | 0.828 |
| topicpair | 3. same topic, opposite stakes | 0.882 | 0.871 | 0.707 | 0.947 | 0.949 | 0.886 |

Reference line, same eval: **probe_iter0**, the 50-row llama70b base probe, scores
**0.947** on anthropic_hh and **0.900** mean.

## Reading

**Two sets beat the base probe on the target split.** `replica` (0.976) and `twin` (0.973)
are both above 0.947, from 600 generated rows and no base data at all. That did not happen on
toolace, where only the best set matched the base probe. anthropic_hh is a much easier split
to write for: it is ordinary chat, and the generator has seen a lot of it.

**`twin` is the best set overall** — top-two on hh *and* the best four-split mean (0.884), the
best dev (0.910), and the best toolace of the five. Writing each conversation twice, once with
a good reply and once with a useless one under the same label, is exactly the structure of the
underlying HH data (`chosen` and `rejected` completions sit in both classes), and it teaches
the probe that reply quality is not the label. It costs nothing on the target split relative
to the plain replica, and it generalises better.

**The specialist/generalist trade-off is much weaker here than on toolace.** On toolace the
toolace ranking and the four-split ranking were nearly opposite; here `twin` and `replica` are
first and second on hh and first and third on the mean. Only `redteam` behaves like a narrow
specialist: 0.928 on hh and 0.700 mean, with toolace at 0.530.

**`redteam` also carries a cue the prompt tried to forbid.** 219 of its 300 high-stakes rows
contain refusal language in the assistant turn, against 6 of 300 low-stakes rows. The prompt
says explicitly that a refusal is not the label and asks for the two classes to be matched on
how cooperative the assistant sounds; the generator did it anyway. The real split does not
work that way, which is the most likely reason this set is the weakest of the five everywhere
except its own split.

**Nobody reproduced the split's length cue, and it did not matter.** In the real split
low-stakes rows are *longer* (730 characters against 590, assistant turns 259 against 157). In
all five generated sets the high-stakes rows are the longer ones. The cue is therefore
inverted relative to eval, so `replica` and `twin` scoring 0.97 cannot be riding on length —
they are reading content.

## Deviations and incidents

- **`twin` was generated twice.** The novelty guard keys a row on its first user turn, which
  is precisely what a twin pair shares, so the first run dropped 99 of 199 rows as duplicates.
  `--dup-key user+reply` was added for it: the key becomes the first user turn plus the final
  assistant turn. The delivered set is 598 of 600 rows in pairs, and all 299 pairs carry a
  consistent label.
- **`topicpair` lost 20 rows to the shape guard** (rows that did not alternate user/assistant),
  the only set with a material shape loss.
- `scripts/generate_toolace_prompts5.py` was renamed `scripts/generate_prompt_variants.py` and
  given `--prompts-md`, `--split` and `--dup-key`, so one script serves both studies.

## Worth doing next

- Error bars on `twin` vs `replica` at n=540 × 8 draws; 0.976 against 0.973 is inside noise,
  but `twin`'s +0.026 on the four-split mean probably is not.
- `twin`'s structure is the transferable finding. Applying it to toolace (the same call
  written with a good and a clumsy assistant turn) is the obvious cross-check, since toolace's
  best sets were the ones that copied the split's form.
- Pool `twin` with `topicpair` — the two best generalists, strong on different splits.
