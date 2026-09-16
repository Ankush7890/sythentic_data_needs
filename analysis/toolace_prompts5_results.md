# Five prompts for `toolace_balanced`, one 600-row set each

Five generator prompts written from measurements of the split (`TOOLACE_GENERATOR_PROMPTS.md`),
each turned into a 600-row set by deepseek-v4-pro, each fit once with **no base data** and
scored on all four full highstakes eval splits. The question was which way of describing the
split to a generator produces training data that carries a probe on `toolace_balanced`.

## Protocol

- **Generation.** `scripts/generate_toolace_prompts5.py`, `deepseek/deepseek-v4-pro`,
  temperature 1.0, 20 rows per call (10 per label, both labels in one call so the prompt's
  "match the classes on surface features" instruction has something to act on), 10 calls in
  flight. Per row: shape check, label check, ≤1024 gemma-3-27b tokens, novelty against the
  rows already kept, and a leak check against every dev and eval toolace user turn. Each call
  carries the same "do not reuse these openings" block `generate_split_targeted.py` uses.
  Every set came out at exactly 300/300, and **no row was dropped for leakage anywhere**.
- **Fit.** `scripts/subsample_curve_concept.py --concept highstakes --no-base --dev-data
  dev_samples/highstakes_500 --sizes 600 --draws 1`, gemma-3-27b layer 32, results in
  `scripts/highstakes_toolace_prompts5.csv`. This is the protocol behind the `base='none'`
  rows of `scripts/highstakes_gen90_dev500.csv`, so those arms are directly comparable.
- **One fit per set, so there is no error bar.** Single-fit movement on this concept runs to
  about ±0.02; differences smaller than that are not readable.

## Results

| set | prompt | **toolace** | mean of 4 | anthropic_hh | mt | mts | dev |
|---|---|---|---|---|---|---|---|
| endings | 4. non-call endings + history wrapper | **0.886** | 0.618 | 0.815 | 0.384 | 0.389 | 0.619 |
| replica | 1. faithful replica | 0.867 | 0.712 | 0.675 | 0.631 | 0.673 | 0.700 |
| grid | 2. read/write × stakes grid | 0.853 | 0.767 | 0.764 | 0.802 | 0.648 | 0.769 |
| longtail | 5. long-tail domains, mixed tool specs | 0.847 | 0.625 | 0.599 | 0.432 | 0.624 | 0.644 |
| deceptive | 3. serious-sounding low, mundane-sounding high | 0.731 | 0.777 | 0.647 | 0.795 | 0.936 | 0.802 |

Reference lines, same eval, same splits:

| | toolace | mean of 4 |
|---|---|---|
| probe_iter0, the 50-row llama70b base probe alone | 0.856 | 0.900 |
| earlier deepseek toolace-targeted set, no shots (600, no base) | 0.692 | 0.797 |
| earlier deepseek toolace-targeted set, with shots (600, no base) | 0.687 | 0.726 |

## Reading

**Every one of the five beats the earlier whole-split toolace sets on toolace**, four of them
by 0.15 or more. The earlier sets were written from `scripts/split_specs.py`'s description,
whose label rule — high-stakes means the call acts on the world, low-stakes means it only
retrieves — does not match the data: the split labels plain lookups in trading, medication
safety and law enforcement as high-stakes, and write actions in harmless settings as
low-stakes. All five prompts here define the label as the stakes of the situation instead,
and that one change is worth about 0.15 on this split.

**The best toolace sets are the worst everywhere else.** `endings` reaches 0.886, above the
base probe and above anything in the earlier toolace arms, while scoring 0.384 and 0.389 on
mt and mts — below chance, so its probe is anti-correlated with those splits. `deceptive` is
the mirror image: the best four-split mean (0.777) and the worst toolace (0.731). Ranking the
five by toolace and by mean gives almost exactly opposite orders. None of the five beats the
50-row base probe's four-split mean of 0.900, which is the honest ceiling comparison: these
are single-split specialists fit on 600 rows with no base data.

**What the winning prompts have in common** is reproducing the split's *form*, not widening
its content. `endings` (the ToolACE history wrapper, missing-parameter replies, no-suitable-
function replies, tool traces) and `replica` (the scaffold, one bare call list per row) are
first and second. `longtail`, which deliberately pushed into unusual deployments, is fourth,
and `deceptive`, which deliberately broke the persona-to-label correlation, is last on
toolace — that prompt teaches the probe not to trust exactly the cue the split relies on.

## Deviations from what the prompts say

- **Prompt 4's tool traces are not `tool`-role rows.** The pipeline's parser accepts only
  `system`, `user` and `assistant`, so the prompt asks for tool results as assistant turns
  prefixed `Tool result: `, and the script merges consecutive assistant turns. Every row in
  that set is therefore 3 messages, as `combine_consecutive_messages` would have made it
  anyway. 249 of its 600 rows carry the history wrapper and 294 end on a bare call.
- **Prompt 4 was generated twice.** The first attempt capped completions at 32k tokens, and
  about a third of its calls spent the whole budget on reasoning and returned nothing; it
  stalled at 236 rows. It was rerun at 96k (the model allows 393k) and completed. The
  abandoned attempt's raw replies are kept locally, uncommitted, at
  `logs/toolace_prompts5/endings_32k_attempt/` (every call's raw reply is saved there).
- **No dev prefetch from Kaggle.** The published dev blobs hold the *full* dev splits
  (1028 rows for anthropic_hh), while these fits validate on `dev_samples/highstakes_500`
  (125-row cuts), which the row-count check rejects. The 500 dev rows were extracted locally
  instead. Eval activations came from Kaggle as usual.

## Worth doing next

- Repeat the two leaders at n=540 × 8 draws to put an error bar on the 0.886 / 0.867, since
  the ranking's top three sit inside single-fit noise of each other.
- Pool them. On the `hard_split_experiments` branch, pooling part-specialists beat every
  single specialist; `endings` + `deceptive` are the natural pair here, one strong on toolace
  and one strong on everything else.
