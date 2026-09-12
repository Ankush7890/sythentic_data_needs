# Two set-level filters on the four llama70b high-stakes arms

Does removing *redundant* rows from a red-team or generated training set make the probe
better at equal training-set size? Two filters, four arms, one protocol.

Filters live in [`scripts/shape_diversity.py`](../scripts/shape_diversity.py); the fits are
[`scripts/highstakes_armfilter.csv`](../scripts/highstakes_armfilter.csv), rendered by
[`scripts/report_armfilter.py`](../scripts/report_armfilter.py).

## The arms

All four are the **same generator** (`meta-llama/llama-3.3-70b-instruct`), the **same
concept** (high-stakes), the **same probe** (`google/gemma-3-27b-it`, layer 32,
`linear_then_softmax`, single) and the **same 50-row base**
(`data/highstakes_llama70b_50.jsonl`), so they differ only in how the rows were obtained.
The three red-team arms come from `origin/experiment_hs_last`; their configs differ by one
knob each, which is what the arm names mean:

| arm | source | N | what it is |
| --- | --- | --- | --- |
| `general` | `probes/hs_gemma27b_llama70b_llamabase_itermemo150/redteam_postprocessed_iter10.jsonl` | 262 | red-team loop, **no** `eval.data_description` anywhere |
| `+desc` | `…_llamabase_evaldesc/…iter10.jsonl` | 40 | adds `eval.data_description` (judge / eval-scope side) |
| `+attacker` | `…_llamabase_evaldesc_attacker/…iter10.jsonl` | 62 | adds `show_eval_data_description: true` — the description also enters the **attacker's** prompt |
| `gen` | `data/highstakes_llama70b_600.jsonl` | 600 | the generator scaffold's one-shot 600-row set |

Every set is exactly class-balanced (131/131, 20/20, 31/31, 300/300). The red-team rows are
`{id, inputs, label}` with `label` ∈ `positive`/`negative`; they were mapped to
`high-stakes`/`low-stakes` and re-serialized into the standard `{inputs, labels}` schema.
A red-team set contains both the attacker's own submissions and the LLM-written
opposite-class partners for them — the filters were applied to the set as it stands, which
is the set that was actually trained on.

## The two filters

**Lexical confounders** (`filter_lexical_confounders`) is tuberlens'
`scripts/preprocessing_redteaming.filter_dataset`, generalized: bag-of-words unigrams
(`min_df=3`, `max_df=0.9`, ≤20k features) → balanced logistic regression → drop the rows
classified with the highest confidence. Three changes were needed to use it here: the
binary label comes from the two class labels actually present (the original compares against
the literal string `"high-stakes"`, which makes it a silent no-op on any other concept); the
percentile cut runs **inside each class**, so class balance cannot drift and become a second
uncontrolled variable; and it reports what it did. `max(predict_proba)` is confidence in the
*predicted* class, so a row the lexical model is confidently **wrong** about is dropped too —
inherited behaviour, kept deliberately.

**Shape mix** (`filter_shape_redundancy`) is new, and maximizes the **variety of
conversation shapes** rather than cutting a tail — a percentile threshold cannot express
"keep one of each". Content is thrown away entirely; the descriptor is the role sequence
(`ua`, `uaua`, `sua`, … — which subsumes turn count, whether a system turn is part of the
row, and which role the row ends on: the whole `SHAPE (exact)` block of
`scripts/split_specs.py` in one string) plus six log-scaled length features. Two stages:

1. **quota over role sequences** — hand the budget out one slot at a time, round-robin over
   the groups that still have rows. As-uniform-as-possible is the maximum-entropy allocation
   subject to what the set contains, so a shape holding 3% of the rows comes out at its fair
   share instead of at 3%.
2. **max-min spread inside each group** — rows sharing a role sequence still differ in
   length, and length is shape (45 characters and 3000 characters are both `ua`). Greedy
   k-center on the standardized length vector, seeded at the group medoid. No RNG anywhere.

The descriptor is computed on the **combine/convert-transformed** messages — the ones the
extraction path actually tokenizes — since `combine_consecutive_messages` merges same-role
neighbours and `uaa` reaches the probe as `ua`. (Measured: the transform is a no-op on all
four of these sets, but it is now checked rather than assumed.)

Both filters run `per_label=True` and return `floor(keep·N)/2` rows per class, so the two
filtered pools are the same size, exactly class-balanced, and a balanced draw from either is
comparable to one from the unfiltered set.

## What the filters removed (`--keep 0.8`)

| arm | pool | shape H<sub>norm</sub> | biggest-shape share | MI(shape;label) bits | mean NN dist | lexical mean conf | BoW train acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `general` | 262 → 208 | 0.792 → **0.853** | 0.473 → **0.346** | 0.0000 → 0.0000 | 0.400 → 0.460 | 0.799 → 0.756 | 0.992 |
| `+desc` | 40 → 32 | 0.878 → **0.915** | 0.375 → **0.281** | 0.0262 → 0.0338 | 0.662 → 0.771 | 0.732 → 0.693 | 0.975 |
| `+attacker` | 62 → 48 | 0.703 → **0.803** | 0.613 → **0.500** | 0.0000 → 0.0000 | 0.773 → 0.917 | 0.777 → 0.740 | 1.000 |
| `gen` | 600 → 480 | 1.000 → 1.000 | 1.000 → 1.000 | 0.0000 → 0.0000 | 0.165 → 0.214 | 0.970 → 0.962 | 1.000 |

Role sequences, before → after:

| arm | before | after |
| --- | --- | --- |
| `general` | `ua` 72, `uau` 36, `uaua` 124, `uauau` 26, `uauaua` 4 | `ua` 72, `uau` 36, `uaua` 70, `uauau` 26, `uauaua` 4 |
| `+desc` | `ua` 10, `uau` 6, `uaua` 15, `uauau` 8, 1 odd | `ua` 9, `uau` 6, `uaua` 8, `uauau` 8, 1 odd |
| `+attacker` | `ua` 4, `uau` 2, `uaua` 38, `uauau` 12, `uauaua` 6 | `ua` 4, `uau` 2, `uaua` 24, `uauau` 12, `uauaua` 6 |
| `gen` | `ua` 600 | `ua` 480 |

Four things to read off this table before any AUROC:

- **`gen` has exactly one shape.** All 600 rows are `ua` — user turn, assistant turn. The
  role-sequence stage has nothing to do there, so the `gen` shape column measures the
  length-spread stage **alone**. (It moves mean nearest-neighbour distance 0.165 → 0.214, so
  it is not idle; it is just not diversifying *sequences*.)
- **`+attacker` is the most shape-collapsed red-team arm** — 61% of its rows are `uaua`,
  against 47% for `general`. Pushing the description into the attacker's prompt made its
  submissions more uniform in shape, not less.
- **Bag-of-words separates every arm essentially perfectly in-sample** (0.975–1.000 train
  accuracy), and on `gen` it does so with 0.970 mean confidence. Lexical shortcuts are
  available in all four sets; they are most blatant in the generated one.
- **MI(shape; label) is 0.0000 on three of four arms.** Shape carries no label information
  there, so flattening the shape distribution cannot be deleting signal — the guard that
  makes the shape filter safe to apply. On `+desc` it is 0.026 bits and *rises* to 0.034.

The two filters keep genuinely different rows, so neither is a proxy for the other:

| arm | pool size | kept by both | lexical only | shape only | Jaccard |
| --- | --- | --- | --- | --- | --- |
| `general` | 208 | 165 | 43 | 43 | 0.657 |
| `+desc` | 32 | 24 | 8 | 8 | 0.600 |
| `+attacker` | 48 | 37 | 11 | 11 | 0.627 |
| `gen` | 480 | 387 | 93 | 93 | 0.675 |

One diagnostic on the per-class cut: a **global** percentile cut (what `filter_dataset`
does) would have taken 70 high-stakes against 50 low-stakes rows out of `gen`, i.e. shifted
the class ratio from 50/50 to 52.6/47.4 while the size is what is nominally being measured.
That is why the cut here is taken within each class.

## Protocol for the AUROC numbers

Per arm: three pools (unfiltered, lexical, shape-mix), each drawn from **8 times** at the
**same n** = 0.6·N, class-balanced n/2 per class, through
`scripts/subsample_curve_concept.py`. Pools are 0.8·N, so every pool can serve every draw
and no pool is exhausted (which would collapse its sd to zero). The three columns of an arm
therefore differ in *which rows were available to draw from* and in nothing else.

    base           data/highstakes_llama70b_50.jsonl (always in full)
    dev            dev_samples/highstakes_500 — the 500-row cut, early stopping + reporting
    eval           eval_sets/highstakes, FULL splits, no subsampling
    probe          gemma-3-27b L32, linear_then_softmax, single, seed 42
    transforms     combine_consecutive_messages = convert_tool_to_assistant = True
    n              general 156, +desc 24, +attacker 36, gen 360

Draws are seeded on `(pool filename, n, draw)`, so the three pools of an arm get
**independent** draws rather than paired ones; Δ carries the standard error of the
difference, `sqrt(sd_a²/8 + sd_b²/8)`.

**Read the two small arms with care.** The probe steps the optimizer every 4th batch of
`ceil(rows/16)`, so `+desc` (50 + 24 = 74 rows → 5 batches) and `+attacker` (86 rows → 6
batches) take **one optimizer step per epoch**: they train (200 epochs, patience 50), but
with an effective batch covering the whole training set. `general` gets 3 steps/epoch and
`gen` 6. All four run at the repo-default `gradient_accumulation_steps=4`, so no
accum-boundary caveat applies, but the small arms are coarser fits than the large ones.

## Results

96/96 fits. Every cell is 8 class-balanced draws; `Δ` is against that arm's unfiltered
column with the standard error of the difference.

### eval mean — the four high-stakes splits, full

| arm | n | original | lexical | Δ lex | shape-mix | Δ shape |
| --- | --- | --- | --- | --- | --- | --- |
| `general` | 156 | 0.9110 ±0.0143 | 0.9141 ±0.0066 | +0.0031 ±0.0056 | 0.9174 ±0.0125 | +0.0063 ±0.0067 |
| `+desc` | 24 | 0.9110 ±0.0236 | 0.9056 ±0.0238 | −0.0054 ±0.0119 | 0.9012 ±0.0146 | −0.0099 ±0.0098 |
| `+attacker` | 36 | 0.8849 ±0.0197 | 0.9015 ±0.0218 | +0.0166 ±0.0104 | 0.9122 ±0.0147 | **+0.0273 ±0.0087** |
| `gen` | 360 | 0.8844 ±0.0222 | 0.8784 ±0.0158 | −0.0060 ±0.0096 | 0.8552 ±0.0373 | **−0.0292 ±0.0153** |

### dev mean — the 500-row dev cut

| arm | n | original | lexical | Δ lex | shape-mix | Δ shape |
| --- | --- | --- | --- | --- | --- | --- |
| `general` | 156 | 0.9062 ±0.0126 | 0.9137 ±0.0123 | +0.0075 ±0.0062 | 0.9140 ±0.0136 | +0.0078 ±0.0065 |
| `+desc` | 24 | 0.9022 ±0.0234 | 0.8924 ±0.0279 | −0.0098 ±0.0129 | 0.8906 ±0.0218 | −0.0115 ±0.0113 |
| `+attacker` | 36 | 0.8766 ±0.0223 | 0.8995 ±0.0272 | +0.0229 ±0.0124 | 0.9186 ±0.0207 | **+0.0420 ±0.0107** |
| `gen` | 360 | 0.8800 ±0.0232 | 0.8683 ±0.0201 | −0.0117 ±0.0108 | 0.8415 ±0.0427 | **−0.0385 ±0.0172** |

**Dev and eval agree in sign on all eight arm × filter cells**, and agree on the ordering of
the four shape-mix effects. With 8 unpaired draws the standard error on a Δ is 0.006–0.015,
so a single cell needs |Δ| > ~0.02 to stand on its own; the across-arm pattern is what the
run actually establishes.

## The headline: whether shape-mix helps is predicted by the set's starting shape entropy

Ordered by the normalized shape entropy of the unfiltered set — the same
`H_norm` column from the diagnostics table above, computed before any filtering:

| arm | H<sub>norm</sub> before | distinct shapes | Δ shape-mix (eval) | Δ shape-mix (dev) |
| --- | --- | --- | --- | --- |
| `+attacker` | 0.703 | 5 | **+0.0273 ±0.0087** (3.1 se) | **+0.0420 ±0.0107** (3.9 se) |
| `general` | 0.792 | 5 | +0.0063 ±0.0067 (0.9 se) | +0.0078 ±0.0065 (1.2 se) |
| `+desc` | 0.878 | 5 | −0.0099 ±0.0098 (−1.0 se) | −0.0115 ±0.0113 (−1.0 se) |
| `gen` | 1.000 | **1** | **−0.0292 ±0.0153** (−1.9 se) | **−0.0385 ±0.0172** (−2.2 se) |

Monotone in all four, on both dev and eval. A line through the four points has
`r = −0.995` (eval) / `r = −0.990` (dev), a slope of −0.19 / −0.26 AUROC per unit of
`H_norm`, and crosses zero at **H<sub>norm</sub> ≈ 0.84** on both.

Four points and one concept, so the constant is not the finding — the sign rule is: **a
shape-diversity filter pays only on a set whose shapes are already collapsed, and costs
AUROC on one that is not.** `+attacker` is the most collapsed set in the experiment (61% of
its rows are `uaua`, because pushing the eval description into the attacker's prompt made
its submissions more uniform) and it is the one arm where rebalancing shapes is worth 0.027
eval AUROC while also *cutting* draw-to-draw variance (±0.0147 against ±0.0197). `+desc`,
whose shapes were already near-uniform, has nothing to gain and loses a little.

Note that the biggest-single-shape share does **not** order the four arms
(0.613 / 0.473 / 0.375 / 1.000 against Δ +0.027 / +0.006 / −0.010 / −0.029): `gen` has the
most concentrated single shape *and* the worst outcome, but it has only one shape to
concentrate. Entropy over the shapes present is the quantity that ranks them, because it
sees the count as well as the concentration.

### Why the single-shape set is harmed: max-min selects outliers

`gen` is 600/600 `ua`, so stage A (the role-sequence quota) has nothing to allocate and the
filter degenerates to stage B alone — greedy k-center on the length vector. **k-center is an
outlier-seeking objective**: on a homogeneous set, "maximum spread" means "prefer the
longest and shortest rows", which is close to the opposite of what training data wants. The
per-split numbers show it plainly:

| `gen` pool | anthropic_hh | mt_balanced | mts_balanced | toolace | mean |
| --- | --- | --- | --- | --- | --- |
| original | 0.9094 ±0.0208 | 0.8181 ±0.0820 | 0.9419 ±0.0238 | 0.8682 ±0.0070 | 0.8844 ±0.0222 |
| shape-mix | 0.9233 ±0.0165 | **0.7310 ±0.0866** | **0.8875 ±0.0604** | 0.8789 ±0.0073 | 0.8552 ±0.0373 |

The loss is concentrated on `mt_balanced` (−0.087) and `mts_balanced` (−0.054), the two
length-extreme splits — one very long document, one many very short turns — and the variance
on `mts_balanced` more than doubles. Meanwhile the two mid-length splits improve slightly.
Selecting length extremes did not teach the extremes; it spent 20% of the budget on rows
that are unrepresentative in the one dimension the filter was optimizing.

The fix, if this filter is used again on a set with few distinct shapes, is to gate stage B:
skip it when a group holds most of the set, or replace the max-min objective with a
**quantile-stratified** pick (sample the length distribution evenly rather than seeking its
extremes). That is untested here.

### Where `+attacker`'s gain comes from

| `+attacker` pool | anthropic_hh | mt_balanced | mts_balanced | toolace | mean |
| --- | --- | --- | --- | --- | --- |
| original | 0.9526 ±0.0183 | 0.8134 ±0.0534 | 0.9247 ±0.0506 | 0.8490 ±0.0183 | 0.8849 ±0.0197 |
| lexical | 0.9639 ±0.0111 | 0.8260 ±0.0601 | **0.9639 ±0.0183** | 0.8523 ±0.0151 | 0.9015 ±0.0218 |
| shape-mix | 0.9559 ±0.0076 | **0.9200 ±0.0460** | 0.9383 ±0.0239 | 0.8345 ±0.0129 | 0.9122 ±0.0147 |

Almost all of the shape-mix gain is **`mt_balanced`, +0.107** — the split that is one very
long clinical document. Cutting `uaua` from 61% to 50% of the set, and letting the
length-extreme picks inside each group through, is exactly a shift toward long single-exchange
rows, and that is the split it buys. It costs a little on `toolace_balanced` (−0.015), the
split whose defining content (a function list in a system prompt) none of these sets contain
anyway. The two filters gain on *different* splits — lexical on `mts_balanced`, shape on
`mt_balanced` — consistent with their 0.63 Jaccard overlap.

### The lexical filter is noise on the mean, but it stabilizes `general`

| arm | Δ lex (eval) | sd original → lexical |
| --- | --- | --- |
| `general` | +0.0031 ±0.0056 (0.6 se) | ±0.0143 → **±0.0066** |
| `+desc` | −0.0054 ±0.0119 (−0.5 se) | ±0.0236 → ±0.0238 |
| `+attacker` | +0.0166 ±0.0104 (1.6 se) | ±0.0197 → ±0.0218 |
| `gen` | −0.0060 ±0.0096 (−0.6 se) | ±0.0222 → ±0.0158 |

No cell reaches 2 se and the signs are split 2/2, so on this evidence dropping the most
lexically-predictable fifth of a set changes the mean by nothing measurable — even though a
bag-of-words model separates every one of these sets at 0.975–1.000 in-sample accuracy, and
`gen` at 0.970 mean confidence. **Lexical separability that high does not mean the probe is
using it**, which is the useful negative result: the confound is available in all four sets
and removing it is not what limits them. What it does do on `general` and `gen` is halve the
draw-to-draw variance (±0.0143 → ±0.0066; ±0.0222 → ±0.0158) — the most confidently
classified rows are also the most redundant ones, so which of them a draw happens to catch
matters less once they are gone.

## What this does not settle

- **One concept, one generator, one probe.** Everything here is high-stakes / llama70b /
  gemma-3-27b L32. The `H_norm` threshold of 0.84 is fitted to four points and should be
  treated as "there is a crossover", not as a number to reuse.
- **Two small arms are coarse fits.** `+desc` (74 training rows) and `+attacker` (86) take
  one optimizer step per epoch, so their probes are trained with an effective batch covering
  the whole set. The arm with the largest effect is one of the two, which is exactly where
  one would want the check repeated at `--accum 1`.
- **Draws are unpaired.** The harness seeds each draw on the pool's filename, so the three
  columns of an arm do not share subsets. Paired draws (the same row indices where the pools
  overlap) would cut the standard error on Δ and are the cheapest way to sharpen every number
  above.
- **`keep=0.8` was never varied.** Whether `+attacker`'s gain grows at `keep=0.6` or reverses
  is unmeasured, and a keep-curve is the obvious next arm.
- **The red-team sets were filtered whole**, attacker submissions and their LLM-written
  opposite-class partners together. The partners are frequently minimal word swaps of their
  originals ($100 → $100,000), which is the single most confounded thing in these sets; a run
  that filters the originals alone would measure something different.
