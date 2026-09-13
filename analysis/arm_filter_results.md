# Two set-level filters on sixteen high-stakes arms

Does removing *redundant* rows from a red-team or generated training set make the probe
better at equal training-set size? Two filters, **four attackers x four arms**, 384 fits,
one protocol.

**Answer: no, and nothing measured here predicts when it would.** Across the sixteen arms
both filters average slightly below zero and neither is distinguishable from no filtering.
Individual arms move by up to 0.045 AUROC in both directions, but every candidate rule for
which arm moves which way — including one this document previously asserted — is refuted by
the full set.

Filters live in [`scripts/shape_diversity.py`](../scripts/shape_diversity.py); the fits are
[`scripts/highstakes_armfilter.csv`](../scripts/highstakes_armfilter.csv), rendered by
[`scripts/report_armfilter.py`](../scripts/report_armfilter.py).

## The arms

Four attackers — `meta-llama/llama-3.3-70b-instruct`, `deepseek/deepseek-v4-pro`,
`openai/gpt-oss-120b`, `nemotron` — each contribute four arms on the **same concept**
(high-stakes) and the **same probe** (`google/gemma-3-27b-it`, layer 32,
`linear_then_softmax`, single). Each attacker's arms are fitted on **its own** 50-row base,
as its red-team run was, so an arm is single-source. The three red-team arms come from
`origin/experiment_hs_last`; their configs differ by one knob each, which is what the arm
names mean:

| arm | source | what it is |
| --- | --- | --- |
| `general` | `probes/hs_gemma27b_<attacker>_<base>_itermemo150/redteam_postprocessed_iter10.jsonl` | red-team loop, **no** `eval.data_description` anywhere |
| `+desc` | `…_evaldesc/…iter10.jsonl` | adds `eval.data_description` (judge / eval-scope side) |
| `+attacker` | `…_evaldesc_attacker/…iter10.jsonl` | adds `show_eval_data_description: true` — the description also enters the **attacker's** prompt |
| `gen` | `data/highstakes_<tag>_600.jsonl` | the generator scaffold's one-shot 600-row set |

Row counts, `general / +desc / +attacker / gen`: llama70b 262 / 40 / 62 / 600, deepseek
394 / 390 / 440 / 600, gpt-oss 442 / 312 / 310 / 600, nemotron 558 / 380 / 368 / 600.
Every set is exactly class-balanced. Per-set shape and lexical diagnostics are in
`scripts/highstakes_armfilter_sets.json`; the exact source path and base file for each is a
field there. The red-team rows are
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

Full per-set numbers are in `scripts/highstakes_armfilter_sets.json`; the shape statistics
that matter for reading the results are the `H_norm` and `top shape %` columns of the results
table below. Three properties hold across all sixteen sets:

- **All four `gen` sets have exactly one shape** — 600/600 rows `ua`, user turn then
  assistant turn, for every attacker. The role-sequence quota has nothing to allocate there,
  so their shape column measures the length-spread stage **alone**.
- **Bag-of-words separates every set essentially perfectly in-sample** (0.975–1.000 train
  accuracy; the `gen` sets at 0.963–0.978 mean confidence). Lexical shortcuts are available
  in all sixteen.
- **MI(shape; label) is at or near zero everywhere** (max 0.009 bits, on `nemotron/general`),
  so flattening the shape distribution is never deleting label signal — the guard that makes
  the shape filter safe to apply, in the sense of not destroying information. Safe is not the
  same as useful, as the results show.

The two filters keep genuinely different rows — Jaccard 0.60–0.68 across the arms — so
neither is a proxy for the other. And a **global** percentile cut (what `filter_dataset`
does) would have taken 70 high-stakes against 50 low-stakes rows out of llama70b's `gen`
alone, shifting the class ratio while size is what is nominally being measured; that is why
the cut here is taken within each class.

The shape distributions differ enormously between attackers, which is what makes the null
result below informative rather than a narrow test: `gpt-oss/+desc` is 89.7% one shape
(H_norm 0.253) while `nemotron/+desc` spreads over 22 sequences with no shape above 20.5%
(H_norm 0.820), out to `uauauauauauauauauauauau`.

## Protocol for the AUROC numbers

Per arm: three pools (unfiltered, lexical, shape-mix), each drawn from **8 times** at the
**same n** = 0.6·N, class-balanced n/2 per class, through
`scripts/subsample_curve_concept.py`. Pools are 0.8·N, so every pool can serve every draw
and no pool is exhausted (which would collapse its sd to zero). The three columns of an arm
therefore differ in *which rows were available to draw from* and in nothing else.

    dev            dev_samples/highstakes_500 — the 500-row cut, early stopping + reporting
    eval           eval_sets/highstakes, FULL splits, no subsampling
    probe          gemma-3-27b L32, linear_then_softmax, single, seed 42
    transforms     combine_consecutive_messages = convert_tool_to_assistant = True
    base           each attacker's OWN 50-row set, always in full
    n              0.6 x N per arm, 24-360 rows

Draws are seeded on `(pool filename, n, draw)`, so the three pools of an arm get
**independent** draws rather than paired ones; Δ carries the standard error of the
difference, `sqrt(sd_a²/8 + sd_b²/8)`.

**Read llama70b's two small arms with care.** The probe steps the optimizer every 4th batch
of `ceil(rows/16)`, so llama70b's `+desc` (50 + 24 = 74 rows → 5 batches) and `+attacker`
(86 rows → 6 batches) take **one optimizer step per epoch**: they train (200 epochs,
patience 50), but with an effective batch covering the whole training set. Every other arm
in the sweep is 186–360 drawn rows and gets 3–6 steps per epoch. All sixteen run at the
repo-default `gradient_accumulation_steps=4`, so no accum-boundary caveat applies, but
llama70b's two small arms are coarser fits than the rest — which matters, because the single
largest effect in the experiment is one of them.

## Results

384 fits: 16 arms x 3 pools x 8 draws. Every cell is 8 class-balanced draws; `Δ` is against
that arm's unfiltered column with the standard error of the difference. Regenerate with
`scripts/report_armfilter.py`.

### eval mean — the four high-stakes splits, full

| attacker | arm | N | n | H_norm | top shape % | original | Δ lex | Δ shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama70b | general | 262 | 156 | 0.792 | 47.3 | 0.9110 ±0.0143 | +0.0031 ±0.0056 | +0.0063 ±0.0067 |
| llama70b | +desc | 40 | 24 | 0.878 | 37.5 | 0.9110 ±0.0236 | −0.0054 ±0.0119 | −0.0099 ±0.0098 |
| llama70b | +attacker | 62 | 36 | 0.703 | 61.3 | 0.8849 ±0.0197 | +0.0166 ±0.0104 | **+0.0273 ±0.0087** |
| llama70b | gen | 600 | 360 | 1.000 | 100.0 | 0.8844 ±0.0222 | −0.0060 ±0.0096 | **−0.0292 ±0.0153** |
| deepseek | general | 394 | 236 | 0.585 | 65.0 | 0.8657 ±0.0089 | −0.0147 ±0.0101 | **−0.0446 ±0.0160** |
| deepseek | +desc | 390 | 234 | 0.531 | 67.9 | 0.8845 ±0.0233 | +0.0062 ±0.0090 | +0.0071 ±0.0094 |
| deepseek | +attacker | 440 | 264 | 0.721 | 31.1 | 0.9022 ±0.0076 | +0.0015 ±0.0040 | −0.0076 ±0.0055 |
| deepseek | gen | 600 | 360 | 1.000 | 100.0 | 0.8965 ±0.0151 | −0.0021 ±0.0067 | −0.0114 ±0.0062 |
| gpt-oss | general | 442 | 264 | 0.349 | 85.1 | 0.8996 ±0.0109 | +0.0016 ±0.0043 | −0.0071 ±0.0046 |
| gpt-oss | +desc | 312 | 186 | 0.253 | 89.7 | 0.8927 ±0.0176 | −0.0030 ±0.0100 | **+0.0166 ±0.0065** |
| gpt-oss | +attacker | 310 | 186 | 0.468 | 73.9 | 0.9220 ±0.0076 | **−0.0149 ±0.0040** | −0.0030 ±0.0049 |
| gpt-oss | gen | 600 | 360 | 1.000 | 100.0 | 0.8705 ±0.0146 | +0.0118 ±0.0075 | +0.0088 ±0.0065 |
| nemotron | general | 558 | 334 | 0.720 | 28.1 | 0.8102 ±0.0136 | −0.0250 ±0.0257 | −0.0213 ±0.0180 |
| nemotron | +desc | 380 | 228 | 0.820 | 20.5 | 0.8604 ±0.0305 | −0.0131 ±0.0127 | +0.0065 ±0.0136 |
| nemotron | +attacker | 368 | 220 | 0.682 | 35.3 | 0.8560 ±0.0211 | −0.0002 ±0.0094 | −0.0043 ±0.0086 |
| nemotron | gen | 600 | 360 | 1.000 | 100.0 | 0.8777 ±0.0118 | +0.0088 ±0.0046 | +0.0027 ±0.0051 |

### Across the sixteen arms, treating an arm as the unit

| filter | mean Δ | median | negative | range | t |
| --- | --- | --- | --- | --- | --- |
| lexical | −0.0022 ±0.0027 | −0.0012 | 9/16 | −0.0250 … +0.0166 | −0.80 |
| shape-mix | −0.0040 ±0.0044 | −0.0036 | 9/16 | −0.0446 … +0.0273 | −0.90 |

Dev agrees: lexical −0.0029 ±0.0034, shape-mix −0.0051 ±0.0053 (10/16 negative), and the
per-arm signs match eval on 13/16 arms for shape-mix.

**Neither filter beats not filtering.** Both point slightly negative, neither reaches one
standard error, and the median arm loses a little under both. At `keep=0.8` on these sets,
the best thing to do with the most lexically-predictable fifth of a set, or with the rows
that make its shape distribution lumpy, is to leave them in.

## No predictor survives

The per-arm effects are not noise — five cells sit past 2 se and they run in both directions
(+0.027, +0.017 against −0.045, −0.021, −0.015). Something real differs between arms. But
every rule proposed during the run failed on later arms:

| candidate predictor of Δ shape-mix | correlation over 16 arms |
| --- | --- |
| shape entropy of the set (H_norm) | −0.140 |
| dominant-shape share | +0.025 |
| distinct shapes in the set | −0.060 |
| rows in the set | −0.375 |
| unfiltered AUROC of the arm | +0.241 |
| bag-of-words train accuracy | +0.158 |

**A retraction.** An earlier version of this document reported, from the four llama70b arms
alone, that Δ shape-mix was monotone in `H_norm` with `r = −0.995` and a crossover at
`H_norm ≈ 0.84`, and proposed it as a rule for when to apply the filter. **That was an
artifact of four points.** Over sixteen arms the correlation is −0.14, and the rule's
sharpest prediction — `gpt-oss/general` at `H_norm` 0.349 should gain +0.092 — came back
−0.007. Two pairs of sibling arms settle it directly: deepseek's `general` and `+desc` have
near-identical shape statistics (H_norm 0.585 vs 0.531, dominant shape 65% vs 68%, same
attacker, same base) and give −0.045 against +0.007; gpt-oss's `general` and `+desc`
(0.349 vs 0.253) give −0.007 against +0.017.

Two other ideas died the same way:

- **Arm type does not carry.** `+attacker` gives +0.0273 / −0.0076 / −0.0030 / −0.0043 across
  llama70b / deepseek / gpt-oss / nemotron. The one large positive is llama70b's 62-row arm,
  the smallest and coarsest fit in the sweep, and it did not reproduce on any other attacker.
- **The single-shape case does not carry either**, and this is the one that had a mechanism
  behind it. With 600/600 rows in one role sequence the quota stage provably cannot act, so
  the filter reduces to greedy k-center on lengths — an outlier-seeking objective. That
  reasoning is still correct about what the filter *does*; it does not fix the *sign* of the
  result. The four `gen` arms give −0.0292 / −0.0114 / +0.0088 / +0.0027: two clear losses,
  two small gains.

The largest correlation left is with **set size** (−0.375), i.e. the filter doing better on
smaller sets — but smaller sets are also the coarser, noisier fits, `|r| = 0.375` over 16
points is well inside what chance produces, and it is the sort of relation this document has
already been burned by once. It is recorded, not believed.

## Variance is not systematically improved either

An early reading of the llama70b arms was that the lexical filter buys stability even when it
does not buy accuracy. Over sixteen arms that does not hold as a rule:

| filter | median sd ratio (filtered / original) | range | inflated |
| --- | --- | --- | --- |
| lexical | 0.89 | 0.43 – 5.28 | 8/16 arms |
| shape-mix | 0.76 | 0.33 – 4.99 | 5/16 arms |

The central tendency is a mild tightening, but the tails are violent in both directions and
land on different arms for the two filters. The extremes: shape-mix on `deepseek/general`
(±0.0089 → ±0.0443) and lexical on `nemotron/general` (±0.0136 → ±0.0715, with draws running
0.645 to 0.857 where the unfiltered pool never leaves 0.79–0.83). Removing a fifth of a set
by either criterion can make which rows you happen to draw matter far more than it did.

## Where the two big effects come from

The two arms that moved most, per eval split — the same filter helping one split and hurting
another is the shape of every large effect here:

`llama70b / +attacker`, the largest gain:

| pool | anthropic_hh | mt_balanced | mts_balanced | toolace | mean |
| --- | --- | --- | --- | --- | --- |
| original | 0.9526 ±0.0183 | 0.8134 ±0.0534 | 0.9247 ±0.0506 | 0.8490 ±0.0183 | 0.8849 ±0.0197 |
| lexical | 0.9639 ±0.0111 | 0.8260 ±0.0601 | **0.9639 ±0.0183** | 0.8523 ±0.0151 | 0.9015 ±0.0218 |
| shape-mix | 0.9559 ±0.0076 | **0.9200 ±0.0460** | 0.9383 ±0.0239 | 0.8345 ±0.0129 | 0.9122 ±0.0147 |

`deepseek / general`, the largest loss:

| pool | anthropic_hh | mt_balanced | mts_balanced | toolace | mean |
| --- | --- | --- | --- | --- | --- |
| original | 0.9159 ±0.0158 | 0.8370 ±0.0292 | 0.9280 ±0.0213 | 0.7818 ±0.0238 | 0.8657 ±0.0089 |
| lexical | 0.9310 ±0.0154 | 0.7906 ±0.0795 | 0.9260 ±0.0333 | 0.7562 ±0.0296 | 0.8509 ±0.0272 |
| shape-mix | 0.9052 ±0.0415 | **0.7406 ±0.1007** | 0.8740 ±0.1010 | 0.7644 ±0.0389 | 0.8210 ±0.0443 |

`mt_balanced` — one very long clinical document per row, no assistant turn — is the split
that moves in both: +0.107 on llama70b's `+attacker`, −0.096 on deepseek's `general`, and it
is also the highest-variance split in the sweep. Since the shape filter's length stage is
precisely what decides how many long single-exchange rows survive, `mt_balanced` is the split
it reaches; which direction it moves is not something the filter controls.

## What this does not settle

- **One concept, one probe, one keep fraction.** Everything is high-stakes / gemma-3-27b L32
  / `keep=0.8`. A keep-curve (0.5, 0.65, 0.8, 0.9) on a handful of arms is the obvious next
  experiment and the cheapest, since the pools are already built.
- **Draws are unpaired.** The harness seeds each draw on the pool's filename, so an arm's
  three columns do not share subsets. Paired draws would cut the standard error on every Δ
  by roughly the correlation between pools and would sharpen the five significant cells.
- **The red-team sets were filtered whole**, attacker submissions and their LLM-written
  opposite-class partners together. The partners are frequently minimal word swaps of their
  originals ($100 → $100,000) — the single most confounded thing in these sets — and a run
  that filters only the originals would measure something different.
- **The filters are one implementation each.** The shape filter's stage B (max-min k-center)
  is outlier-seeking by construction; a quantile-stratified length pick, or gating stage B
  when one shape dominates, is a different filter that this experiment says nothing about.
- **16 arms, 8 draws each is enough to reject a large effect, not a small one.** A true
  +0.005 from either filter would be invisible at this resolution.
