# The coverage test on dev samples, and whether it predicts the synthetic half-gain size

**Answer, in three lines.** On real, exactly-labelled samples the concepts order as the
generated-set study said: a split's own kind is least replaceable under *instruction*
(median gain kept without it, G_dev = 0.58), then *harmful* (0.71), then *high-stakes*
(0.88), and cross-kind transfer runs the same way (0.63 / 0.74 / 0.87). So the
non-substitution is a property of the concepts, not only of the generator's rendering of
the kinds. But no dev-sample statistic predicts a split's synthetic half-gain size better
than knowing its concept — the null the brief expected, now with exact labels.

*(1,200 fits: 248 for Part A over three concepts, 952 for Part C under instruction. Brief:
`docs/dev_coverage_task.md`. Code: `scripts/dc_dev.py`. Every number below is final.)*

## Method, in one paragraph

Each concept's dev set (`dev_samples/`) is cut by split — the kind is exact, it is the
split a sample comes from — into three training arms per split `s`: **own** (every dev
sample of `s`), **others** (every dev sample of the concept's other splits: set-minus-kind
with exact labels) and **all** (the whole dev set, one per concept). Each arm is fit once
at its full class-balanced size `n_arm = min(2·min(n_pos, n_neg), 590)`, eight draws,
through `scripts/subsample_curve_concept.py` unedited, `--no-base`, accumulation
`ceil(n/16)` at batch 16 (one optimiser step per epoch, the `direction_count` regime, rows
tagged `none+ga<K>bs16`). Since the dev samples are the *training* data here, every fit
early-stops on the concept's DeepSeek-V4-Pro detailed generated set
(`data/<concept>_deepseekv4pro_evaldesc[shape]_600.jsonl`), the same file for all three
arms. `own` and `all` fits are scored on every eval split of the concept (so one `own` fit
is one row of a transfer matrix); `others` fits on the target split only
(`dc_run_curve.py --eval-split`). With `A_x(s)` the eight-draw mean eval AUROC on `s`:
`G_dev = (a_others − 0.5)/(a_all − 0.5)` is the gain from chance kept without the split's
own kind, and `t_other_dev` is the mean of the other splits' `own` probes on `s` (the
off-diagonal column mean of the transfer matrix). No curve is fit in Part A.

## Part A — per concept (medians over splits)

| concept | splits | a_own | a_others | a_all | a_all − a_others | **G_dev** | **t_other_dev** |
|---|---|---|---|---|---|---|---|
| instruction | 6 | 0.986 | 0.780 | 0.983 | 0.199 | **0.58** | **0.63** |
| harmful | 4 | 0.992 | 0.852 | 0.994 | 0.124 | **0.71** | **0.74** |
| high-stakes | 4 | 0.988 | 0.918 | 0.983 | 0.052 | **0.88** | **0.87** |
| *generated sets (`direction_count`)* | | | | | | *G 0.67 / 0.90 / 0.95* | *t_other 0.81 / 0.85 / 0.95* |

Real samples of a split's own kind take the probe to the ceiling everywhere (a_own
0.85–1.00; 0.98–0.99 medians), and so does the whole dev set (a_all). The concepts differ
only in what is left when the split's own samples are removed.

## Part A — per split

`± sd` is the across-draw sd of G_dev with the draws of the two arms paired by index
(they are unrelated fits, so this is a spread, not a paired quantity).

| concept | split | n_own | a_own | a_others | a_all | G_dev | t_other_dev |
|---|---|---|---|---|---|---|---|
| instruction | anthropic_harmless_refusal | 68 | 1.000 | 0.953 | 1.000 | 0.91 ± 0.06 | 0.557 |
| instruction | bbq_substitution | 68 | 0.901 | 0.954 | 0.985 | 0.94 ± 0.03 | 0.768 |
| instruction | hc_context_drift | 66 | 1.000 | 0.733 | 0.974 | 0.49 ± 0.09 | 0.682 |
| instruction | hc_contradiction | 68 | 0.987 | 0.748 | 0.968 | 0.53 ± 0.07 | 0.719 |
| instruction | mm_substitution | 68 | 0.984 | 0.812 | 0.990 | 0.64 ± 0.02 | 0.584 |
| instruction | oig_context_drift | 66 | 0.979 | 0.701 | 0.981 | 0.42 ± 0.03 | 0.537 |
| harmful | ai_dilemmas | 46 | 0.996 | 0.935 | 0.998 | 0.87 ± 0.06 | 0.753 |
| harmful | ant_hh | 44 | 0.848 | 0.648 | 0.833 | 0.44 ± 0.08 | 0.612 |
| harmful | balanced_refusal | 134 | 1.000 | 0.769 | 0.992 | 0.55 ± 0.04 | 0.735 |
| harmful | daily_dilemmas | 66 | 0.987 | 0.956 | 0.995 | 0.92 ± 0.01 | 0.771 |
| high-stakes | anthropic_hh_balanced | 590 | 0.987 | 0.907 | 0.985 | 0.84 ± 0.05 | 0.854 |
| high-stakes | mt_balanced | 278 | 0.992 | 0.930 | 0.981 | 0.89 ± 0.05 | 0.881 |
| high-stakes | mts_balanced | 274 | 0.988 | 0.985 | 0.994 | 0.98 ± 0.01 | 0.922 |
| high-stakes | toolace_balanced | 328 | 0.898 | 0.830 | 0.882 | 0.86 ± 0.06 | 0.756 |

`n_others` is 336–338 (instruction), 156–246 (harmful) and 590 (high-stakes, capped);
`n_all` is 404 / 290 / 590. Transfer matrices: `scripts/dc_dev_transfer_<concept>.csv`.

**The transfer matrices** (rows: training split, columns: eval split) say *which* kinds
cover which, and they are not symmetric:

- *instruction*: a probe trained on real refusal samples is near chance on every other
  split (0.39–0.59), yet refusal itself is learned from the other five (0.95). The reverse
  also fails: BBQ samples score refusal at **0.25**, a direction that is *reversed* there.
  The two context-drift splits and the contradiction split feed BBQ well (0.91–0.94), but
  oig_context_drift is fed by nothing (0.50–0.60 from every other split) and hc_context_drift
  only by oig (0.85; the rest 0.52–0.77).
- *harmful*: the two dilemma splits cover each other almost completely (ai_dil→daily
  0.98, daily→ai_dil 1.00), and AI dilemmas also covers balanced refusal (0.95) where the
  other two do not (0.59, 0.67). Ant-HH is covered by nothing (0.58–0.68 from any single
  other split).
- *high-stakes*: the three chatbot/medical splits cover each other (0.79–0.98); toolace
  is the least covered (0.72–0.83 from the others).

## Part A — the predictions, stated before looking (brief §A5), and whether each held

| prediction | result | held? |
|---|---|---|
| G_dev well below 1 under *instruction* and near 1 under the other two | instruction 0.58; high-stakes 0.88 (0.84–0.98); **harmful 0.71**, with Ant-HH 0.44 and balanced-refusal 0.55 | **partly** — instruction ✓, high-stakes ✓, harmful ✗ |
| Harmless-refusal collapses without refusal samples (a_others near 0.5) | a_others **0.953**, G_dev 0.91 | **✗** |
| t_other_dev ordered instruction < harmful < high-stakes | 0.63 < 0.74 < 0.87 (medians; means 0.64 / 0.72 / 0.85) | **✓** |
| Under high-stakes and harmful, a_others within a few hundredths of a_all | high-stakes gaps 0.08 / 0.05 / 0.01 / 0.05; harmful 0.06 / 0.19 / 0.22 / 0.04 | **✗** (only high-stakes MTS and harmful daily-dilemmas) |

**Reading.** The ordering the generated-set study found is reproduced on real data by
both statistics, so it is a property of the concepts and not of how one generator wrote
the kinds. What does *not* reproduce is the size of the contrast: on real data every
concept loses more without a split's own kind than its generated sets did (G 0.58 / 0.71
/ 0.88 against 0.67 / 0.90 / 0.95), and *harmful* moves most — it sits between the other
two rather than next to *high-stakes*. The generated *harmful* kinds covered one another
better than the real ones do.

**The harmless-refusal reversal.** On the generated sets, removing refusal samples cost
the refusal split nearly all of its gain (11% kept under DeepSeek, flat under GPT-OSS). On
real data it keeps 91% — and no single other split gets it there (best single split
0.88, oig_context_drift); the five together do. A plausible reading, not tested here, is
that the real negatives of the other kinds include enough declining or non-answering
assistants to carry the direction, and the generated ones did not. The *converse* is
where refusal is special on real data: refusal samples teach nothing that transfers
(0.39–0.59 on the other splits).

**Ant-HH** (the known exception): on real data its own kind is the better route, a_own
**0.848** against a_others **0.648** (G_dev 0.44), and it is also the one split no arm
takes near the ceiling (a_all 0.833). Under the generated sets its own kind was worse than
the rest of the set; on real data it is the other way round.

## Part B — does any of it predict the synthetic half-gain size?

Target: `log_m_detailed`, each split's median log10 half-gain size over its four
detailed-prompt curves (`knee_predictor.load_targets`). Statistics exactly as
`direction_count._link_to_paper_m` computes them, reusing `knee_predictor`'s functions.
Fourteen splits (MM-substitution now has a predictor value). **With MM in, the
concept-only baseline is 0.284** (it was 0.273 over the thirteen splits of
`dc_link_stats.csv`); every bar below is against 0.284, the baseline on the same points.

| predictor | n | ρ | p (perm.) | 95% CI | LOO RMSE | concept-only | pred + concept | ρ within instruction (p) | bar 1: \|ρ\|≥0.6, p<0.05 | bar 2: beats concept | bar 3: sign holds in instruction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **t_other_dev** | 14 | **−0.723** | **0.005** | [−0.83, −0.40] | 0.368 | 0.284 | 0.287 | −0.31 (0.58) | **✓** | ✗ | ✓ |
| G_dev | 14 | −0.415 | 0.14 | [−0.65, −0.15] | 0.445 | 0.284 | 0.257 | **−0.77** (0.10) | ✗ | ✗ | ✓ |
| a_others | 14 | −0.433 | 0.12 | [−0.65, −0.16] | 0.471 | 0.284 | 0.247 | **−0.77** (0.10) | ✗ | ✗ | ✓ |
| a_all − a_others | 14 | +0.385 | 0.18 | [+0.16, +0.62] | 0.438 | 0.284 | 0.281 | **+0.77** (0.10) | ✗ | ✗ | ✓ |
| a_own | 14 | −0.187 | 0.52 | [−0.42, +0.13] | 0.516 | 0.284 | 0.323 | +0.09 (0.92) | ✗ | ✗ | ✗ |
| R_dev (Part C) | 14 | +0.665 | 0.014 | [+0.29, +0.78] | 0.428 | 0.284 | 0.296 | +0.09 (0.92) | **✓** | ✗ | ✓ |

R_dev's row is over all fourteen splits, from the Part C runs under harmful and
high-stakes (see "Part C under harmful and high-stakes"). Over the six instruction splits
alone, as first run, it was ρ = +0.086 (p = 0.92), LOO 0.470 against the six-split grand
mean 0.354.

**Verdict: no predictor clears all three bars.** `t_other_dev` clears the correlation bar
(ρ = −0.72, p = 0.005, the same strength as the generated-set `t_other`'s −0.79) and its
sign holds within *instruction*, but its leave-one-split-out error (0.37) is worse than
concept identity's (0.28): it is the concept mean in disguise, and adding it to concept
identity changes nothing (0.287). The within-*instruction* part of the target is carried
by a different statistic: **G_dev and a_others rank the six instruction splits at
ρ = −0.77** (p = 0.10 at six points), where the generated-set R and t_other had 0.0; with
concept identity they lower the LOO error from 0.284 to 0.257 / 0.247 — the only
predictors here that add more than a few thousandths to the concept. That is a lead, not a result: six
points, p = 0.10, and not one of the brief's three bars. As expected up front, the
between-concept ordering reproduces and nothing beats concept identity.

## Part C — curves for a dev-sample R, instruction only

Run because Part B's within-instruction |ρ| for G_dev is 0.77 ≥ 0.5 (the brief's trigger).
Same arm files, same regime and validation set; ladders `own` 2 4 6 10 20 30 50, `others`
and `all` 2 4 6 10 20 30 50 80 110 170 300; eight draws; `own`/`others` scored on the
target split, `all` unrestricted. 952 fits. Every curve is fit with the reference
log-logistic (`fit_curves_ref.fit_curve`, as `direction_count --stage analyse` does:
`flat` = in-range gain < 0.02, `censored` = m at or below the smallest size), R's interval
is `direction_count._paired_ratio_ci` (400 paired replicates), and `G_dev_fit` is the
*fitted* gain kept without the kind, (U − L)_others / (U − L)_all — the generated-set
study's G.

| split | m_own [95%] | m_all [95%] | m_others | **R_dev = m_all/m_own** [95%] | G_dev_fit [95%] | G_dev (Part A) |
|---|---|---|---|---|---|---|
| anthropic_harmless_refusal | 4.9 [3.0, 8.9] | 31 [13, 43] | 37 | **6.3** [2.3, 13.3] | 0.85 [0.53, 1.25] | 0.91 |
| bbq_substitution | 22 [11, 34] | 37 [22, 51] | 31 | **1.7** [0.9, 3.2] | 1.08 [0.71, 1.36] | 0.94 |
| hc_context_drift | 19 [16, 22] | 84 [71, 99] | 55 | **4.5** [3.5, 5.8] | 0.45 [0.40, 0.61] | 0.49 |
| hc_contradiction | 15 [11, 19] | 99 [77, 108] | 66 | **6.8** [4.5, 8.7] | 0.46 [0.39, 0.58] | 0.53 |
| mm_substitution | 13 [8, 20] | 66 [56, 77] | 117 | **4.9** [3.5, 8.0] | 0.55 [0.44, 0.67] | 0.64 |
| oig_context_drift | 34 [29, 40] | 99 [84, 108] | 117 | **3.0** [2.5, 3.5] | 0.46 [0.36, 1.11] | 0.42 |

(Intervals from `scripts/dc_dev_partc_ratios.csv` / `dc_dev_partc_fits.csv`.) No arm is
flat, none is censored, all six rows are usable.

**Median R_dev = 4.7** (range 1.7–6.8), against the generated sets' 5.30 under
*instruction*: on real data too, a sample of a split's own kind is worth about five mixed
ones, and every row but BBQ has its whole interval above 1. **Median G_dev_fit = 0.51**,
against 0.67 on the generated sets, and the fitted and from-chance versions of G agree
split by split (Spearman 0.83 over the six; oig_context_drift moves from last to third). And R_dev does
**not** rank the synthetic target within *instruction* (ρ = +0.09) — the same 0.0 the
generated-set R had. The within-concept signal Part B found is in *how much* gain the
other kinds leave behind, not in how fast the split's own kind is learned.

## Part C under harmful and high-stakes

Run for `docs/dev_coverage_partc_task.md`, to put a real-data number on the paper's
cross-concept coverage share. Same arm files, regime (`--no-base --grad-accum ceil(n/16)
--batch-size 16`, eight draws), validation sets and restriction rule as the instruction
run; every arm takes the full ladder 2 4 6 10 20 30 50 80 110 170 350 590 cut at its
class-balanced size (`dc_dev.py --stage partc-fit --concept <c>`). *Harmful*: 77 cells,
616 fits; *high-stakes*: 102 cells, 816 fits; no failures. The instruction rows are the
ones above, untouched (its legacy ladders are kept, so `--concept instructions` still
reproduces them byte for byte). Same fitter, flat rule and censoring rule; nothing was
changed to move a floor-pinned m.

| split | m_own [95%] | m_all [95%] | m_others | **R_dev = m_all/m_own** [95%] | G_dev_fit [95%] | G_dev (Part A) |
|---|---|---|---|---|---|---|
| *harmful* ai_dilemmas | 7.5 [5.4, 10.5] | 20 [17, 26] | 17 | **2.7** [1.8, 4.1] | 0.88 [0.78, 0.99] | 0.87 |
| *harmful* ant_hh | 9.6 [6.3, 11.4] | 66 [16, 108] | 71 | **6.8** [1.5, 12.2] | 0.22 [0.12, 1.30] | 0.44 |
| *harmful* balanced_refusal | 11.4 [3.0, 20.4] | 19 [8, 24] | 8.2 | **1.6** [0.6, 6.8] | 0.64 [0.34, 1.08] | 0.55 |
| *harmful* daily_dilemmas | 8.2 [6.4, 10.5] | 19 [15, 22] | 26 | **2.3** [1.6, 3.2] | 1.12 [0.85, 1.20] | 0.92 |
| *high-stakes* anthropic_hh | **3.0** [3.0, 22] | 4.6 [3.0, 10.5] | 77 | **1.5** [0.2, 3.2] | 0.89 [0.82, 1.68] | 0.84 |
| *high-stakes* mt | 13.5 [3.0, 20.4] | 19 [15, 24] | 13 | **1.4** [0.8, 5.8] | 1.01 [0.65, 1.30] | 0.89 |
| *high-stakes* mts | 9.6 [3.0, 20.4] | 12 [7, 24] | 17 | **1.3** [0.5, 4.9] | 0.76 [0.61, 1.23] | 0.98 |
| *high-stakes* toolace | 20.4 [7.5, 60.4] | 66 [34, 100] | 47 | **3.2** [1.0, 9.5] | 1.13 [0.53, 1.62] | 0.86 |

No arm is flat; all eight rows are usable. **One `m_own` sits at the grid floor of 3**
(*high-stakes* anthropic_hh) and **four
`m_own` intervals reach it** (one *harmful*, three *high-stakes*). The `censored` flag
fires nowhere, and cannot: it is "m at or below the smallest measured size", the
smallest size is 2 and the grid starts at 3, so the floor count is the column that
catches a pinned m.

### Per concept (medians over usable splits; 95% bootstrap over splits, 4,000 resamples)

| concept | usable | m_own (kind-only) | m_all (mixed) | R_dev | G_dev_fit | m_own at floor / interval at floor | generated sets: kind-only / mixed / R |
|---|---|---|---|---|---|---|---|
| *instruction* | 6/6 | 16.7 [9.2, 27.9] | 75 [34, 99] | 4.68 [2.30, 6.53] | 0.51 [0.45, 0.96] | 0 / 1 | 11 / 117 / 5.30 |
| *harmful* | 4/4 | 8.9 [7.5, 11.4] | 19.6 [18.8, 65.5] | 2.51 [1.65, 6.80] | 0.76 [0.22, 1.12] | 0 / 1 | 7.2 / 18 / 2.12 |
| *high-stakes* | 4/4 | 11.5 [3.0, 20.4] | 15.6 [4.6, 65.5] | 1.46 [1.28, 3.21] | 0.95 [0.76, 1.13] | 1 / 3 | 6.9 / 7.9 / 1.18 |

The generated-set ordering of R reproduces on real data, at similar sizes: *instruction*
4.7 vs 5.3, *harmful* 2.5 vs 2.1, *high-stakes* 1.5 vs 1.2. The mixed sizes of *harmful*
match (19.6 vs 18); *instruction*'s real m_all is lower (75 vs 117) and *high-stakes*'s
higher (15.6 vs 7.9), so the real mixed gap is narrower. Real `m_own` runs above the
generated kind-only m in all three concepts (16.7 / 8.9 / 11.5 against 11 / 7.2 / 6.9).

### The real-data coverage share (`scripts/dc_dev_partc_share.csv`)

| comparison | gap_all (mixed) | gap_own (kind-only) | **share = 1 − gap_own/gap_all** [95%] | generated sets |
|---|---|---|---|---|
| *instruction* vs mean(*harmful*, *high-stakes*) | 0.632 | 0.217 | **0.66** [−0.48, 1.39] | 0.79 (gaps 0.99 / 0.21) |
| *instruction* vs *harmful* | 0.582 | 0.273 | 0.53 [−1.98, 2.51] | 0.76 |
| *instruction* vs *high-stakes* | 0.682 | 0.160 | 0.76 [0.01, 2.20] | 0.81 |

**The point estimate is 0.66, against 0.79 on generated sets**, and the kind-only gap is
the generated one almost exactly (0.217 vs 0.21); the difference sits in the mixed gap
(0.63 vs 0.99). The interval is uninformative, though: with four to six splits per
concept a resample that shrinks gap_all towards 0 sends the ratio anywhere, so the data
are consistent with the paper's four fifths and with much less. Only *instruction* vs
*high-stakes* excludes zero, and only just.

**The floor does not move the point estimate.** The brief's bound applies only if most
`m_own` values are floor-pinned; here it is one of eight. The pinned value is the
smallest of *high-stakes*' four `m_own` (3, 9.6, 13.5, 20.4), and a median of four is the
mean of the middle two whatever the smallest one is, so a true m_own below 3 at
anthropic_hh leaves the *high-stakes* median (11.5), gap_own and the share unchanged.
Treating it as exactly 3 (`share_floor3`) gives the same 0.66 [−0.48, 1.39]. Strictly,
bootstrap resamples that draw anthropic_hh two or more times would have a lower
*high-stakes* median, so only the interval (not the point estimate) is an upper bound in
the brief's sense.

**Per kind, *instruction* is still the slowest, but only modestly.** Its median `m_own`
(16.7) exceeds *harmful*'s (8.9) and *high-stakes*' (11.5), a ratio of 1.9 and 1.4 (1.5
on generated sets, 11 vs 7). The intervals overlap (*instruction* [9.2, 27.9] against
[7.5, 11.4] and [3.0, 20.4]), so "per kind the concepts are alike, and *instruction*'s
large mixed m is mostly coverage" holds in direction on real data but not with a margin
four splits per concept can confirm.

### Part B with R_dev over all fourteen splits

R_dev now exists for every split, so it is scored against the same 0.284 concept-only
baseline as the other predictors: **ρ = +0.665, p = 0.014, 95% CI [+0.29, +0.78]; LOO
0.428 against concept 0.284 (pred + concept 0.296); within instruction +0.09 (p = 0.92).**
It clears the correlation bar and the sign bar, and fails the one that matters: like
`t_other_dev`, it ranks the concepts, and concept identity alone does that better.
The verdict of Part B stands: nothing beats concept identity.

## The difference-of-means statistic on dev samples

*(Brief: `docs/dev_coverage_dom_task.md`. Code: `scripts/dc_dev_geometry.py`. No probe is
trained and no curve is fit; the whole stage runs in seconds on pooled features.)*

**Why.** Table `dc_main` put two transfer AUROCs side by side that are not the same
statistic: the generated row (`t_other`) is a class-mean direction per LLM-tagged kind
scored on the other kinds' generated samples; the real row (`t_other_dev`) is a trained,
early-stopped probe scored on the other eval splits. This section computes the
generated-row statistic, unchanged, on the dev samples.

**Method.** Features are the token-mean of Gemma-3-27B-IT layer 32 over the attention
mask, float32 (`direction_count.pool_set`, over the `own` arm files so the high-stakes MTS
assistant-first fix is in). Per concept the dev splits are stacked (404 / 290 / 1,908
rows), standardised on their own mean and sd, and each split gets one unit
difference-of-means direction `d_s` (every split clears `MIN_KIND_PER_CLASS = 20`; the
smallest is Ant-HH, 22 / 22). `T[s, s']` scores `d_s` on split `s'`'s dev samples, `E[s, e]`
on eval split `e` pushed through the *dev* standardiser; `d_all` is the whole dev set's
direction and `d_-s` the set-minus-split one. `G_dom = (AUROC(d_-s) − 0.5) / (AUROC(d_all) −
0.5)`. Everything else — `_auroc`, `_standardiser`, `_direction`, `SEED`, the five-fold
held-out diagonal and its RNG stream — is imported from or mirrors
`direction_count.geometry_for_set`, which the code reproduces exactly (all 272 `T_ij` and
`cos_ij` cells of `dc_geometry.csv`) when fed the LLM tags. The eval features are the knee
study's pool, rebuilt from the Kaggle blobs on this machine; the rebuilt
`knee_pooled_manifest.json` is byte-identical to the committed one.

**One departure: the diagonal on paired splits.** Every *instruction* dev split and three
of the four *harmful* ones open with each user message exactly twice, once per class
(every prompt of those ten splits; Ant-HH and all of high-stakes are unpaired). The brief's row-level five folds
then put each held-out row's opposite-class twin in the training fold, and since the class
means are dominated by prompt content the direction points the held-out row at the wrong
class: **0.07–0.11** on hc_context_drift, hc_contradiction, oig_context_drift,
ai_dilemmas and daily_dilemmas, whose directions score 0.84–0.98 on their own *eval*
split. The generated sets are unpaired, so there the row folds were fine (0.96–1.00). The
row-fold value is kept as specified (`t_own`); `t_own_grp` holds a prompt's pair out
together (the key is the first non-empty user message: hc_context_drift's two rows also
differ in a later user turn, and the assistant-first fix gives 263 MTS rows the same blank
opener), and it is the one quoted below. It changes nothing but the diagonal (and
`t_gap`); every off-diagonal and eval number is fit-on-one-split, score-on-another.

### Four transfer numbers per concept

Means over splits, as `dc_neff.csv` defines them (probe: mean, median in brackets).
Generated rows are the mean over the four generators' detailed sets.

| concept | gen. direction on gen. samples (`t_other`) | gen. direction on eval (`e_other`) | **dev direction on dev (`t_other`)** | **dev direction on eval (`e_other`)** | probe on eval (`t_other_dev`) |
|---|---|---|---|---|---|
| instruction | 0.82 | 0.60 | **0.61** | **0.63** | 0.64 (0.63) |
| harmful | 0.85 | 0.63 | **0.63** | **0.63** | 0.72 (0.74) |
| high-stakes | 0.95 | 0.75 | **0.62** | **0.62** | 0.85 (0.87) |

Own-split and whole-set numbers, same layout:

| concept | gen. `t_own` | dev `t_own_grp` (row folds) | gen. `e_own` | dev `e_own` | gen. `e_all` | dev `e_all` | dev n_eff (gen.) | dev mean / max off-diag cos |
|---|---|---|---|---|---|---|---|---|
| instruction | 0.98 | 0.92 (0.40) | 0.71 | 0.89 | 0.72 | 0.81 | 4.89 (3.75) | 0.11 / 0.71 |
| harmful | 0.98 | 0.96 (0.49) | 0.75 | 0.92 | 0.77 | 0.77 | 3.51 (3.78) | 0.10 / 0.41 |
| high-stakes | 1.00 | 0.90 (0.89) | 0.91 | 0.90 | 0.91 | 0.93 | 3.64 (3.56) | 0.09 / 0.43 |

**Reading.** With the classifier held fixed, the real splits transfer **alike across the
three concepts**: a split's class-mean direction scores another split at 0.61 / 0.63 /
0.62 on dev samples and 0.63 / 0.63 / 0.62 on eval, flat to within 0.03, where both
existing rows order the concepts (0.82 / 0.85 / 0.95 generated, 0.64 / 0.72 / 0.85 probe).
The dev directions are also near-orthogonal everywhere (mean off-diagonal cos 0.09–0.11;
the only large pair is hc_context_drift–hc_contradiction, 0.71, and mt–mts, 0.43). So the
concept ordering of the generated `t_other` is a property of the tagged generated kinds,
and the ordering of the probe's `t_other_dev` is something the optimiser adds on real
data — neither is in the class means of the real splits. The paper's "transfer says the
same without curves" sentence needs that qualification.

High-stakes' flat number has one clear source: **Anthropic-HH** is orthogonal to the other
three splits (cos 0.005 / −0.019 / −0.015; its direction scores them 0.41–0.55 and theirs
score it 0.45–0.51, dev and eval alike), and it is 54% of the dev set. Among MT, MTS and ToolACE alone the
off-diagonal mean is 0.77 on dev and 0.75 on eval — high-stakes' generated `e_other`
exactly, and still above either other concept. That is a statement about one split, not a
rescue of the ordering: *instruction* and *harmful* each have such pairs too
(hc_context_drift→refusal 0.92, ai_dilemmas→balanced_refusal 0.98).

### Per split

`t_other` / `e_other`: the off-diagonal *column* mean (the other splits' directions scored
on s), the analogue of `t_other_dev`. `G_dom` on eval (dev in brackets; the dev
denominator is in-sample, see below). Probe columns copied from `dc_dev_cells.csv`;
generated columns are the per-split medians over generators behind `dc_link_stats.csv`
(mm_substitution has none).

| concept | split | n | `t_own_grp` | `t_other` | `e_own` | `e_other` | `G_dom` eval (dev) | probe `t_other_dev` | probe `G_dev` | gen. `t_other` / `e_gap` |
|---|---|---|---|---|---|---|---|---|---|---|
| instruction | anthropic_harmless_refusal | 68 | 0.98 | 0.575 | 1.00 | 0.623 | **−0.42** (−0.48) | 0.557 | 0.91 | 0.82 / 0.49 |
| instruction | bbq_substitution | 68 | 0.95 | 0.610 | 0.79 | 0.751 | 0.50 (−0.06) | 0.768 | 0.94 | 0.82 / −0.07 |
| instruction | hc_context_drift | 66 | 0.88 | 0.623 | 0.86 | 0.626 | 0.88 (0.84) | 0.682 | 0.49 | 0.75 / −0.01 |
| instruction | hc_contradiction | 68 | 0.85 | 0.668 | 0.84 | 0.637 | 0.76 (0.83) | 0.719 | 0.53 | 0.82 / 0.01 |
| instruction | mm_substitution | 68 | 0.99 | 0.568 | 0.98 | 0.568 | **−0.45** (−0.36) | 0.584 | 0.64 | — |
| instruction | oig_context_drift | 66 | 0.90 | 0.588 | 0.86 | 0.562 | 0.41 (0.43) | 0.537 | 0.42 | 0.82 / 0.15 |
| harmful | ai_dilemmas | 46 | 0.95 | 0.660 | 0.91 | 0.652 | 0.91 (0.92) | 0.753 | 0.87 | 0.85 / −0.03 |
| harmful | ant_hh | 44 | 0.95 | 0.590 | 0.83 | 0.602 | 0.24 (−1.44) | 0.612 | 0.44 | 0.85 / 0.19 |
| harmful | balanced_refusal | 134 | 0.94 | 0.577 | 0.96 | 0.577 | **−0.39** (−0.34) | 0.735 | 0.55 | 0.85 / 0.34 |
| harmful | daily_dilemmas | 66 | 1.00 | 0.704 | 0.98 | 0.699 | 0.41 (0.31) | 0.771 | 0.92 | 0.85 / −0.03 |
| high-stakes | anthropic_hh_balanced | 1028 | 0.95 | 0.480 | 0.96 | 0.468 | −0.08 (−0.04) | 0.854 | 0.84 | 0.97 / 0.26 |
| high-stakes | mt_balanced | 278 | 0.95 | 0.735 | 0.93 | 0.732 | 0.88 (0.92) | 0.881 | 0.89 | 0.97 / 0.11 |
| high-stakes | mts_balanced | 274 | 0.90 | 0.704 | 0.91 | 0.714 | 0.77 (0.70) | 0.922 | 0.98 | 0.97 / 0.11 |
| high-stakes | toolace_balanced | 328 | 0.79 | 0.565 | 0.80 | 0.550 | 0.25 (0.25) | 0.756 | 0.86 | 0.97 / 0.13 |

Transfer matrices: `scripts/dc_dev_dom_transfer_<concept>_{dev,eval}.csv`.

`G_dom` is a ratio of two gains over chance and is ill-conditioned where `d_all` itself is
weak on the split (Ant-HH: `d_all` 0.61 on dev in-sample, 0.47 held out — whence −1.44
and, in the CSV, 5.50 under the held-out denominator `G_dom_dev_ho`). The eval form, whose
denominators are 0.66–1.00, is the one to read; medians over splits **0.46 / 0.33 / 0.51**
(instruction / harmful / high-stakes) against the probe's G_dev 0.58 / 0.71 / 0.88.

### The predictions, stated before looking, and whether each held

| prediction | result | held? |
|---|---|---|
| `t_other_dev_dom` ordered instruction < harmful < high-stakes | 0.605 / 0.633 / 0.621 (medians 0.60 / 0.63 / 0.63); size-matched 0.605 / 0.633 ± 0.007 / 0.616 ± 0.036 | **✗** — flat; high-stakes is not the most transferable |
| `e_other` of real directions above the generated directions' 0.60 / 0.63 / 0.75 | 0.628 / 0.632 / 0.616 | **✗** — level with generated for instruction (+0.03) and harmful (+0.01), **below** it for high-stakes (−0.13) |
| `e_own` well above 0.71 / 0.75 / 0.91 | 0.886 / 0.919 / 0.900 | **partly** — instruction ✓ (+0.18), harmful ✓ (+0.17), high-stakes ✗ (equal) |
| Harmless-refusal's direction near chance or reversed on the other instruction splits; BBQ's reversed on refusal | refusal → others 0.43 / 0.57 / 0.54 / **0.24** / 0.57 on dev, 0.59 / 0.55 / 0.54 / **0.26** / 0.52 on eval (probe 0.39–0.59); BBQ → refusal **0.23** dev / **0.33** eval (probe 0.25); mm → refusal also reversed (0.21 / 0.18) | **✓** — a property of the samples, not of the optimiser |
| `G_dom` well below 1 under instruction, near 1 under the other two | eval medians 0.46 / 0.33 / 0.51; no concept near 1 | **partly** — instruction ✓, harmful ✗, high-stakes ✗ |

**The refusal asymmetry is sharper under the class mean than under the probe.** The
set-minus-refusal direction is *reversed* on refusal (0.26 dev, 0.29 eval), where the
probe trained on the same five splits reached 0.953 (G_dev 0.91). Two of the five other
splits point the opposite way on refusal (BBQ, MM: 0.18–0.33) and three the right way
(hc_context_drift 0.92–0.96, hc_contradiction and oig 0.75–0.86); their class means
cancel and then some, and the probe finds the three. So the Part A finding that refusal
is carried by the other splits together (G_dev 0.91) is a property of training — the
samples' class means alone do not carry it. The converse half of the asymmetry, that
refusal samples teach nothing transferable, is the samples' own.

### Size check

The dev splits hold 44–1,028 samples against the generated kinds' 43–211, and a class
mean over more samples is less noisy. Steps 2–4 were re-run with every split
class-balanced to min(its balanced size, 100), 20 draws seeded off `SEED` (the full-dev
standardiser kept). Every *instruction* split is already at or below 100 (66–68), so its
size-matched run is the full-size one exactly. The rest:

| concept | `t_other` full | size-matched (mean ± sd) | `e_other` full | size-matched |
|---|---|---|---|---|
| harmful | 0.633 | 0.633 ± 0.007 | 0.632 | 0.633 ± 0.002 |
| high-stakes | 0.621 | 0.616 ± 0.036 | 0.616 | 0.614 ± 0.024 |

Full and size-matched differ by less than one across-draw sd everywhere, so size is not
what flattens the concepts; **the tables above use the full-size values**. The
size-matched row of every statistic is `gen = dev_n100` in `scripts/dc_dev_neff.csv` and
the `*_n100` / `sd_*_n100` columns of `scripts/dc_dev_dom_cells.csv`.

## Caveats

- **The validation set is off-distribution by design.** Every fit early-stops on the
  concept's DeepSeek-V4-Pro detailed generated set, never on a dev or eval split. It is
  one fixed file per concept across all arms, so the early-stopping signal is one thing
  throughout, but it is not the eval distribution: a probe that would score higher on
  the eval splits at a later epoch can be stopped before it. Base-probe AUROC on the three
  validation sets: 0.881 / 0.950 / 0.987. These CSVs are not comparable with any run that
  validated on `dev_samples/highstakes_500` and are not mixed with one.
- **The 590 cap** binds only under *high-stakes*: `own(anthropic_hh)` (1,028 available),
  all four `others` arms (880–1,634) and `all` (1,908) are drawn at 590. The *high-stakes*
  `others` and `all` arms therefore see 590 of a larger pool, and differ in *which* 590 by
  draw; the other two concepts' arms use every sample they have.
- **Class balance of the smallest dev sets.** *Harmful* Ant-HH (22/22) and AI-dilemmas
  (23/23) are drawn whole; eight draws there differ only in order and initialisation, so
  their across-draw sd understates the sampling uncertainty a larger dev set would have.
  Every *instruction* split is 33/33 or 34/34.
- **G_dev is a gain from chance, not a fitted gain.** Part A fits no curve, so its G is
  (a_others − 0.5)/(a_all − 0.5) at one size, where the generated-set G is a ratio of
  fitted (U − L). The two are close where Part C measures both (instruction), but the
  from-chance version is inflated when a_all itself is far from the ceiling (Ant-HH,
  toolace) and it can exceed 1 (it does not here).
- **a_own and a_all sit near the AUROC ceiling**, so G_dev's denominator is nearly the
  same for every split and the statistic is carried by a_others — which is why G_dev and
  a_others have identical ranks within *instruction*. The instruction refusal split's
  `own` and `all` arms have sd = 0: AUROC 1.0 on every draw (the same fits vary on the
  other splits), not a probe that took no step. The high-stakes `all` arm at accumulation
  37 has non-zero sd at every split.
- **Assistant-first MTS rows.** 263 of the 274 *high-stakes* `mts_balanced` dev rows open
  (system, assistant, user, …). tuberlens' `LabelledDataset.load_from` — the loader behind
  every eval and dev split — always inserts a blank user turn before such an opening; the
  harness's in-memory training path does not, and gemma's chat template refuses the
  conversation. The arm files therefore carry the `load_from` representation of those rows
  (`dc_dev.fix_assistant_first`), verified identical to `load_from`'s output for all 2,602
  dev rows, so a dev sample is the same token sequence as training data as it is as
  validation data. Every other row is copied verbatim.
- **Half-gain sizes are on the reference fitter's grid** (90 log-spaced m in [3, 5000],
  ~9% apart), which is why several Part C m's recur exactly (31.0, 36.6, 65.6, 99.4,
  117.5). The bootstrap intervals include that resolution.
- **`oig_omission` is excluded everywhere**: from the arms and from the eval (the
  unrestricted instruction fits score a directory of symlinks to the six other eval
  splits). `eval_sets/` was not touched.
- **Fourteen splits in three concepts** cannot separate a per-split predictor from a
  concept-level constant; the within-instruction tests have six points and p ≥ 0.10 at
  best.

## Outputs

`scripts/dc_dev.py` (stages `arms | prefetch | warm | fit | cells | link | partc-fit |
partc-analyse | partc-share`, the Part C stages taking `--concept`), `scripts/run_dc_dev.sh`,
`scripts/run_dc_dev_partc.sh` and `scripts/run_dc_dev_partc2.sh`;
`scripts/dc_dev_arms.csv`; the Part A fit CSVs `scripts/dc_dev_<concept>_{own,all}.csv`
and `scripts/dc_dev_<concept>_others_<split>.csv`; `scripts/dc_dev_cells.csv` and
`scripts/dc_dev_transfer_<concept>.csv`; `scripts/dc_dev_link_stats.csv` and
`scripts/dc_dev_scatter.csv` (the columns of `dc_scatter.csv` with the dev predictors in
place of the geometry ones); Part C's `scripts/dc_dev_partc_{instructions,hu_harm,highstakes}_*.csv`,
`scripts/dc_dev_partc_fits.csv`, `scripts/dc_dev_partc_ratios.csv`,
`scripts/dc_dev_partc_share.csv`.
The difference-of-means section: `scripts/dc_dev_geometry.py` (stages `pool | geometry`),
the pooled dev features `scripts/dc_pooled/dcdev_<concept>_<split>_own_{mean,labels,ntokens}.npy`,
`scripts/dc_dev_geometry.csv` (the layout of `dc_geometry.csv`, plus `minus_<split>` rows),
`scripts/dc_dev_neff.csv` (the layout of `dc_neff.csv`, `gen = dev` and `dev_n100`, plus
`t_own_grp` and the size-matched `sd_*`), `scripts/dc_dev_dom_transfer_<concept>_{dev,eval}.csv`
and `scripts/dc_dev_dom_cells.csv`.

Wall-clock per fit (one RTX 3090, gemma-3-27b layer 32, activations cached): instruction
own 13 s / others 17–18 s / all 21 s; harmful 14 s / 12–17 s / 20 s; high-stakes 99 s /
29–71 s / 115 s (unrestricted high-stakes fits read all 47 GB of eval blobs); Part C
instruction 11 s, harmful 9 / 10 / 13 s (own / others / all; 1.9 h), high-stakes
18 / 19 / 85 s (8.5 h).
Extraction of the 2,602 dev samples and the three 600-row validation blobs took ~2 h.
