# Predicting the learning-curve knee from the evaluation split alone

**Verdict: no predictor works.** Of 118 cheap properties computed from the 14 evaluation
splits' cached Gemma-3-27B-IT layer-32 activations, none meets the three conditions set
for this study — ρ ≥ 0.6 at p < 0.05 over the 14 splits, beating the concept-only
leave-one-split-out baseline, and keeping its sign inside the instructions concept. The
best of them fails the second condition badly, and a family-wise permutation test says
its correlation is what the best of 118 noise predictors would give anyway
(family-wise p = 0.44). Knowing which of the three concepts a split belongs to predicts
its knee roughly twice as well as any activation statistic measured here, and nothing
measured here adds to it.

## Method

For each of the 14 splits (`oig_omission` excluded throughout) the published per-split
activation blob — validated against model name, layer and row count before use — was
reduced to two matrices, the token-mean and the last real token of every row, with
padding read off the attention mask rather than assumed; no model was ever loaded and
no synthetic data was generated. From those matrices, under 5-fold stratified CV with
seed 20260917 and averaged over folds, came: (**A**) an in-distribution few-shot curve
— logistic regression on k ∈ {2…256} class-balanced samples drawn from the training
folds, 32 draws each, held-out AUROC, with the paper's own log-logistic fitter
(`scripts/fit_curves_ref.py`, imported, not re-implemented) applied to that curve to
give `m_ID`; (**B**) the share of full logistic regression's AUROC gain that a single
difference-of-means direction recovers, `r1`, plus the same direction estimated from
two samples per class over 200 draws; (**C**) held-out AUROC of logistic regression on
the top-d principal components and the `d95` / `d90` at which it saturates; (**D**)
label-aware geometry — Fisher ratio along the difference-of-means direction, centroid
cosine, within-class participation ratio, within- versus between-class cosine, and for
the nine class-paired splits the within-pair distance relative to the between-pair
spread; and (**E**) controls needing no activations at all. The target is the median
over the four `detailed`-prompt Gemma curves of log₁₀ m, flat curves (fitted in-range
gain < 0.02, the reference fitter's own rule) dropped first and knees at or below the
smallest measured size clamped to 10, which is why every headline number is a rank
statistic. All 14 splits have four non-flat detailed curves. Confidence intervals
resample the curves behind each split's median and jitter each curve's log₁₀ m by the
bootstrap sd the fitter already stored, so censoring and draw noise both propagate.

Reproduce with `scripts/knee_predictor.py --stage pool|features|analyse`; outputs are
`scripts/knee_predictors.csv`, `knee_predictor_curves.csv`, `knee_predictor_stats.csv`
and `knee_predictor_scatter.csv`.

## Table 1 — every predictor against the primary target (median log₁₀ m, detailed, 14 splits)

`LOO` is leave-one-split-out RMSE in log₁₀ units from the predictor alone, `+conc` from
predictor plus concept, `ΔR²` the curve-level gain over generator + prompt (§ Curve
level), `ρ_ins` the rank correlation inside the six instructions splits. Mean-pooled
features unless the name ends `_last`.

| | predictor | ρ | p | 95% CI | LOO | +conc | ΔR² | ρ_ins |
|---|---|---|---|---|---|---|---|---|
| A | `m_ID` (C=0.1) | **+0.604** | **0.027** | [+0.36, +0.74] | 0.380 | 0.274 | 0.264 | +0.638 |
| A | `m_ID` (C=1) | +0.564 | 0.039 | [+0.32, +0.73] | 0.484 | 0.295 | 0.139 | +0.657 |
| A | `log n90_ID` | +0.257 | 0.379 | [−0.04, +0.48] | 0.526 | 0.293 | 0.065 | +0.257 |
| A | `U_ID` (ID ceiling) | −0.046 | 0.879 | [−0.27, +0.22] | 0.489 | 0.293 | 0.039 | −0.087 |
| A | `k_ID` (ID steepness) | +0.484 | 0.076 | [+0.25, +0.72] | 0.477 | 0.287 | 0.112 | +0.754 |
| A | `AUROC_ID(k=2)` | −0.547 | 0.048 | [−0.71, −0.29] | 0.563 | 0.278 | 0.085 | −0.600 |
| A | `AUROC_ID(k=8)` | −0.552 | 0.045 | [−0.71, −0.27] | 0.490 | 0.276 | 0.129 | −0.657 |
| A | `AUROC_ID(k=32)` | −0.459 | 0.104 | [−0.65, −0.19] | 0.470 | 0.275 | 0.135 | −0.600 |
| B | `AUROC_full` | −0.393 | 0.163 | [−0.60, −0.11] | 0.500 | 0.279 | 0.039 | −0.600 |
| B | `AUROC_dom` | −0.429 | 0.129 | [−0.64, −0.15] | 0.496 | 0.282 | 0.100 | −0.600 |
| B | `AUROC_dom2` | −0.525 | 0.057 | [−0.70, −0.26] | 0.521 | 0.275 | 0.102 | −0.600 |
| B | `r1` (one-direction share) | −0.464 | 0.099 | [−0.66, −0.19] | 0.505 | 0.286 | 0.085 | −0.600 |
| C | `log d95` | +0.480 | 0.085 | [+0.22, +0.68] | 0.514 | 0.290 | 0.070 | +0.338 |
| C | `log d90` | +0.459 | 0.099 | [+0.21, +0.67] | 0.542 | 0.296 | 0.049 | +0.131 |
| C | `AUROC_pca(d=1)` | −0.582 | 0.033 | [−0.76, −0.30] | 0.527 | 0.291 | 0.163 | −0.257 |
| C | `AUROC_pca(d=8)` | −0.477 | 0.088 | [−0.69, −0.24] | 0.476 | 0.294 | 0.124 | −0.543 |
| D | Fisher ratio along dom | −0.182 | 0.533 | [−0.43, +0.13] | 0.589 | 0.293 | 0.003 | −0.257 |
| D | centroid cosine | +0.371 | 0.187 | [+0.09, +0.56] | 0.592 | 0.283 | 0.024 | +0.371 |
| D | within-class part. ratio | +0.284 | 0.325 | [+0.04, +0.51] | 0.524 | 0.285 | 0.027 | −0.543 |
| D | within − between cosine | −0.468 | 0.095 | [−0.65, −0.22] | 0.602 | 0.283 | 0.030 | −0.600 |
| D | centroid dist. / within sd | −0.490 | 0.078 | [−0.67, −0.23] | 0.530 | 0.272 | 0.081 | −0.600 |
| D | within-pair / between (9) | −0.233 | 0.555 | — | 0.598 | 0.285 | — | −0.543 |
| E | split size | −0.320 | 0.262 | [−0.59, −0.07] | 0.703 | 0.290 | 0.057 | −0.621 |
| E | mean tokens | +0.002 | 1.000 | [−0.27, +0.24] | 0.482 | 0.280 | 0.020 | +0.600 |
| E | mean turns | −0.100 | 0.735 | [−0.29, +0.19] | 0.516 | 0.312 | 0.001 | +0.621 |
| E | class-paired (0/1) | +0.499 | 0.081 | [+0.24, +0.68] | 0.451 | 0.322 | 0.161 | — |
| E | fraction truncated | −0.501 | 0.072 | [−0.71, −0.23] | 0.480 | 0.283 | 0.039 | — |
| A′ | `m_ID` last-token | −0.201 | 0.492 | [−0.51, +0.01] | 0.593 | 0.616 | 0.036 | +0.207 |
| B′ | `AUROC_full` last-token | +0.377 | 0.177 | [+0.12, +0.60] | 0.501 | 0.351 | 0.066 | +0.058 |
| B′ | `r1` last-token | +0.367 | 0.187 | [+0.10, +0.60] | 0.462 | 0.282 | 0.118 | +0.143 |
| C′ | `log d95` last-token | −0.287 | 0.308 | [−0.55, −0.05] | 0.472 | 0.297 | 0.092 | −0.030 |
| D′ | Fisher ratio last-token | +0.279 | 0.323 | [+0.00, +0.53] | 0.498 | 0.291 | 0.035 | +0.143 |

**Baselines to beat: concept-only LOO RMSE = 0.284; grand-mean LOO RMSE = 0.501.**

## Table 2 — the baseline, and why nothing clears it

| model | LOO RMSE (log₁₀ units) |
|---|---|
| grand mean of the other 13 splits | 0.501 |
| **concept identity alone** (mean of the other splits of its concept) | **0.284** |
| best single predictor alone (`m_ID`, C=0.1) | 0.380 |
| best single predictor + concept | 0.274 |
| number of the 118 predictors whose LOO beats 0.284 | **0** |

Concept identity cuts the error almost in half; the best activation statistic recovers
only about a third of that, and adding it to concept improves 0.284 to 0.274 — a
3.5% reduction on 14 points, which is noise. Not one of the 118 predictors beats the
concept-only baseline on its own.

**Family-wise check.** Ten predictors reach a nominal p < 0.05 where 5.9 are expected by
chance. Permuting the target 10,000 times and taking the largest |ρ| across all 118
predictors each time, the median of that null maximum is 0.585 — against an observed
best of 0.604. **Family-wise p = 0.44.** The winner is not distinguishable from the best
of 118 noise predictors.

## Table 3 — within instructions only (6 splits)

This is where the knee spans more than an order of magnitude (log₁₀ m from 1.67 to 2.56)
and where concept identity cannot help, so it is the sharpest test.

| predictor | ρ inside instructions | p | sign holds |
|---|---|---|---|
| `k_ID` (C=1) | +0.754 | 0.109 | yes |
| `AUROC_ID(k=8)` | −0.657 | 0.171 | yes |
| `m_ID` (C=1) | +0.657 | 0.174 | yes |
| `m_ID` (C=0.1) | +0.638 | 0.196 | yes |
| split size | −0.621 | 0.259 | yes |
| mean tokens | +0.600 | 0.239 | yes |
| `AUROC_full` | −0.600 | 0.236 | yes |
| `r1` | −0.600 | 0.239 | yes |
| within − between cosine | −0.600 | 0.239 | yes |
| centroid dist. / within sd | −0.600 | 0.239 | yes |
| `log d95` | +0.338 | 0.666 | yes |
| `AUROC_pca(d=1)` | −0.257 | 0.656 | yes |

Signs hold, which is the one condition the leading predictors do satisfy. But with six
points nothing can reach p < 0.05 at these effect sizes — the smallest p here is 0.109 —
so this table constrains almost nothing. It is reported because the brief asks for it,
not because it decides anything.

## Curve level (130 non-flat Gemma curves)

| model of log₁₀ m | R² | gain over generator + prompt |
|---|---|---|
| generator + prompt | 0.150 | — |
| generator + prompt + **split** (the ceiling) | 0.616 | **0.466** |
| generator + prompt + `m_ID` (C=0.1) | 0.414 | 0.264 |
| generator + prompt + `AUROC_pca(d=1)` | 0.313 | 0.163 |
| generator + prompt + `r1` | 0.235 | 0.085 |
| generator + prompt + `AUROC_full` | 0.189 | 0.039 |

The split factor's 0.466 reproduces the paper's 42–45%. The best single predictor buys
0.264 of it — 57% of the ceiling from one number instead of 13 dummies. That is the most
favourable number in this study, and it should be read with the family-wise result in
mind: `m_ID` was chosen as the winner *after* seeing all 118, and the two predictors one
would have nominated in advance — full-model AUROC, and the one-direction share `r1` —
buy 0.039 and 0.085.

## Per-split values (sorted by target)

| split | concept | log₁₀ m | `m_ID` | `AUROC_full` | `r1` | `d95` | rows | paired |
|---|---|---|---|---|---|---|---|---|
| anthropic_hh_balanced | highstakes | 1.000 | 3.0 | 0.984 | +0.94 | 4 | 2984 | no |
| balanced_refusal | hu_harm | 1.057 | 3.5 | 0.997 | +0.91 | 4 | 400 | yes |
| daily_dilemmas | hu_harm | 1.129 | 33.6 | 0.930 | +0.46 | 128 | 196 | yes |
| mts_balanced | highstakes | 1.137 | 4.6 | 0.994 | +0.92 | 4 | 86 | no |
| mt_balanced | highstakes | 1.184 | 5.8 | 0.996 | +0.94 | 4 | 604 | no |
| toolace_balanced | highstakes | 1.191 | 14.6 | 0.911 | +0.72 | 32 | 734 | no |
| ai_dilemmas | hu_harm | 1.328 | 33.6 | 0.787 | −1.39 | 5376 | 136 | yes |
| ant_hh | hu_harm | 1.618 | 5.4 | 0.948 | +0.87 | 8 | 134 | no |
| mm_substitution | instructions | 1.672 | 20.4 | 0.975 | +0.83 | 128 | 200 | yes |
| bbq_substitution | instructions | 1.744 | 24.1 | 0.984 | +0.74 | 128 | 200 | yes |
| anthropic_harmless_refusal | instructions | 1.907 | **3.0** | **1.000** | +0.98 | **1** | 200 | yes |
| hc_context_drift | instructions | 2.052 | 77.4 | 0.808 | −1.12 | 128 | 194 | yes |
| hc_contradiction | instructions | 2.161 | 77.4 | 0.734 | −1.10 | 5376 | 200 | yes |
| oig_context_drift | instructions | 2.559 | 60.3 | 0.947 | −0.01 | 128 | 194 | yes |

**`anthropic_harmless_refusal` is the counterexample that breaks the hypothesis.** It is
the *easiest* split in-distribution by every measure taken here — held-out AUROC 1.000,
an in-distribution knee at the bottom of the grid, and a single principal direction
enough to reach 95% of the gain — and yet its synthetic knee is the third highest of the
fourteen (m ≈ 81). `ant_hh` is a milder version of the same thing: easy in-distribution
(AUROC 0.948, `m_ID` 5.4, `d95` 8) and the hardest knee in its concept. Whatever makes a
split need a lot of generated data, it is not that the split is hard to separate.

## What this supports

- **The paper's split effect reproduces.** Split identity explains 0.466 of the residual
  R² of log₁₀ m over generator + prompt across 130 non-flat Gemma curves, squarely in the
  42–45% the paper reports.
- **Concept is the usable part of that effect.** Leave-one-split-out, concept identity
  alone cuts RMSE from 0.501 to 0.284 log₁₀ units. If you must guess a knee before
  generating, guess your concept's mean; that is the state of the art this study leaves
  standing.
- **A weak, consistently signed association between the in-distribution knee and the
  synthetic knee.** `m_ID` correlates at ρ = +0.60 (p = 0.027) over 14 splits, keeps its
  sign within instructions, holds up on the `targeted` secondary target (ρ = +0.62,
  p = 0.037; `m_ID` at C=1 gives +0.70, p = 0.012), and buys 57% of the split ceiling at
  curve level. It is the only thing here that looks like a signal.

## What this does not support

- **Any of it as a usable predictor.** The association above does not survive the
  family-wise null (p = 0.44), does not beat concept identity (LOO 0.380 vs 0.284), and
  adds essentially nothing on top of it (0.284 → 0.274). All three of the brief's
  conditions must hold; the second fails outright.
- **"Hard in-distribution ⇒ needs more synthetic data."** This was the study's main
  hypothesis and the data contradict it directly:
  `anthropic_harmless_refusal` is perfectly separable in-distribution and near the top
  of the knee ranking. In-distribution difficulty and synthetic-data appetite are
  different things.
- **Geometry.** The whole of group D is the weakest family in the table. The Fisher ratio
  along the difference-of-means direction — the most natural "how separable is this"
  statistic available — has ρ = −0.18, p = 0.53, and a curve-level ΔR² of 0.003.
- **Last-token pooling.** Uniformly worse than mean pooling, and degenerate for the
  in-distribution knee: all six instructions splits land at the bottom of the size grid,
  so `m_ID` last-token has no variance to correlate with inside the concept.
- **Anything beyond `detailed`.** On `general` (9 splits) and `llm` (6 splits) nothing
  reaches p < 0.05. The one target where many predictors appear to pass all the filters
  is `targeted` (52 of 121) — but there the concept-only baseline is *worse* than the
  grand mean (0.533 vs 0.521), so the bar is degenerate, and each split's target is a
  single curve rather than a median over four. That result should not be believed.
- **Activations earning their keep.** Two predictors that need no activations at all —
  whether the split is class-paired (ρ = +0.50) and what fraction of its rows hit the
  1024-token cap (ρ = −0.50) — sit among the leaders. Nothing in the activation geometry
  clears what you can read off the JSONL.

## Caveats

- **Fourteen points.** Every split-level number rests on 14 observations, and with 118
  predictors the multiple-comparison burden is what the family-wise test measures: the
  best of 118 noise predictors correlates at 0.585 in the median. Read Table 1's
  per-predictor p-values as descriptive only.
- **The classifier here is not the paper's probe.** Mean-pooled logistic regression on
  standardised features is not tuberlens' softmax-pooled head trained with Adam and
  early stopping. A geometry that mean pooling cannot see is not ruled out by this
  study, and the in-distribution curve measures a different learner from the one whose
  knees are the target.
- **Censoring, concentrated in high-stakes.** The smallest size ever measured was 10, so
  every knee at or below it is a tie at log₁₀ m = 1.0. Nine of the 56 non-flat
  detailed-prompt curves are censored this way — eight of them high-stakes — and
  `anthropic_hh_balanced`'s median sits exactly on the floor. Six of the fourteen splits
  have a median at or under 1.2, so the bottom third of the target range is compressed
  into near-ties. The rank statistics handle that correctly but cannot recover
  information that was never measured. Most of the real spread lives in the six
  instructions splits, which is exactly where n = 6 makes significance unreachable.
- **Standardisation in the few-shot curve uses the whole training fold**, not the k
  drawn samples. Estimating 5376 feature means from two samples would make the curve a
  measurement of standardisation noise. The held-out fold is untouched either way, but
  this makes `AUROC_ID(k)` optimistic relative to a true k-sample practitioner.
- **Some splits cannot reach k = 256.** `mts_balanced` (43 rows per class) tops out at
  k = 64, so its in-distribution curve is fitted over fewer sizes than the others'.
- **Predictor F was not computed.** The brief allows the 50-sample base probe's per-split
  AUROC only if it is already present in `scripts/*_gen90*.csv` or
  `scripts/*_pooled_size_curve.csv` on `origin/per_split_studies`. It is not: every row
  in those files is either base + generated data (n ≥ 30) or generated-only
  (`base = none`); there is no base-only row. The `ownbase_only_*_50.pkl` probes on that
  branch would give it from cached activations without fitting anything, which is the
  obvious follow-up, but scoring them was outside what the brief permitted here.
- **The Kaggle fetcher is not committed.** `.gitignore` on this repo purged
  `fetch_kaggle_eval_activations.py` from history deliberately; the version used here
  (rebuilt for the three *eval* concepts — the copy on `origin/devsamples_kfold_cloud`
  points at the *dev* blobs) stays untracked in respect of that decision. Recover it
  from that branch and re-apply the `CONCEPTS` table if the blobs need refetching.
