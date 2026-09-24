# Task: the coverage test on dev samples, and whether it predicts the synthetic half-gain size

## Setup

Repo: `https://github.com/acheckervarty7890/probe_auto_improvement` (private; you have access). Check out branch `dev_coverage` (cut from `direction_count` at 52bf3c72, so the harness, the `dc_*` study scripts, the eval sets, the dev sets, the generated sets and, if you are the machine that ran `direction_count`, every activation cache are all here). Bootstrap with `bash scripts/setup_env.sh` if `.venv_claude` does not exist; run every Python command as `.venv_claude/bin/python`. Read `CLAUDE.md` and `analysis/direction_count.md` before writing code. Commit as you go on `dev_coverage` and push; results live in the CSVs, in `analysis/dev_coverage.md`, and in the final commit message.

**GPU stages:** activations of the dev samples if they are not cached (about 2,600 conversations, one model load; `direction_count --stage warm` and the `--dev-data` runs of the paper's curves will already have cached most or all of them), then probe-head fits on cached activations. High-stakes fits are the slow ones (the four eval blobs are 47 GB; `scripts/dc_run_curve.py --eval-split` exists to avoid reading all four when only one is needed).

## Why

`analysis/direction_count.md` (and Section 5.2 / Appendix E of the paper) cuts each *generated* detailed set into kind-only, mixed and set-minus-kind arms and finds that a split learns from its own kind about as fast under every concept (half-gain size 7--11 samples), but that the concept's *other* kinds substitute for it under *high-stakes* and *harmful* (median R = 1.2 and 2.1, gain kept without the kind G = 0.95 and 0.90) and not under *instruction* (R = 5.3, G = 0.67). Two things that study cannot say:

1. **Is the non-substitution a property of the concept or of the generator's rendering of the kinds?** The kinds there are LLM-tagged samples of one generated set. The dev samples are real, in-distribution, and their kind is exact: it is the split they come from.
2. **Does any of it predict a split's synthetic half-gain size out of sample?** In `scripts/dc_link_stats.csv` both R and the cross-kind transfer AUROC `t_other` correlate with the per-split target (Spearman 0.71 and -0.79 over fourteen splits) but neither beats the concept's mean in leave-one-split-out error (0.37 vs 0.27), and both have Spearman 0.0 *within* the six *instruction* splits. Dev samples are what a practitioner would have in hand before generating anything, so a dev-sample version of these statistics is the version of the predictor that matters.

Part A answers 1 and produces the statistics for 2 with no curve fitting and no half-gain sizes, so none of the grid-floor problem that makes R noisy. Part B tests 2 with the machinery `knee_predictor.py` and `direction_count.py --stage analyse` already have. Part C (conditional) fits curves for a dev-sample R, under *instruction* only, where a kind-only half-gain size resolves.

The dev sets (`dev_samples/`): *instruction* six splits of 66--68 samples (`oig_omission.jsonl` is excluded, as everywhere in the paper); *harmful* `hu_ha/dev_{ai_dilemmas,ant_hh,balanced_refusal,daily_dilemmas}` with 46, 44, 134, 66; *high-stakes* four splits of 274--1,028. All are `{inputs, labels}` JSONLs with the concept's own label strings and are class-balanced or nearly so.

## Part A (run first): full-size arms on dev samples, no curves

Run these two statistics before anything that involves a half-gain size. Neither needs a curve fit, so neither carries the grid-floor problem that makes R noisy. Concept order throughout: **instruction first, then harmful, then high-stakes.**

**Step 1. Within-concept transfer matrix.** One probe per split trained on all its dev samples, scored on every eval split of the concept. Fourteen probes (eight draws each). This is the real-data analogue of the cross-kind transfer AUROC that already orders the concepts 0.81, 0.85, 0.95 (`t_other` in `scripts/dc_neff.csv`, which used per-kind difference-of-means directions; here it is the probe itself).

**Step 2. Set-minus-kind at full size.** One probe per split trained on every other split's dev samples, scored on the target split. Fourteen more probes. Compare its gain to the all-dev probe's gain (one probe per concept, three more): that is G with exact labels.

If those two reproduce the ordering, the curves for R are optional and run only under *instruction*, where a kind-only half-gain size resolves (Part C). Sections A1--A5 spell out the files, the validation set, the fit commands and the statistics for these two steps.


### A1. Materialise the arms

For every concept and every split `s` of it, three training files in the untracked `.dc_work/`, named `dcdev_<concept>_<split>_{own,others,all}.jsonl` (the `all` file is per concept, not per split, so write it once):

- `own`: every dev sample of `s` (Step 1).
- `others`: every dev sample of the concept's other splits (five under *instruction*, three otherwise). This is the set-minus-kind arm with exact labels (Step 2).
- `all`: the concept's whole dev set (404 / 290 / 1,908 samples). This is the mixed arm, the denominator of G (Step 2).

Write a manifest `scripts/dc_dev_arms.csv` with the columns of `scripts/dc_arms.csv` (`concept, gen, arm, split, file, n, n_pos, n_neg, balanced, sizes, skipped`; put `dev` in `gen`). The harness draws class-balanced, so the size each arm is fit at is `n_arm = min(2 * min(n_pos, n_neg), 590)`. The cap of 590 keeps every arm within the ladder the `direction_count` mixed arm used, and matters only for the *high-stakes* `others` and `all` arms.

Add a small `--stage` or a separate script (`scripts/dc_dev.py` is fine) rather than editing `direction_count.py`'s existing stages; reuse its helpers by import.

### A2. The validation set

The harness uses `--dev-data` as the early-stopping set. Here the dev samples are the *training* data, so pass a different validation set: the DeepSeek-V4-Pro detailed set of the concept,

```
--dev-data data/<concept>_deepseekv4pro_evaldesc_600.jsonl
```

(`instructions_deepseekv4pro_evaldesc_600.jsonl`, `hu_harm_deepseekv4pro_evaldescshape_600.jsonl`, `highstakes_deepseekv4pro_evaldesc_600.jsonl`; a single JSONL is accepted). It is already cached, disjoint from every eval set, and off-distribution to the target, so it cannot leak. Use the same file for all three arms and all three concepts' runs so the early-stopping signal is one thing throughout. Do **not** early-stop on any dev split, any eval split, or the training arm itself. State in the commit message that this is the validation set, and never mix it with `dev_samples/highstakes_500` in one CSV.

### A3. Fits

One `subsample_curve_concept.py` call per (arm file), at the single size `n_arm`, eight draws, `--no-base`, the `direction_count` optimiser regime (`--grad-accum ceil(n_arm/16) --batch-size 16`, one optimiser step per epoch; this is what `direction_count.py --stage fit` passes and it tags the rows `none+ga<K>bs16`). Eight draws at full size differ only in draw order and initialisation, which is the seed spread you want.

- `own` arms: **unrestricted** eval, so each fit scores every eval split of the concept. The per-split columns of one `own` fit are one row of the transfer matrix: probe trained on real samples of `s`, scored on every split of the concept. Out: `scripts/dc_dev_<concept>_own.csv`.
- `all` arms: unrestricted (one fit reads for all splits). Out: `scripts/dc_dev_<concept>_all.csv`.
- `others` arms: restricted to the target split through `scripts/dc_run_curve.py --eval-split <s>` (the other columns of an `others` fit are not needed, and under *high-stakes* they are the whole cost). Out: `scripts/dc_dev_<concept>_others_<split>.csv` (the restricted runner writes its own file because the header differs).

Count: *instruction* 6 x 8 own + 6 x 8 others + 8 all = 104 fits; *harmful* 72; *high-stakes* 72; **248 fits**. Order: *instruction* first (it is the concept the whole question is about, and the one where Part C may follow), then *harmful*, then *high-stakes* (the slow one). Within a concept run Step 1 (`own`) before Step 2 (`others`, then `all`), and commit and push after each concept so a partial result is already usable. Set `AGENTIC_REDTEAM_MAX_MEMORY` as the other run scripts do. The harness resumes on `(tag, file, n, draw)`, so a killed run loses at most one fit; do not launch the fits through a driver you might later kill by its parent pid alone (the fit worker outlives its parent).

*Harmful* has the smallest dev sets (Ant-HH 22/22), so it is where a class-balance edge case would show; check its `own` rows land at n = 44 and 46 before moving on. If the `all` arm under *high-stakes* at n = 590 with accumulation 37 returns an across-draw sd of exactly 0 at any split, the probe took no optimiser step; report it rather than change the regime.

### A4. Statistics per split (write `scripts/dc_dev_cells.csv`)

With `A_x(s)` the mean over eight draws of eval AUROC on split `s` for arm `x`:

- `a_own`: `A_own(s)`, the in-distribution ceiling of real data.
- `a_others`: `A_others(s)`, the set-minus-kind level.
- `a_all`: `A_all(s)`, the mixed level.
- `G_dev = (a_others - 0.5) / (a_all - 0.5)`: gain kept without the split's own kind, gain measured from chance since nothing is fitted here.
- `t_other_dev`: the mean over the concept's other splits `s'` of `A_own(s')` scored on `s`, i.e. the column mean of the transfer matrix off the diagonal. This is the dev-sample analogue of `t_other` in `dc_neff.csv`, which used per-kind difference-of-means directions; here it is the probe itself.
- `t_own_dev`: the diagonal, `A_own(s)` on `s` (equals `a_own`; keep both names so the CSV reads like `dc_neff.csv`).
- `sd_*`: the across-draw sd for each of the above.

Also write the full transfer matrix per concept, `scripts/dc_dev_transfer_<concept>.csv` (rows: training split, columns: eval split, mean over draws).

### A5. What Part A is testing (state these as predictions before looking)

From the generated-set study, if non-substitution is a property of the concept:

- `G_dev` well below 1 under *instruction* (the generated-set medians are 0.67 vs 0.90 and 0.95) and near 1 under the other two.
- Harmless-refusal collapses without refusal samples: `a_others` near 0.5 there (the generated set kept 11% of the gain under DeepSeek and went flat under GPT-OSS).
- `t_other_dev` ordered *instruction* < *harmful* < *high-stakes* (the generated-set directions gave 0.81, 0.85, 0.95).
- Under *high-stakes* and *harmful*, `a_others` within a few hundredths of `a_all`.

If instead `G_dev` and `t_other_dev` are alike across concepts on real data, the generated-set result is about the generated kinds not covering one another, not about the concept; say so plainly in `analysis/dev_coverage.md`. Either outcome is a result.

Ant-HH is the known exception (its own generated kind was worse than the rest of the set); report its `a_own` vs `a_others` separately.

## Part B: does it predict the synthetic half-gain size?

The target is the one `knee_predictor.py` and `direction_count.py --stage analyse` already use: `log_m_detailed`, each split's median log10 half-gain size over its four detailed-prompt curves (`knee_predictor.load_targets`, `PRIMARY_TARGET`), with the bootstrap over the curves behind each median. Fourteen splits (MM-substitution now has a value on the predictor side too, since it has a dev file; it had no kind-only arm in the generated study).

For each of `G_dev`, `t_other_dev`, `a_others`, `a_own`, and `a_all - a_others`, compute exactly what `_link_to_paper_m` in `direction_count.py` computes for R and `t_other` (it imports the functions from `knee_predictor`): Spearman with tie-averaged ranks, permutation p, the bootstrap interval that resamples the curves behind each split's median, leave-one-split-out RMSE of a linear prediction of the target from the predictor alone, from concept identity alone, from the grand mean, and from predictor plus concept, `beats_concept`, and Spearman with its p **within the six *instruction* splits**. Write `scripts/dc_dev_link_stats.csv` with the columns of `scripts/dc_link_stats.csv`, and a scatter file with the columns of `scripts/dc_scatter.csv`.

The bar, as in the paper's Appendix D: rho >= 0.6 at p < 0.05, LOO error below concept identity's (0.27), and a sign that holds within *instruction*. Report every predictor against all three, and say which, if any, clears all three. The expected outcome, to be honest about it up front: the between-concept ordering reproduces and nothing beats 0.27, because the concept mean already carries the between-concept part and the residual is within *instruction*. A null with exact labels is worth having; do not tune anything to avoid it.

## Part C (conditional): curves for a dev-sample R, *instruction* only

Run only if Part B shows within-*instruction* signal for `G_dev` or `t_other_dev` (|rho| >= 0.5 over the six splits, either sign). Otherwise skip and say so.

Ladders through the same regime as `direction_count.py --stage fit` (one `subsample_curve_concept.py` call per size, `--grad-accum ceil(n/16) --batch-size 16`, eight draws, `--no-base`, the Part A validation set): `own` at 2 4 6 10 20 30 50 (a 68-sample dev split gives at most 66 balanced, so the ladder stops at 50); `others` and `all` at 2 4 6 10 20 30 50 80 110 170 300 (the `others` files hold about 336 samples, `all` 404). Restricted eval for `own` and `others`, unrestricted for `all`. About 6 x 7 x 8 + 6 x 11 x 8 + 11 x 8 = 952 fits. Fit every curve with the log-logistic of `direction_count.py --stage analyse` (same flat rule `gain_obs < 0.02`, same censoring floor, same grid), and report per split `m_own, m_all, m_others, R_dev = m_all / m_own, G_dev` with their bootstrap intervals, then add `R_dev` to the Part B table. Do not run Part C under *harmful* or *high-stakes*: the in-distribution kind-only half-gain sizes there sit at the fitter's floor (`analysis/knee_predictor.md`, `m_ID`), so R would be a ratio of two unresolved numbers.

## Outputs (commit and push on `dev_coverage`)

- `scripts/dc_dev_arms.csv`, the fit CSVs `scripts/dc_dev_<concept>_{own,all}.csv` and `scripts/dc_dev_<concept>_others_<split>.csv`, `scripts/dc_dev_cells.csv`, `scripts/dc_dev_transfer_<concept>.csv`, `scripts/dc_dev_link_stats.csv`, the scatter file, and Part C's CSVs if run.
- The script(s) you wrote, and any wrapper `run*.sh` (gitignored; `git add -f`).
- `analysis/dev_coverage.md`: the method in a paragraph, the per-concept table (median `a_own`, `a_others`, `a_all`, `G_dev`, `t_other_dev`), the per-split table, the Part B table with the three bars, the Part A predictions and whether each held, the caveats (validation set, the 590 cap, class balance of the smallest dev sets, that `G_dev` is a gain from chance rather than a fitted gain).
- Final commit message: the per-concept table, the Part B verdict in one line per predictor, wall-clock per fit per concept, and whether Part C ran. Intermediate commits after each concept of Part A (instruction, harmful, high-stakes, in that order) with that concept's transfer matrix and its `a_own / a_others / a_all / G_dev` rows.

## Do not

- Edit `subsample_curve_concept.py`, `dc_run_curve.py`, `direction_count.py`'s existing stages, `knee_predictor.py`, or `fit_base_plus_concept.py`.
- Delete, reorder or refit any row of any existing CSV (`dc_*.csv`, `knee_*.csv`, the `*_size_curve.csv` files).
- Early-stop on dev samples, on an eval split, or on the training arm; and never pass `dev_samples/highstakes_500` or the full high-stakes dev set as `--dev-data` in this study.
- Touch `eval_sets/`. Every number here is scored on the eval splits exactly as the paper's curves are.
- Include `oig_omission` anywhere.
