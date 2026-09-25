# Task: Part C of the dev-sample coverage test under *harmful* and *high-stakes*

## Setup

Same repo, same branch `dev_coverage` (continue from the tip; `analysis/dev_coverage.md` and `scripts/dc_dev.py` are the state of play). Same environment (`.venv_claude/bin/python`), same rules (`CLAUDE.md`). Commit as you go on `dev_coverage` and push. No new activations are needed: every dev sample and the three validation blobs were extracted for Part A.

## Why

`docs/dev_coverage_task.md` ran Part C (size ladders for a dev-sample R = m_all / m_own) under *instruction* only, and told you not to run it under the other two concepts because their in-distribution kind-only half-gain sizes sit at the fitter's floor. That is still expected, and this brief asks for the runs anyway, for one reason: the paper's headline from the generated-set study is a *cross-concept* share, "four fifths of the gap in half-gain size between *instruction* and the other two concepts is coverage", computed as

```
gap_mixed = log10 m_mixed(instr) - mean(log10 m_mixed(harm), log10 m_mixed(hs))     # 0.99 on generated sets
gap_kind  = log10 m_kind(instr)  - mean(log10 m_kind(harm),  log10 m_kind(hs))      # 0.21
share     = 1 - gap_kind / gap_mixed                                                 # 0.79
```

with each m the median over the concept's used cells. That number cannot be checked on real data without m_all and m_own for all three concepts. If m_own under *harmful* and *high-stakes* pins at the floor, the real-data share is still informative: it becomes a bound (see "Reading the result" below), and the paper can say so instead of saying nothing.

## What to run

Part C ladders for the eight remaining splits, through the existing `partc-fit` and `partc-analyse` stages of `scripts/dc_dev.py`, generalised to take a concept. Same arm files as Part A (`.dc_work/dcdev_<concept>_<split>_{own,others,all}.jsonl`), same regime (`--grad-accum ceil(n/16) --batch-size 16`, eight draws, `--no-base`), same validation set (the concept's DeepSeek-V4-Pro detailed set; never a dev or eval split, never `dev_samples/highstakes_500`), same restriction rule (`own` and `others` scored on the target split through `dc_run_curve.py --eval-split`, `all` unrestricted).

### Ladders

Every size of the paper's ladder `2 4 6 10 20 30 50 80 110 170 350 590` that a class-balanced draw of the arm can supply (`n <= balanced` in `scripts/dc_dev_arms.csv`), which is the rule `partc_jobs()` already applies; only the ladder tuples change. Concretely:

| concept | arm | balanced size | ladder |
|---|---|---|---|
| *harmful* | own ai_dilemmas, ant_hh | 46, 44 | 2 4 6 10 20 30 |
| *harmful* | own daily_dilemmas | 66 | 2 4 6 10 20 30 50 |
| *harmful* | own balanced_refusal | 134 | 2 4 6 10 20 30 50 80 110 |
| *harmful* | others (156--246) | | 2 ... 110, plus 170 where balanced >= 170 |
| *harmful* | all | 290 | 2 4 6 10 20 30 50 80 110 170 |
| *high-stakes* | own mt, mts, toolace | 278, 274, 328 | 2 4 6 10 20 30 50 80 110 170 |
| *high-stakes* | own anthropic_hh | 1,028 (draws capped at 590) | 2 4 6 10 20 30 50 80 110 170 350 590 |
| *high-stakes* | others, all | 880--1,908 (draws capped at 590) | 2 4 6 10 20 30 50 80 110 170 350 590 |

Use `LADDER_FULL = (2, 4, 6, 10, 20, 30, 50, 80, 110, 170, 350, 590)` for every arm and let the `n <= balanced` rule cut it (the `balanced` column of `dc_dev_arms.csv` is the pre-cap size, so under *high-stakes* the rule admits the whole ladder for every arm but the three smaller `own` arms); keep the *instruction* results exactly as they are (their CSVs are done; the generalised stage must skip cells already in a CSV, which the harness's resume key does for you).

Count, roughly: *harmful* 29 own + 40 others + 10 all cells x 8 draws = 630 fits at 12--20 s; *high-stakes* 42 own + 48 others + 12 all cells x 8 = 816 fits, at 30--70 s restricted and about 115 s for the twelve unrestricted `all` cells. About 2 h and 14 h respectively. **Run *harmful* first, then *high-stakes*.** Commit and push after each concept's `partc-analyse`.

### Code

Generalise rather than duplicate: `PARTC_CONCEPT` becomes a `--concept` argument of the `partc-fit` and `partc-analyse` stages (default `instructions` so the old invocation still means what it meant), `partc_csv` names files by that concept, and the ladder tuples become the single full ladder above. Do not edit `subsample_curve_concept.py`, `dc_run_curve.py`, `fit_curves_ref.py`, `direction_count.py`, or any Part A stage. The *high-stakes* `all` arm is unrestricted and reads all four eval blobs (47 GB) per fit, as in Part A; if memory is tight run it alone.

### Analysis

`partc-analyse` for each concept writes rows to `scripts/dc_dev_partc_ratios.csv` (append to the existing six *instruction* rows; same columns: `m_own, m_all, m_others` with intervals, `R_dev` with the paired-ratio interval, `G_dev_fit`, the flat and censored flags) and `scripts/dc_dev_partc_fits.csv`. Same fitter, same flat rule (in-range gain under 0.02), same censoring flag (m at or below the smallest measured size). Then:

1. **Per concept:** median `m_own`, `m_all`, `R_dev`, `G_dev_fit` over the usable splits, with a bootstrap over splits (4,000 resamples) on each median, and the count of splits whose `m_own` (and whose `m_own` interval) sits at the grid floor of 3 or is flagged censored. Put the generated-set values (`scripts/dc_summary`-style: 11 / 7.2 / 6.9 kind-only, 117 / 18 / 7.9 mixed, R 5.30 / 2.12 / 1.18) beside them.
2. **The real-data share.** Compute `gap_all`, `gap_own` and `share` exactly as above with `m_all` for mixed and `m_own` for kind-only, medians over usable splits per concept. Interval: bootstrap over splits within each concept (4,000 resamples), reported as a 95% interval on `share`. Also report the two single-concept versions (*instruction* against *harmful* alone, against *high-stakes* alone), as Appendix E of the paper does (0.76 and 0.81 on generated sets).
3. **Part B, extended.** Re-run `--stage link` so `R_dev` is scored over all fourteen splits (it was six), with the same three bars as before.

Write a section "Part C under harmful and high-stakes" into `analysis/dev_coverage.md` with the per-concept table, the share with its interval, the floor counts, and the per-split table for the eight new splits (same columns as the *instruction* one there).

## Reading the result (write this reasoning into the analysis, adapted to what you find)

- If `m_own` under *harmful* and *high-stakes* is **not** at the floor for most splits: the share is a real number; compare it to 0.79 and give the interval.
- If `m_own` **is** at the floor (3) or censored for most of those splits: the true `m_own` there is at most what was reported, so `gap_own` as computed is a **lower bound** on the true kind-only gap and the share as computed is an **upper bound** on the true coverage share. Say that, give the bound, and give the share you get if the floor-pinned values are treated as exactly 3. Do not move the grid floor to resolve it.
- Either way, report whether *instruction*'s `m_own` (median 13--22 on real data, from the six rows already in `dc_dev_partc_ratios.csv`) exceeds the other two concepts' `m_own`, which is the "per kind the concepts are alike" claim (11 vs 7 on generated sets).

## Outputs (commit and push on `dev_coverage`)

- `scripts/dc_dev_partc_{hu_harm,highstakes}_*.csv` (fit CSVs), the appended `scripts/dc_dev_partc_ratios.csv` and `scripts/dc_dev_partc_fits.csv`, the refreshed `scripts/dc_dev_link_stats.csv` and scatter file, a new `scripts/dc_dev_partc_share.csv` (gap_all, gap_own, share, interval, the two single-concept versions, floor counts).
- The generalised `scripts/dc_dev.py` and a `run_dc_dev_partc2.sh` (gitignored; `git add -f`).
- The new section in `analysis/dev_coverage.md`.
- Final commit message: the per-concept table, the share with interval and the floor counts, the extended Part B line for `R_dev`, wall-clock per fit per concept.

## Do not

- Refit, delete or reorder any existing row of any CSV; the six *instruction* Part C rows and every Part A row stay as they are.
- Change the fitter's grid, the flat rule, or the censoring rule to make a floor-pinned `m_own` look resolved.
- Early-stop on anything but the concept's DeepSeek-V4-Pro detailed set.
- Touch `eval_sets/`, or include `oig_omission` anywhere.
