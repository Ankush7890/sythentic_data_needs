# Task: fill the high-stakes general-prompt size curve (Figure 1, top-left panel)

## Setup

Repo: `https://github.com/acheckervarty7890/probe_auto_improvement` (private; you have access). Check out branch `hs_general_fill` (cut from `direction_count`, so the fitting harness, the high-stakes eval sets, the 500-sample dev cut, the four generators' 50-sample bases and their detailed sets are all here, and if you are the machine that ran the `direction_count` high-stakes arms, their activations are already in the per-sample cache). Bootstrap with `bash scripts/setup_env.sh` if `.venv_claude` does not exist; run every Python command as `.venv_claude/bin/python`. Read `CLAUDE.md` before writing code. Commit as you go on `hs_general_fill` and push; results live in the CSV and in the final commit message.

**One GPU stage** (step 4: Gemma-3-27B-IT layer-32 activations of the four general-prompt 600-sample sets, about 2,400 conversations, one model load). Everything else is probe-head fits on cached activations. Each high-stakes fit scores all four high-stakes eval splits, which is what makes this concept slower than the other two; with the 500-sample dev cut a fit takes about 90-120 s and holds about 7 GiB, so two can run side by side on a 24 GB card.

## Why

Figure 1 of the paper plots, per concept and generator, mean eval AUROC against training-set size at n = 10, 30, 80, 110, 170, 350, 590 under the **pooled** protocol: every training row is drawn from the generator's 650-row pool (its own 50-sample general-prompt base plus its 600 generated samples), `--no-base`, class-balanced, eight draws per size. *Harmful* and *instruction* have every size for all four generators (`origin/per_split_studies:scripts/{hu_harm,instructions}_pooled_size_curve.csv`, 384 rows each). The high-stakes pooled run stopped early on 2026-09-15: `origin/per_split_studies:scripts/highstakes_pooled_size_curve.csv` has 96 rows, i.e. both arms at n = 10, 30, 80 only, four draws each. So the high-stakes general-prompt curves in the figure have points at 10, 30, 80 and a 590 point borrowed from the fixed-base protocol, and **n = 110, 170, 350 are missing for all four generators** (and 590 is not on the pooled protocol).

## What to run (required)

General arm only, four generators, n = 110, 170, 350, 590, `--grad-accum 4`, eight draws, `--no-base`, dev set `dev_samples/highstakes_500`: 4 x 4 x 8 = **128 fits**, tagged `base = none+ga4` in the CSV. This is exactly the phase-2 protocol the other two concepts ran (`origin/per_split_studies:run_pooled_sizecurve2.sh`, whose header comment explains why accumulation is held at the default 4 above n = 80).

### Steps

1. **Fetch the four general-prompt sets**, which this branch does not carry (it has only the `_evaldesc_600` detailed sets and the `_50` bases):
   ```
   git checkout origin/per_split_studies -- data/highstakes_llama70b_600.jsonl data/highstakes_gptoss_600.jsonl data/highstakes_nemotron_600.jsonl data/highstakes_deepseekv4pro_600.jsonl
   ```
   Commit them. Each is 300/300 with fields `inputs` (JSON string of messages) and `labels`.

2. **Seed the output CSV with the 96 existing rows**, so the new rows append under the header the paper's reader expects and the resume key protects the old rows:
   ```
   git show origin/per_split_studies:scripts/highstakes_pooled_size_curve.csv > scripts/highstakes_pooled_size_curve.csv
   ```
   Never delete, reorder, or refit those 96 rows.

3. **Build the pools** (deterministic from `data/`, written to the untracked `.pool_work/`):
   ```
   .venv_claude/bin/python scripts/build_pooled_sets.py --concept highstakes
   ```
   Expect eight files `pool_highstakes_<gen>_{general,evaldesc}_650.jsonl`, 325/325 each; report any deduplicated rows.

4. **Warm the activation cache** (GPU, one load):
   ```
   .venv_claude/bin/python scripts/warm_pooled_sets.py --concept highstakes
   ```
   The `--dry-run` flag prints how many conversations are uncached first; expect about 2,400 (the four general sets) if the detailed sets and bases are already cached, about 4,800 otherwise.

5. **Fits.** Per generator, in this order (Llama first: it is the generator with the least spread in the other concepts' curves, so its four points validate the setup):
   ```
   for gen in llama70b deepseekv4pro gptoss nemotron; do
     .venv_claude/bin/python scripts/subsample_curve_concept.py --concept highstakes \
        .pool_work/pool_highstakes_${gen}_general_650.jsonl \
        --no-base --dev-data dev_samples/highstakes_500 --grad-accum 4 \
        --sizes 110 170 350 590 --draws 8 \
        --out scripts/highstakes_pooled_size_curve.csv \
        >> logs/pooled_sizecurve_highstakes.log 2>&1
   done
   ```
   Pass `--grad-accum 4` explicitly even though 4 is the default: it is what writes the `none+ga4` tag. Set `SYNTHETIC_PROBE_DATA_MAX_MEMORY` as the other run scripts do (`0=22GiB,cpu=45GiB`). Two generators may run concurrently. The harness resumes: re-running the same command skips `(tag, file, n, draw)` rows already in the CSV, so a kill loses at most one fit. If you write a wrapper script, note that `run*.sh` is gitignored and must be `git add -f`-ed. Do not launch the fits through a driver you might later kill by its parent pid alone: the fit worker is a subprocess and outlives its parent, and orphaned workers are what filled the card during the `direction_count` run.

6. **Check** before pushing the final commit:
   - `.venv_claude/bin/python scripts/report_pooled_curve.py --concept highstakes` (copy the script from `origin/per_split_studies:scripts/report_pooled_curve.py` if it is not on this branch) shows, for every generator, eight draws at 110, 170, 350, 590 on tag `none+ga4`.
   - No size has an across-draw sd of exactly 0: that is the untrained-probe failure (zero optimiser steps) and means the regime is wrong.
   - The pooled 590 mean should sit within about 0.02 of the fixed-base 590 point already in the paper for the same generator (`scripts/highstakes_gen90_dev500.csv`, `highstakes_<gen>_600.jsonl` on base `highstakes_<gen>_50.jsonl`, n = 540): Llama 0.849, DeepSeek 0.900, GPT-OSS 0.877, Nemotron 0.879. A larger gap is a finding, not an error; report it rather than rerun.
   - The curve should be monotone within noise from the existing 80 point (Llama 0.863, DeepSeek 0.847, GPT-OSS 0.852, Nemotron 0.841 at n = 80, ga5 rows).

## Optional, lower priority (state in the commit message whether these were run)

- **P2, 64 fits.** n = 80 at `--grad-accum 4` for both arms (all eight pools, eight draws), so high-stakes uses the same regime at 80 as the other concepts; the paper currently falls back to the `ga5` rows at 80 for this concept only. Same command with `--sizes 80` on all eight pools.
- **P3, 96 fits.** Top up n = 10, 30, 80 from four to eight draws on both arms, with the phase-1 regimes: n = 80 `--grad-accum 5`, n = 30 `--grad-accum 2`, n = 10 `--grad-accum 1 --batch-size 16` (one size per command, as in `origin/per_split_studies:run_pooled_sizecurve.sh`). `--draws 8` computes draws 0-7 and the resume key skips the existing 0-3, so exactly four new draws land per cell and no existing row moves.

Run the required 128 fits and push before starting either optional block.

## Outputs (commit and push on `hs_general_fill`)

- `scripts/highstakes_pooled_size_curve.csv` (96 existing rows plus 128 new, plus any optional rows).
- `data/highstakes_<gen>_600.jsonl` (the four fetched sets) and any wrapper script.
- Final commit message: a table of mean +- sd eval AUROC per generator at 10, 30, 80, 110, 170, 350, 590 (general arm), the row count per tag, the wall-clock per fit, and the answers to the checks above.

## Do not

- Edit `subsample_curve_concept.py`, `fit_base_plus_concept.py`, `build_pooled_sets.py`, or `warm_pooled_sets.py`.
- Refit or delete any existing row of any CSV, or touch `dc_*.csv`, `knee_*.csv`, or the other concepts' curves.
- Use the full 1,908-sample high-stakes dev set: every high-stakes small-n curve in the paper used `dev_samples/highstakes_500`, and mixing the two would change the early-stopping signal within one curve.
