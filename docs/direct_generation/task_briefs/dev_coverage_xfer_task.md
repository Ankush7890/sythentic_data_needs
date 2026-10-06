# Task: probe transfer scored on the classifier's own source — dev split to dev split, generated kind to generated kind

## Setup

Same repo, same branch `dev_coverage` (continue from the tip; `scripts/dc_dev.py`, `scripts/dc_dev_geometry.py`, `scripts/direction_count.py` and `analysis/dev_coverage.md` are the state of play). Same environment (`.venv_claude/bin/python`), same rules (`CLAUDE.md`). Commit as you go on `dev_coverage` and push. **Concept order: *instruction*, then *harmful*, then *high-stakes*; commit and push after each.**

**GPU stages:** one extraction pass to build the scoring blobs (about 2,000 dev rows plus the twelve generated sets' tagged rows, about 6,500 conversations; Part A's 2,602 dev samples plus three 600-row blobs took about 2 h, so budget 5 h), then probe-head fits on cached activations.

## Why

Every probe transfer number so far is scored on the eval splits: Part A's `t_other_dev` is a probe trained on one dev split scored on the *other eval splits* (`dc_dev.py`, `concept_cells`, the `eval_<split>` columns), and `docs/dev_coverage_gen_probe_task.md` asks for the same with generated kinds as the training arms. The direction statistic is the one scored on the classifier's own source: on the generated row it is a kind's direction scored on the other kinds' *generated* samples (0.82 / 0.85 / 0.95), and `dc_dev_geometry.py` scored dev-split directions on the other *dev* splits and found it flat (0.61 / 0.63 / 0.62; commit ef086f60). So the five numbers in play differ in the classifier and in the scoring set at once, and the two "same source" cells for the probe are missing. This task fills them:

- **dev to dev**: a probe trained on all dev samples of split `s`, scored on the dev samples of every other split of the concept.
- **generated to generated**: a probe trained on the tagged samples of kind `k` of one generated set, scored on the other kinds' samples of the same set.

With these, each row of `dc_main` has the probe scored on its own source and on the eval splits, and the direction scored on its own source and on the eval splits.

## Scoring outside the harness

The harness (`subsample_curve_concept.py`) scores every fit on the concept's eval dir and deletes the probe pickle (`out_pkl.unlink`) before the next draw, so the transfer scores have to be taken while the pickle exists. Do not edit the harness. Write `scripts/dc_xfer.py` that, as `dc_run_curve.py` and `dc_dev.py --stage harness` do, imports the harness and calls its `main()` with the concept's `eval_dir` replaced, and additionally wraps `synthetic_probe_data.evaluation.evaluate_probe` (the harness imports it inside `main`, so patch the attribute on that module before calling `main`) with a function that first scores the pickle on the transfer directory through `synthetic_probe_data.retrain.score_probe_on_dev(out_pkl, <dir>, concept.base_cache, combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT)`, appends one row `(samples, base, n, draw, <one column per file in the directory>)` to a side CSV, then calls the original. `score_probe_on_dev` returns per-file AUROC for a directory of JSONLs (`_per_split_auroc`), reads the directory's activation blob if it exists and extracts it otherwise, and the blob is keyed on the files' names and bytes, so each scoring directory is extracted once and reused by every fit. Make the side CSV resumable on the same `(samples, base, n, draw)` key the harness uses, and skip a fit whose row is already in both files.

**Scoring directories** (in `.dc_work/`, symlinks where the file exists, never copies of anything in `eval_sets/`):

- `xfer_dev_<concept>/`: one symlink per dev split of the concept, the `own` arm files from Part A (`.dc_work/dcdev_<concept>_<split>_own.jsonl`; rebuild with `dc_dev.py --stage arms` if absent — they carry the assistant-first fix, `dev_samples/` does not). *Instruction* six files (404 rows), *harmful* four (290). For *high-stakes* the four splits hold 274-1,028 rows and a 1,908-row blob is about 21 GB, held in memory at every scoring call; write instead four files each cut class-balanced to 150 rows (600 rows, seeded off `dc_dev.SEED` or 20260917 if it has none, the cut recorded in the manifest), which also matches the generated sets' size.
- `xfer_gen_<concept>_<gen>/`: one symlink per kind file of the set, `.dc_work/dc_<concept>_<gen>_<split>_kind.jsonl` (rebuild with `direction_count.py --stage arms` if absent), including kinds with fewer than 60 tagged rows and kinds with one class only (a single-class file scores NaN and is left out; it still needs a column). Twelve directories, about 520-600 rows each.

Build every blob before the fits, in a `--stage warm` that calls `score_probe_on_dev` once per directory with the concept's base probe, as `dc_dev.py --stage warm` does for its validation blob, and report the base probe's AUROC per file as the sanity check. Fifteen blobs, about 6 GB each; delete a concept's generated-set blobs after its fits are done and committed if disk is short, and say so.

## Arms, validation sets and fits

**Dev arms** (the 14 `own` arms of Part A, at `n_arm = min(2 * min(n_pos, n_neg), 590)`, eight draws, `--no-base`, `--grad-accum ceil(n/16) --batch-size 16`): the same command `dc_dev.py --stage fit` builds for an `own` arm, early-stopped on the concept's DeepSeek-V4-Pro detailed set (`dc_dev.val_dir(concept)`), with the eval restricted to the split's own eval split (`dc_run_curve.restricted_eval_dir`; the eval column is a by-product here and unrestricted *high-stakes* eval is the whole cost) and the transfer directory `xfer_dev_<concept>`. Part A's pickles were not kept, so these are refits; 112 fits. Out: `scripts/dc_xfer_dev_<concept>_own.csv` (harness rows) and `scripts/dc_xfer_dev_<concept>_scores.csv` (side rows).

**Generated arms** (the `kind` arms with at least 60 tagged rows, the `MIN_KIND_TAGGED` rule: 19 + 16 + 16 = 51 arms, at `n = 2 * min(n_pos, n_neg)`, eight draws, same regime): early-stopped on the concept's dev samples, the harness default and what every generated curve in the paper used (`dev_samples/instructions`, `dev_samples/hu_ha`, and `--dev-data dev_samples/highstakes_500` for *high-stakes*), never on the DeepSeek detailed set, which is the source of DeepSeek's own arms. Eval restricted to the kind's own split; transfer directory `xfer_gen_<concept>_<gen>`. 408 fits. Out: `scripts/dc_xfer_gen_<concept>__<gen>_own.csv` and `..._scores.csv`, one pair per (concept, generator), never shared between processes.

If `docs/dev_coverage_gen_probe_task.md` has not been run when you start, its `kind` arms are these same 51 arms with unrestricted eval: run them once through `dc_xfer.py` with unrestricted eval and both briefs are served by one set of fits (write the harness rows where that brief expects them). If it has been run, its pickles are gone and the fits here are separate; say which happened.

Neither validation set overlaps a scoring directory or a training arm: the dev arms early-stop on generated samples and are scored on dev samples; the generated arms early-stop on dev samples and are scored on generated samples. Never early-stop on a scoring directory, an eval split or the training arm. Set `SYNTHETIC_PROBE_DATA_MAX_MEMORY` as the other run scripts do; the harness resumes per fit, so a killed run loses at most one.

## Statistics

The diagonal of both matrices is in-sample (a probe trained on all of `s` scored on `s`), so it is reported and flagged, never averaged into anything; the off-diagonal is the statistic.

- `scripts/dc_xfer_dev_<concept>.csv`: rows training split, columns dev split, eight-draw means. Per split, `t_dev2dev(s)` = the off-diagonal column mean (other splits' probes scored on `s`'s dev samples) with its across-draw sd; per concept the median over splits.
- `scripts/dc_xfer_gen_<concept>__<gen>.csv`: rows training kind, columns kind, same. Per (generator, split) `t_gen2gen`; per split the median over generators with the generator range; per concept the median over splits. `mm_substitution` has a column and no row.
- `scripts/dc_xfer_cells.csv`: one row per split with `t_dev2dev`, `t_gen2gen` (median over generators), and beside them the existing numbers for the same split: `t_other_dev` from `dc_dev_cells.csv` (probe, dev to eval), `t_other_dom` and the eval-side `e_other_dom` from the `dc_dev_dom_*` files (direction, dev to dev and dev to eval), `t_other` and `e_gap` from `dc_link_stats.csv` (direction, generated to generated and to eval), and `t_other_gen` from `dc_gen_cells.csv` if that brief has run.
- `scripts/dc_xfer_summary.csv`: the per-concept table with all of those as columns, two classifiers by two scoring sets by two sources.
- A Spearman of `t_dev2dev` and of `t_gen2gen` with the per-split `log_m_detailed` target, as `dc_dev.py --stage link` computes it, appended to a new `scripts/dc_xfer_link_stats.csv` with the columns of `dc_dev_link_stats.csv`.

## Predictions, stated before looking

- `t_gen2gen` ordered *instruction* < *harmful* < *high-stakes*: the direction on the same samples gives 0.82 / 0.85 / 0.95 and the leave-one-kind-out ladders agree, so a trained probe should too, and at a higher level than the direction.
- `t_dev2dev` is the open cell. The direction was flat here (0.61 / 0.63 / 0.62) while the probe on eval was ordered (0.63 / 0.74 / 0.87). If the probe on dev is ordered, the flat direction result is about the class-mean classifier and the ordering is a property of the samples under any trained read-out; if the probe on dev is also flat, the real-sample ordering lives only in the eval scoring and the paper has to say which scoring set its real row uses and why.
- The refusal asymmetry holds in both dev matrices (a refusal-trained probe near chance on the other *instruction* dev splits, BBQ reversed on refusal) and is weaker on generated kinds.
- `t_dev2dev` and `t_gen2gen` differ from their eval-scored counterparts by less than the spread across splits within a concept, otherwise the scoring set is doing as much work as the classifier and the paper's caption has to name it.

## Outputs (commit and push on `dev_coverage`, one commit per concept and a final one)

- `scripts/dc_xfer.py` and any wrapper `run*.sh` (`git add -f`); a manifest `scripts/dc_xfer_arms.csv` with the columns of `dc_dev_arms.csv` plus `xfer_dir`, and the *high-stakes* scoring cut listed per file.
- The fit and score CSVs, the two matrix families, `dc_xfer_cells.csv`, `dc_xfer_summary.csv`, `dc_xfer_link_stats.csv`.
- A section in `analysis/dev_coverage.md`, "Transfer scored on the classifier's own source": the method in a paragraph (the wrapper, the scoring directories, the *high-stakes* cut, the two validation sets and why neither leaks), the per-concept summary table with all eight cells, the per-split table, the two dev matrices side by side (probe and direction), the predictions and whether each held, wall-clock per fit and per blob.
- Commit message per concept: that concept's `t_dev2dev` and `t_gen2gen` medians beside `t_other_dev` and `t_other`; the final one: the summary table and one line on whether the ordering holds on the probe's own source for each of the two rows.

## Do not

- Edit `subsample_curve_concept.py`, `dc_run_curve.py`, `direction_count.py`, `dc_dev.py`'s existing stages, `dc_dev_geometry.py`, `knee_predictor.py` or `fit_base_plus_concept.py`; wrap, do not patch files.
- Add rows to any existing CSV (`dc_curves_*`, `dc_dev_*`, `dc_gen_*`, `dc_neff`, `dc_geometry`, `dc_arms`).
- Early-stop on a scoring directory, an eval split or the training arm; put `highstakes_500`-validated and DeepSeek-validated rows in one file.
- Copy anything out of `eval_sets/`; include `oig_omission` anywhere (it is in `dev_samples/instructions`, so the harness reads it as validation for the generated arms as every paper curve did, and it stays out of every scoring directory, arm and matrix).
