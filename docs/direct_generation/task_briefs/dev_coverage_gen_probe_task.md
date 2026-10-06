# Task: the probe transfer statistic on generated samples, the way Part A ran it on dev samples

## Setup

Same repo, same branch `dev_coverage` (continue from the tip; `scripts/dc_dev.py`, `scripts/direction_count.py` and `analysis/dev_coverage.md` are the state of play). Same environment (`.venv_claude/bin/python`), same rules (`CLAUDE.md`). Commit as you go on `dev_coverage` and push. This is the companion of `docs/dev_coverage_dom_task.md`: that brief carries the generated-set *direction* statistic over to the dev samples; this one carries the dev-sample *probe* statistic over to the generated sets. Together they let Table `dc_main` report one classifier per row instead of two.

**GPU stages:** probe-head fits on cached activations only. Every generated set's activations are in the per-sample cache from `direction_count.py --stage warm`, and the eval blobs are the ones every fit reads. *High-stakes* is the slow concept (four eval blobs, 47 GB; the unrestricted fits here read all four).

## Why

Part A of `docs/dev_coverage_task.md` trained one probe per dev split at its full class-balanced size (`own`), one on the concept's other splits (`others`), one on the whole dev set (`all`), eight draws each, and read them as levels: `t_other_dev` is the off-diagonal column mean of the own-probe transfer matrix on the eval splits, `G_dev = (a_others - 0.5) / (a_all - 0.5)`. The generated-set study (`direction_count.py`) never did this: its kind-only and leave-one-kind-out arms are size *ladders* that stop at 80 and at the ladder's top, its `t_other` is a difference-of-means direction scored on generated samples, and under *high-stakes* the kind arms were scored on their own split only. So the real row of `dc_main` has a probe transfer number and the generated row has a direction one. This task produces the probe version on the generated sets: full-size `kind`, `loko` and `mixed` arms, eight draws, the eval-split transfer matrix, and the same cell statistics as `scripts/dc_dev_cells.csv`.

## Arms

The arm files are the ones `direction_count.build_arms` writes from the committed tags in `data/kind_tags/`: `.dc_work/dc_<concept>_<gen>_{mixed,<split>_kind,<split>_loko}.jsonl`, rows copied verbatim from the generated set, disagreed rows dropped from `kind` and `loko`. Rebuild them if `.dc_work/` is empty (`direction_count.py --stage arms`, deterministic). Do not overwrite `scripts/dc_arms.csv`, which now holds only the *high-stakes* rows of a later run; write the manifest for this task to `scripts/dc_gen_arms.csv` with the same columns, `sizes` holding the single size each arm is fit at:

- `kind` = the `own` arm: every tagged sample of the kind, `n = 2 * min(n_pos, n_neg)` (62-211 across sets; no cap bites). Skip a kind with fewer than 60 tagged samples, the `MIN_KIND_TAGGED` rule, which drops `mm_substitution` everywhere and `hc_context_drift` under llama70b, as in the study.
- `loko` = the `others` arm: the set minus the kind, `n = min(2 * min(n_pos, n_neg), 590)` (300-452).
- `mixed` = the `all` arm at 590: **do not refit**. The eight draws at n = 590 in `scripts/dc_curves_<concept>__<gen>.csv` (`samples = dc_<concept>_<gen>_mixed.jsonl`) are unrestricted, on the validation set below, and are the rows to read.

Count of new arm fits: `kind` 19 + 16 + 16 = 51, `loko` 24 + 16 + 16 = 56; 107 arms x 8 draws = **856 fits**.

## Validation set

The concept's dev samples, the harness default, as every generated-set curve in the paper and in `direction_count.py --stage fit` used them: `dev_samples/instructions` (`oig_omission` is in that directory, and the harness reads it for early stopping as it always has; it is excluded from the *eval* side, below), `dev_samples/hu_ha`, and `dev_samples/highstakes_500` for *high-stakes* (`--dev-data dev_samples/highstakes_500`, as `scripts/dc_chain_highstakes.sh` passes it). This is the mirror image of Part A, which trained on real samples and early-stopped on the DeepSeek-V4-Pro detailed set: in both runs the early-stopping set is from the other source and disjoint from the training arm and from every eval split. Never early-stop on an eval split or on the training arm. State the validation set in the commit message; never put a row validated on the full high-stakes dev set into a CSV that holds `highstakes_500` rows.

**Sensitivity check, cheap, *instruction* only:** refit the `kind` arms of llama70b, gptoss and nemotron (14 arms, 112 fits) early-stopped on the DeepSeek-V4-Pro detailed set instead (`--dev-data` the `dc_dev.val_dir("instructions")` directory, the exact file Part A used). DeepSeek's own arms cannot be checked this way, since its detailed set is their source. If `t_other` moves by less than the across-draw sd, the two rows of `dc_main` are comparable as they stand; if it moves by more, report both and say which validation set the table uses.

## Fits

The command `dc_dev.py --stage fit` builds, with the arm file swapped and the validation set above: `--no-base`, `--grad-accum ceil(n/16) --batch-size 16` (one optimiser step per epoch, rows tagged `none+ga<K>bs16`), `--sizes <n> --draws 8`. Add a `--stage gen-fit` to `dc_dev.py` (reuse `fit_jobs`/`stage_fit` with a manifest argument) or a small `scripts/dc_gen.py`; do not edit `direction_count.py`'s stages or the harness.

- `kind` arms: **unrestricted** eval through `dc_dev.py --stage harness` (the concept's eval dir minus `oig_omission`), so one fit scores every eval split of the concept and is one row of the transfer matrix. Out: `scripts/dc_gen_<concept>__<gen>_own.csv`.
- `loko` arms: restricted to the target split through `dc_run_curve.py --eval-split <split>`, as Part A's `others` arms were. Out: `scripts/dc_gen_<concept>__<gen>_others_<split>.csv`.

Order: *instruction* (all four generators), then *harmful*, then *high-stakes*; within a concept `kind` before `loko`; commit and push after each concept. The harness resumes on `(samples, n, draw)`, so give every (concept, generator) its own output file and never share one between two processes. Set `SYNTHETIC_PROBE_DATA_MAX_MEMORY` as the other run scripts do. Under *high-stakes* the 16 unrestricted `kind` arms are the cost (128 fits reading four blobs); run them last.

## Statistics (write `scripts/dc_gen_cells.csv`)

One row per (concept, generator, split), the columns of `scripts/dc_dev_cells.csv` with `gen` added, and `A_x(s)` the eight-draw mean eval AUROC on split `s`:

- `a_own = A_kind(s)`, `a_others = A_loko(s)`, `a_all = A_mixed(s)` at 590.
- `G_gen = (a_others - 0.5) / (a_all - 0.5)`.
- `t_other_gen`: mean over the concept's other kinds `k'` (those with a `kind` arm) of `A_kind(k')` scored on `s`, the off-diagonal column mean of the transfer matrix within one generator; `t_own_gen = a_own`. `mm_substitution` has a column (the other kinds' probes scored on it) and no row.
- `sd_*` across draws, `n_draws_*`.

Then `scripts/dc_gen_transfer_<concept>__<gen>.csv` (rows training kind plus `mixed`, columns eval split, draw means), and `scripts/dc_gen_summary.csv`: per split the median over generators of each statistic, per concept the median over splits, beside `t_other_dev`, `G_dev`, `a_own`, `a_others`, `a_all` from `dc_dev_cells.csv` and `t_other`, `e_other` from `dc_neff.csv`. Also the Spearman between `t_other_gen` (median over generators) and the per-split `log_m_detailed` target, computed as `dc_dev.py --stage link` computes it for `t_other_dev`, appended as new rows to a new `scripts/dc_gen_link_stats.csv` with the columns of `dc_dev_link_stats.csv`.

## Predictions, stated before looking

- `t_other_gen` ordered *instruction* < *harmful* < *high-stakes*, as the direction version (0.82 / 0.85 / 0.95) and the real-sample probe version (0.63 / 0.74 / 0.87) both are.
- `a_own` on generated kinds below the real `own` arms' 0.99-1.00: the direction study found the whole-set direction already beat any kind's direction on the eval splits (`e_all` 0.72 / 0.77 / 0.91 against `e_own` 0.71 / 0.75 / 0.91), and the kind-only ladders under *instruction* stop short of the mixed level.
- `G_gen` below 1 under *instruction* and near 1 under the other two, matching the ladder-based `G` of 0.67 / 0.90 / 0.95 within a few hundredths, since the same arms and levels are behind both.
- Ant-HH: the generated cell is the reverse of the real one (`R = 6.8, G = 0.22` on real data; near 1 on generated), so `a_others` should sit close to `a_all` there.
- The refusal asymmetry seen on real samples (a refusal-trained probe near chance on the other *instruction* splits) should be weaker or absent on generated kinds, where the refusal kind kept 11% of the gain under DeepSeek and went flat under GPT-OSS in the ladder study: say what the matrix shows.

If `t_other_gen` does not order the concepts, the direction statistic and the probe statistic disagree on the generated sets, and the paper's "transfer says the same without curves" sentence rests on the direction alone; say so.

## Outputs (commit and push on `dev_coverage`)

- `scripts/dc_gen_arms.csv`, the fit CSVs `scripts/dc_gen_<concept>__<gen>_own.csv` and `..._others_<split>.csv`, the sensitivity-check CSVs `scripts/dc_gen_instructions__<gen>_own_dsval.csv`, `scripts/dc_gen_cells.csv`, `scripts/dc_gen_transfer_<concept>__<gen>.csv`, `scripts/dc_gen_summary.csv`, `scripts/dc_gen_link_stats.csv`.
- The stage or script you wrote, and any wrapper `run*.sh` (`git add -f`).
- A section in `analysis/dev_coverage.md`, "The probe transfer statistic on generated samples": the method in a paragraph, the per-concept table with the generated-probe and real-probe numbers side by side (`a_own`, `a_others`, `a_all`, `G`, `t_other`), the per-split table (median over generators, with the generator range), the sensitivity check, the predictions and whether each held, wall-clock per fit per concept.
- Commit message: the per-concept table and one line on whether the ordering holds under the probe on generated samples.

## Do not

- Refit the mixed arm, or add rows to any `dc_curves_*.csv`, `dc_dev_*.csv`, `dc_arms.csv`, `dc_neff.csv` or other existing CSV.
- Edit `subsample_curve_concept.py`, `dc_run_curve.py`, `direction_count.py`, `knee_predictor.py` or `fit_base_plus_concept.py`.
- Early-stop on an eval split or on the training arm; mix `highstakes_500`-validated and full-dev-validated rows in one file.
- Touch `eval_sets/` or `data/kind_tags/`, or include `oig_omission` on the eval side anywhere.
