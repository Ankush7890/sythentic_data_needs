# Task: the difference-of-means transfer statistic on dev samples, scored on dev samples and on the eval splits

## Setup

Same repo, same branch `dev_coverage` (continue from the tip, 19408c4e or later; `scripts/dc_dev.py` and `analysis/dev_coverage.md` are the state of play). Same environment (`.venv_claude/bin/python`), same rules (`CLAUDE.md`). Commit as you go on `dev_coverage` and push. No probe is trained anywhere in this task and no curve is fit. The only GPU-side work is `knee_predictor.py --stage pool` if `scripts/knee_pooled/` is missing on your machine (it is untracked); it is resumable and skips splits already pooled.

## Why

The paper's Table `dc_main` puts two transfer AUROCs side by side, one per row, and they are not the same statistic:

- **Generated row** (`t_other` in `scripts/dc_neff.csv`, from `direction_count.py --stage geometry`): a unit difference-of-means direction per LLM-tagged kind, on the generated set's own standardised mean-pooled activations, scored as a linear projection on the *other kinds' generated samples*. No training. Concept means 0.82 / 0.85 / 0.95 (*instruction* / *harmful* / *high-stakes*).
- **Real row** (`t_other_dev` in `scripts/dc_dev_cells.csv`, from `dc_dev.py`): a *trained probe* (Adam, early-stopped on the DeepSeek-V4-Pro detailed set) on one split's dev samples, scored on the concept's *other eval splits*. Medians 0.63 / 0.74 / 0.87.

Three things differ at once: the classifier (class-mean direction vs trained probe), the kind label (tagger vs source split), and the scoring set (other kinds' generated samples vs other eval splits). The ordering agrees but the levels are not comparable. This task computes the generated-row statistic, unchanged, on the dev samples, so that the real row can be reported with the same classifier, and it scores those directions on both the other splits' dev samples (the exact analogue of `t_other`) and on the eval splits (the analogue of `e_other`, which `dc_neff.csv` already carries for the generated sets: 0.60 / 0.63 / 0.75).

## What to compute

Reproduce `geometry_for_set` in `scripts/direction_count.py` with the dev split in the role of the kind. Import its helpers (`_standardiser`, `_direction`, `_auroc`, `_load_eval_features`, `MIN_KIND_PER_CLASS`, `N_GEOM_FOLDS`, `SEED`) rather than copying them; add a `--stage geometry` to `scripts/dc_dev.py` or write `scripts/dc_dev_geometry.py`. Do not edit `direction_count.py`.

**Features.** The token-mean of Gemma-3-27B-IT layer 32 over the attention-mask positions, float32, exactly what `direction_count.pool_set` computes from the per-sample cache. Every dev sample is already in that cache from `dc_dev.py --stage warm` (2,634 conversations; if any is missing, run that stage, it is one model load per concept). Pool from the `own` arm files (`.dc_work/dcdev_<concept>_<split>_own.jsonl`, rebuilt by `dc_dev.py --stage arms` if absent), not from `dev_samples/`, because the arm files carry the assistant-first fix and the cache key is computed over the fixed rows. Write `scripts/dc_pooled/dcdev_<concept>_<split>_own_{mean,labels,ntokens}.npy`, the layout `pool_set` uses. The eval splits' features are `scripts/knee_pooled/<eval_stem>_mean.npy` through `_load_eval_features`, as in the generated study. `oig_omission` is excluded from everything, as everywhere in the paper.

**Per concept:**

1. Stack the concept's dev splits into one matrix `X` (404 / 290 / 1,908 rows), fit the standardiser on it (`mu, sd = _standardiser(X)`, the analogue of "the set's own standardised features"), `Z = (X - mu) / sd`.
2. One unit difference-of-means direction per split `d_s = _direction(Z[split == s], y[split == s])`, and the whole-dev-set direction `d_all = _direction(Z, y)`. Apply `MIN_KIND_PER_CLASS = 20` as the generated study does; every dev split clears it (the smallest is Ant-HH at 22 / 22), so no split is dropped. Record `n_pos`, `n_neg` per split.
3. **T on dev samples** (the analogue of `T_ij`): `T[s, s']` = AUROC of `Z[split == s'] @ d_s` against `y[split == s']`. The diagonal is the five-fold held-out value with the generated study's `_held_out_auroc` logic and `SEED`, so it is comparable with the off-diagonal. `t_own_dev_dom` = mean of the diagonal, `t_other_dev_dom` = mean of the off-diagonal, per concept; per split, the off-diagonal *column* mean (other splits' directions scored on `s`), which is what `t_other_dev` in `dc_dev_cells.csv` is for the probe.
4. **E on eval splits** (the analogue of `E_is`): push every eval split of the concept through the *same* `mu, sd`, then `E[s, e]` = AUROC of `Ze @ d_s` on eval split `e`, plus the `d_all` row. `e_own`, `e_other`, `e_all` per concept as in `dc_neff.csv`; per split, `e_gap_<split>` and `t_gap_<split>` with the generated study's definitions.
5. **Set-minus-split direction** (cheap, and the fit-free analogue of `G`): `d_-s = _direction(Z[split != s], y[split != s])`, scored on `s`'s dev samples and on `s`'s eval split. Report `G_dom = (AUROC(d_-s) - 0.5) / (AUROC(d_all) - 0.5)` on both, next to `G_dev` from `dc_dev_cells.csv`.
6. `n_eff`, `mean_off_cos`, `max_off_cos` of the Gram matrix of the split directions, for completeness (expected 3-4 everywhere, as on the generated sets; it is not the point of this task).

**Size check.** The generated kinds hold 43-211 tagged samples and the dev splits 44-1,028, and a class-mean over more samples is a less noisy direction. Re-run steps 2-4 with every dev split subsampled, class-balanced, to the smaller of its balanced size and 100, over 20 draws seeded off `SEED`, and report the draw mean and sd beside the full-size value. If the full-size and the size-matched `t_other_dev_dom` differ by more than the across-draw sd, the size-matched one is the comparable number; say which one the tables use.

## Predictions, stated before looking

- `t_other_dev_dom` ordered *instruction* < *harmful* < *high-stakes*, as both existing versions are (0.82 / 0.85 / 0.95 generated; 0.63 / 0.74 / 0.87 probe on real).
- `e_other` on real directions above the generated directions' 0.60 / 0.63 / 0.75, since a dev split is in-distribution to its eval split; `e_own` well above 0.71 / 0.75 / 0.91 for the same reason.
- Harmless-refusal: its dev direction near chance or reversed on the other *instruction* splits, and BBQ's direction reversed on refusal (the probe transfer matrix `scripts/dc_dev_transfer_instructions.csv` has 0.39-0.59 and 0.25). If the class-mean direction shows the same asymmetry it is a property of the samples, not of the optimiser.
- `G_dom` well below 1 under *instruction*, near 1 under the other two.

If instead the real-sample directions transfer alike across concepts, say so plainly: it would mean the generated-set `t_other` ordering is about the tagged kinds and the probe-based `t_other_dev` ordering is about training, and the paper's "transfer says the same without curves" sentence has to be qualified. Either outcome is a result.

## Outputs (commit and push on `dev_coverage`)

- `scripts/dc_dev_geometry.csv` with the columns of `scripts/dc_geometry.csv` (`gen = dev`, `set = dcdev_<concept>`, `kind_i`/`kind_j` the split stems, `T_ij` for dev-on-dev, `E_is` for dev-on-eval, plus rows for `kind_i = minus_<split>` from step 5).
- `scripts/dc_dev_neff.csv` with the columns of `scripts/dc_neff.csv` (`gen = dev`), one row per concept at full size and one per concept size-matched (`gen = dev_n100`).
- `scripts/dc_dev_dom_transfer_<concept>_dev.csv` and `..._eval.csv`: rows training split (plus `all` and `minus_<split>`), columns scored split, the format of `scripts/dc_dev_transfer_<concept>.csv`.
- `scripts/dc_dev_dom_cells.csv`: per split, `t_other_dom`, `t_own_dom`, `e_own_dom`, `e_other_dom`, `G_dom_dev`, `G_dom_eval`, their size-matched versions with sd, beside `t_other_dev` and `G_dev` copied from `dc_dev_cells.csv` and `t_other` and `e_gap` from `dc_link_stats.csv` for the same split.
- A section in `analysis/dev_coverage.md`, "The difference-of-means statistic on dev samples": the method in a paragraph, one table with the four transfer numbers per concept in one place (generated direction on generated samples; generated direction on eval; dev direction on dev; dev direction on eval; probe on eval), the per-split table, the predictions and whether each held, and the size check.
- Commit message: the per-concept table and one line on whether the ordering and the refusal asymmetry hold under the class-mean direction.

## Do not

- Train a probe, fit a curve, or touch `dc_dev_cells.csv`, `dc_dev_transfer_*.csv`, `dc_neff.csv`, `dc_geometry.csv` or any other existing CSV.
- Edit `direction_count.py`, `knee_predictor.py`, `subsample_curve_concept.py`, `dc_run_curve.py` or `fit_base_plus_concept.py`.
- Standardise the eval splits on their own statistics; they go through the dev set's `mu, sd`, as the generated study pushes them through the generated set's.
- Touch `eval_sets/`, or include `oig_omission` anywhere.
