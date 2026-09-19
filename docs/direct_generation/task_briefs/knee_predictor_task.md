# Task: build a pre-generation predictor of the learning-curve knee, per evaluation split

## Setup

Repo: `https://github.com/acheckervarty7890/probe_auto_improvement` (private; you have access). Clone it and check out branch `knee_predictor` (already exists, cut from `main`). Bootstrap the environment with `bash scripts/setup_env.sh` from inside the checkout: it creates `.venv_claude`, clones the private `acheckervarty7890/tuberlens` at branch `iterative_pipeline_2` and installs it editable, then installs `requirements.txt` and this repo. Run every Python command as `.venv_claude/bin/python`. No GPU is required; torch is only used to read activation blobs. Read `CLAUDE.md` before writing code. Commit as you go on `knee_predictor` and push; this repo keeps results in commit messages and in `analysis/*.md`, so the final commit message must contain the full result summary.

## Why

The paper this feeds fits every synthetic-data learning curve with a log-logistic A(n) = L + (U-L)/(1+(n/m)^-k). The half-gain size m is the "knee". Its finding is that the evaluation split explains 42–45% of the variance of log m while the generator, probe model, and prompt explain a few percent. What it lacks is a way to know the knee *before generating any data*. Your job: compute cheap properties of each evaluation split from its cached Gemma-3-27B-IT layer-32 activations and test which of them predicts the split's knee. No synthetic data, no LLM calls, no model load.

## Inputs (all in the repo)

1. **Targets: `scripts/knee_fits.csv`.** Columns: `model,concept,split,recipe,variant,gen,sizes,L,U,U_lo,U_hi,m,m_lo,m_hi,k,n90,n90_lo,n90_hi,gain_obs,rmse,U_bvar,lm_bvar`. Use `model == gemma27b` only (140 rows). `recipe` is one of `detailed`, `general`, `llm`, `targeted`; `gen` is the generator. `split` values equal the eval JSONL stems below. A curve is **flat** if its fitted in-range gain is below 0.02; `scripts/fit_curves_ref.py` is the reference fitter and holds the exact rule. Reproduce that rule, do not invent another. An `m` of 10 or below means "at or below the smallest measured size", i.e. left-censored: use rank statistics and treat those as ties at 10.
   - Primary target per split: `log10 m` median over the four **detailed**-prompt curves (non-flat only).
   - Secondary targets: the same for `general`, for the five `llm` curves (six splits only), for `targeted`; and the `n90` medians.
   - Keep the curve-level table (one row per non-flat Gemma curve) for the curve-level check below.

2. **Eval splits and labels.** `eval_sets/highstakes/{anthropic_hh_balanced,mt_balanced,mts_balanced,toolace_balanced}.jsonl`, `eval_sets/hu_ha/{eval_ai_dilemmas,eval_ant_hh,eval_balanced_refusal,eval_daily_dilemmas}.jsonl`, `eval_sets/instructions/{anthropic_harmless_refusal,bbq_substitution,hc_context_drift,hc_contradiction,mm_substitution,oig_context_drift}.jsonl`. **Exclude `oig_omission` everywhere.** 14 splits. Each row has `inputs` (a JSON string of `[{role, content}, ...]`) and `labels`. Positive labels are `high-stakes`, `harmful_to_human`, `assistant_follows_the_instruction`; confirm against a probe pickle's `pos_class_label` under `probes/` rather than trusting this list.

3. **Cached activations (Kaggle, owner `anku7890`).** Fetch with
   `.venv_claude/bin/python scripts/fetch_kaggle_eval_activations.py --concept {highstakes|hu_harm|instructions} --cache-dir eval_activations_<concept>`
   (auth via `KAGGLE_CONFIG_DIR` pointing at the directory holding `kaggle.json`, or `KAGGLE_API_TOKEN`; the inventory is `eval_activations_manifest.json`, concepts `highstakes_eval`, `hu_harm_eval`, `instructions_eval`). Blobs land at `<cache-dir>/<split>-acts_full.pt`: Gemma-3-27B-IT layer 32, computed with `combine_consecutive_messages=True, convert_tool_to_assistant=True`. Never point an eval and a dev concept at the same cache dir; do not fetch the `_dev` concepts at all.
   - To attach a blob to a dataset, reuse `_load_split` from `git show origin/devsamples_kfold_cloud:scripts/eval_kfold_cv.py` (it validates the blob header against model, layer, and row count; keep that validation). Per-sample activations arrive as `dataset.other_fields["activations"]`, a list of `[tokens, hidden]` tensors in file order; `git show origin/hello_kitty:scripts/settling_text_features.py` (function `geometry`) shows the mean-pooling pattern.
   - Memory: `anthropic_hh_balanced` alone is about 30 GB (2984 rows × 1024 × 5376 × bf16). Load one split at a time, immediately reduce each sample to (a) the mean over tokens and (b) the last token, save `scripts/knee_pooled/<split>_{mean,last}.npy` (float32) plus `<split>_labels.npy` and `<split>_ntokens.npy`, free the blob, and never hold two splits at once. Everything downstream runs on these small matrices. Make this stage resumable (skip a split whose .npy files exist). Commit the `.npy` files only if the total is under 200 MB; otherwise commit a manifest with shapes and hashes.

## Predictors per split

All from the pooled matrices; 5-fold stratified CV wherever a fit is involved, fixed seed, mean over folds. Do each for mean-pooled and last-token features; mean-pooled is primary.

A. **In-distribution few-shot curve.** sklearn `LogisticRegression` (standardised features, C=1; also C=0.1) on k ∈ {2, 4, 8, 16, 32, 64, 128, 256} class-balanced samples drawn from the training folds, 32 draws per k, AUROC on the held-out fold. Fit the same log-logistic to this curve by porting the fitter from `scripts/fit_curves_ref.py`, not a re-implementation → `m_ID`, `n90_ID`, `U_ID`. Record `AUROC_ID(k)` at every k. Hypothesis: the synthetic knee tracks the in-distribution knee.

B. **One-direction share.** Under CV: AUROC of the difference-of-means direction (fit on training folds, project held-out) vs AUROC of full logistic regression. `r1 = (AUROC_dom − 0.5) / (AUROC_full − 0.5)`. Also the difference of means from 2 samples per class, averaged over 200 draws (`AUROC_dom2`).

C. **Direction count.** PCA on the training folds' features (centred). For d ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, all}: logistic regression on the top-d PCs, held-out AUROC. `d95` = smallest d whose AUROC reaches 95% of the all-dimension gain over 0.5; also `d90`. Keep the full curve per split.

D. **Geometry.** Fisher ratio along the difference-of-means direction; cosine between class centroids; participation ratio of the within-class covariance; mean pairwise cosine within vs between classes; and for class-paired splits (three `hu_ha` splits and all six `instructions` splits share the prefix within a pair; recover pairs with `load_split` from `git show origin/hello_kitty:scripts/analyze_row_settling.py`) the norm of the within-pair difference relative to the between-pair spread.

E. **Controls that need no activations.** Split size, mean token count, class-paired or not, mean number of turns, concept identity.

F. **The practitioner's cheap probe** (report separately, since it uses synthetic data): the 50-sample base probe's per-split AUROC, read from the per-split columns of `scripts/*_gen90*.csv` and `scripts/*_pooled_size_curve.csv` on `origin/per_split_studies`. Use it only if it is already there; do not fit anything to get it.

## Analysis

1. Spearman ρ between every predictor and the primary target over the 14 splits, with a permutation p-value and a 95% bootstrap CI that resamples the *curves* behind each split's median (so censoring at 10 and draw noise propagate). Same for the secondary targets. One table, predictors as rows.
2. **Baseline to beat:** concept identity alone (predict a split's log m from the mean of the other splits of its concept, leave-one-split-out). For each predictor, leave-one-split-out linear prediction of log m from the predictor alone and from predictor + concept; report LOO RMSE in log10 units against the concept-only baseline and the grand-mean baseline.
3. **Within-instruction check** (6 splits): rank correlation of each predictor with the target inside the concept, since that is where the knee spans more than an order of magnitude and concept identity cannot help.
4. **Curve level** (about 120 non-flat Gemma curves): linear model `log m ~ predictor + generator + prompt`, the R² gain over `generator + prompt`, compared with `split` as a factor (the ceiling, 42–45% in the paper).
5. State plainly which predictor, if any, meets all of: ρ ≥ 0.6 with p < 0.05 on 14 splits, beats the concept-only LOO baseline, and keeps its sign within instruction. If none does, the null is the result; do not soften it.

## Outputs (commit and push all of them)

- `scripts/knee_predictor.py` (feature extraction + fits, resumable, `--stage pool|features|analyse`).
- `scripts/knee_predictors.csv` (one row per split, every predictor, every target), `scripts/knee_predictor_curves.csv` (the A and C curves, long format), `scripts/knee_predictor_stats.csv` (the tables of §1–2).
- `scripts/knee_predictor_scatter.csv` with columns `split,concept,predictor_value,log_m,log_m_lo,log_m_hi` for the best predictor, ready to plot.
- `analysis/knee_predictor.md`: method in one paragraph, the two tables, the within-instruction table, the curve-level R², a "what this supports / does not support" section, and caveats (mean-pooled logistic regression is not the paper's softmax-pooled tuberlens head; 14 points; censoring at 10 for most high-stakes and harmful curves).
- Final commit message = the summary table and the verdict.

## Guardrails

- Never load Gemma or any LLM. If a blob is missing, fetch it; if the fetch fails, stop and report which split is missing.
- Keep file order as the join key everywhere (blob, labels, pairs), as the settling scripts do.
- Fix seeds and state them. Every CV number is a mean over folds; every few-shot number a mean over draws with its sd.
- Report the numbers you got, including nulls and anything that contradicts the paper's claim.
- Do not modify any existing script, CSV, or analysis file on other branches; everything new lives on `knee_predictor`.
