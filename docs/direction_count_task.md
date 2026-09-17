# Task: does the number of directions the generated data must cover set the half-gain size?

## Setup

Repo: `https://github.com/acheckervarty7890/probe_auto_improvement` (private; you have access). Clone it and check out branch `direction_count` (already exists, cut from `knee_predictor`, so every input of that study is here too). Bootstrap with `bash scripts/setup_env.sh` from inside the checkout: it creates `.venv_claude`, clones the private `acheckervarty7890/tuberlens` at branch `iterative_pipeline_2` and installs it editable, then installs `requirements.txt` and this repo. Run every Python command as `.venv_claude/bin/python`. Read `CLAUDE.md` before writing code. Commit as you go on `direction_count` and push; this repo keeps results in commit messages and in `analysis/*.md`, so the final commit message must contain the full result summary.

**One stage needs a GPU** (stage 1: Gemma-3-27B-IT layer-32 activations of twelve 600-sample generated sets, about 7,200 conversations; a single 24 GB card with the repo's default `AGENTIC_REDTEAM_MAX_MEMORY` is enough). Every other stage is probe-head fits and numpy on cached activations. Stage 0 needs an LLM API for tagging.

## Why

The paper fits every synthetic-data learning curve with a log-logistic A(n) = L + (U−L)/(1+(n/m)^−k); m is the *half-gain size*. Concept identity explains 34% of the variance of log m and the evaluation split beyond the concept 8%; the generator, probe model and prompt explain a few percent. The `knee_predictor` study (`analysis/knee_predictor.md`) showed that no property of the evaluation split's own activations predicts m: the split easiest to separate in-distribution (Harmless-refusal, AUROC 1.000 on one principal component) has one of the largest half-gain sizes. So whatever sets m is not in the evaluation activations alone.

The remaining hypothesis is about the *generated* data. The detailed prompt writes each batch for one of the concept's **kinds**, one kind per evaluation split, in round-robin (see `eval_kinds()` and `focus_kind` in `scripts/generate_instructions_dataset.py`). If the kinds of *high-stakes* share one direction in activation space, every generated sample serves every split and the curve saturates early. If the six kinds of *instruction* are separate directions, n mixed samples give one split only about n/6 useful ones, so its half-gain size on the mixed set should be several times its half-gain size on samples of its own kind alone. The probe is a single direction by construction (one linear map per token, softmax-pooled), so the question is not the rank of the probe but how many distinct directions the training data has to cover before the one direction the probe learns serves the split.

Predictions, in the order they are tested:
1. The **ratio** R_s = m(mixed set) / m(kind-only set) on split s is near 1 for the *high-stakes* and *harmful* splits and well above 1 (up to about 6) for the *instruction* splits.
2. R_s tracks the **effective number of directions** among the concept's kinds in the generated set (a cosine statistic, no training).
3. Removing split s's kind from the set (**leave-one-kind-out**) barely moves the curve on s for *high-stakes* and *harmful* and flattens it for *instruction*.
4. Either R_s or the kind-direction alignment predicts the paper's per-split m better than concept identity alone (the baseline `knee_predictor.md` used).

If kind-only and mixed curves have the same half-gain size, the data need is not about covering kinds but about within-kind difficulty, and the hypothesis is rejected. That is a publishable null next to the `knee_predictor` null; do not soften it.

## Inputs (all on this branch)

1. **Targets: `scripts/knee_fits.csv`** (columns `model,concept,split,recipe,variant,gen,sizes,L,U,U_lo,U_hi,m,m_lo,m_hi,k,n90,n90_lo,n90_hi,gain_obs,rmse,U_bvar,lm_bvar`). Use `model == gemma27b`, `recipe == detailed`; the per-split target is the median log10 m over the four generators, flat curves dropped, m ≤ 10 tied at 10. **`scripts/fit_curves_ref.py`** is the reference fitter (grid least squares over draws, 200-replicate bootstrap, flat = fitted in-range gain < 0.02). Port it; do not write another.

2. **Detailed-prompt generated sets** (300/300, fields `inputs` as a JSON string of messages and `labels`; **no kind field**):
   `data/highstakes_<gen>_evaldesc_600.jsonl`, `data/instructions_<gen>_evaldesc_600.jsonl`, `data/hu_harm_<gen>_evaldescshape_600.jsonl` for `<gen>` in `llama70b gptoss nemotron deepseekv4pro`; the generators' own 50-sample general-prompt bases are `data/<concept>_<gen>_50.jsonl`. These are byte-identical to the sets behind the paper's detailed curves (`origin/per_split_studies`, `scripts/<concept>_gen90*.csv` and `scripts/<concept>_pooled_size_curve.csv`, which you may read for cross-checks but must not refit).

3. **Kind descriptions.** `EVAL_DATA_DESCRIPTION` in `scripts/generate_instructions_dataset.py` (six numbered kinds) and in `scripts/generate_highstakes_dataset.py` (four kinds); `eval_kinds()` splits the numbered items. The *harmful* kinds (the `evaldescshape` variant, which adds the measured turn lengths) are not a constant in `scripts/generate_hu_harm_dataset.py`; the four kinds are:
   1. AN AI DECIDING IN AN INSTITUTIONAL ROLE (the assistant is cast as a system with a call to make and states it in one line; the decision protects the people it lands on or trades their well-being away).
   2. AN EVERYDAY REQUEST ANSWERED STRAIGHT (a chatty exchange in which the assistant declines nothing; the request is harmless or damaging).
   3. A REQUEST THAT SHOULD NOT SIMPLY BE CARRIED OUT (the reply declines in one short line or supplies the procedure).
   4. AN INTERPERSONAL DILEMMA (a first-person account with two courses open and one line of advice).
   The exact shaped paragraph (with the measured turn lengths) is `EVAL_DATA_DESCRIPTION` in `git show origin/generator_experiment_1:scripts/generate_hu_harm_dataset.py`, and also in `git show origin/human_harm_last:analysis/refit_studies/descriptions.json`; quote it verbatim in your tagging prompt.

   **Kind → evaluation split.** Verify each row against ten samples of the split before relying on it, and record the verification in the analysis note:

   | concept | kind | split stem |
   |---|---|---|
   | instructions | 1 unjustified refusal | `anthropic_harmless_refusal` |
   | instructions | 2 biased substitution | `bbq_substitution` |
   | instructions | 3 context drift, retrieved-document | `hc_context_drift` |
   | instructions | 4 contradicting the source | `hc_contradiction` |
   | instructions | 5 exaggeration / embellishment | `mm_substitution` |
   | instructions | 6 context drift, dialogue | `oig_context_drift` |
   | highstakes | 1 open-ended chatbot conversation | `anthropic_hh_balanced` |
   | highstakes | 2 clinical document | `mt_balanced` |
   | highstakes | 3 doctor-patient dialogue | `mts_balanced` |
   | highstakes | 4 tool-calling agent | `toolace_balanced` |
   | hu_harm | 1 AI in an institutional role | `eval_ai_dilemmas` |
   | hu_harm | 2 everyday request answered straight | `eval_ant_hh` |
   | hu_harm | 3 request that should not be carried out | `eval_balanced_refusal` |
   | hu_harm | 4 interpersonal dilemma | `eval_daily_dilemmas` |

4. **Evaluation splits and dev sets.** `eval_sets/{highstakes,hu_ha,instructions}/*.jsonl` and `dev_samples/{highstakes,hu_ha,instructions}/`; **exclude `oig_omission` everywhere** (14 splits). `scripts/fit_base_plus_concept.py::CONCEPTS` holds the paths and positive labels (`high-stakes`, `harmful_to_human`, `assistant_follows_the_instruction`).

5. **Evaluation activations** (Gemma-3-27B-IT layer 32, Kaggle owner `anku7890`, inventory `eval_activations_manifest.json`, concepts `highstakes_eval`, `hu_harm_eval`, `instructions_eval`). The fetcher `scripts/fetch_kaggle_eval_activations.py` that `scripts/knee_predictor.py --stage pool` calls is **not tracked**; rebuild it from `src/agentic_redteam/kaggle_activations.py` exactly as the previous agent did (auth via `KAGGLE_CONFIG_DIR` or `KAGGLE_API_TOKEN`; blobs at `<cache-dir>/<split>-acts_full.pt`; never point an eval and a dev concept at one cache dir; do not fetch `_dev` concepts). Then `--stage pool` gives `scripts/knee_pooled/<split>_{mean,last}.npy` plus labels; mean-pooled is what stage 2 uses.

6. **The fitting harness** (verbatim from `origin/per_split_studies`, do not edit):
   - `scripts/subsample_curve_concept.py --concept C <set.jsonl> --sizes ... --draws 8 --no-base --grad-accum K --batch-size 16 --out <csv>`: class-balanced draws seeded on (file stem, n, draw), early stopping on the concept's dev set, one CSV row per fit with `dev_mean`, `eval_mean` and every per-split column `eval_<stem>` (for `hu_ha` the stems already carry `eval_`; see `split_column`). Resumes on `(samples, n, draw)`.
   - **Optimiser regime:** accumulation = ceil(n/16) with batch 16, so every size takes exactly one optimiser step per epoch (`run_pooled_sizecurve.sh` explains why; at the default accumulation of 4 any set under 49 samples takes zero steps and returns the untrained probe). Use that rule at every size in this study, including the large ones, so the three arms share one regime; tag it in the CSV as the harness does.
   - `scripts/warm_set_activations.py --concept C <set.jsonl>` extracts a set's activations into the per-sample cache (one 27B load per call); `scripts/warm_pooled_sets.py --concept C` does all of a concept's pools in one load after `scripts/build_pooled_sets.py --concept C`. Once warmed, every fit is a probe-head fit on cached activations.

## Stage 0: tag every generated sample with its kind

Two independent LLM passes per sample (the same model with the kinds listed in two different orders, temperature 0), each given the concept's numbered kinds verbatim and the conversation, returning the kind index or `none`. Keep a sample when both passes agree; report the agreement rate, the per-kind counts per set (round-robin predicts about 100 per kind for *instruction* and 150 for the other two), and the `none` rate. Hand-read 60 tagged samples per concept (ten per kind, random) and record your agreement with the tagger in the analysis note. Output `data/kind_tags/<set stem>.csv` with columns `row,kind,split,pass1,pass2,agree`. A rule-based tagger may be used as a cross-check only. Samples the two passes disagree on are dropped from the kind-only and leave-one-kind-out arms but kept in the mixed arm.

## Stage 1: activations of the generated sets (GPU)

`scripts/build_pooled_sets.py --concept C` then `scripts/warm_pooled_sets.py --concept C` for the three concepts (this covers the twelve detailed sets and the twelve 50-sample bases in three model loads). While the blobs are in memory, also write mean-over-tokens pooled features per sample, `scripts/dc_pooled/<set stem>_mean.npy` (float32) with `_labels.npy` and `_ntokens.npy`, using the same pooling and message transforms as `scripts/knee_predictor.py::pool_split` (`combine_consecutive_messages=True, convert_tool_to_assistant=True`), so generated and evaluation features are comparable. Never hold two sets' full activations at once; make the stage resumable; commit the `.npy` files if under 200 MB total, otherwise a manifest with shapes and hashes.

## Stage 2: direction geometry (CPU, no training)

Per set (concept × generator), on standardised mean-pooled features (standardiser fit on the set):
- **Per-kind direction** d_k = mean(positive, kind k) − mean(negative, kind k), unit-normalised, for every kind with at least 20 samples per class.
- **Cosine matrix** between kinds and the **effective number of directions** n_eff = (Σλ)² / Σλ², λ the eigenvalues of the Gram matrix of the unit d_k. Also the full-set direction d_all and its cosine with each d_k.
- **Transfer matrix** T[i, j]: AUROC on kind j's samples of the projection onto d_i, with d_i fit on the other four folds of a 5-fold split when i = j; and the same against the evaluation splits, E[i, s]: AUROC on evaluation split s (from `scripts/knee_pooled`) of the projection onto d_i.
- **Per-split alignment** a_s = cos(d_kind(s), d_all), and the share of E[all, s] − 0.5 that E[kind(s), s] − 0.5 recovers.
Report per concept: n_eff, mean off-diagonal cosine, the matrices, per generator and averaged. Prediction: n_eff near 1 for *high-stakes* and *harmful*, above 2 for *instruction*.

## Stage 3: the causal test (probe fits on cached activations)

For each concept, generator and evaluation split s, three arms drawn from the **same** detailed set, materialised as JSONLs under `.dc_work/` so the harness's resume keys stay distinct:
- (a) **kind-only**: `dc_<concept>_<gen>_<split>_kind.jsonl`, the samples tagged kind(s);
- (b) **mixed**: `dc_<concept>_<gen>_mixed.jsonl`, the whole set (one file per set, scored on every split at once);
- (c) **leave-one-kind-out**: `dc_<concept>_<gen>_<split>_loko.jsonl`, the set minus kind(s).

Sizes: 2, 4, 6, 10, 20, 30, 50, 80 for (a) (capped at the kind's balanced count; skip a kind with fewer than 60 tagged samples and say so), and 2, 4, 6, 10, 20, 30, 50, 80, 110, 170, 350, 590 for (b) and (c) (capped at the arm's count). Eight draws per size, `--no-base`, accumulation ceil(n/16) with batch 16 at every size. Sizes below 10 exist to uncensor the *high-stakes* and *harmful* half-gain sizes, which the paper's curves left at "≤ 10". Output `scripts/dc_curves_<concept>.csv` per concept via `--out`.

**Order of work:** *instruction* × `deepseekv4pro` first (six splits × three arms; about 1,400 fits) and push its result before anything else, because it answers the question on its own. Then the other three generators on *instruction*, then *high-stakes* and *harmful* on all four.

Fit every (concept, generator, split, arm) curve with the ported fitter, target column `eval_<s>`: m_a, m_b, m_c with bootstrap intervals and the flat rule, into `scripts/dc_fits.csv` (the `knee_fits.csv` schema plus `arm,target_split,kind_n`).

## Analysis

1. **Ratio.** R_s = m_b / m_a per split and generator, with a bootstrap interval that resamples draws in both arms. Table per concept: median R and range over splits and generators; state censoring (any m at the smallest size). Compare R with n_eff from stage 2: Spearman over the 14 splits × 4 generators, and over the six *instruction* splits alone.
2. **Leave-one-kind-out.** G_s = (U_c − L_c) / (U_b − L_b) and m_c / m_b per split. Shared directions predict G_s near 1 and m_c near m_b; separate directions predict a flat or much slower curve (c).
3. **Coverage versus per-kind difficulty.** Compare m_a across concepts. If the *instruction* kind-only half-gain sizes fall to the *high-stakes* / *harmful* level, the concept effect in the paper is coverage; if they stay higher, part of it is per-kind difficulty. Report the fraction of log m_b − log m_a(other concepts) that coverage explains.
4. **Link to the paper's m.** Spearman ρ of R_s and of a_s with the split's median log10 m from `knee_fits.csv`, permutation p, bootstrap CI over the curves behind each median; leave-one-split-out RMSE from the predictor alone and with concept identity against the concept-only baseline (reuse the code and baselines in `scripts/knee_predictor.py --stage analyse`); the within-*instruction* rank correlation on six splits.
5. **Verdict.** The hypothesis is supported if (i) the *instruction* median R is above 2 and the *high-stakes* and *harmful* medians below 1.5 with non-overlapping intervals, and (ii) R tracks n_eff with ρ ≥ 0.6 at p < 0.05 or a_s beats the concept-only baseline. Report which of (i) and (ii) hold; if neither, the null is the result.

## Outputs (commit and push all of them)

- `scripts/direction_count.py` (`--stage tag|warm|geometry|arms|fit|analyse|all`, resumable at every stage).
- `data/kind_tags/*.csv`; `scripts/dc_pooled/` (or its manifest); `scripts/dc_geometry.csv` (one row per set per kind pair: cosine, T, E) and `scripts/dc_neff.csv` (one row per set); `scripts/dc_curves_<concept>.csv`; `scripts/dc_fits.csv`; `scripts/dc_ratios.csv` (`concept,gen,split,m_a,m_a_lo,m_a_hi,m_b,m_b_lo,m_b_hi,m_c,m_c_lo,m_c_hi,R,R_lo,R_hi,G,n_eff,a_s,kind_n`); `scripts/dc_scatter.csv` (`split,concept,R,n_eff,a_s,log_m,log_m_lo,log_m_hi`).
- `analysis/direction_count.md`: method in one paragraph, the tagging audit, the geometry table, the ratio table, the leave-one-kind-out table, the coverage-versus-difficulty split, the link to m with baselines, a "what this supports / does not support" section, and caveats (tagger noise; mean-pooled directions are not the softmax-pooled head; kind-only sets are 100 to 150 samples so their curves stop at 80; sizes below 10 use one optimiser step per epoch).
- Final commit message = the ratio table, the geometry table, and the verdict.

## Guardrails

- Do not edit the harness (`subsample_curve_concept.py`, `fit_base_plus_concept.py`, the warm scripts); wrap it. Do not refit the paper's existing curves. Do not touch the `knee_predictor` outputs.
- Never subsample an evaluation split; never point dev and eval concepts at one cache; never load the LLM except in stage 1 (activations) and stage 0 (tagging via API).
- If the GPU stage is impossible, finish stage 0, commit the tags and the tagging audit, and stop with a report of what is missing.
- Skip sizes above a kind's balanced count rather than drawing with replacement, and list every skipped cell.
