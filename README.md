# How Much Synthetic Data Does an Activation Probe Need?

Code and data for the paper *How Much Synthetic Data Does an Activation Probe Need?
Learning Curves for Three Concepts on Fourteen Distributions*.

The paper trains linear activation probes on Gemma-3-27B-IT (layer 32; also Qwen3-8B,
Mistral-NeMo-12B and Llama-3.2-1B) for three monitoring concepts — *high-stakes*,
*harmful*, *instruction* — using **only synthetic training samples written by an LLM**, and
measures learning curves on fourteen held-out evaluation distributions. Training sets come
from two recipes:

1. **Direct prompt generation** (`direct_generation/`) — a generator LLM writes labelled
   samples from a prompt (a one-line concept prompt, a paragraph describing the traffic, or
   an LLM-written per-split prompt); a probe is fit on *n* of them and scored per split.
   This is the recipe behind every learning curve in the paper.
2. **Red-teaming scaffold** (`redteam_scaffold/`) — an attacker LLM searches for samples
   the current probe gets wrong, a judge labels them, the probe is retrained, repeat. The
   project started here; the paper reports it as one arm.

`main` holds code and data only. Every result (learning-curve CSVs, red-team arm outputs,
loop runs, per-iteration probes), the wrappers each study was run with, and the study
write-ups are on the **`results`** branch, which is `main` plus `results/` and `docs/`
(see *Results branch* below).

## Layout

```
eval_sets/<concept>/     the 14 evaluation splits (+ oig_omission, excluded from the paper)
dev_samples/<concept>/   dev sets: early stopping and per-batch ΔAUROC
data/                    every generated training set (both recipes) and the red-team subset bases
direct_generation/       the paper's protocol: generate → fit → learning curve
  src/synthetic_probe_data/   generate→score→retrain→guide loop, probe fitting, activation caches
  scripts/                    set generators and the learning-curve harness (24 scripts, listed in its README)
  configs/                    generator/judge/probe configs (markdown + YAML frontmatter)
  *_GENERATOR_PROMPTS.md      the five LLM-written prompts per split (read by generate_prompt_variants.py)
redteam_scaffold/        the attacker / judge / retrain scaffold
  src/agentic_redteam/        attacker, tools, judge, contrastive preprocessing, retrain
  scripts/                    run_redteam, iterative_retrain, subset-base and draw helpers
  configs/                    every arm's config (attacker + judge prompts)
```

`eval_sets`, `dev_samples` and `data` exist once, at the root. Each subproject contains
symlinks to them (`direct_generation/eval_sets -> ../eval_sets`, ...) because its scripts
resolve paths relative to their own directory; on Windows enable `core.symlinks`. Probes
are not on `main`: the base probes (`probe_iter0.pkl` per concept) and every later one are
on the `results` branch under `results/direct_generation/probes/`; the learning-curve
harness expects them at `direct_generation/probes/gen_gemma27b_<concept>/probe_iter0.pkl`
(copy or symlink them there, or regenerate them with the loop).

The two subprojects are independent Python packages (`synthetic_probe_data` and
`agentic_redteam`), each with its own `pyproject.toml`, `requirements.txt` and `README.md`.
Run their scripts from inside the subproject directory.

## Setup

Both packages need [tuberlens](https://github.com/blandfort/tuberlens) (the fork
`acheckervarty7890/tuberlens`, branch `iterative_pipeline_2`, was used), PyTorch with CUDA
for activation extraction, and API keys for the LLM providers used (`ANTHROPIC_API_KEY`
and/or `OPENROUTER_API_KEY`).

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -U pip setuptools wheel
git clone --branch iterative_pipeline_2 https://github.com/acheckervarty7890/tuberlens.git .venv/src/tuberlens
pip install -e .venv/src/tuberlens
pip install -r direct_generation/requirements.txt      # pinned versions of the paper's env
pip install -e direct_generation                       # synthetic_probe_data
pip install -e redteam_scaffold                        # agentic_redteam (optional)
```

Precomputed eval/dev activations for Gemma-3-27B-IT were served from Kaggle during the
runs (`kaggle:` config section, `KAGGLE_CONFIG_DIR`); without them every script extracts
locally into `cache_*/` directories (large, gitignored).

## Concepts, splits, generators

| concept (paper) | code name | eval splits (`eval_sets/<dir>/`) |
|---|---|---|
| high-stakes | `highstakes` | `anthropic_hh_balanced`, `mt_balanced`, `mts_balanced`, `toolace_balanced` |
| harmful | `hu_harm` (eval dir `hu_ha`) | `eval_ai_dilemmas`, `eval_daily_dilemmas`, `eval_balanced_refusal`, `eval_ant_hh` |
| instruction | `instructions` | `anthropic_harmless_refusal`, `bbq_substitution`, `mm_substitution`, `hc_contradiction`, `hc_context_drift`, `oig_context_drift` |

A fifteenth split, `instructions/oig_omission`, is shipped but excluded from every number
in the paper. `dev_samples/highstakes_500/` is the 125-per-split high-stakes dev cut used
by the later studies.

Generators: Llama-3.3-70B-Instruct (`llama70b`), GPT-OSS-120B (`gptoss`),
DeepSeek-V4-Pro (`deepseekv4pro`), Nemotron-3-Ultra-550B (`nemotron`). Generated sets are
named `data/<concept>_<generator>_<variant>_<n>.jsonl`: `_50` = the general-prompt base
set, `_evaldesc_600` = the detailed-prompt set, `tgtnone`/`tgtshot`/`tgtmin` = split-targeted
sets (measured shape / few-shot anchor / shape-free), `_p5_`/`_hh5_`/... = the five
LLM-written prompt variants per split, `<concept>_base_<letters>.jsonl` = the red-team
subset bases (one letter per generator).

Probe models: `PROBE_PROFILE=gemma27b|qwen8b|llama1b|mistralnemo12b` selects the
extraction model and layer in `direct_generation/scripts/fit_base_plus_concept.py`.

## The pipeline

Every learning curve in the paper is: `scripts/generate_*_dataset.py` or
`generate_split_targeted.py` or `generate_prompt_variants.py` (write a set) →
`build_pooled_sets.py` (base ∪ generated pools) → `warm_pooled_sets.py` /
`warm_set_activations.py` (one extraction-model load per set) →
`subsample_curve_concept.py` (draw *n*, fit, score per split; pure cache hits) →
`report_pooled_curve.py` / `fit_curves_ref.py` (means per size; the log-logistic fit and
the half-gain size *m*). `fit_base_plus_concept.py` holds the per-concept table (base
data, dev/eval dirs, cache dirs, labels) and the probe-model profiles. The base probes were
written by `scripts/iterative_generate.py configs/gen_gemma27b_<concept>.md` and live on
the `results` branch.

The red-team arms are `redteam_scaffold/scripts/iterative_retrain.py <config>` with the
configs under `redteam_scaffold/configs/<generator>_<concept>_gemma27b_<base>_<arm>.md`;
the subset-base grids use `build_subset_base_activations.py` and `fit_combined_draws.py`.

## Results branch

`git checkout results` adds, on top of `main` (and merges in the experiment branches' history):

```
results/direct_generation/scripts/       every learning-curve CSV (<concept>_pooled_size_curve.csv, *_gen90*.csv,
                                         <profile>_<concept>_pooled_size_curve.csv, *_prompts5*.csv, *_tgtmin_*.csv,
                                         dc_*.csv, knee_*.csv, settling CSVs) and the per-study scripts that wrote them
results/direct_generation/run_scripts/   the run_*.sh wrapper each study was launched with
results/direct_generation/results_gen_gemma27b_*/   the generate→score→retrain loop runs behind the base probes
results/direct_generation/probes/        the base probes (gen_gemma27b_<concept>[_nemotron]/probe_iter0.pkl) and every per-iteration / per-study probe
results/direct_generation/analysis/refit_studies/   per-draw fit JSONs (harm split-targeted and combination studies)
results/redteam_scaffold/results_*/      per arm: *_comparison.csv (per-iteration eval AUROC) and *_probing_f{n,p}.jsonl (attempts)
results/redteam_scaffold/results_*_combined_draws/  the subset-base grids
results/redteam_scaffold/{scripts,run_scripts,analysis}/   the arms' one-off scripts, wrappers and refit JSONs
docs/direct_generation/                  study write-ups (direction_count.md, knee_predictor.md, prompts5_*.md,
                                         label audits, settling reports), task_briefs/, prompts/ (prompt dumps)
```

Paper section → results (paths on the `results` branch, under `results/direct_generation/`
unless noted):

| paper | study | results |
|---|---|---|
| Sec. 5.1 and 5.3, Fig. 1; App. "Values behind Figures" | learning curves, general and detailed prompts, Gemma-27B | `scripts/{highstakes,hu_harm,instructions}_pooled_size_curve.csv`, `*_gen90*.csv`, `*_size_curve.csv`, `*_ownbase_size_curve.csv`; wrappers `run_pooled_sizecurve.sh`, `run_persplit_fits.sh`, `run_evaldesc_fits.sh` |
| Sec. 5.1; App. "Other probe models" | Qwen3-8B, Mistral-NeMo-12B, Llama-3.2-1B | `scripts/{qwen8b,mistralnemo12b,llama1b}_<concept>_pooled_size_curve.csv`; `run_<profile>_sizecurve.sh` |
| Sec. 5.1, Fig. 2; App. "Fitted learning curves" | log-logistic fits, half-gain size, variance decomposition | `scripts/knee_fits.csv` (fitter: `direct_generation/scripts/fit_curves_ref.py`) |
| Sec. 5.2; App. "Coverage" | direction-count / coverage study | `scripts/dc_fits.csv`, `dc_ratios.csv`, `dc_geometry.csv`, `dc_curves_*.csv`; `scripts/direction_count.py`, `dc_run_curve.py`, `dc_chain_*.sh`; `docs/direct_generation/direction_count.md` |
| Sec. 5.4; App. "Split-targeted generation" | split-targeted sets, shape-free arm, arm filters | `scripts/*_tgtmin_size_curve*.csv`, `*_armfilter*`; `analysis/refit_studies/hu_harm_split_targeted_*/`, `hu_harm_combination/`; `run_tgt*.sh`, `run_*_hu_harm.sh`; `docs/direct_generation/split_targeted_*.md`, `arm_filter_results.md` |
| Sec. 5.4; App. "LLM-written prompt variants" | five LLM-written prompts per split, label audits | `scripts/*_prompts5.csv`, `*_prompts5_sizecurve.csv`, `prompts5_*_{eval_scores,manual_verdicts}.csv`; `run_*prompts5*.sh`; `docs/direct_generation/prompts5_*.md`, `toolace_*.md`, `*_label_audit.md` |
| Sec. 5.5; App. "Red-teaming scaffold" | red-team arms per generator, subset-base grids | `results/redteam_scaffold/results_*/*_comparison*.csv`, `results_*_combined_draws/combined_draws*.csv`; wrappers `results/redteam_scaffold/run_scripts/` |
| App. "Can the half-gain size be predicted before generating?" | 118 split properties vs. half-gain size (null) | `scripts/knee_predictor*.csv`, `knee_pooled_manifest.json`; `scripts/knee_predictor.py`; `docs/direct_generation/knee_predictor.md` |
| App. "Per-sample settling", "Below 60 samples" | settling and sub-60 curves on the instruction splits | `scripts/instructions_row_settling.csv`, `instructions_*_small_*.csv`, `instructions_parts_size_curve*.csv`; `scripts/analyze_row_settling.py`; `run_instr*.sh`, `run_hcdrift_small_curve.sh`; `docs/direct_generation/instructions_row_settling.md` and the `instructions_*.txt` reports |

## Branches

`main` has one line of history: the scaffold's development and the restructure commit.
The `results` branch is an octopus merge of `main` and the experiment branches whose
content the repository carries, so their full history (every intermediate CSV, log, probe
and failsafe checkpoint) is reachable from `results` and stays off `main`:

| branch merged into `results` | study |
|---|---|
| `dev_new_scaffolding` | the commit that replaced the red-team scaffold with the generation loop |
| `per_split_studies` | Gemma-27B pooled and fixed-base learning curves (Fig. 1) |
| `qwen8b`, `llama1b`, `mistralnemo12b` | the same curves on the other probe models |
| `per_split_studies2` | shape-free split-targeted sets, arm filters |
| `generator_experiment_1` | harm split-targeted arms and the combination study |
| `toolace_stuff` | five LLM-written prompts per split, label audits |
| `hello_kitty` | per-sample settling, sub-60 curves (also carries the ToolACE-parts study) |
| `direction_count` | coverage / direction-count study (built on `main` + `knee_predictor`) |
| `knee_predictor` | pre-generation predictors of the half-gain size |
| `experiment_hs_last`, `experiment_instruction_last`, `human_harm_last` | the red-team arms and subset-base grids per concept |

Not in `main` (kept as branches): `hard_split_experiments` (ToolACE cut into parts, not in
the paper), `oig_omission_experiment` (the split the paper excludes), `generalization_tests`
(pooling, ceilings, ensembles), `ceiling_analysis`, `devsamples_kfold_cloud` /
`devsamples_kfold_fixes` (k-fold ceilings), `high_stakes_old` / `human_harm_old` /
`instruction_old` (snapshots of the retired red-team experiment branches, under
`old_branches/`), `arch-cluster-ablation`, `architecture_related_experiment`,
`intact-pair-nonlinear`, `ensemble_speeduptest`, `redteam-activation-publishing`,
`first_instruction_experiment`, `tests_new`, `backup/experiment8_cloud-failsafe-20260730`,
and the `archive/experiment*_cloud` tags.

Left on the experiment branches only: run logs (`logs/`, `logs_archive/`), per-sample
score dumps (`data/instructions_row_scores/*.npz` on `hello_kitty`), pooled feature dumps
(`scripts/dc_pooled/*.npy` on `direction_count`), per-batch candidate probes
(`probes/*/candidates/`), the red-team arms' per-iteration probes, and the red-team runs'
bookkeeping sidecars.

## Notes

- `direct_generation`'s package was renamed from `agentic_redteam` to `synthetic_probe_data`
  when the two lineages were brought together; its environment variables changed prefix
  accordingly (`SYNTHETIC_PROBE_DATA_MAX_MEMORY`, `SYNTHETIC_PROBE_DATA_TRUNCATE_LAYERS`,
  `SYNTHETIC_PROBE_DATA_STAGE_ACTIVATIONS`, `SYNTHETIC_PROBE_DATA_FIT_STAGING_RESERVE_GIB`).
  `redteam_scaffold` keeps `agentic_redteam` and `AGENTIC_REDTEAM_*`.
- Where two branches carried different versions of the same script, `main` has the newest
  superset (`fit_base_plus_concept.py` with all four probe profiles, `subsample_curve_concept.py`
  with `--eval-splits`, `split_specs.py` / `generate_split_targeted.py` with the shape-free
  `--minimal` arm). The ToolACE-parts variant of the last two is on `hello_kitty`.
- The ToolACE evaluation split (`eval_sets/highstakes/toolace_balanced.jsonl`, derived from
  the public ToolACE dataset) and a few generated sets contain strings shaped like
  credentials (example AWS keys, a GitHub-token-shaped string). They come from the source
  data and the generator, not from any account.
