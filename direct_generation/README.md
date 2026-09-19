# direct_generation — synthetic training sets and learning curves for activation probes

The paper's protocol. A generator LLM writes labelled conversations for a probe's concept;
probes are fit on *n* of them and scored per evaluation split; the resulting learning curves
are the paper's main object. The package `synthetic_probe_data` (formerly `agentic_redteam`;
see the repository README) holds the generation loop and the probe-fitting / activation-cache
machinery that every study script builds on.

Two things live here:

1. **The generate → score → retrain → guide loop** (`scripts/iterative_generate.py`), which
   wrote each concept's base probe (`probes/gen_gemma27b_<concept>/probe_iter0.pkl`).
2. **The learning-curve harness** (`scripts/subsample_curve_concept.py` and friends) behind
   the paper's figures. The generated sets live in `../data/`; the results and the
   per-study scripts are on the `results` branch (see the end of this file).

## The loop

For a current probe *P* with mean dev AUROC *A*, one iteration does:

1. **Directions.** Take the `n_batches` directions written for this iteration — the
   judge's, from the previous iteration; at iteration 0 the generator proposes them.
2. **Generate.** `n_batches` concurrent generator calls, batch *k* under direction *k*
   by `models[k % len(models)]`, each returning `batch_size` samples (`batch_size/2` per
   class). Over-long samples (`max_sample_tokens`, counted with the probe's tokenizer),
   malformed ones and duplicates of anything generated earlier in the run are dropped;
   a short batch gets up to `max_retries` in-context top-up asks.
3. **Warm the activation cache** for every new sample in one extraction-model load.
4. **Score each batch on its own.** Train a candidate probe on
   base ∪ accepted-so-far ∪ batch (pure cache hit — no model load), read its per-split
   dev AUROC, Δ = mean − *A*. Δ > `min_auroc_gain` ⇒ **accepted**;
   |Δ| ≤ `exhausted_gain` ⇒ flagged **exhausted** for the judge; Δ < 0 ⇒ harmful.
5. **Union retrain.** Train `probe_iter{i+1}.pkl` on base ∪ every accepted batch so
   far; its dev AUROC is the next baseline. Nothing accepted ⇒ the probe carries over.
6. **Judge.** Every batch (direction, sample excerpts, per-split Δ, verdict) goes to
   the judge, which rewrites a bounded rolling memo and writes the next iteration's
   `n_batches` directions.
7. Optional `--eval` on the eval splits.

`loop.iterations` (or `--iterations`) repeats this. Everything is resumable at batch
granularity from the sidecars in `output.run_dir`. The probe itself decides what gets
kept: a batch enters the training set only if a probe trained on it scores higher on the
dev set than the current probe.

Both LLM roles can be driven by `claude_sdk` (Anthropic Python SDK) or `openrouter` (the
`openai` SDK pointed at [OpenRouter](https://openrouter.ai/)); generator models may mix
providers. No tools, MCP or shell are involved — every call is a single chat completion.

## Setup

See the repository README for the environment (tuberlens fork, pinned
`requirements.txt`, `pip install -e direct_generation`). Keys:

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # any `provider: claude_sdk` section
export OPENROUTER_API_KEY=sk-or-...       # any `provider: openrouter` section
```

Memory limits
for the truncated extraction model are read from `SYNTHETIC_PROBE_DATA_MAX_MEMORY`
(e.g. `0=22GiB,cpu=45GiB`); `SYNTHETIC_PROBE_DATA_TRUNCATE_LAYERS`,
`SYNTHETIC_PROBE_DATA_STAGE_ACTIVATIONS` and
`SYNTHETIC_PROBE_DATA_FIT_STAGING_RESERVE_GIB` tune extraction and fit staging.

## Run the loop

```bash
python scripts/iterative_generate.py configs/gen_gemma27b_highstakes.md \
  --base-training-data data/highstakes_llama70b_50.jsonl \
  --probe-out-dir probes/generate_example \
  --eval --eval-dataset-dir eval_sets/highstakes      # --eval is optional
```

`--base-training-data` trains the initial probe (unless `probe.path` warm-starts one)
and is part of every retrain. A **dev set is required** (`validation.dev_data` or
`--dev-data`): it is both the fit's early-stopping validation set and the set every
batch's ΔAUROC is read on, and must be disjoint from the eval splits. The three shipped
dev sets are under `dev_samples/<concept>/`, the eval splits under `eval_sets/<concept>/`.

Outputs, per run:

| where | what |
| --- | --- |
| `<probe-out-dir>/probe_iter{N}.pkl` | the probe iteration N starts from (`probe_iter0` = initial) |
| `<probe-out-dir>/candidates/probe_iter{i}_batch{k}.pkl` | the per-batch candidate probes |
| `<run_dir>/batches.jsonl` | every batch: direction, samples, AUROC before/after, Δ, accepted/exhausted |
| `<run_dir>/guidance.jsonl` | the judge's memo + directions per iteration |
| `<run_dir>/auroc_history.csv` | one row per (iteration, batch): the ΔAUROC ledger |
| `<run_dir>/accepted_iter{N}.jsonl` | the accepted samples `probe_iter{N}` was trained with |
| `<run_dir>/runlog.jsonl` | lifecycle / error events |

The runs that produced the paper's base probes (configs
`configs/gen_gemma27b_<concept>[_gptoss|_nemotron].md`) are on the `results` branch under
`results/direct_generation/results_gen_gemma27b_*/`, the probes they wrote under
`results/direct_generation/probes/`; the harness reads them from `probes/` in this directory.

## Config

A markdown file: YAML frontmatter + `# Generator` and `# Judge` sections holding the
two system prompts. See `configs/example_generate.md` for every key; the essentials:

```yaml
generator:
  provider: openrouter
  models: [meta-llama/llama-3.3-70b-instruct]   # batch k → models[k % len]
  n_batches: 5          # n
  batch_size: 20        # m (even)
  concurrency: 5
  max_sample_tokens: 1024
judge:
  provider: openrouter
  model: openai/gpt-5.1-chat
  memo_word_budget: 400
probe:                  # from-scratch fields; or `path:` to warm-start
  model: google/gemma-3-27b-it
  layer: 32
  pos_class_label: high-stakes
  neg_class_label: low-stakes
  description: ...
loop:
  iterations: 3
  min_auroc_gain: 0.0
  exhausted_gain: 0.002
validation:
  dev_data: ../dev_samples/highstakes
output:
  run_dir: ../results/my_run
```

## The learning-curve harness

```
scripts/generate_{highstakes,hu_harm,instructions}_dataset.py   write a set from the general or detailed prompt
scripts/generate_split_targeted.py + split_specs.py             split-targeted sets (measured shape, few-shot anchor, --minimal)
scripts/generate_prompt_variants.py                             the five LLM-written prompts per split
scripts/inspect_generated_set.py                                length / label / overlap checks on a set
scripts/build_pooled_sets.py, build_qwen8b_pools.py             base ∪ generated pools per (concept, generator, arm)
scripts/warm_pooled_sets.py, warm_set_activations.py            one extraction-model load per set → per-conversation cache
scripts/subsample_curve_concept.py                              draw n samples, fit, score every split (cache hits only)
scripts/fit_base_plus_concept.py                                per-concept table (base data, dev/eval dirs, caches, labels) and PROBE_PROFILE
scripts/report_pooled_curve.py, fit_curves_ref.py               mean/sd per size; the log-logistic fit (half-gain size m)
scripts/make_probe_templates.py, make_qwen8b_probe_templates.py untrained probe templates for the non-Gemma profiles
scripts/extract_eval_activations.py, publish_kaggle_eval_activations.py   the precomputed eval/dev activations
scripts/verify_generation_loop.py, verify_fit_staging.py, verify_ensemble_fusion.py   self-checks (fake LLMs / fake fits)
```

`PROBE_PROFILE=gemma27b|qwen8b|llama1b|mistralnemo12b` selects the extraction model and
layer; Gemma has trained base probes, the other three use templates and their own
`.pool_work_<profile>/` and cache directories.

## Studies and results

`main` carries only this pipeline and the shared data (`eval_sets/`, `dev_samples/` and
`data/` are symlinks to the repository root). The learning-curve CSVs, the
per-study scripts, the `run_*.sh` wrapper each study was launched with, the loop runs
behind the base probes and the study write-ups are on the `results` branch under
`results/direct_generation/` and `docs/direct_generation/`; the repository README maps
every paper section to them.

## Verifying

```bash
# Loop bookkeeping with fake LLMs and fake fits — no GPU, model or key needed
python scripts/verify_generation_loop.py --mode fake
# Same, with real llama-1b fits on a 50-row base set and a small dev cut
python scripts/verify_generation_loop.py --mode real
```
