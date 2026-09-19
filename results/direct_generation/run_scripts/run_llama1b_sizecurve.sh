#!/usr/bin/env bash
# LLAMA-3.2-1B-INSTRUCT POOLED SIZE CURVE — every training row resampled, 52 pools, 8 draws.
#
# Probed model: meta-llama/Llama-3.2-1B-Instruct, layer 8 (the middle of 16), same probe spec as
# the gemma probes (linear_then_softmax, batch 16, lr 5e-3, 200 epochs, patience 50). PROBE_PROFILE
# switches fit_base_plus_concept.CONCEPTS to probes/llama1b_<concept>/ and cache_llama1b_<concept>/.
# No probe is trained on a base set: fits inherit model/layer/labels/spec from a metadata-only
# template. There are no Kaggle activations for this model, so dev and eval are extracted
# locally by the first fit of each concept and cached. The Llama-3 chat template's system-header
# date is pinned (fit_base_plus_concept.PINNED_DATE_STRING) so no extraction depends on the clock.
#
# POOLS (scripts/build_qwen8b_pools.py --out-dir .pool_work_llama1b; the same 52 pools as qwen8b):
# each 600-row generated set ∪ the same generator's 50-row base, drawn with --no-base, so NOTHING is held fixed across draws.
#   highstakes   16 = general x4 + evaldesc x4 + tgtnone x4 splits + tgtmin x4 splits
#   instructions 20 = general x4 + evaldesc x4 + tgtnone x6 splits + tgtmin x6 splits
#   hu_harm      16 = general x4 + evaldescshape x4 + full x4 splits + minimal x4 splits
#
# SIZES 590 350 170 110 80 30 10, class-balanced draws seeded on (pool stem, n, draw). The
# optimizer regime is the pooled curve's (run_pooled_sizecurve*.sh): at n<=80 accumulation =
# ceil(n/16) so every size takes exactly one step per epoch (at the default 4, n=30 and n=10
# take ZERO steps and return the untrained probe); above 80 the default accumulation 4, passed
# explicitly so rows tag base='none+ga4'.
#
#   n=590 350 170 110 -> --grad-accum 4
#   n=80              -> --grad-accum 5
#   n=30              -> --grad-accum 2
#   n=10              -> --grad-accum 1 --batch-size 16
#
# High-stakes validates on dev_samples/highstakes_500, as every high-stakes curve in this repo
# does (the full 1908-row dev set is resident for every epoch of every fit).
#
# hu_harm 16x7x8 = 896, instructions 20x7x8 = 1120, highstakes 16x7x4 = 448 fits.
# Resumable: re-running skips (base, samples, n, draw)
# rows already in scripts/llama1b_<concept>_pooled_size_curve.csv.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export PROBE_PROFILE=llama1b
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-8}"
# High-stakes runs 4 draws per cell, as the qwen8b curve settled on: every high-stakes fit re-scores
# the whole (largest) eval set, so its fits cost several times the other two concepts'.
HS_DRAWS="${HS_DRAWS:-4}"
CONCEPTS="${CONCEPTS:-hu_harm instructions highstakes}"
SIZES="${SIZES:-590 350 170 110 80 30 10}"
POOL_DIR=.pool_work_llama1b
mkdir -p logs

ls "$POOL_DIR"/pool_*.jsonl >/dev/null 2>&1 || $PY scripts/build_qwen8b_pools.py --out-dir "$POOL_DIR"
# Metadata-only probe templates (no base set, no training); see the script's docstring.
$PY scripts/make_probe_templates.py || exit 1

for concept in $CONCEPTS; do
    out="scripts/llama1b_${concept}_pooled_size_curve.csv"
    log="logs/llama1b_sizecurve_${concept}.log"
    devflag="" draws="$DRAWS"
    [ "$concept" = highstakes ] && devflag="--dev-data dev_samples/highstakes_500" && draws="$HS_DRAWS"

    echo ">>> $(date -Is)  ${concept}: warming the per-sample cache for every pool"
    $PY scripts/warm_pooled_sets.py --concept "$concept" --pool-dir "$POOL_DIR" \
        >> "logs/llama1b_warm_${concept}.log" 2>&1 \
        || { echo ">>> $(date -Is)  FAILED warm for ${concept}"; continue; }

    pools=$(ls "$POOL_DIR"/pool_${concept}_*.jsonl)
    for n in $SIZES; do
        case $n in
            80) extra="--grad-accum 5" ;;
            30) extra="--grad-accum 2" ;;
            10) extra="--grad-accum 1 --batch-size 16" ;;
            *)  extra="--grad-accum 4" ;;
        esac
        echo ">>> $(date -Is)  ${concept} n=$n ($extra) over $(echo "$pools" | wc -l) pools"
        $PY scripts/subsample_curve_concept.py --concept "$concept" $pools \
            --no-base $devflag $extra --sizes "$n" --draws "$draws" --out "$out" \
            >> "$log" 2>&1 \
            || echo ">>> $(date -Is)  FAILED ${concept} n=$n"
    done
    echo ">>> $(date -Is)  ${concept} done — $(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 )) rows in $out"
done
echo ">>> $(date -Is)  llama1b size curve finished."
