#!/usr/bin/env bash
# QWEN3-8B POOLED SIZE CURVE — every training row resampled, 52 pools, 8 draws.
#
# Probed model: Qwen/Qwen3-8B, layer 18 (the middle of 36), same probe spec as the gemma
# probes (linear_then_softmax, batch 16, lr 5e-3, 200 epochs, patience 50). PROBE_PROFILE
# switches fit_base_plus_concept.CONCEPTS to probes/qwen8b_<concept>/ and cache_qwen8b_<concept>/.
# No probe is trained on a base set: fits inherit model/layer/labels/spec from a metadata-only
# template. There are no Kaggle activations for this model, so dev and eval are extracted
# locally by the first fit of each concept and cached.
#
# POOLS (scripts/build_qwen8b_pools.py -> .pool_work_qwen8b/): each 600-row generated set ∪ the
# same generator's 50-row base, drawn with --no-base, so NOTHING is held fixed across draws.
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
# 52 pools x 7 sizes x 8 draws = 2912 fits. Resumable: re-running skips (base, samples, n, draw)
# rows already in scripts/qwen8b_<concept>_pooled_size_curve.csv.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export PROBE_PROFILE=qwen8b
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-8}"
CONCEPTS="${CONCEPTS:-hu_harm instructions highstakes}"
SIZES="${SIZES:-590 350 170 110 80 30 10}"
POOL_DIR=.pool_work_qwen8b
mkdir -p logs

ls "$POOL_DIR"/pool_*.jsonl >/dev/null 2>&1 || $PY scripts/build_qwen8b_pools.py
# Metadata-only probe templates (no base set, no training); see the script's docstring.
$PY scripts/make_qwen8b_probe_templates.py || exit 1

for concept in $CONCEPTS; do
    out="scripts/qwen8b_${concept}_pooled_size_curve.csv"
    log="logs/qwen8b_sizecurve_${concept}.log"
    devflag=""
    [ "$concept" = highstakes ] && devflag="--dev-data dev_samples/highstakes_500"

    echo ">>> $(date -Is)  ${concept}: warming the per-sample cache for every pool"
    $PY scripts/warm_pooled_sets.py --concept "$concept" --pool-dir "$POOL_DIR" \
        >> "logs/qwen8b_warm_${concept}.log" 2>&1 \
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
            --no-base $devflag $extra --sizes "$n" --draws "$DRAWS" --out "$out" \
            >> "$log" 2>&1 \
            || echo ">>> $(date -Is)  FAILED ${concept} n=$n"
    done
    echo ">>> $(date -Is)  ${concept} done — $(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 )) rows in $out"
done
echo ">>> $(date -Is)  qwen8b size curve finished."
