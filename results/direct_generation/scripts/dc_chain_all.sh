#!/usr/bin/env bash
# The whole study's fits, generators run CONCURRENTLY, in the brief's concept order.
#
# WHY CONCURRENT. A fit here is not GPU-bound: it holds 2.7 GiB of the card at 30-45%
# utilisation and spends most of its wall-clock loading and scoring cached activations
# (5.5 GiB of blobs per fit for instructions). On this box — 88 cores, 251 GiB — four
# generators run side by side for the price of one, which takes instructions from ~28 h
# to ~7 h. Each generator writes its own CSV (--out-tag); the harness appends a row per
# fit and resumes off its --out file, so sharing one would corrupt both the file and the
# resume. --stage analyse reads every dc_curves_<concept>*.csv.
#
# highstakes runs THREE at a time, not four: its restricted eval still loads one blob per
# fit and anthropic_hh_balanced alone is 33 GiB.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv_claude/bin/python
mkdir -p logs

run_concept () {           # concept, parallel width, extra flags
    local concept="$1" width="$2"; shift 2
    local extra="$*" pids=() gens=(deepseekv4pro gptoss nemotron llama70b) i=0
    for gen in "${gens[@]}"; do
        # instructions x deepseekv4pro is already running, started before --out-tag
        # existed and writing to the shared CSV; leave it alone and wait for it below.
        if [ "$concept" = instructions ] && [ "$gen" = deepseekv4pro ]; then continue; fi
        local tag="--out-tag $gen"
        echo ">>> $(date -Is) fits: $concept x $gen"
        # shellcheck disable=SC2086
        $PY scripts/direction_count.py --stage fit --concepts "$concept" \
            --generators "$gen" $tag $extra >> "logs/dc_fit_${concept}_${gen}.log" 2>&1 &
        pids+=($!)
        i=$((i + 1))
        if [ "$((i % width))" -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
    done
    [ "${#pids[@]}" -gt 0 ] && wait "${pids[@]}"
    echo ">>> $(date -Is) $concept fits done"
}

run_concept instructions 3

echo ">>> $(date -Is) waiting for the instructions deepseekv4pro fits already running"
while pgrep -f "[-]-concepts instructions --generators deepseekv4pro" >/dev/null; do
    sleep 60
done
echo ">>> $(date -Is) instructions done"

for concept in hu_harm highstakes; do
    echo ">>> $(date -Is) warm: $concept"
    $PY scripts/direction_count.py --stage warm --concepts "$concept" --no-bases \
        >> "logs/dc_warm_${concept}.log" 2>&1 || { echo ">>> FAILED warm $concept"; continue; }
    $PY scripts/direction_count.py --stage geometry --concepts "$concept" \
        >> "logs/dc_geometry_${concept}.log" 2>&1
    $PY scripts/direction_count.py --stage arms --concepts "$concept" \
        >> "logs/dc_arms_${concept}.log" 2>&1
    if [ "$concept" = highstakes ]; then
        run_concept highstakes 3 --highstakes-dev dev_samples/highstakes_500 --restrict-eval
    else
        run_concept hu_harm 4
    fi
done
echo ">>> $(date -Is) all concepts done"
