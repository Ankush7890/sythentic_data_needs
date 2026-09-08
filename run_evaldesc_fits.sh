#!/usr/bin/env bash
# Fit + evaluate the eval-description-steered 600-row sets, under EXACTLY the protocol the
# unsteered sets were measured under (scripts/{instructions_gen90,highstakes_gen90_dev500}.csv):
#
#   own base       each generator's own 50-row set, so every arm is single-source
#   n = 540        90% of the 600, 8 seeded class-balanced draws -> a noise bar per set
#   n = 600 x1     the whole set, for the headline number (added here for the unsteered
#                  sets too, so the two are compared at the same n as well as at 540)
#   dev            instructions: dev_samples/instructions; highstakes: dev_samples/highstakes_500
#                  (the 500-row cut the unsteered highstakes arms already used — the full
#                  1908-row dev set is resident every epoch and costs ~20x per fit)
#
# Rows append to the SAME CSVs the unsteered arms wrote, keyed on (base, samples, n, draw),
# so this is also the restart path: re-run and it skips what is already there.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
GENS="llama70b gptoss deepseekv4pro nemotron"
mkdir -p logs

wait_for_sets() {  # concept — block until that concept's four steered sets are written
    local concept=$1 g f missing
    while :; do
        missing=""
        for g in $GENS; do
            f="data/${concept}_${g}_evaldesc_600.jsonl"
            # The generator writes the file once, at the end, so a non-empty file is a
            # finished one. Guard on the row count anyway: a short set means the model
            # could not fill its half and the arm is not comparable.
            [ -s "$f" ] || missing="$missing $g"
        done
        [ -n "$missing" ] || break
        echo ">>> $(date -Is)  waiting on $concept generation:$missing"
        sleep 120
    done
}

run() {  # concept set_file gen sizes draws
    local concept=$1 set_file=$2 gen=$3 sizes=$4 draws=$5
    [ -s "$set_file" ] || { echo ">>> $(date -Is)  MISSING $set_file — skipping"; return 0; }
    local devflag=""
    local out="scripts/instructions_gen90.csv"
    if [ "$concept" = highstakes ]; then
        devflag="--dev-data dev_samples/highstakes_500"
        out="scripts/highstakes_gen90_dev500.csv"
    fi
    echo ">>> $(date -Is)  $concept $(basename "$set_file") sizes=$sizes draws=$draws"
    $PY scripts/subsample_curve_concept.py --concept "$concept" "$set_file" \
        --base-data "data/${concept}_${gen}_50.jsonl" $devflag \
        --sizes $sizes --draws "$draws" --out "$out" \
        >> "logs/fit_evaldesc_${concept}.log" 2>&1
}

for concept in instructions highstakes; do
    wait_for_sets "$concept"
    # PASS A — full set, steered. First touch of these rows, so this is also what pays the
    # gemma-3-27b extraction; the 540-draw fits after it are pure cache hits.
    for g in $GENS; do run "$concept" "data/${concept}_${g}_evaldesc_600.jsonl" "$g" 600 1; done
    # PASS B — full set, unsteered counterpart (activations already cached).
    for g in $GENS; do run "$concept" "data/${concept}_${g}_600.jsonl" "$g" 600 1; done
    # PASS C — 8 draws at 90%, steered.
    for g in $GENS; do run "$concept" "data/${concept}_${g}_evaldesc_600.jsonl" "$g" 540 8; done
done
echo ">>> $(date -Is)  all eval-description fits finished."
