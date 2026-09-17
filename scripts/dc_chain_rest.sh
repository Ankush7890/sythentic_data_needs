#!/usr/bin/env bash
# The rest of the study, in the brief's order, once the instructions fits are done.
#
#   hu_harm    extract (no 50-row bases: no arm uses them, and the card is the constraint)
#              -> pool -> geometry -> arms -> fits, four generators
#   highstakes the same, with two differences: the fits use dev_samples/highstakes_500,
#              the 500-row doubly-balanced dev cut the paper's own small-n pooled curves
#              used (the full 1908-row dev set is resident every epoch and is what makes
#              this concept ~20x the others), and the kind-only / leave-one-kind-out arms
#              are scored on their own target split alone (--restrict-eval), because the
#              four eval blobs are 47 GB and those arms read one column of them.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv_claude/bin/python
mkdir -p logs

# Wait for the instructions CHAIN, not for a fit process: that chain runs one process per
# generator, and between two of them there is a gap in which a "no fit is running" check
# would be true and this script would start an extraction onto a card the next fit wants.
echo ">>> $(date -Is) waiting for the instructions chain"
while pgrep -f "[d]c_chain_instructions.sh" >/dev/null; do
    sleep 120
done

for concept in hu_harm highstakes; do
    echo ">>> $(date -Is) warm: $concept"
    $PY scripts/direction_count.py --stage warm --concepts "$concept" --no-bases \
        >> "logs/dc_warm_${concept}.log" 2>&1 || { echo ">>> FAILED warm $concept"; continue; }
    echo ">>> $(date -Is) geometry + arms: $concept"
    $PY scripts/direction_count.py --stage geometry --concepts "$concept" \
        >> "logs/dc_geometry_${concept}.log" 2>&1
    $PY scripts/direction_count.py --stage arms --concepts "$concept" \
        >> "logs/dc_arms_${concept}.log" 2>&1
    extra=""
    [ "$concept" = highstakes ] && extra="--highstakes-dev dev_samples/highstakes_500 --restrict-eval"
    for gen in deepseekv4pro gptoss nemotron llama70b; do
        echo ">>> $(date -Is) fits: $concept x $gen"
        # shellcheck disable=SC2086
        $PY scripts/direction_count.py --stage fit --concepts "$concept" --generators "$gen" $extra \
            >> "logs/dc_fit_${concept}_${gen}.log" 2>&1 \
            || echo ">>> $(date -Is) FAILED $concept x $gen"
    done
done
echo ">>> $(date -Is) all concepts done"
