#!/usr/bin/env bash
# Hand the GPU from the extraction stage to the instructions fits without an idle gap.
#
# --stage warm extracts the three concepts in one process; --stage fit needs the card to
# itself (a fit stages its activations on the GPU, and the extraction holds 22 GiB). So
# this waits for the eight instructions sets to be extracted AND pooled, stops the warm
# process — it is resumable per conversation, so hu_harm and highstakes lose nothing —
# and runs the instructions fits, deepseekv4pro first as the brief asks.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv_claude/bin/python
mkdir -p logs

echo ">>> $(date -Is) waiting for the instructions extraction"
until grep -q "instructions: .* newly extracted" logs/dc_warm.log 2>/dev/null; do sleep 60; done
echo ">>> $(date -Is) waiting for the eight instructions sets to be pooled"
until [ "$(grep -c 'rows x' logs/dc_warm.log)" -ge 8 ]; do sleep 30; done

echo ">>> $(date -Is) stopping the warm stage"
pkill -f "direction_count.py --stage warm" || true
sleep 30

for gen in deepseekv4pro llama70b gptoss nemotron; do
    echo ">>> $(date -Is) fits: instructions x $gen"
    $PY scripts/direction_count.py --stage fit --concepts instructions --generators "$gen" \
        >> "logs/dc_fit_instructions_${gen}.log" 2>&1 \
        || echo ">>> $(date -Is) FAILED instructions x $gen"
done
echo ">>> $(date -Is) instructions fits done"
