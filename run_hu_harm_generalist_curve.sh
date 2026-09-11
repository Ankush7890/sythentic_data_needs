#!/usr/bin/env bash
# hu_harm generalist size curve — the third concept's counterpart to
# run_generalist_sizecurve.sh. 4 generators x n=600(x1) and 540/300/120/60/30 (x8 draws).
#
# USES THE `evaldescshape` SETS, not `evaldesc`. Both exist on generator_experiment_1 (another
# session generated them); the shaped variant is the one comparable to the instructions and
# high-stakes generalists, which also carried the shape machinery alongside the description.
# All four are 600 rows, 300/300, two-turn throughout — hu_harm's four kinds are all
# two-message exchanges — and verified disjoint from eval, dev, own base and the unsteered set.
#
# Appends to scripts/hu_harm_gen90.csv, which already holds the four UNSTEERED arms at
# n=540 x 8 — the baseline this curve is read against. Own 50-row base per generator, the
# concept's own dev set (dev_samples/hu_ha, 290 rows), all four splits scored.
#
# The first pass pays the extraction for 2400 rows; every later draw is a cache hit.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
OUT=scripts/hu_harm_gen90.csv

for gen in llama70b gptoss deepseekv4pro nemotron; do
    f="data/hu_harm_${gen}_evaldescshape_600.jsonl"
    b="data/hu_harm_${gen}_50.jsonl"
    [ -s "$f" ] && [ -s "$b" ] || { echo ">>> $(date -Is)  MISSING $f or $b"; continue; }
    for spec in "600 1" "540 8" "300 8" "120 8" "60 8" "30 8"; do
        set -- $spec
        echo ">>> $(date -Is)  hu_harm ${gen} n=$1 draws=$2"
        $PY scripts/subsample_curve_concept.py --concept hu_harm "$f" \
            --base-data "$b" --sizes "$1" --draws "$2" --out "$OUT" \
            >> logs/hu_harm_generalist_curve.log 2>&1 \
            || echo ">>> $(date -Is)  FIT FAILED ${gen} n=$1"
    done
    echo ">>> $(date -Is)  hu_harm generalist done: $gen"
done
echo ">>> $(date -Is)  all hu_harm generalist fits finished."
