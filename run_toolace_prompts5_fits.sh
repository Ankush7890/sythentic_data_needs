#!/usr/bin/env bash
# One fit per prompt set: n=600, one draw, NO base data, validated on dev_samples/highstakes_500,
# scored on all four full highstakes eval splits. Same protocol as the no-base rows in
# scripts/highstakes_gen90_dev500.csv, so the earlier toolace arms are the comparison.
#
# This workspace has no activation cache, so the FIRST fit of each set extracts its 600 rows
# (and, on the very first fit, the 500 dev rows) on gemma-3-27b. Later fits are cache hits.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
OUT=scripts/highstakes_toolace_prompts5.csv
mkdir -p logs

for f in data/highstakes_deepseekv4pro_p5_*_600.jsonl; do
    [ -s "$f" ] || continue
    echo ">>> $(date -Is)  fitting $(basename "$f")"
    $PY scripts/subsample_curve_concept.py --concept highstakes "$f" --no-base \
        --dev-data dev_samples/highstakes_500 --sizes 600 --draws 1 --out "$OUT" \
        >> logs/fit_p5.log 2>&1 || echo ">>> $(date -Is)  FIT FAILED $(basename "$f")"
done
echo ">>> $(date -Is)  all fits finished"
