#!/usr/bin/env bash
# One fit per hh prompt set: n=600, one draw, NO base data, dev_samples/highstakes_500,
# scored on all four full highstakes eval splits — the protocol of the base='none' rows in
# scripts/highstakes_gen90_dev500.csv and of the toolace prompt study.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
OUT=scripts/highstakes_anthropic_hh_prompts5.csv
mkdir -p logs

for f in data/highstakes_deepseekv4pro_hh5_*_600.jsonl; do
    [ -s "$f" ] || continue
    echo ">>> $(date -Is)  fitting $(basename "$f")"
    $PY scripts/subsample_curve_concept.py --concept highstakes "$f" --no-base \
        --dev-data dev_samples/highstakes_500 --sizes 600 --draws 1 --out "$OUT" \
        >> logs/fit_hh5.log 2>&1 || echo ">>> $(date -Is)  FIT FAILED $(basename "$f")"
done
echo ">>> $(date -Is)  all hh fits finished"
