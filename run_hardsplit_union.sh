#!/usr/bin/env bash
# hard_split_experiments, round 3: ALL eight specialist sets pooled — round 1's four 600-row part
# sets and round 2's four 300-row sub-part sets, 3600 rows. NO base data. The full union once,
# then eight 90% resamples (90% of each set, class-balanced). Every row is already extracted.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export MAX_MEMORY="${MAX_MEMORY:-${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}}"
PY=.venv_claude/bin/python
SETS=$(ls data/toolace_parts/highstakes_deepseekv4pro_tgtnone_toolace_*_600.jsonl \
          data/toolace_parts/highstakes_deepseekv4pro_tgtnone_{lookup,finance}_*_300.jsonl)
echo ">>> $(date -Is) union of $(echo $SETS | wc -w) sets"
$PY scripts/fit_toolace_parts.py $SETS --union --sizes 100 90 --draws 1 8 \
    --out scripts/highstakes_toolace_union.csv
echo ">>> $(date -Is) all union fits finished."
