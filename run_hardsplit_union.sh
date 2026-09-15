#!/usr/bin/env bash
# hard_split_experiments, round 3: ALL eight specialist sets pooled — round 1's four 600-row part
# sets and round 2's four 300-row sub-part sets, 3600 rows. NO base data. ONE fit on the full union (no resampled draws — a fit on
# 3600 rows trains host-resident and takes far longer than the 300-600-row arms). Every row is
# already extracted.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export MAX_MEMORY="${MAX_MEMORY:-${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}}"
PY=.venv_claude/bin/python
SETS=$(ls data/toolace_parts/highstakes_deepseekv4pro_tgtnone_toolace_*_600.jsonl \
          data/toolace_parts/highstakes_deepseekv4pro_tgtnone_{lookup,finance}_*_300.jsonl)
echo ">>> $(date -Is) union of $(echo $SETS | wc -w) sets"
$PY scripts/fit_toolace_parts.py $SETS --union --sizes 100 --draws 1 \
    --out scripts/highstakes_toolace_union.csv
echo ">>> $(date -Is) union fit finished."
