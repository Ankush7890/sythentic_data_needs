#!/usr/bin/env bash
# hard_split_experiments, round 3c: size curve on the merged 3600-row specialist pool — eight
# random class-balanced draws at 1200 and at 1800 rows (600 is round 3b, same CSV). Draws are
# seeded on (file stem, n, draw), so sizes are independent draws, not nested. NO base data.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export MAX_MEMORY="${MAX_MEMORY:-${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}}"
PY=.venv_claude/bin/python
$PY scripts/fit_toolace_parts.py data/toolace_parts/highstakes_deepseekv4pro_tgtnone_union8_3600.jsonl \
    --arm union600 --sizes 1200 1800 --draws 8 8 --out scripts/highstakes_toolace_union600.csv
echo ">>> $(date -Is) union 1200/1800-row draws finished."
