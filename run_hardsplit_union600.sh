#!/usr/bin/env bash
# hard_split_experiments, round 3b: the eight specialist sets merged into one 3600-row pool
# (data/toolace_parts/highstakes_deepseekv4pro_tgtnone_union8_3600.jsonl), then eight random
# class-balanced draws of 600 from it — draws over the MERGED pool, so each set's share varies
# by chance. NO base data. Every row is already extracted.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export MAX_MEMORY="${MAX_MEMORY:-${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}}"
PY=.venv_claude/bin/python
$PY scripts/fit_toolace_parts.py data/toolace_parts/highstakes_deepseekv4pro_tgtnone_union8_3600.jsonl \
    --arm union600 --sizes 600 --draws 8 --out scripts/highstakes_toolace_union600.csv
echo ">>> $(date -Is) union 600-row draws finished."
