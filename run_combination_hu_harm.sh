#!/usr/bin/env bash
# COMBINATION STUDY: pool one set per eval split, three arms x four per-split sizes.
# 3 arms x (8 best + 8 n30 + 8 n300 + 1 n600) = 75 fits. Dev is all four splits.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
mkdir -p logs
echo ">>> $(date -Is)  combination study starting"
$PY scripts/fit_hu_harm_combination.py --arms arm1 cleanPL minimal \
    --modes best n30 n300 n600 --draws 8 --accum 4 \
    >> logs/combination_hu_harm.log 2>&1
echo ">>> $(date -Is)  combination study finished (rc=$?)"
