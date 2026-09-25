#!/usr/bin/env bash
# Part C of docs/dev_coverage_task.md (instruction only), after Part A's fits free the GPU.
# Part B's within-instruction |rho| for G_dev is 0.77 >= 0.5, which is the brief's trigger.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv_claude/bin/python
until grep -q "Part A done" logs/dcdev_chain.log; do sleep 60; done
echo ">>> $(date -Is) Part B (before Part C)"
$PY scripts/dc_dev.py --stage link >> logs/dcdev_link.log 2>&1
echo ">>> $(date -Is) Part C fits"
$PY scripts/dc_dev.py --stage partc-fit >> logs/dcdev_partc_fit.log 2>&1
echo ">>> $(date -Is) Part C analyse"
$PY scripts/dc_dev.py --stage partc-analyse >> logs/dcdev_partc_analyse.log 2>&1
$PY scripts/dc_dev.py --stage link >> logs/dcdev_link.log 2>&1
echo ">>> $(date -Is) Part C done"
