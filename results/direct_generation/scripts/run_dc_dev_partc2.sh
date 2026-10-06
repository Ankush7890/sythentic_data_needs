#!/usr/bin/env bash
# Part C under harmful then high-stakes (docs/dev_coverage_partc_task.md), then the share
# and Part B with R_dev over all fourteen splits. Fits skip cells already in a CSV.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv_claude/bin/python
for c in hu_harm highstakes; do
  echo ">>> $(date -Is) Part C fits $c"
  $PY scripts/dc_dev.py --stage partc-fit --concept $c >> logs/dcdev_partc2_fit_$c.log 2>&1
  echo ">>> $(date -Is) Part C analyse $c"
  $PY scripts/dc_dev.py --stage partc-analyse --concept $c >> logs/dcdev_partc2_analyse.log 2>&1
  echo ">>> $(date -Is) Part C $c done"
done
$PY scripts/dc_dev.py --stage partc-share >> logs/dcdev_partc2_share.log 2>&1
$PY scripts/dc_dev.py --stage link >> logs/dcdev_link.log 2>&1
echo ">>> $(date -Is) Part C2 done"
