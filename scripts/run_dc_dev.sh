#!/usr/bin/env bash
# The dev-coverage study, Part A, end to end on one GPU (docs/dev_coverage_task.md).
#
# warm (every dev sample + the three validation blobs, one model load per concept) ->
# fits instruction -> harmful -> high-stakes, each concept's `own` before `others`
# before `all`, then that concept's cells. The fit stage resumes per fit off the
# harness's own CSVs, so re-running this after a kill loses at most one fit.
#
# High-stakes waits for the Kaggle prefetch to finish: its eval blobs are 47 GB and a fit
# that met a half-downloaded split would start a second download into the same staging dir.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv_claude/bin/python
mkdir -p logs

echo ">>> $(date -Is) warm"
$PY scripts/dc_dev.py --stage warm >> logs/dcdev_warm.log 2>&1 || { echo "warm FAILED"; exit 1; }

for c in instructions hu_harm highstakes; do
    if [ "$c" = highstakes ]; then
        echo ">>> $(date -Is) waiting for the Kaggle prefetch"
        while pgrep -f "dc_dev.py --stage prefetch" > /dev/null; do sleep 60; done
    fi
    echo ">>> $(date -Is) fits: $c"
    $PY scripts/dc_dev.py --stage fit --concepts "$c" >> "logs/dcdev_fit_$c.log" 2>&1 \
        || echo ">>> $(date -Is) fit stage for $c exited non-zero"
    $PY scripts/dc_dev.py --stage cells --concepts "$c" >> "logs/dcdev_cells.log" 2>&1 \
        || echo ">>> $(date -Is) cells for $c FAILED"
    echo ">>> $(date -Is) done: $c"
done
echo ">>> $(date -Is) Part A done"
