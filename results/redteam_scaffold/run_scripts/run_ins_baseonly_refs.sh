#!/usr/bin/env bash
set -e
# The no-red-team REFERENCE probe for each subset base (50 rows for a singleton, 100 for a
# pair, 150 for a triple, 200 for the whole rotation). Every red-team cell is read against
# what its own base trains alone, and the 200-row reference is the wrong yardstick for a
# 100-row cell — so without these the "did red-teaming help at all?" question cannot be
# asked on the subset-base grid.
#
# The four SINGLETON fits also extract the four per-attacker base-activation blobs, which is
# what build_subset_base_activations.py merges the pair/triple blobs out of. So this script
# runs BEFORE the subset-base sweep, not after it.
#
# 14 fits, only 200 forward passes in total (the four 50-row cuts; every other base is a
# merge of those).
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python

while pgrep -f "fit_combined_draws\.py --combos" >/dev/null 2>&1; do sleep 60; done
echo ">>> $(date -Is)  GPU free; fitting the singleton base-only references (extracts the four 50-row blobs)"
$PY scripts/fit_combined_draws.py --sizes 1 --draws 0 --fraction 0.9 --base-mode subset --base-only

echo ">>> $(date -Is)  merging the pair/triple base blobs"
$PY scripts/build_subset_base_activations.py --verify --sizes 2 3

echo ">>> $(date -Is)  fitting the pair/triple base-only references"
$PY scripts/fit_combined_draws.py --sizes 2 3 --draws 0 --fraction 0.9 --base-mode subset --base-only
echo ">>> $(date -Is)  base-only references finished."
