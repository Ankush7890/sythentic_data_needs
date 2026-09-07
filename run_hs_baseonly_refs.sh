#!/usr/bin/env bash
set -e
# The no-red-team REFERENCE probe for each subset base (100 rows for a pair, 150 for a
# triple, 200 for the whole rotation). Every red-team cell is read against what its own base
# trains alone, and the 200-row reference is the wrong yardstick for a 100-row cell — so
# without these the "did red-teaming help at all?" question cannot be asked on the
# subset-base grid.
#
# Ten fits, no extraction (the merged base blobs are already cached), ~40 min. Runs only
# after the subset-base sweep releases the GPU: two fits on one 24 GB card will OOM.
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
# Wait until no fit is actually running. Deliberately NOT `kill -0 $SWEEP_PID`: that
# succeeds on a ZOMBIE too, and the sweep's parent shell exits without reaping it, so the
# loop waited forever on a process that had already finished (10 min of idle GPU).
# Presence of a real worker process is the thing we actually care about.
while pgrep -f "fit_combined_draws\.py --combos" >/dev/null 2>&1; do sleep 60; done
echo ">>> $(date -Is)  subset-base sweep done; fitting base-only references"
# --draws 0 means no red-team cells; --base-only is what does the work here.
.venv_claude/bin/python scripts/fit_combined_draws.py --sizes 2 3 --draws 0 \
    --fraction 0.9 --base-mode subset --base-only
echo ">>> $(date -Is)  base-only references finished."
