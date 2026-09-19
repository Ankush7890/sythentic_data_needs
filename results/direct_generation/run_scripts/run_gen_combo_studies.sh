#!/usr/bin/env bash
set -e
# The two GENERATOR combination studies, queued behind the red-teaming one.
#
#   i)  instructions   every subset of the four generators' 600-row sets, pooled, no base
#   ii) highstakes     the same
#
# WAITS FOR THE INSTRUCTION-FOLLOWING COMBINATION STUDY TO FINISH FIRST. One 24 GB card, and
# a second fit on it does not merely halve throughput — the staged training + dev
# activations are sized against what is free at fit time, so an overlapping run pushes both
# host-resident and costs far more than the fit it added.
#
# HIGHSTAKES USES THE 500-ROW DEV SUBSAMPLE. Its full dev set is 1908 rows = 21 GB of
# gemma-27b activations, which takes the whole card and leaves the training rows to be
# gathered from the host every epoch — measured elsewhere in this branch at ~20x the cost of
# the other concepts. Both dev blobs are already cached, so this is a choice about speed
# only, and it is the same dev set <concept>_gen90.csv's highstakes rows were fit against.
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-8}"

echo ">>> $(date -Is)  waiting for the red-team combination study to release the GPU"
while pgrep -f "run_ins_(combined_draws|study_rest|baseonly_refs|subsetbase_draws|subset_draws)\.sh" >/dev/null 2>&1 \
   || pgrep -f "fit_combined_draws\.py" >/dev/null 2>&1; do sleep 120; done
echo ">>> $(date -Is)  GPU free"

echo ">>> $(date -Is)  STUDY i: instructions — generator subsets, 600 rows each, no base"
$PY scripts/gen_combo_draws.py --concept instructions --draws "$DRAWS" --fraction 0.9 --full-pool

echo ">>> $(date -Is)  STUDY ii: highstakes — generator subsets, 600 rows each, no base"
$PY scripts/gen_combo_draws.py --concept highstakes --dev-data dev_samples/highstakes_500 \
    --draws "$DRAWS" --fraction 0.9 --full-pool

echo ">>> $(date -Is)  both generator combination studies complete."
