#!/usr/bin/env bash
# P2: n=80 at --grad-accum 4 for BOTH arms, all eight high-stakes pools, 8 draws.
#
# WHY. Phase 1 fit n=80 at --grad-accum 5 (the ceil(n/16) small-end rule), so high-stakes is
# the one concept whose n=80 point in Figure 1 is on a different optimizer regime from the
# rest of its own curve — 110/170/350/590 are all ga4. The paper currently falls back to the
# ga5 rows at 80 for this concept only. These 64 fits give 80 a ga4 reading so the whole
# high-stakes curve can be read on one regime, matching hu_harm and instructions.
#
# The existing ga5 rows at n=80 are NOT touched: the resume key is
# (base_label, file, n, draw) and base_label here is `none+ga4`, distinct from `none+ga5`,
# so these 64 fits append alongside them and the paper can use either.
#
# BOTH arms (all eight pools, general and evaldesc), unlike the required block, because the
# two arms are compared against each other at each x and a regime change on one arm only
# would confound that comparison.
#
# Run as two streams of four pools under setsid (general arm and evaldesc arm), which share
# the 24 GB card at ~6.4 GiB of live tensors each.
#
# WATCH WHAT YOU GREP when checking whether a stream is up. `pgrep -af subsample_curve |
# grep -v eval` silently hides the ENTIRE evaldesc arm, because "evaldesc" contains "eval".
# That filter made four healthy evaldesc streams look dead here, each was relaunched, and
# five fits ended up competing for one card at ~122 GB RSS before the mistake was caught.
# Nothing had crashed and nothing needed relaunching. Match on the pool basename instead.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
OUT=scripts/highstakes_pooled_size_curve.csv
LOG=logs/pooled_sizecurve_highstakes.log
DRAWS="${DRAWS:-8}"
mkdir -p logs

for pool in "$@"; do
    P=".pool_work/${pool}"
    [ -s "$P" ] || { echo ">>> $(date -Is)  MISSING $P"; continue; }
    echo ">>> $(date -Is)  P2 n=80 ga4 $(basename "$P")"
    $PY -u scripts/subsample_curve_concept.py --concept highstakes "$P" \
        --no-base --dev-data dev_samples/highstakes_500 --grad-accum 4 \
        --sizes 80 --draws "$DRAWS" --out "$OUT" \
        >> "$LOG" 2>&1 \
        || echo ">>> $(date -Is)  FAILED $(basename "$P")"
done
echo ">>> $(date -Is)  P2 stream finished [$*]"
