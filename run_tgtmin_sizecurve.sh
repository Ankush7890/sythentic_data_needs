#!/usr/bin/env bash
# Size curve for the six SHAPE-FREE instruction sets: how few of the 600 rows are needed?
#
# Sizes are the requested fractions of 600 — 50% 20% 10% 5% 2.5% — plus a bare n=10, eight
# independent class-balanced draws each. 6 sets x 6 sizes x 8 draws = 288 fits.
#
# Each fit is `own 50-row base + n drawn rows`, so n=10 still trains on 60 rows and clears
# tuberlens' ~49-row optimizer-step threshold; below that the pytorch probe never takes a
# step and the number would be meaningless rather than small.
#
# Activations are already cached for all six sets from the 90-fit run, so this is probe-head
# fits on warm caches throughout — no extraction.
#
# WAITS for the family-B backfill to exit: one 24 GB card, one model at a time. The pattern
# is anchored on the invocation, not the bare script name — an unanchored `pgrep -f` also
# matches a shell that merely QUOTES the name, which stalled the backfill for 100 minutes.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG="${TAG:-deepseekv4pro}"
OUT="${OUT:-scripts/instructions_tgtmin_size_curve.csv}"
SIZES="${SIZES:-300 120 60 30 15 10}"

while pgrep -f "bash \./run_tgtnone_backfill\.sh" > /dev/null; do
    sleep 60
done
echo ">>> $(date -Is)  card free; starting the size curve -> $OUT"

SPLITS="anthropic_harmless_refusal bbq_substitution hc_context_drift hc_contradiction
mm_substitution oig_context_drift"

for split in $SPLITS; do
    set_file="data/instructions_${TAG}_tgtmin_${split}_600.jsonl"
    [ -s "$set_file" ] || { echo ">>> $(date -Is)  MISSING $set_file"; continue; }
    echo ">>> $(date -Is)  size curve: $split"
    # One call per set, all sizes at once: the script resumes on (base, samples, n, draw), so
    # a kill costs at most the fit in flight and a re-run is a no-op.
    $PY scripts/subsample_curve_concept.py --concept instructions "$set_file" \
        --base-data "data/instructions_${TAG}_50.jsonl" \
        --sizes $SIZES --draws 8 --out "$OUT" \
        >> logs/sizecurve_tgtmin_instructions.log 2>&1 \
        || echo ">>> $(date -Is)  CURVE FAILED for $split"
    echo ">>> $(date -Is)  size curve done: $split ($(grep -c "_tgtmin_${split}_" "$OUT" 2>/dev/null || echo 0) rows)"
done
echo ">>> $(date -Is)  all shape-free size curves finished."
