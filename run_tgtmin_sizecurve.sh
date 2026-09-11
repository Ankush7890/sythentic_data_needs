#!/usr/bin/env bash
# Size curve for the SHAPE-FREE sets of one concept: how few of the 600 rows are needed?
#
#     ./run_tgtmin_sizecurve.sh                      # instructions (six sets, 288 fits)
#     CONCEPT=highstakes ./run_tgtmin_sizecurve.sh   # high-stakes  (four sets, 192 fits)
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
CONCEPT="${CONCEPT:-instructions}"
SIZES="${SIZES:-300 120 60 30 15 10}"
OUT="${OUT:-scripts/${CONCEPT}_tgtmin_size_curve.csv}"

# High-stakes fits validate on the 500-row dev subsample, exactly as its 90-fit run did —
# the full dev_samples/highstakes is 1908 rows and is resident for every fit. Instructions
# takes the concept default. Getting this wrong would make the curve incomparable to the
# n=540 reference rows it is read against.
DEVFLAG=""
[ "$CONCEPT" = highstakes ] && DEVFLAG="--dev-data dev_samples/highstakes_500"

case "$CONCEPT" in
    instructions) SPLITS="anthropic_harmless_refusal bbq_substitution hc_context_drift
                          hc_contradiction mm_substitution oig_context_drift" ;;
    highstakes)   SPLITS="anthropic_hh_balanced mt_balanced mts_balanced toolace_balanced" ;;
    *) echo "unknown CONCEPT=$CONCEPT"; exit 2 ;;
esac

while pgrep -f "bash \./run_tgtnone_backfill\.sh" > /dev/null; do
    sleep 60
done
echo ">>> $(date -Is)  card free; starting the $CONCEPT size curve -> $OUT"

for split in $SPLITS; do
    set_file="data/${CONCEPT}_${TAG}_tgtmin_${split}_600.jsonl"
    [ -s "$set_file" ] || { echo ">>> $(date -Is)  MISSING $set_file"; continue; }
    echo ">>> $(date -Is)  size curve: $split"
    # One call per set, all sizes at once: the script resumes on (base, samples, n, draw), so
    # a kill costs at most the fit in flight and a re-run is a no-op.
    $PY scripts/subsample_curve_concept.py --concept "$CONCEPT" "$set_file" \
        --base-data "data/${CONCEPT}_${TAG}_50.jsonl" $DEVFLAG \
        --sizes $SIZES --draws 8 --out "$OUT" \
        >> "logs/sizecurve_tgtmin_${CONCEPT}.log" 2>&1 \
        || echo ">>> $(date -Is)  CURVE FAILED for $split"
    echo ">>> $(date -Is)  size curve done: $split ($(grep -c "_tgtmin_${split}_" "$OUT" 2>/dev/null || echo 0) rows)"
done
echo ">>> $(date -Is)  all shape-free $CONCEPT size curves finished."
