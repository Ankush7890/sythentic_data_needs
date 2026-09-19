#!/usr/bin/env bash
# The sub-60 curve for the other two arms, so the takeoff analysis has the same footing
# hc_context_drift got: 10, 20, ... 120 in even steps, all at accumulation 1.
#
# run_hcdrift_small_curve.sh documents why every point below 60 must run at accumulation 1
# (the default takes ZERO optimizer steps under ~49 rows and returns the probe at
# initialisation) and why the overlap points are re-run here rather than borrowed from the
# default-accumulation curve.
#
# THE SPACING IS GEOMETRIC, NOT UNIFORM. A learning curve is roughly linear in log n, so a
# uniform grid spends most of its fits where the curve is already flat: on hc_context_drift
# the twelve uniform points bought four nearly-identical readings between 80 and 120 and
# only three below 40, which is where a sixth of the split had already taken off. These
# eight sizes step by ~1.4x instead
#
#     10  15  20  30  40  60  80  120
#
# which puts four points under 40, covers the same 10-120 range, and costs 8 sizes rather
# than 12 -- 512 fits instead of 768. Seven of the eight also sit on hc_context_drift's
# uniform grid, so the three arms stay directly comparable size by size.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-/home/ubuntu/.kaggle}"
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-32}"
SIZES="${SIZES:-10 15 20 30 40 60 80 120}"
mkdir -p logs

SETS="${SETS:-data/instructions_deepseekv4pro_tgtmin_oig_context_drift_600.jsonl
data/instructions_deepseekv4pro_tgtnone_mm_substitution_600.jsonl}"

for f in $SETS; do
    [ -s "$f" ] || { echo ">>> MISSING $f"; exit 1; }
done

echo ">>> $(date -Is)  small curve: sizes '$SIZES', $DRAWS draws, accumulation 1"
$PY scripts/fit_instructions_parts.py $SETS --sizes $SIZES --draws "$DRAWS" --grad-accum 1 \
    --out scripts/instructions_small_curve.csv \
    --row-scores data/instructions_row_scores \
    >> logs/instrsmall.log 2>&1 \
    || echo ">>> $(date -Is)  FAILED"
echo ">>> $(date -Is)  done ($(grep -c . scripts/instructions_small_curve.csv) lines)"
