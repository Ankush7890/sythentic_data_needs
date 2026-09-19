#!/usr/bin/env bash
# The instructions per-part size curve: five split-targeted sets, five sizes, eight draws.
#
# THE ARMS (all deepseek-v4-pro, all fit on the drawn rows ALONE — no base data):
#   + shape info (tgtnone)  hc_context_drift, hc_contradiction, mm_substitution
#   shape-free   (tgtmin)   hc_context_drift, oig_context_drift
#
# THE GRID. 30 60 120 300 540 rows, 8 class-balanced draws each = 200 fits. Run in two
# passes for the reason run_tgtmin_sizecurve_nobase.sh documents: the probe is
# batch_size 16 x gradient_accumulation_steps 4, so with no base data
#
#     n=540 -> 34 batches -> 8 steps/epoch      n=60 ->  4 batches -> 1 step /epoch
#     n=300 -> 19 batches -> 4 steps/epoch      n=30 ->  2 batches -> ZERO steps
#     n=120 ->  8 batches -> 2 steps/epoch
#
# so n=30 at the default returns the probe at INITIALISATION. Pass 1 runs 60..540 at the
# default, which keeps them comparable to every other no-base number in this campaign;
# pass 2 runs n=30 at accumulation 1, where it actually trains. The two go to separate
# CSVs and MUST NOT be pooled.
#
# Every fit also scores the four equal-size parts of each of the four cut eval splits
# (scripts/make_instructions_parts.py), from the same forward pass as the split's own AUROC.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-/home/ubuntu/.kaggle}"
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-8}"
BIG="${BIG:-540 300 120 60}"
SMALL="${SMALL:-30}"
mkdir -p logs

SETS="data/instructions_deepseekv4pro_tgtnone_hc_context_drift_600.jsonl
data/instructions_deepseekv4pro_tgtnone_hc_contradiction_600.jsonl
data/instructions_deepseekv4pro_tgtnone_mm_substitution_600.jsonl
data/instructions_deepseekv4pro_tgtmin_hc_context_drift_600.jsonl
data/instructions_deepseekv4pro_tgtmin_oig_context_drift_600.jsonl"

for f in $SETS; do
    [ -s "$f" ] || { echo ">>> $(date -Is)  MISSING $f"; exit 1; }
done

echo ">>> $(date -Is)  pass 1 (default accumulation): sizes '$BIG', $DRAWS draws"
$PY scripts/fit_instructions_parts.py $SETS --sizes $BIG --draws "$DRAWS" \
    --out scripts/instructions_parts_size_curve.csv \
    >> logs/instrparts_fits.log 2>&1 \
    || echo ">>> $(date -Is)  PASS 1 FAILED"
echo ">>> $(date -Is)  pass 1 done ($(grep -c . scripts/instructions_parts_size_curve.csv) lines)"

echo ">>> $(date -Is)  pass 2 (accumulation 1): sizes '$SMALL', $DRAWS draws"
$PY scripts/fit_instructions_parts.py $SETS --sizes $SMALL --draws "$DRAWS" --grad-accum 1 \
    --out scripts/instructions_parts_size_curve_accum1.csv \
    >> logs/instrparts_fits.log 2>&1 \
    || echo ">>> $(date -Is)  PASS 2 FAILED"
echo ">>> $(date -Is)  all instructions part-curve fits finished."
