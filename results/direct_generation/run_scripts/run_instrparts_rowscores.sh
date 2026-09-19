#!/usr/bin/env bash
# Per-ROW scores for the three arms the settling analysis needs.
#
#   shape-free (tgtmin)   hc_context_drift, oig_context_drift
#   + shape info (tgtnone) mm_substitution
#
# 4 sizes x 16 draws x 3 arms = 192 fits. Sixteen draws, not eight: a per-SAMPLE placement
# value has a much wider across-draw spread than a 50-row part AUROC, and the settling
# threshold has to be read in units of that spread. Eight draws puts the per-row standard
# error around 0.05-0.09, which is too coarse to call a settling point on a single row.
#
# Draws 0..7 are the SAME row subsets as scripts/instructions_parts_size_curve.csv (the
# draw key is <stem>:<n>:<draw> and nothing about it changed), so the overlap is a direct
# reproduction check on the published curves — see scripts/report_rowscores.py --repro.
# Draws 8..15 are new.
#
# n=30 is deliberately absent. It ran at accumulation 1, a different optimizer regime, and
# must not sit on the same curve; run_instrparts_fits.sh documents why.
#
# Default accumulation throughout, so every fit here is comparable to the published points.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-/home/ubuntu/.kaggle}"
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-16}"
SIZES="${SIZES:-540 300 120 60}"
mkdir -p logs

SETS="data/instructions_deepseekv4pro_tgtmin_hc_context_drift_600.jsonl
data/instructions_deepseekv4pro_tgtmin_oig_context_drift_600.jsonl
data/instructions_deepseekv4pro_tgtnone_mm_substitution_600.jsonl"

for f in $SETS; do
    [ -s "$f" ] || { echo ">>> $(date -Is)  MISSING $f"; exit 1; }
done

echo ">>> $(date -Is)  row-score fits: sizes '$SIZES', $DRAWS draws, 3 arms"
$PY scripts/fit_instructions_parts.py $SETS --sizes $SIZES --draws "$DRAWS" \
    --out scripts/instructions_rowscores_size_curve.csv \
    --row-scores data/instructions_row_scores \
    >> logs/instrparts_rowscores.log 2>&1 \
    || echo ">>> $(date -Is)  ROW-SCORE PASS FAILED"
echo ">>> $(date -Is)  done ($(grep -c . scripts/instructions_rowscores_size_curve.csv) lines,"\
     "$(find data/instructions_row_scores -name '*.npz' | wc -l) score files)"
