#!/usr/bin/env bash
# Below 60 rows: the shape-free hc_context_drift curve at 10, 20, ... 120, uniform.
#
# hc_context_drift settles at 120 (analysis/instructions_row_settling.md), so the whole
# interesting range is below where the published curve starts. This walks it in even steps.
#
# WHY EVERY POINT HERE RUNS AT ACCUMULATION 1. The inherited spec is batch_size 16 x
# gradient_accumulation_steps 4, and the DataLoader does not drop its last partial batch,
# so with no base data:
#
#       n    batches   steps/epoch @ accum 4   steps/epoch @ accum 1
#      10        1              0                        1
#      20        2              0                        2
#      40        3              0                        3
#      50        4              1                        4
#     120        8              2                        8
#
# Everything under 50 rows takes ZERO optimizer steps at the default and returns the probe
# at INITIALISATION — the gradient is zeroed at the top of every epoch and never applied.
# So the sub-60 range is only measurable at accumulation 1.
#
# And because accumulation 1 is a different optimizer regime, the OVERLAP points (60, 120)
# are run at accumulation 1 too, rather than borrowed from the default-accumulation curve.
# That keeps this curve internally comparable end to end, and leaves two sizes at which it
# can be read against the published one — a comparison to report, not to quietly rely on.
# These go to their own CSV and must not be pooled with the default-accumulation rows.
#
# 12 sizes x 32 draws = 384 fits, one arm. Draw seeding is unchanged, so draw d at size n
# is the same rows as anywhere else in this campaign.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-/home/ubuntu/.kaggle}"
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-32}"
SIZES="${SIZES:-10 20 30 40 50 60 70 80 90 100 110 120}"
mkdir -p logs

SET=data/instructions_deepseekv4pro_tgtmin_hc_context_drift_600.jsonl
[ -s "$SET" ] || { echo ">>> MISSING $SET"; exit 1; }

echo ">>> $(date -Is)  hc_drift small curve: sizes '$SIZES', $DRAWS draws, accumulation 1"
$PY scripts/fit_instructions_parts.py "$SET" --sizes $SIZES --draws "$DRAWS" --grad-accum 1 \
    --out scripts/instructions_hcdrift_small_curve.csv \
    --row-scores data/instructions_row_scores \
    >> logs/hcdrift_small.log 2>&1 \
    || echo ">>> $(date -Is)  FAILED"
echo ">>> $(date -Is)  done ($(grep -c . scripts/instructions_hcdrift_small_curve.csv) lines)"
