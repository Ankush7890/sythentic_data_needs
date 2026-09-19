#!/usr/bin/env bash
# SIZE CURVE, TINY END: n=15 and n=10, arms 2 (cleanPL) and 3 (minimal).
#
# --accum 1 IS MANDATORY HERE, and it is not a free choice. tuberlens steps the optimizer
# only when (batch_idx + 1) % gradient_accumulation_steps == 0. At batch_size=16 a 15-row or
# 10-row set is ONE batch per epoch, so accum 2 (which was correct at n=30, two batches)
# would give zero optimizer steps and the fit would silently return its seeded
# initialisation — the sd=0 failure from earlier in this campaign. accum 1 gives exactly one
# step per epoch accumulated over the whole training set: the same regime as n=30/accum 2
# and n=60/accum 4.
#
#   n=60  accum 4 -> 4 batches -> 1 step/epoch
#   n=30  accum 2 -> 2 batches -> 1 step/epoch
#   n=15  accum 1 -> 1 batch   -> 1 step/epoch
#   n=10  accum 1 -> 1 batch   -> 1 step/epoch
#
# --sizes (not --fracs) because 15/600 and 10/600 both round to the same f{pct} tag; results
# are tagged n15b / n10b instead.
#
# 7 conditions x 2 sizes x 8 draws = 112 fits.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
mkdir -p logs
LOG=logs/sizecurve_tiny_hu_harm.log

REFUSAL_CONDS="refusal_cleanPL_600 request_minimal_600 request_minimal2_600"
DILEMMA_CONDS="ai_dilemmas_cleanPL_600 daily_dilemmas_cleanPL_600 ai_dilemmas_minimal_600 daily_dilemmas_minimal_600"

for spec in "refusal:$REFUSAL_CONDS" "dilemmas:$DILEMMA_CONDS"; do
    pair=${spec%%:*}; conds=${spec#*:}
    echo ">>> $(date -Is)  $pair n=15,10 (accum 1)"
    $PY scripts/fit_hu_harm_split_targeted.py --pair "$pair" --conditions $conds \
        --sizes 15 10 --draws 8 --balanced --accum 1 --skip-full >> "$LOG" 2>&1
    echo ">>> $(date -Is)  $pair done (rc=$?)"
done
echo ">>> $(date -Is)  tiny size curve finished."
