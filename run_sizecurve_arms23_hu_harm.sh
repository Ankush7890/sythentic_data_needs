#!/usr/bin/env bash
# SIZE CURVE for arm 2 (cleanPL) and arm 3 (minimal), human-harm concept.
#
# The same 8-draw resampling protocol already recorded for the arm-1 targeted sets and the
# unsteered generic control: class-BALANCED draws at 50 / 20 / 10 / 5 % of each 600-row set,
# 8 independent draws per size, generated-rows-only fits, pair-restricted dev, every fit
# scored on ALL FOUR eval splits. --skip-full because the 100% fits are already committed.
#
# --accum: 4 everywhere EXCEPT 5%. At 5% a set is 30 rows and the inherited
# gradient_accumulation_steps of 4 means batches/epoch < accum, so optimizer.step() is never
# called and the fit returns its initialisation — the degenerate result that produced sd=0
# the first time this curve was run. The threshold is 49 rows; accum 2 clears it.
#
# 7 conditions x 4 sizes x 8 draws = 224 fits.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
mkdir -p logs
LOG=logs/sizecurve_arms23_hu_harm.log

REFUSAL_CONDS="refusal_cleanPL_600 request_minimal_600 request_minimal2_600"
DILEMMA_CONDS="ai_dilemmas_cleanPL_600 daily_dilemmas_cleanPL_600 ai_dilemmas_minimal_600 daily_dilemmas_minimal_600"

curve () {  # pair conds
    local pair=$1 conds=$2
    echo ">>> $(date -Is)  $pair 50/20/10% (accum 4)"
    $PY scripts/fit_hu_harm_split_targeted.py --pair "$pair" --conditions $conds \
        --fracs 0.5 0.2 0.1 --draws 8 --balanced --accum 4 --skip-full >> "$LOG" 2>&1
    echo ">>> $(date -Is)  $pair 50/20/10% done (rc=$?)"
    echo ">>> $(date -Is)  $pair 5% (accum 2)"
    $PY scripts/fit_hu_harm_split_targeted.py --pair "$pair" --conditions $conds \
        --fracs 0.05 --draws 8 --balanced --accum 2 --skip-full >> "$LOG" 2>&1
    echo ">>> $(date -Is)  $pair 5% done (rc=$?)"
}

curve refusal  "$REFUSAL_CONDS"
curve dilemmas "$DILEMMA_CONDS"
echo ">>> $(date -Is)  arms 2/3 size curve finished."
