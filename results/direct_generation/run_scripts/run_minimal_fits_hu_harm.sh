#!/usr/bin/env bash
# ARM 3 FITS. Same protocol as the clean ablation: generated-rows-only fits, pair-restricted
# dev, full fit + 4 draws at 90%, --accum 4, every fit scored on ALL FOUR eval splits.
# One pair at a time — a single GPU, and each pair warms its own activation cache.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
mkdir -p logs

echo ">>> $(date -Is)  refusal pair: request_minimal x2 (ONE prompt, two draws)"
$PY scripts/fit_hu_harm_split_targeted.py --pair refusal --accum 4 \
    --conditions request_minimal_600 request_minimal2_600 \
    >> logs/fit_minimal_hu_harm.log 2>&1
echo ">>> $(date -Is)  refusal pair done (rc=$?)"

echo ">>> $(date -Is)  dilemmas pair: ai_dilemmas_minimal + daily_dilemmas_minimal"
$PY scripts/fit_hu_harm_split_targeted.py --pair dilemmas --accum 4 \
    --conditions ai_dilemmas_minimal_600 daily_dilemmas_minimal_600 \
    >> logs/fit_minimal_hu_harm.log 2>&1
echo ">>> $(date -Is)  dilemmas pair done (rc=$?)"
echo ">>> $(date -Is)  arm 3 fits finished."
