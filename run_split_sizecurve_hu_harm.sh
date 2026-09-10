#!/usr/bin/env bash
# Size curve over ALL FOUR hu_ha splits' split-targeted sets: 8 balanced draws at 50%, 20%,
# 10% and 5% of each set, scored on that set's own pair of eval splits.
#
# generic_600 is carried at every fraction as the CONTROL. It is not itself split-targeted,
# so it is an addition to the ask — but the whole lesson of 1c8040e1 is that the targeted
# sets only mean something against a same-generator set written without the description, and
# a curve of targeted data alone cannot say whether targeting is worth it at any size.
#
# Draws are CLASS-BALANCED here (see draw_subset): 5% of 600 is 30 rows, and a uniform draw
# would let the class ratio wander alongside the size being measured. They are tagged f{pct}b
# and carry balanced:true so they are never confused with the earlier uniform f90 rows.
#
# Every row of every set is already in the per-conversation activation cache, so this loads
# no model: it is ~224 probe-head fits on cached activations plus cached evals.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -f .env2 ]; then set -a; . ./.env2; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
FRACS="0.5 0.2 0.1 0.05"
mkdir -p logs

echo ">>> $(date -Is)  pair 1: ant_hh / balanced_refusal"
$PY scripts/fit_hu_harm_split_targeted.py --pair refusal --skip-full --balanced \
    --fracs $FRACS --draws 8 \
    --conditions ant_hh_600 refusal_v2_600 refusal_v1_600_CONTAMINATED generic_600 \
    >> logs/sizecurve_refusal.log 2>&1

echo ">>> $(date -Is)  pair 2: ai_dilemmas / daily_dilemmas"
$PY scripts/fit_hu_harm_split_targeted.py --pair dilemmas --skip-full --balanced \
    --fracs $FRACS --draws 8 \
    --conditions ai_dilemmas_600 daily_dilemmas_600 generic_600 \
    >> logs/sizecurve_dilemmas.log 2>&1

echo ">>> $(date -Is)  size curve finished."
