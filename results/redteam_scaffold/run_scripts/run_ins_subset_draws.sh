#!/usr/bin/env bash
set -e
# ATTACKER-SUBSET sweep (FIXED 200-row base): every 1-, 2- and 3-attacker combination, in
# each of the three configurations, resampled at 90%. With the 4-attacker grid from
# run_ins_combined_draws.sh, this fills in the k-curve: does pooling more attackers' red-team
# data help, and does the configuration effect hold at every k, or only when all four are
# combined?
#
# 4 singletons + 6 pairs + 4 triples = 14 subsets per configuration, 42 in all. FOUR draws
# each rather than eight: the k-curve's error bars come from the subsets in a cell as much as
# from the draws within one, so 4 x 6 = 24 fits per (configuration, k=2) cell buys more than
# 8 x 3 would. fit_combined_draws.py resumes on (combo, draw), so deepening any cell to 8
# later is a re-run of this script with DRAWS=8, not a recompute.
#
# The base is the same 200 rows at every k — see build_combos() for why that is the right
# control and not a shortcut. run_ins_subsetbase_draws.sh is the companion design that
# scales the base with k instead.
#
# ONE PROCESS PER (combo, draw), as in run_ins_combined_draws.sh: the per-draw leak scales
# with the activation set. Ordered by k, then configuration, so an interrupted sweep still
# leaves whole cells rather than fragments of all of them.
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-4}"

SINGLES="g d l n"
PAIRS="gd gl gn dl dn ln"
TRIPLES="gdl gdn gln dln"
for codes_set in "$SINGLES" "$PAIRS" "$TRIPLES"; do
    for g in memo desc att; do
        for codes in $codes_set; do
            for d in $(seq 0 $((DRAWS-1))); do
                echo ">>> $(date -Is)  ${g}_${codes} draw $d"
                $PY scripts/fit_combined_draws.py --combos "${g}_${codes}" --draws $((d+1)) --fraction 0.9
            done
        done
    done
done
echo ">>> $(date -Is)  subset sweep finished."
$PY scripts/fit_combined_draws.py --sizes 1 2 3 4 --draws 0 --fraction 0.9
