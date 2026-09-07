#!/usr/bin/env bash
set -e
# The PRACTITIONER-FAITHFUL variant of the subset sweep: a k-attacker subset trains on just
# those k attackers' 50-row base cuts (50*k rows), not on all 200. That is what someone who
# had actually run those k attackers would have had.
#
# The companion run (run_hs_subset_draws.sh, --base-mode fixed) holds the base at 200 rows
# for every k so that ONLY the red-team pool varies — the controlled comparison. Both are
# worth having: the fixed-base grid isolates the effect of the red-team data, this one
# reports what the whole pipeline would have produced. They agree by construction at k=4.
#
# NO EXTRACTION. scripts/build_subset_base_activations.py has already written every subset's
# base-activation blob by MERGING the four cached per-attacker blobs — verified bit-identical
# against the independently extracted 200-row blob. The red-team activations (4573 blobs) and
# the contrastive pairs were already on disk. So this run loads no model at all.
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-4}"

# Wait for the fixed-base sweep to finish — one GPU, so they cannot overlap. Waiting on
# its PID rather than a pgrep pattern: the pattern would also match this script's own
# command line and any shell that happens to quote it.
FIXED_PID="${FIXED_PID:-0}"
while [ "$FIXED_PID" != 0 ] && kill -0 "$FIXED_PID" 2>/dev/null; do sleep 60; done
while pgrep -f "fit_combined_draws\.py --combos" >/dev/null 2>&1; do sleep 60; done
echo ">>> $(date -Is)  fixed-base sweep done; starting subset-base sweep"

for codes_set in "gd gl dl gn dn ln" "gdl gdn gln dln"; do
    for g in memo desc att; do
        for codes in $codes_set; do
            for d in $(seq 0 $((DRAWS-1))); do
                echo ">>> $(date -Is)  ${g}_${codes} draw $d (subset base)"
                $PY scripts/fit_combined_draws.py --combos "${g}_${codes}" \
                    --draws $((d+1)) --fraction 0.9 --base-mode subset
            done
        done
    done
done
echo ">>> $(date -Is)  subset-base sweep finished."
$PY scripts/fit_combined_draws.py --sizes 2 3 --draws 0 --fraction 0.9 --base-mode subset
