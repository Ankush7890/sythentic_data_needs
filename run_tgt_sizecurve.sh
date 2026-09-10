#!/usr/bin/env bash
# SIZE CURVE on the split-targeted NO-SHOT sets: 8 draws per size for all ten splits.
# Default sizes 300/120/60/30 (50/20/10/5% of 600); override with SIZES=, as the 2.5%
# point (n=15) was.
#
# No extraction: every row of these sets is already in the per-conversation activation cache
# from the full-size arms, and the cache is keyed on the conversation, so a subset is a pure
# hit. Same protocol as every other arm here — own 50-row base, class-balanced draws seeded
# on (stem, n, draw), all splits scored, appended to the same two CSVs.
#
# Waits for the main fit driver to exit so the two never contend for the GPU.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG=deepseekv4pro
SIZES="${SIZES:-300 120 60 30}"

while pgrep -f "[r]un_tgt_fits.sh" >/dev/null; do sleep 120; done
echo ">>> $(date -Is)  main fits finished; starting size curve"

ARMS="
instructions anthropic_harmless_refusal
instructions bbq_substitution
instructions hc_context_drift
instructions hc_contradiction
instructions mm_substitution
instructions oig_context_drift
highstakes anthropic_hh_balanced
highstakes mt_balanced
highstakes mts_balanced
highstakes toolace_balanced
"

echo "$ARMS" | while read -r concept split; do
    [ -n "${concept:-}" ] || continue
    set_file="data/${concept}_${TAG}_tgtnone_${split}_600.jsonl"
    [ -s "$set_file" ] || { echo ">>> $(date -Is)  MISSING $set_file"; continue; }
    devflag=""; out="scripts/instructions_gen90.csv"
    if [ "$concept" = highstakes ]; then
        devflag="--dev-data dev_samples/highstakes_500"
        out="scripts/highstakes_gen90_dev500.csv"
    fi
    for n in $SIZES; do
        echo ">>> $(date -Is)  size curve ${split} n=${n} draws=8"
        $PY scripts/subsample_curve_concept.py --concept "$concept" "$set_file" \
            --base-data "data/${concept}_${TAG}_50.jsonl" $devflag \
            --sizes "$n" --draws 8 --out "$out" \
            >> "logs/sizecurve_${concept}.log" 2>&1 \
            || echo ">>> $(date -Is)  FIT FAILED ${split} n=${n}"
    done
    echo ">>> $(date -Is)  size curve done: $concept/$split"
done
echo ">>> $(date -Is)  all size-curve fits finished."
