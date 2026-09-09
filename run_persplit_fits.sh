#!/usr/bin/env bash
# Extract + fit the per-split steered sets, ARM BY ARM: wait for a set, extract it, fit it,
# move on. Generation of arm k+1 runs against the extraction and fits of arm k, and results
# land per arm instead of all at the end.
#
# Protocol is identical to the mixed steered arms (own 50-row deepseek base, n=540 x 8 draws
# plus one full n=600 fit, appended to the same two CSVs), so a per-split arm can be read
# against the mixed arm and against the unsteered arm without refitting anything.
#
# Every fit scores ALL splits, so the CSV holds the full transfer matrix: what a set written
# for one split does to the other six.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG="${TAG:-deepseekv4pro}"

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
    set_file="data/${concept}_${TAG}_${split}_600.jsonl"
    while [ ! -s "$set_file" ]; do sleep 120; done
    devflag=""; out="scripts/instructions_gen90.csv"
    if [ "$concept" = highstakes ]; then
        devflag="--dev-data dev_samples/highstakes_500"
        out="scripts/highstakes_gen90_dev500.csv"
    fi
    echo ">>> $(date -Is)  extracting $set_file"
    $PY scripts/warm_set_activations.py --concept "$concept" "$set_file" \
        >> "logs/warm_persplit_${concept}.log" 2>&1 \
        || echo ">>> $(date -Is)  EXTRACTION FAILED for $set_file"
    for spec in "600 1" "540 8"; do
        set -- $spec
        echo ">>> $(date -Is)  fitting $split sizes=$1 draws=$2"
        $PY scripts/subsample_curve_concept.py --concept "$concept" "$set_file" \
            --base-data "data/${concept}_${TAG}_50.jsonl" $devflag \
            --sizes "$1" --draws "$2" --out "$out" \
            >> "logs/fit_persplit_${concept}.log" 2>&1 \
            || echo ">>> $(date -Is)  FIT FAILED for $split ($1/$2)"
    done
    echo ">>> $(date -Is)  arm done: $concept/$split"
done
echo ">>> $(date -Is)  all per-split fits finished."
