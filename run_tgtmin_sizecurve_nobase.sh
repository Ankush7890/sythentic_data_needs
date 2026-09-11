#!/usr/bin/env bash
# Size curve for the SHAPE-FREE sets with NO BASE TRAINING DATA.
#
# WHY NO BASE. Every earlier curve fit `generic 50-row base + n targeted rows`, so at the
# small end the base is most of the training set and the number says more about the mixture
# than about the targeted set. These fits are the drawn rows ALONE.
#
# THE 49-ROW FLOOR, and why this runs two passes. The probe is linear_then_softmax with
# batch_size 16 and gradient_accumulation_steps 4, and it steps the optimizer only on every
# 4th batch of ceil(rows/16). Without the base:
#
#     n=300 -> 19 batches -> 4 steps/epoch        n=30 -> 2 batches -> ZERO steps
#     n=120 ->  8 batches -> 2 steps/epoch        n=15 -> 1 batch   -> ZERO steps
#     n= 60 ->  4 batches -> 1 step /epoch        n=10 -> 1 batch   -> ZERO steps
#
# A fit that never steps returns the probe at INITIALISATION, so at the default accumulation
# the bottom half of this curve would be noise around an untrained probe rather than a
# measurement. Pass 1 runs every size at the default, which keeps it comparable to every
# other number in this campaign; pass 2 re-runs the three degenerate sizes at accum=1, where
# they actually train. The two go to separate CSVs and must not be pooled.
#
# Instructions first, then high-stakes, as asked. Activations are warm for all ten sets.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG="${TAG:-deepseekv4pro}"
SIZES="${SIZES:-300 120 60 30 15 10}"
SMALL="${SMALL:-30 15 10}"

INSTR_SPLITS="anthropic_harmless_refusal bbq_substitution hc_context_drift hc_contradiction
mm_substitution oig_context_drift"
HS_SPLITS="anthropic_hh_balanced mt_balanced mts_balanced toolace_balanced"

run_pass () {   # <concept> <splits> <sizes> <out> [extra flags...]
    local concept="$1" splits="$2" sizes="$3" out="$4"; shift 4
    local devflag=""
    [ "$concept" = highstakes ] && devflag="--dev-data dev_samples/highstakes_500"
    for split in $splits; do
        local set_file="data/${concept}_${TAG}_tgtmin_${split}_600.jsonl"
        [ -s "$set_file" ] || { echo ">>> $(date -Is)  MISSING $set_file"; continue; }
        echo ">>> $(date -Is)  ${concept}/${split} -> $(basename "$out")  sizes='$sizes' $*"
        $PY scripts/subsample_curve_concept.py --concept "$concept" "$set_file" \
            --no-base $devflag "$@" --sizes $sizes --draws 8 --out "$out" \
            >> "logs/sizecurve_nobase_${concept}.log" 2>&1 \
            || echo ">>> $(date -Is)  CURVE FAILED for ${concept}/${split}"
        echo ">>> $(date -Is)  done: ${concept}/${split} ($(grep -c "_tgtmin_${split}_" "$out" 2>/dev/null || echo 0) rows in $(basename "$out"))"
    done
}

run_pass instructions "$INSTR_SPLITS" "$SIZES" scripts/instructions_tgtmin_size_curve_nobase.csv
echo ">>> $(date -Is)  instructions pass 1 (default accumulation) finished."
run_pass instructions "$INSTR_SPLITS" "$SMALL" scripts/instructions_tgtmin_size_curve_nobase_accum1.csv --accum 1
echo ">>> $(date -Is)  instructions finished."

run_pass highstakes "$HS_SPLITS" "$SIZES" scripts/highstakes_tgtmin_size_curve_nobase.csv
echo ">>> $(date -Is)  highstakes pass 1 (default accumulation) finished."
run_pass highstakes "$HS_SPLITS" "$SMALL" scripts/highstakes_tgtmin_size_curve_nobase_accum1.csv --accum 1
echo ">>> $(date -Is)  all no-base size curves finished."
