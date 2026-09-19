#!/usr/bin/env bash
# Extract + fit + eval the SHAPE-FREE (tgtmin) sets, under the SAME protocol as every other
# arm in these CSVs: own 50-row base, one full n=600 fit plus n=540 x 8 draws, every dev and
# eval split scored. Only the set preparation differs, which is the question.
#
# Runs CONCURRENTLY with run_tgt_minimal_gen.sh: generation is API-bound and this is
# GPU-bound, so they do not contend. Ten arms x 9 fits = 90 rows when complete.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG="${TAG:-deepseekv4pro}"
STATE="$(mktemp)"; trap 'rm -f "$STATE"' EXIT

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

# WORK-STEALING, not strict order. The generator produces arms at very different speeds, and
# a strict in-order consumer leaves the GPU idle waiting for arm k while arms k+1.. are
# already on disk. This loops the whole arm list repeatedly and takes whatever is READY AND
# UNFITTED, so the card only idles when nothing at all is available.
fitted () {   # an arm is done when its 9 rows (one n=600 + eight n=540) are in the CSV
    local base="$1" csv="$2" n
    [ -f "$csv" ] || return 1
    # `grep -c` on a miss prints 0 AND exits 1, so `|| echo 0` would append a SECOND line and
    # the comparison would see "0\n0". Default the variable instead of falling back.
    n=$(grep -c "^${base}," "$csv" 2>/dev/null); n=${n:-0}
    [ "$n" -ge 9 ]
}

settled () {  # the generator writes its output in one go at the END, but a set caught
              # mid-write would be fitted short and silently. 30 s of no growth is settled.
    local f="$1" age
    age=$(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || echo 0) ))
    [ "$age" -ge 30 ]
}

pending=1
while [ "$pending" -gt 0 ]; do
    echo "$ARMS" | while read -r concept split; do
        [ -n "${concept:-}" ] || continue
        set_file="data/${concept}_${TAG}_tgtmin_${split}_600.jsonl"
        devflag=""; out="scripts/instructions_gen90.csv"
        if [ "$concept" = highstakes ]; then
            devflag="--dev-data dev_samples/highstakes_500"
            out="scripts/highstakes_gen90_dev500.csv"
        fi
        base="$(basename "$set_file")"
        fitted "$base" "$out" && continue
        [ -s "$set_file" ] && settled "$set_file" \
            || { echo "$concept/$split" >> "$STATE"; continue; }
        echo ">>> $(date -Is)  extracting $set_file ($(wc -l < "$set_file") rows)"
        $PY scripts/warm_set_activations.py --concept "$concept" "$set_file" \
            >> "logs/warm_tgtmin_${concept}.log" 2>&1 \
            || echo ">>> $(date -Is)  EXTRACTION FAILED for $set_file"
        for spec in "600 1" "540 8"; do
            set -- $spec
            echo ">>> $(date -Is)  fitting tgtmin/${split} sizes=$1 draws=$2"
            $PY scripts/subsample_curve_concept.py --concept "$concept" "$set_file" \
                --base-data "data/${concept}_${TAG}_50.jsonl" $devflag \
                --sizes "$1" --draws "$2" --out "$out" \
                >> "logs/fit_tgtmin_${concept}.log" 2>&1 \
                || echo ">>> $(date -Is)  FIT FAILED for tgtmin/${split} ($1/$2)"
        done
        echo ">>> $(date -Is)  arm done: $concept/tgtmin/$split"
    done
    # The `while read` runs in a subshell, so the count of arms not yet ready comes back
    # through a file rather than a variable.
    pending=$(wc -l < "$STATE" 2>/dev/null || echo 0)
    : > "$STATE"
    [ "$pending" -gt 0 ] && { echo ">>> $(date -Is)  $pending arm(s) not yet ready; waiting"; sleep 120; }
done
echo ">>> $(date -Is)  all shape-free fits finished."
