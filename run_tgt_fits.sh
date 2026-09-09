#!/usr/bin/env bash
# Extract + fit the split-targeted sets, arm by arm, under the SAME protocol as every other
# arm in these CSVs: own 50-row base, n=540 x 8 draws plus one full n=600 fit, all splits
# scored. Only the set preparation differs, which is the question.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
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

# WORK-STEALING, not strict order. The generator produces arms at very different speeds — a
# no-shot arm carries more dedup pressure and can take 5x a shot arm — and a strict in-order
# consumer leaves the GPU idle waiting for arm k while arms k+1.. are already on disk. This
# loops the whole arm list repeatedly and takes whatever is READY AND UNFITTED, so the card
# only idles when nothing at all is available.
fitted () {   # an arm is done when its 9 rows (one n=600 + eight n=540) are in the CSV
    local base="$1" csv="$2"
    # `grep -c` on a miss prints 0 AND exits 1, so `|| echo 0` appends a SECOND line and the
    # comparison sees "0\n0". Default the variable instead of falling back.
    local n
    [ -f "$csv" ] || return 1
    n=$(grep -c "^${base}," "$csv" 2>/dev/null); n=${n:-0}
    [ "$n" -ge 9 ]
}

pending=1
while [ "$pending" -gt 0 ]; do
    pending=0
    did_work=0
    echo "$ARMS" | while read -r concept split; do
        [ -n "${concept:-}" ] || continue
        for variant in tgtshot tgtnone; do
            set_file="data/${concept}_${TAG}_${variant}_${split}_600.jsonl"
            devflag=""; out="scripts/instructions_gen90.csv"
            if [ "$concept" = highstakes ]; then
                devflag="--dev-data dev_samples/highstakes_500"
                out="scripts/highstakes_gen90_dev500.csv"
            fi
            base="$(basename "$set_file")"
            fitted "$base" "$out" && continue
            [ -s "$set_file" ] || { echo "$concept/$variant/$split" >> "$STATE"; continue; }
            echo ">>> $(date -Is)  extracting $set_file"
            $PY scripts/warm_set_activations.py --concept "$concept" "$set_file" \
                >> "logs/warm_tgt_${concept}.log" 2>&1 \
                || echo ">>> $(date -Is)  EXTRACTION FAILED for $set_file"
            for spec in "600 1" "540 8"; do
                set -- $spec
                echo ">>> $(date -Is)  fitting ${variant}/${split} sizes=$1 draws=$2"
                $PY scripts/subsample_curve_concept.py --concept "$concept" "$set_file" \
                    --base-data "data/${concept}_${TAG}_50.jsonl" $devflag \
                    --sizes "$1" --draws "$2" --out "$out" \
                    >> "logs/fit_tgt_${concept}.log" 2>&1 \
                    || echo ">>> $(date -Is)  FIT FAILED for ${variant}/${split} ($1/$2)"
            done
            echo ">>> $(date -Is)  arm done: $concept/$variant/$split"
        done
    done
    # The `while read` runs in a subshell, so the count of still-ungenerated arms comes back
    # through a file rather than a variable.
    pending=$(wc -l < "$STATE" 2>/dev/null || echo 0)
    : > "$STATE"
    [ "$pending" -gt 0 ] && { echo ">>> $(date -Is)  $pending arm(s) not yet generated; waiting"; sleep 120; }
done
echo ">>> $(date -Is)  all split-targeted fits finished."
