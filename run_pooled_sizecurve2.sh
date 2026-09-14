#!/usr/bin/env bash
# PHASE 2 of the pooled curve: put n>=80 on the repo's DEFAULT optimizer regime, and fill
# the unsteered arm's x-axis on its OWN base.
#
# WHY accum 4 HERE AND NOT ceil(n/16). The ceil rule exists to rescue the small end, where
# the default gradient_accumulation_steps=4 (batch_size 16, so one step per 64 samples) takes
# ZERO optimizer steps and the fit returns the untrained probe. Above ~64 rows the default
# steps fine, and every point already committed in scripts/<concept>_gen90*.csv ran at it:
#
#     n=110 ->  7 batches -> 1 step/epoch        n=350 -> 22 batches -> 5 steps/epoch
#     n=170 -> 11 batches -> 2 steps/epoch       n=590 -> 37 batches -> 9 steps/epoch
#
# Holding accum at 4 means the pooled rows differ from the gen90 rows in exactly ONE way —
# whether the 50 base rows are resampled — which is the thing being measured. Applying
# ceil(n/16) up here instead would change the regime by 5-9x in optimizer steps at the same
# time, and the two effects could not be separated afterwards. --grad-accum 4 is passed
# EXPLICITLY (not left to default) so the rows tag as base='none+ga4' and can never be
# confused with the accum-5 n=80 rows phase 1 wrote.
#
# WHY THE UNSTEERED ARM ONLY ABOVE n=80. scripts/<concept>_size_curve.csv is the unsteered
# arm's existing curve and it is unusable here: it fits every attacker's set on the LLAMA70B
# 50 rows, not on that attacker's own base, so it is not comparable to the steered arm (which
# was always own-base) or to anything else in this study. On own base the unsteered arm has
# only n=590 and 650. These fits give it the rest of the axis.
#
# Per concept: 8 pools x n=80  = 64 fits, plus 4 unsteered pools x 4 sizes = 128 fits.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-8}"
CONCEPTS="${CONCEPTS:-hu_harm instructions}"
BIG_SIZES="${BIG_SIZES:-110 170 350 590}"
mkdir -p logs

run_cell () {   # <concept> <pool> <n> <out> <log>
    local concept="$1" pool="$2" n="$3" out="$4" log="$5" devflag=""
    [ "$concept" = highstakes ] && devflag="--dev-data dev_samples/highstakes_500"
    echo ">>> $(date -Is)  ${concept} $(basename "$pool") n=$n (--grad-accum 4)"
    $PY scripts/subsample_curve_concept.py --concept "$concept" "$pool" \
        --no-base $devflag --grad-accum 4 --sizes "$n" --draws "$DRAWS" --out "$out" \
        >> "$log" 2>&1 \
        || echo ">>> $(date -Is)  FAILED ${concept} $(basename "$pool") n=$n"
}

for concept in $CONCEPTS; do
    out="scripts/${concept}_pooled_size_curve.csv"
    log="logs/pooled_sizecurve_${concept}.log"

    # n=80 on the default regime, BOTH arms, so the two arms share one rule at that x.
    for pool in .pool_work/pool_${concept}_*.jsonl; do
        [ -s "$pool" ] || { echo ">>> $(date -Is)  MISSING $pool"; continue; }
        run_cell "$concept" "$pool" 80 "$out" "$log"
    done

    # The rest of the axis, UNSTEERED arm only.
    for pool in .pool_work/pool_${concept}_*_general_*.jsonl; do
        [ -s "$pool" ] || { echo ">>> $(date -Is)  MISSING $pool"; continue; }
        for n in $BIG_SIZES; do
            run_cell "$concept" "$pool" "$n" "$out" "$log"
        done
    done
    echo ">>> $(date -Is)  ${concept} phase 2 done — $(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 )) rows in $out"
done
echo ">>> $(date -Is)  phase 2 finished."
