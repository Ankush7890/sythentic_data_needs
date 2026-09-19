#!/usr/bin/env bash
# SMALL-n SIZE CURVE WITH NOTHING HELD FIXED, both arms, own 50-row base pooled in.
#
# WHY THIS EXISTS. Every size curve in this repo so far fits `fixed 50-row base + n drawn
# generated rows`. At the small end that base is most of the training set and is the SAME 50
# rows in every draw, so the spread across draws is not the spread of a training set of that
# size — it understates it, and the mean is anchored to one particular base. Here the base is
# pooled into the set it is drawn from (scripts/build_pooled_sets.py) and --no-base fits the
# draw alone, so all n rows are resampled on every draw.
#
# The unsteered arm uses each generator's OWN 50 rows, not the llama70b 50 that
# scripts/<concept>_size_curve.csv used for every attacker — that matches the steered arm in
# scripts/<concept>_gen90*.csv, which was always own-base, so the two arms are comparable.
#
# ACCUMULATION = ceil(n/16), i.e. exactly ONE optimizer step per epoch at every size.
# tuberlens steps on every `accum`-th batch of its train DataLoader, which is built WITHOUT
# drop_last, so the batch count is ceil(rows/batch_size) including a short final batch:
#
#     n=80 -> 5 batches at bs16 -> --grad-accum 5 -> 1 step/epoch
#     n=30 -> 2 batches at bs16 -> --grad-accum 2 -> 1 step/epoch
#     n=10 -> 1 batch  at bs16 -> --grad-accum 1 -> 1 step/epoch
#
# n=10 also passes --batch-size 16 explicitly: --grad-accum 1 otherwise drops the batch to 8
# (two steps/epoch), which is a different regime from the other two sizes. At the DEFAULT
# accumulation of 4, n=30 and n=10 would take ZERO steps and the fit would return the
# UNTRAINED probe — the sd=0 failure this repo has hit twice.
#
# Rows are tagged base='none+ga<K>' (and 'none+ga1bs16' at n=10), so they can never be
# pooled with the fixed-base rows in the existing curves.
#
# Per concept: 8 pools x 3 sizes x 8 draws = 192 fits.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
DRAWS="${DRAWS:-8}"
CONCEPTS="${CONCEPTS:-hu_harm instructions}"
mkdir -p logs

for concept in $CONCEPTS; do
    out="scripts/${concept}_pooled_size_curve.csv"
    log="logs/pooled_sizecurve_${concept}.log"
    devflag=""
    [ "$concept" = highstakes ] && devflag="--dev-data dev_samples/highstakes_500"
    for pool in .pool_work/pool_${concept}_*.jsonl; do
        [ -s "$pool" ] || { echo ">>> $(date -Is)  MISSING $pool"; continue; }
        for n in 80 30 10; do
            case $n in
                80) extra="--grad-accum 5" ;;
                30) extra="--grad-accum 2" ;;
                10) extra="--grad-accum 1 --batch-size 16" ;;
            esac
            echo ">>> $(date -Is)  ${concept} $(basename "$pool") n=$n ($extra)"
            $PY scripts/subsample_curve_concept.py --concept "$concept" "$pool" \
                --no-base $devflag $extra --sizes "$n" --draws "$DRAWS" --out "$out" \
                >> "$log" 2>&1 \
                || echo ">>> $(date -Is)  FAILED ${concept} $(basename "$pool") n=$n"
        done
    done
    echo ">>> $(date -Is)  ${concept} done — $(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 )) rows in $out"
done
echo ">>> $(date -Is)  all pooled size curves finished."
