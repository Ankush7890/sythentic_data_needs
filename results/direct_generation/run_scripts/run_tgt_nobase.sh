#!/usr/bin/env bash
# The instruction split-targeted arms fit WITHOUT base data — the generated set alone.
#
# Every other arm in these CSVs trains on the generator's own 50-row base PLUS n generated
# rows, so at n=540 the fit sees 590 rows and at n=15 it sees 65. That makes the small end
# of the size curve mostly base. These runs drop the base entirely, so n IS the training set
# and the generated data has to carry the probe on its own. It is also how the hu_harm study
# on `human_harm_last` fit its split-targeted sets, which makes the two comparable.
#
# Rows land in the same CSV with base='none', which keeps the resume key distinct from the
# with-base rows, so both live side by side and pair up cell for cell.
#
#   pass 1  both variants, n=600 x1 and n=540 x8   -> mirrors the main table
#   pass 2  no-shot only, n=300/120/60/30/15 x8    -> the size curve without a base
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG=deepseekv4pro
OUT=scripts/instructions_gen90.csv
SPLITS="anthropic_harmless_refusal bbq_substitution hc_context_drift hc_contradiction mm_substitution oig_context_drift"

fit () {  # set_file sizes draws
    $PY scripts/subsample_curve_concept.py --concept instructions "$1" --no-base \
        --sizes $2 --draws "$3" --out "$OUT" >> logs/nobase_instructions.log 2>&1 \
        || echo ">>> $(date -Is)  FIT FAILED $(basename "$1") sizes=$2"
}

echo ">>> $(date -Is)  PASS 1: both variants at n=600 and n=540"
for split in $SPLITS; do
    for variant in tgtshot tgtnone; do
        f="data/instructions_${TAG}_${variant}_${split}_600.jsonl"
        [ -s "$f" ] || { echo ">>> $(date -Is)  MISSING $f"; continue; }
        echo ">>> $(date -Is)  no-base ${variant}/${split}"
        fit "$f" 600 1
        fit "$f" 540 8
    done
done

echo ">>> $(date -Is)  PASS 2: no-shot size curve without base"
for split in $SPLITS; do
    f="data/instructions_${TAG}_tgtnone_${split}_600.jsonl"
    [ -s "$f" ] || continue
    # 300/120/60 only. The fit uses batch_size 16 with gradient_accumulation_steps 4, so an
    # optimizer step happens once per 64 samples: a training set with fewer than four batches
    # (under ~49 rows) takes ZERO steps and the fit returns the UNTRAINED probe. Measured — at
    # n=30 and n=15 every draw returned byte-identical dev 0.4802 / target 0.6409 while the row
    # counts differed correctly. Those points are not measurable without a base, and the 32
    # rows already written were removed. With a base they are fine (n=15 -> 65 rows).
    for n in 300 120 60; do
        echo ">>> $(date -Is)  no-base curve ${split} n=${n}"
        fit "$f" "$n" 8
    done
done
echo ">>> $(date -Is)  all no-base fits finished."
