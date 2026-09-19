#!/usr/bin/env bash
# The high-stakes split-targeted arms fit WITHOUT base data — the counterpart to
# run_tgt_nobase.sh / run_tgt_nobase_small.sh on the instruction concept.
#
#   pass 1  both variants, n=600 x1 and n=540 x8, default optimizer config   ->  72 fits
#   pass 2  no-shot curve, n=300/120/60 x8, default config                   ->  96 fits
#   pass 3  no-shot only, n=30/15/10 x8, --grad-accum 1                      ->  96 fits
#
# Pass 3 needs the override: batch_size 16 x gradient_accumulation_steps 4 means one
# optimizer step per 64 samples, so a training set under ~49 rows takes ZERO steps and the
# fit returns the UNTRAINED probe. Those rows are tagged base='none+ga1' and must not be read
# as a continuation of passes 1-2, whose config is the default.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG=deepseekv4pro
OUT=scripts/highstakes_gen90_dev500.csv
SPLITS="anthropic_hh_balanced mt_balanced mts_balanced toolace_balanced"

fit () {  # set_file sizes draws [extra flags]
    $PY scripts/subsample_curve_concept.py --concept highstakes "$1" --no-base \
        --dev-data dev_samples/highstakes_500 --sizes $2 --draws "$3" --out "$OUT" ${4:-} \
        >> logs/nobase_highstakes.log 2>&1 \
        || echo ">>> $(date -Is)  FIT FAILED $(basename "$1") sizes=$2 ${4:-}"
}

echo ">>> $(date -Is)  PASS 1: both variants, n=600 and n=540"
for split in $SPLITS; do
    for variant in tgtshot tgtnone; do
        f="data/highstakes_${TAG}_${variant}_${split}_600.jsonl"
        [ -s "$f" ] || { echo ">>> $(date -Is)  MISSING $f"; continue; }
        echo ">>> $(date -Is)  no-base ${variant}/${split}"
        fit "$f" 600 1
        fit "$f" 540 8
    done
done

echo ">>> $(date -Is)  PASS 2: no-shot curve without base"
for split in $SPLITS; do
    f="data/highstakes_${TAG}_tgtnone_${split}_600.jsonl"
    [ -s "$f" ] || continue
    for n in 300 120 60; do
        echo ">>> $(date -Is)  no-base curve ${split} n=${n}"
        fit "$f" "$n" 8
    done
done

# PASS 3 is NO-SHOT ONLY, so the whole size curve (300 down to 10) is one variant end to end.
# The shots/no-shots comparison lives at n=600/540 in pass 1, as it does on instructions.
echo ">>> $(date -Is)  PASS 3: n=30/15/10 at grad_accum=1, no-shot only"
for split in $SPLITS; do
    for variant in tgtnone; do
        f="data/highstakes_${TAG}_${variant}_${split}_600.jsonl"
        [ -s "$f" ] || continue
        for n in 30 15 10; do
            echo ">>> $(date -Is)  ga1 ${variant}/${split} n=${n}"
            fit "$f" "$n" 8 "--grad-accum 1"
        done
    done
done
echo ">>> $(date -Is)  all high-stakes no-base fits finished."
