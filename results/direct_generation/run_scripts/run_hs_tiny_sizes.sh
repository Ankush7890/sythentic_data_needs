#!/usr/bin/env bash
# The far end of the high-stakes size curve: n=10 and n=5 (1.7% and 0.8% of 600), 8 draws
# each, four splits, 64 fits.
#
# At these sizes the generated set is a garnish on the 50-row base — 5 rows is 9% of the
# training data — so these points approach "base probe alone". That is exactly the useful
# limit for toolace_balanced, whose curve rises monotonically as its targeted data shrinks
# (0.7269 at n=540 -> 0.7694 at n=15): if the trend is real, these should keep climbing
# toward the base-only probe.
#
# Waits for the running size-curve driver so the two never contend for the card.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG=deepseekv4pro

while pgrep -f "[r]un_tgt_sizecurve.sh" >/dev/null; do sleep 60; done
echo ">>> $(date -Is)  previous size curve finished; starting n=10 and n=5"

for split in anthropic_hh_balanced mt_balanced mts_balanced toolace_balanced; do
    set_file="data/highstakes_${TAG}_tgtnone_${split}_600.jsonl"
    [ -s "$set_file" ] || { echo ">>> $(date -Is)  MISSING $set_file"; continue; }
    for n in 10 5; do
        echo ">>> $(date -Is)  tiny size ${split} n=${n} draws=8"
        $PY scripts/subsample_curve_concept.py --concept highstakes "$set_file" \
            --base-data "data/highstakes_${TAG}_50.jsonl" --dev-data dev_samples/highstakes_500 \
            --sizes "$n" --draws 8 --out scripts/highstakes_gen90_dev500.csv \
            >> logs/sizecurve_highstakes.log 2>&1 \
            || echo ">>> $(date -Is)  FIT FAILED ${split} n=${n}"
    done
    echo ">>> $(date -Is)  tiny sizes done: $split"
done
echo ">>> $(date -Is)  all tiny-size fits finished."
