#!/usr/bin/env bash
# The small end of the NO-BASE instruction curve: n=30, 15, 10, both variants, 8 draws.
#
# These sizes cannot be fit under the inherited optimizer config. It uses batch_size 16 with
# gradient_accumulation_steps 4, so one optimizer step costs 64 samples and a training set
# under ~49 rows takes ZERO steps — the fit returns the UNTRAINED probe (measured: dev 0.4802
# and target 0.6409 identical across all 8 draws at both n=30 and n=15). With a base they are
# fine, since n=15 there means 65 rows.
#
# So these run at --grad-accum 1 (and batch_size 8, since n=10 is under one batch of 16).
# THAT IS A DIFFERENT ESTIMATOR: rows are tagged base='none+ga1' and must NOT be read against
# the n=60+ no-base rows above them, whose config is the default. The discontinuity at n=60
# in any curve joining them is a config change, not a data effect.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG=deepseekv4pro
for split in anthropic_harmless_refusal bbq_substitution hc_context_drift hc_contradiction \
             mm_substitution oig_context_drift; do
    for variant in tgtnone tgtshot; do
        f="data/instructions_${TAG}_${variant}_${split}_600.jsonl"
        [ -s "$f" ] || { echo ">>> $(date -Is)  MISSING $f"; continue; }
        for n in 30 15 10; do
            echo ">>> $(date -Is)  ga1 ${variant}/${split} n=${n}"
            $PY scripts/subsample_curve_concept.py --concept instructions "$f" \
                --no-base --grad-accum 1 --sizes "$n" --draws 8 \
                --out scripts/instructions_gen90.csv \
                >> logs/nobase_small.log 2>&1 \
                || echo ">>> $(date -Is)  FIT FAILED ${variant}/${split} n=${n}"
        done
    done
done
echo ">>> $(date -Is)  all small no-base fits finished."
