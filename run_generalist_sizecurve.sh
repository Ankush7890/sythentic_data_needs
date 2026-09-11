#!/usr/bin/env bash
# SIZE CURVE on the GENERALIST sets — the mixed eval-description 600-row sets every
# split-targeted arm is compared against. 8 draws at n=300/120/60/30 (50/20/10/5% of 600),
# all four generators, both concepts.
#
# Same protocol as the specialists' with-base curve: each generator's own 50-row base, the
# concept's dev set, all splits scored, appended to the same two CSVs. With the base present
# even n=30 is 80 training rows, so every point clears the ~49-row floor below which the
# inherited batch_size 16 x accumulation 4 takes zero optimizer steps.
#
# No extraction: these sets were all extracted for their n=540 arms and the cache is keyed
# per conversation, so every draw is a pure hit.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
GENS="deepseekv4pro llama70b gptoss nemotron"

for concept in instructions highstakes; do
    devflag=""; out=scripts/instructions_gen90.csv
    if [ "$concept" = highstakes ]; then
        devflag="--dev-data dev_samples/highstakes_500"
        out=scripts/highstakes_gen90_dev500.csv
    fi
    for gen in $GENS; do
        f="data/${concept}_${gen}_evaldesc_600.jsonl"
        b="data/${concept}_${gen}_50.jsonl"
        [ -s "$f" ] && [ -s "$b" ] || { echo ">>> $(date -Is)  MISSING $f or $b"; continue; }
        for n in 300 120 60 30; do
            echo ">>> $(date -Is)  generalist ${concept}/${gen} n=${n}"
            $PY scripts/subsample_curve_concept.py --concept "$concept" "$f" \
                --base-data "$b" $devflag --sizes "$n" --draws 8 --out "$out" \
                >> "logs/generalist_curve_${concept}.log" 2>&1 \
                || echo ">>> $(date -Is)  FIT FAILED ${gen} n=${n}"
        done
    done
done
echo ">>> $(date -Is)  all generalist size-curve fits finished."
