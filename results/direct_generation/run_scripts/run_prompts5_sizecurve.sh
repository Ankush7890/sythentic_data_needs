#!/usr/bin/env bash
# SIZE CURVE for the ten prompt-variant sets: 8 draws at each of 590/300/150/75/40/20/10,
# NO base data, scored on the TARGET SPLIT ONLY.
#
# Each study validates on its own split's 125 dev rows (dev_samples/highstakes_500_<split>/)
# and scores only that split, because nothing here asks how the sets travel — that question is
# already answered in analysis/*_prompts5_results.md at n=600 on all four splits. Those n=600
# rows are therefore NOT on this curve: they used the four-split dev set, which drives early
# stopping, so they are a different probe.
#
# Under ~49 training rows the inherited spec (batch_size 16 x gradient_accumulation_steps 4)
# takes ZERO optimizer steps and the fit returns the UNTRAINED probe, so n=40/20/10 run with
# --grad-accum 1 and land tagged base='none+ga1'. Do not read across that boundary as one line.
#
# No extraction: every row of all ten sets is already in the per-conversation activation cache
# from the n=600 fits, and a subset is a pure cache hit.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
BIG="${BIG:-590 300 150 75}"
TINY="${TINY:-40 20 10}"
DRAWS="${DRAWS:-8}"
mkdir -p logs

fit () {  # set_file dev_dir split out_csv sizes [extra]
    $PY scripts/subsample_curve_concept.py --concept highstakes "$1" --no-base \
        --dev-data "$2" --eval-splits "$3" --sizes $5 --draws "$DRAWS" --out "$4" ${6:-} \
        >> logs/sizecurve_prompts5.log 2>&1 \
        || echo ">>> $(date -Is)  FIT FAILED $(basename "$1") sizes=$5 ${6:-}"
}

run_study () {  # glob dev_dir split out_csv
    for f in $1; do
        [ -s "$f" ] || continue
        echo ">>> $(date -Is)  $(basename "$f")  sizes $BIG"
        fit "$f" "$2" "$3" "$4" "$BIG"
        echo ">>> $(date -Is)  $(basename "$f")  sizes $TINY (grad-accum 1)"
        fit "$f" "$2" "$3" "$4" "$TINY" "--grad-accum 1"
    done
}

run_study "data/highstakes_deepseekv4pro_p5_*_600.jsonl" dev_samples/highstakes_500_toolace \
          toolace_balanced scripts/highstakes_toolace_prompts5_sizecurve.csv
run_study "data/highstakes_deepseekv4pro_hh5_*_600.jsonl" dev_samples/highstakes_500_anthropic_hh \
          anthropic_hh_balanced scripts/highstakes_anthropic_hh_prompts5_sizecurve.csv

echo ">>> $(date -Is)  size curve finished"
