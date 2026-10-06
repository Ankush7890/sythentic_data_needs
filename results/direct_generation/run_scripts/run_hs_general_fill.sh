#!/usr/bin/env bash
# Fill the high-stakes GENERAL arm of the pooled size curve at n = 110/170/350/590.
#
# This is the phase-2 protocol of origin/per_split_studies:run_pooled_sizecurve2.sh, run for
# the one concept that stopped early: highstakes. Read that script's header for WHY accum is
# held at 4 above n=80 (the ceil(n/16) rule exists only to rescue the small end, where the
# default takes zero optimizer steps; applying it up here would change the regime by 5-9x in
# optimizer steps at the same time as the pooling, and the two could not be separated).
# --grad-accum 4 is passed EXPLICITLY so the rows tag `none+ga4` and can never be confused
# with the accum-5 n=80 rows phase 1 wrote.
#
# 4 generators x 4 sizes x 8 draws = 128 fits, appended to scripts/highstakes_pooled_size_curve.csv
# under the 96 rows phase 1 already wrote there (the resume key skips anything present, so a
# kill loses at most one fit and no existing row ever moves).
#
# highstakes needs --dev-data dev_samples/highstakes_500: every high-stakes small-n curve in
# the paper used that 500-row cut, and the full 1,908-row dev set would change the
# early-stopping signal within one curve.
#
# PROCESS GROUP. The fit is in-process here (subsample_curve_concept.py spawns nothing), but
# run this under `setsid` and kill it with `kill -TERM -<pgid>`, never by the wrapper's pid
# alone — an orphaned fit holding ~7 GiB is what filled the card during the direction_count
# run.
#
#   setsid bash run_hs_general_fill.sh llama70b &        # validate the setup on Llama first
#   setsid bash run_hs_general_fill.sh deepseekv4pro gptoss nemotron &
#
# With no arguments it runs all four, two at a time.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
OUT=scripts/highstakes_pooled_size_curve.csv
LOG=logs/pooled_sizecurve_highstakes.log
SIZES="${SIZES:-110 170 350 590}"
DRAWS="${DRAWS:-8}"
GENS="${*:-llama70b deepseekv4pro gptoss nemotron}"
mkdir -p logs

run_gen () {   # <generator>
    local gen="$1" pool=".pool_work/pool_highstakes_${1}_general_650.jsonl"
    [ -s "$pool" ] || { echo ">>> $(date -Is)  MISSING $pool"; return 1; }
    echo ">>> $(date -Is)  highstakes ${gen} general n=${SIZES} draws=${DRAWS} (--grad-accum 4)"
    $PY -u scripts/subsample_curve_concept.py --concept highstakes "$pool" \
        --no-base --dev-data dev_samples/highstakes_500 --grad-accum 4 \
        --sizes $SIZES --draws "$DRAWS" --out "$OUT" \
        >> "$LOG" 2>&1 \
        || echo ">>> $(date -Is)  FAILED highstakes ${gen}"
    echo ">>> $(date -Is)  done ${gen}"
}

for gen in $GENS; do
    run_gen "$gen"
done
echo ">>> $(date -Is)  finished [$GENS] — $(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 )) rows in $OUT"
