#!/usr/bin/env bash
# P3: top n=10, 30 and 80 up from four draws to eight, BOTH arms, on the PHASE-1 regimes.
#
# Phase 1 ran these three sizes at four draws each, on the ceil(n/16) small-end rule that
# rescues the low end from taking zero optimizer steps:
#
#     n=10 -> --grad-accum 1 --batch-size 16   (tag none+ga1bs16)
#     n=30 -> --grad-accum 2                   (tag none+ga2)
#     n=80 -> --grad-accum 5                   (tag none+ga5)
#
# Those regimes are reproduced EXACTLY here, one size per command, because the point is to
# put more draws under the existing points, not to make new ones: any change to accum or
# batch size would write a different tag and leave the original four draws stranded at four.
# `--draws 8` computes draws 0-7 and the resume key (base_label, file, n, draw) skips the
# existing 0-3, so exactly four new draws land per cell and no existing row moves.
#
# 8 pools x 3 sizes x 4 new draws = 96 fits. Small n, so these are the fastest fits here.
#
# Run as two streams (one arm each) under setsid. When checking whether a stream is up, do
# NOT filter with `grep -v eval`: it hides the whole evaldesc arm, because "evaldesc"
# contains "eval". Match on the pool basename.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
OUT=scripts/highstakes_pooled_size_curve.csv
LOG=logs/pooled_sizecurve_highstakes.log
DRAWS="${DRAWS:-8}"
mkdir -p logs

run_cell () {   # <pool-basename> <n> <extra flags...>
    local pool="$1" n="$2"; shift 2
    local P=".pool_work/${pool}"
    [ -s "$P" ] || { echo ">>> $(date -Is)  MISSING $P"; return 1; }
    echo ">>> $(date -Is)  P3 n=$n [$*] $(basename "$P")"
    $PY -u scripts/subsample_curve_concept.py --concept highstakes "$P" \
        --no-base --dev-data dev_samples/highstakes_500 "$@" \
        --sizes "$n" --draws "$DRAWS" --out "$OUT" \
        >> "$LOG" 2>&1 \
        || echo ">>> $(date -Is)  FAILED n=$n $(basename "$P")"
}

for pool in "$@"; do
    run_cell "$pool" 80 --grad-accum 5
    run_cell "$pool" 30 --grad-accum 2
    run_cell "$pool" 10 --grad-accum 1 --batch-size 16
done
echo ">>> $(date -Is)  P3 stream finished [$*]"
