#!/usr/bin/env bash
# Backfill the family-B sets that are ON DISK but never fitted, so the shape-free arm has the
# control it needs.
#
# tgtmin removes TWO things from tgtshot at once: the dev few-shot anchor and the shape. Only
# tgtnone (measured description, anchor dropped, shape kept) separates them, and the earlier
# family-B run died with 4 sets generated and 3 fitted. This fits whatever tgtshot/tgtnone
# sets are on disk and short of their 9 rows.
#
# WAITS for the tgtmin fit driver to exit first. One 24 GB card, one gemma-3-27b at a time: a
# concurrent extraction would OOM rather than overlap.
#
# The wait pattern is anchored on the INVOCATION (`bash ./<script>`), not on the bare script
# name. An unanchored `pgrep -f` also matches any shell whose command line merely QUOTES the
# name — the heredoc that wrote this file did exactly that, and the first version of this
# script waited 100 minutes on an echo of its own source before anyone noticed.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
TAG="${TAG:-deepseekv4pro}"

while pgrep -f "bash \./run_tgtmin_fits\.sh" > /dev/null; do
    sleep 120
done
echo ">>> $(date -Is)  tgtmin fits finished; backfilling family-B sets"

for set_file in data/*_tgtshot_*_600.jsonl data/*_tgtnone_*_600.jsonl; do
    [ -s "$set_file" ] || continue
    case "$set_file" in
        data/instructions_*) concept=instructions; devflag=""; out=scripts/instructions_gen90.csv ;;
        data/highstakes_*)   concept=highstakes
                             devflag="--dev-data dev_samples/highstakes_500"
                             out=scripts/highstakes_gen90_dev500.csv ;;
        *) continue ;;
    esac
    base="$(basename "$set_file")"
    n=$(grep -c "^${base}," "$out" 2>/dev/null); n=${n:-0}
    [ "$n" -ge 9 ] && { echo ">>> $(date -Is)  SKIP $base ($n rows)"; continue; }
    echo ">>> $(date -Is)  extracting $base ($(wc -l < "$set_file") rows)"
    $PY scripts/warm_set_activations.py --concept "$concept" "$set_file" \
        >> "logs/warm_tgtmin_${concept}.log" 2>&1 \
        || echo ">>> $(date -Is)  EXTRACTION FAILED for $base"
    for spec in "600 1" "540 8"; do
        set -- $spec
        echo ">>> $(date -Is)  fitting $base sizes=$1 draws=$2"
        $PY scripts/subsample_curve_concept.py --concept "$concept" "$set_file" \
            --base-data "data/${concept}_${TAG}_50.jsonl" $devflag \
            --sizes "$1" --draws "$2" --out "$out" \
            >> "logs/fit_tgtmin_${concept}.log" 2>&1 \
            || echo ">>> $(date -Is)  FIT FAILED for $base ($1/$2)"
    done
    echo ">>> $(date -Is)  arm done: backfill/$base"
done
echo ">>> $(date -Is)  family-B backfill finished."
