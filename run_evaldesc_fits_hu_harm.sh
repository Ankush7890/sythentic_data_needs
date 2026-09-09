#!/usr/bin/env bash
# Fit + evaluate the eval-description-steered hu_harm sets, under EXACTLY the protocol the
# unsteered hu_harm sets were measured under (scripts/hu_harm_gen90.csv):
#
#   own base       each generator's own 50-row set, so every arm is single-source
#   n = 540        90% of the 600, 8 seeded class-balanced draws -> a noise bar per set
#   n = 600 x1     the whole set, for the headline number (run here for the unsteered sets
#                  too, so the two arms are compared at the same n as well as at 540)
#   dev            dev_samples/hu_ha, used whole — 290 rows, so unlike highstakes there is
#                  no reduced dev cut and the CSV needs no _dev500 variant
#
# Rows append to the SAME CSV the unsteered arms wrote, keyed on (base, samples, n, draw),
# so this is also the restart path: re-run and it skips what is already there.
#
# ONE DIFFERENCE from run_evaldesc_fits.sh, and it is only about this box: that run was on
# a machine whose per-sample activation cache already held every unsteered row, so pass B
# was free. Here the cache is cold, so the unsteered sets are warmed FIRST — they are on
# disk already, while the steered ones are still being written, so this is GPU time that
# would otherwise be spent waiting on the generator.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -f .env2 ]; then set -a; . ./.env2; set +a; fi
TAGSUFFIX="${TAGSUFFIX:-evaldesc}"      # which steered arm to fit; see run_evaldesc_gen_hu_harm.sh
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
GENS="llama70b gptoss deepseekv4pro nemotron"
CSV=scripts/hu_harm_gen90.csv
mkdir -p logs

warm() {  # set_file
    local f=$1
    [ -s "$f" ] || { echo ">>> $(date -Is)  MISSING $f — not warmed"; return 0; }
    echo ">>> $(date -Is)  extracting $f"
    $PY scripts/warm_set_activations.py --concept hu_harm "$f" \
        >> logs/warm_evaldesc_hu_harm.log 2>&1 \
        || echo ">>> $(date -Is)  EXTRACTION FAILED for $f"
}

run() {  # set_file gen sizes draws
    local set_file=$1 gen=$2 sizes=$3 draws=$4
    [ -s "$set_file" ] || { echo ">>> $(date -Is)  MISSING $set_file — skipping"; return 0; }
    echo ">>> $(date -Is)  $(basename "$set_file") sizes=$sizes draws=$draws"
    $PY scripts/subsample_curve_concept.py --concept hu_harm "$set_file" \
        --base-data "data/hu_harm_${gen}_50.jsonl" \
        --sizes $sizes --draws "$draws" --out "$CSV" \
        >> logs/fit_evaldesc_hu_harm.log 2>&1
}

# PASS 0 — warm the unsteered sets (already on disk) while generation runs.
for g in $GENS; do warm "data/hu_harm_${g}_600.jsonl"; done
# PASS 1 — warm each steered set as it lands, in generation order. ONE process touches the
# GPU at a time (warming and fitting are both here; the generator does not use it at all).
for g in $GENS; do
    f="data/hu_harm_${g}_${TAGSUFFIX}_600.jsonl"
    while [ ! -s "$f" ]; do sleep 120; done
    warm "$f"
done
echo ">>> $(date -Is)  all four steered sets extracted"

# PASS A — full set, steered (the headline).
for g in $GENS; do run "data/hu_harm_${g}_${TAGSUFFIX}_600.jsonl" "$g" 600 1; done
# PASS B — full set, unsteered counterpart, at the same n.
for g in $GENS; do run "data/hu_harm_${g}_600.jsonl" "$g" 600 1; done
# PASS C — 8 draws at 90%, steered (the noise bar the unsteered arms already have).
for g in $GENS; do run "data/hu_harm_${g}_${TAGSUFFIX}_600.jsonl" "$g" 540 8; done
echo ">>> $(date -Is)  all hu_harm eval-description fits finished."
