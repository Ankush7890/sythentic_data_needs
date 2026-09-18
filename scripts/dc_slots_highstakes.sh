#!/usr/bin/env bash
# high-stakes fits, TWO generators in flight at a time.
#
# The width is a measurement, not a guess. A high-stakes fit holds 6.7 GiB of the card at
# 3% utilisation and spends its wall-clock on 47 GiB of cached eval activations, so two
# fit side by side (13.4 GiB) with headroom. FOUR does not: the card goes to 24.2/24 GiB
# and, this being WSL2, the driver oversubscribes into host memory and pages over PCIe
# instead of raising — the eval collapses from ~90 s a fit to 135 s a BATCH, 187 batches
# to the split. See `_to_device_for_fit` in CLAUDE.md for the same failure.
#
# `setsid` matters as much as the width. `--stage fit` drives the harness through
# subprocess.run, so killing the stage leaves its fit worker alive, holding the card:
# four such orphans are what produced the collapse above, and the card only came back
# when they were killed by pid. Each generator here gets its own process group, so
# `kill -- -<pgid>` takes the worker with it.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv_claude/bin/python
WIDTH="${WIDTH:-2}"
mkdir -p logs

run_one () {
    local gen="$1"
    echo ">>> $(date -Is) start highstakes x $gen"
    setsid $PY scripts/direction_count.py --stage fit --concepts highstakes \
        --generators "$gen" --out-tag "$gen" \
        --highstakes-dev dev_samples/highstakes_500 --restrict-eval \
        >> "logs/dc_fit_highstakes_${gen}.log" 2>&1
    echo ">>> $(date -Is) done highstakes x $gen (exit $?)"
}

running=0
for gen in "$@"; do
    run_one "$gen" &
    running=$((running + 1))
    if [ "$running" -ge "$WIDTH" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait
echo ">>> $(date -Is) highstakes: every requested generator done"
