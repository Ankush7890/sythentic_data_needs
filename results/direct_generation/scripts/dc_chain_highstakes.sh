#!/usr/bin/env bash
# high-stakes, ONE generator at a time. Not a style choice — a measurement.
#
# The other two concepts fit four generators side by side at no cost per fit: a fit holds
# ~2.7 GiB of the card and spends its wall-clock on cached activations. high-stakes does
# not. Its eval blobs are 47 GiB (anthropic_hh_balanced alone is 33), its dev set is
# staged on the card for every epoch, and this box is WSL2, where the driver
# OVERSUBSCRIBES into host memory and pages over PCIe instead of raising (the failure mode
# CLAUDE.md records under `_to_device_for_fit`). Measured here: one process runs a fit in
# ~150 s; a fourth concurrent process pushed the card to 24.2/24 GiB and the eval collapsed
# to 135 s PER BATCH — 187 batches to a split — and the processes never recovered when the
# others were killed. So: strictly sequential, and each generator's cells are resumable,
# so whatever this gets through is complete work rather than a partial curve.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY=.venv_claude/bin/python
mkdir -p logs

for gen in "$@"; do
    echo ">>> $(date -Is) highstakes x $gen"
    $PY scripts/direction_count.py --stage fit --concepts highstakes --generators "$gen" \
        --out-tag "$gen" --highstakes-dev dev_samples/highstakes_500 --restrict-eval \
        >> "logs/dc_fit_highstakes_${gen}.log" 2>&1 \
        || echo ">>> $(date -Is) FAILED highstakes x $gen"
done
echo ">>> $(date -Is) highstakes done"
