#!/usr/bin/env bash
# The n=0 floor for the size curves: deepseek's own 50-row base alone, same dev, same
# protocol. Waits for the tiny-size driver so nothing contends for the card.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export SYNTHETIC_PROBE_DATA_MAX_MEMORY="${SYNTHETIC_PROBE_DATA_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$SYNTHETIC_PROBE_DATA_MAX_MEMORY}"
PY=.venv_claude/bin/python
while pgrep -f "[r]un_hs_tiny_sizes.sh" >/dev/null; do sleep 60; done
echo ">>> $(date -Is)  base-only control"
$PY scripts/fit_base_only_dev.py --concept instructions \
    --base-data data/instructions_deepseekv4pro_50.jsonl 2>&1 | grep -vE "^\[|it/s|Processing"
$PY scripts/fit_base_only_dev.py --concept highstakes \
    --base-data data/highstakes_deepseekv4pro_50.jsonl \
    --dev-data dev_samples/highstakes_500 2>&1 | grep -vE "^\[|it/s|Processing"
echo ">>> $(date -Is)  base-only control done"
