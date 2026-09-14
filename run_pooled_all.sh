#!/usr/bin/env bash
# Driver: hu_harm curves (cache already warm) -> instructions warm -> instructions curves.
# High-stakes is deliberately NOT here; it is the expensive concept and comes after these two.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
echo ">>> $(date -Is)  START hu_harm curves"
CONCEPTS=hu_harm ./run_pooled_sizecurve.sh
echo ">>> $(date -Is)  START instructions warm"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
.venv_claude/bin/python scripts/warm_pooled_sets.py --concept instructions \
    >> logs/warm_pool_instructions.log 2>&1
echo ">>> $(date -Is)  START instructions curves"
CONCEPTS=instructions ./run_pooled_sizecurve.sh
echo ">>> $(date -Is)  ALL DONE"
