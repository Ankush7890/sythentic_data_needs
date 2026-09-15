#!/usr/bin/env bash
# hard_split_experiments fits — NO base data anywhere. Waits for the eval/dev prep and all four
# part sets, then: (1) each specialist alone, (2) the four pooled — 150 rows per set at
# n=600, 135 per set at n=540.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export MAX_MEMORY="${MAX_MEMORY:-${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}}"
PY=.venv_claude/bin/python
OUT=scripts/highstakes_toolace_parts.csv
SETS=""
for part in toolace_ops toolace_lookup toolace_finance toolace_roledef; do
    SETS="$SETS data/toolace_parts/highstakes_deepseekv4pro_tgtnone_${part}_600.jsonl"
done
until grep -aq "^{'anthropic_hh_balanced'\|'mean'" logs/hardsplit_prep.log 2>/dev/null \
      && ! pgrep -f prep_hardsplit_eval.py >/dev/null; do sleep 60; done
for s in $SETS; do until [ -s "$s" ]; do sleep 60; done; done
echo ">>> $(date -Is) prep and sets ready"
for s in $SETS; do
    echo ">>> $(date -Is) inspect $s"
    $PY scripts/inspect_generated_set.py --concept highstakes "$s" 2>&1 | sed 's/^/>>>   /'
done
echo ">>> $(date -Is) (1) specialists"
$PY scripts/fit_toolace_parts.py $SETS --sizes 600 540 --draws 1 8 --check --out "$OUT"
echo ">>> $(date -Is) (2) pooled"
$PY scripts/fit_toolace_parts.py $SETS --pool --sizes 600 540 --draws 1 8 --out "$OUT"
echo ">>> $(date -Is) all hard-split fits finished."
