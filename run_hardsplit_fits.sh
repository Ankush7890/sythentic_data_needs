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
until grep -aq "top-up finished" logs/hardsplit_topup.log 2>/dev/null; do sleep 60; done
echo ">>> $(date -Is) prep and sets ready"
# Gate: every row must carry its part's scaffold and every set must be exactly 300/300. The
# first generation passed inspect_generated_set.py with ~220 plain-chat rows in two sets.
$PY - $SETS <<'PY' || { echo ">>> $(date -Is) SCAFFOLD GATE FAILED — not fitting"; exit 1; }
import collections, json, sys
sys.path[:0] = ["scripts", "src"]
from generate_split_targeted import has_required
from split_specs import SPLIT_SPECS
bad = 0
for path in sys.argv[1:]:
    part = path.split("tgtnone_")[1].rsplit("_600", 1)[0]
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    miss = sum(not has_required(json.loads(r["inputs"]), SPLIT_SPECS["highstakes"][part]) for r in rows)
    lab = collections.Counter(r["labels"] for r in rows)
    print(f">>>   gate {part}: {len(rows)} rows {dict(lab)}, {miss} without scaffold")
    bad += miss + (lab != collections.Counter({"high-stakes": 300, "low-stakes": 300}))
sys.exit(1 if bad else 0)
PY
for s in $SETS; do
    echo ">>> $(date -Is) inspect $s"
    $PY scripts/inspect_generated_set.py --concept highstakes "$s" 2>&1 | sed 's/^/>>>   /'
done
echo ">>> $(date -Is) (1) specialists"
$PY scripts/fit_toolace_parts.py $SETS --sizes 600 540 --draws 1 8 --check --out "$OUT"
echo ">>> $(date -Is) (2) pooled"
$PY scripts/fit_toolace_parts.py $SETS --pool --sizes 600 540 --draws 1 8 --out "$OUT"
echo ">>> $(date -Is) all hard-split fits finished."
