#!/usr/bin/env bash
# hard_split_experiments, round 2: toolace_lookup and toolace_finance each cut in two
# (scripts/make_toolace_subparts.py). One 300-row NO-SHOT specialist per sub-part,
# deepseek-v4-pro, then fits with NO base data: n=300 x 1 and n=270 x 8 per set, dev
# highstakes_500, scored on every eval split, toolace's four parts and the four sub-parts.
#
# Generation (API) runs in parallel; each set is gated and extracted (GPU) as soon as it lands,
# so extraction does not wait for the slowest generation. Fits start once all four are warm.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export MAX_MEMORY="${MAX_MEMORY:-${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}}"
PY=.venv_claude/bin/python
PARTS="lookup_media lookup_utility finance_markets finance_money"
OUT=scripts/highstakes_toolace_subparts.csv
set_path () { echo "data/toolace_parts/highstakes_deepseekv4pro_tgtnone_${1}_300.jsonl"; }
mkdir -p logs

for p in $PARTS; do
    f=$(set_path "$p")
    [ -s "$f" ] && continue
    $PY scripts/generate_split_targeted.py --concept highstakes --split "$p" \
        --model deepseek/deepseek-v4-pro --n 300 --no-shots --out "$f" \
        > "logs/hardsplit_sub_gen_${p}.log" 2>&1 &
done

gate () {   # every row carries the scaffold, exactly 150/150
    $PY - "$1" "$2" <<'PY'
import collections, json, sys
sys.path[:0] = ["scripts", "src"]
from generate_split_targeted import has_required
from split_specs import SPLIT_SPECS
part, path = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
miss = sum(not has_required(json.loads(r["inputs"]), SPLIT_SPECS["highstakes"][part]) for r in rows)
lab = collections.Counter(r["labels"] for r in rows)
print(f">>>   gate {part}: {len(rows)} rows {dict(lab)}, {miss} without scaffold", flush=True)
sys.exit(1 if miss or lab != collections.Counter({"high-stakes": 150, "low-stakes": 150}) else 0)
PY
}

warmed=""
while [ "$(echo $warmed | wc -w)" -lt 4 ]; do
    progressed=0
    for p in $PARTS; do
        case " $warmed " in *" $p "*) continue ;; esac
        f=$(set_path "$p")
        # a set is ready when its file exists and its generator has exited
        if [ -s "$f" ] && ! ps -eo args | grep -q "[g]enerate_split_targeted.py .*--split $p "; then
            echo ">>> $(date -Is) $p generated"
            gate "$p" "$f" || { echo ">>> $(date -Is) GATE FAILED for $p — stopping"; exit 1; }
            $PY scripts/inspect_generated_set.py --concept highstakes --min-per-class 150 "$f" 2>&1 | sed 's/^/>>>   /'
            $PY scripts/warm_set_activations.py --concept highstakes "$f" >> logs/hardsplit_sub_warm.log 2>&1 \
                || { echo ">>> $(date -Is) EXTRACTION FAILED for $p"; exit 1; }
            echo ">>> $(date -Is) $p warm"
            warmed="$warmed $p"; progressed=1
        fi
    done
    [ "$progressed" = 1 ] || sleep 60
done

SETS=""; for p in $PARTS; do SETS="$SETS $(set_path "$p")"; done
echo ">>> $(date -Is) fitting"
$PY scripts/fit_toolace_parts.py $SETS --sizes 300 270 --draws 1 8 --out "$OUT"
echo ">>> $(date -Is) all sub-part fits finished."
