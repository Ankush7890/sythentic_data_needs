#!/usr/bin/env bash
# The two mts_balanced arms, re-run after the turn-range fix. Both drivers had already passed
# this split when its spec enforced exactly 6 messages against a description offering 6-12,
# which dropped 266 items for 119 kept; the arms are regenerated here rather than left short.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
for variant in tgtshot tgtnone; do
    flag=""; [ "$variant" = tgtnone ] && flag="--no-shots"
    out="data/highstakes_deepseekv4pro_${variant}_mts_balanced_600.jsonl"
    [ -s "$out" ] && { echo ">>> $(date -Is)  SKIP $out"; continue; }
    echo ">>> $(date -Is)  generating $out"
    # batch 2, not the default 5: this split's rows are 6-12 messages, and five of them in
    # one reply overran the output budget — 17 of 24 calls came back with EMPTY content.
    $PY scripts/generate_split_targeted.py --concept highstakes --split mts_balanced \
        --model deepseek/deepseek-v4-pro --n 600 --batch-size 2 --call-budget-factor 25 \
        $flag --out "$out" \
        > "logs/gen_highstakes_${variant}_mts_balanced.log" 2>&1
    echo ">>> $(date -Is)  $out: $(wc -l < "$out" 2>/dev/null || echo 0) rows"
    $PY scripts/inspect_generated_set.py --concept highstakes "$out" \
        --compare data/highstakes_deepseekv4pro_50.jsonl 2>&1 | sed 's/^/>>>   /'
done
echo ">>> $(date -Is)  mts_balanced arms regenerated."
