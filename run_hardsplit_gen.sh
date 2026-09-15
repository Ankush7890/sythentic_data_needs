#!/usr/bin/env bash
# hard_split_experiments: one 600-row NO-SHOT specialist per toolace part, deepseek-v4-pro —
# the generator behind every earlier toolace specialist. The four parts are generated in
# parallel (API only, no GPU). Part definitions: scripts/make_toolace_parts.py.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
GEN="${GEN:-deepseek/deepseek-v4-pro}"
TAG="${TAG:-deepseekv4pro}"
mkdir -p logs data/toolace_parts
for part in toolace_ops toolace_lookup toolace_finance toolace_roledef; do
    out="data/toolace_parts/highstakes_${TAG}_tgtnone_${part}_600.jsonl"
    if [ -s "$out" ]; then echo ">>> $(date -Is) SKIP $out"; continue; fi
    (
        echo ">>> $(date -Is) generating $out"
        $PY scripts/generate_split_targeted.py --concept highstakes --split "$part" \
            --model "$GEN" --n 600 --no-shots --out "$out" > "logs/hardsplit_gen_${part}.log" 2>&1
        echo ">>> $(date -Is) $out: $(wc -l < "$out" 2>/dev/null || echo 0) rows"
    ) &
done
wait
echo ">>> $(date -Is) all toolace part generation finished."
