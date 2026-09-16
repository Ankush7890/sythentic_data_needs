#!/usr/bin/env bash
# One 600-row toolace set per prompt in TOOLACE_GENERATOR_PROMPTS.md, deepseek-v4-pro.
# All five run at once; each call asks for 20 rows (10 per label) and takes ~10 min, so the
# concurrency is what sets the wall-clock.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
export OPENROUTER_TIMEOUT_S="${OPENROUTER_TIMEOUT_S:-1800}"
PY=.venv_claude/bin/python
mkdir -p logs data

names=(1:replica 2:grid 3:deceptive 4:endings 5:longtail)
for spec in "${names[@]}"; do
    k="${spec%%:*}"; name="${spec##*:}"
    out="data/highstakes_deepseekv4pro_p5_${name}_600.jsonl"
    [ -s "$out" ] && { echo ">>> SKIP $out"; continue; }
    $PY scripts/generate_toolace_prompts5.py --prompt "$k" --n 600 --concurrency 10 \
        --out "$out" > "logs/gen_p5_${name}.log" 2>&1 &
done
wait
echo ">>> $(date -Is) all five sets finished"
for f in data/highstakes_deepseekv4pro_p5_*_600.jsonl; do echo "$(wc -l < "$f") $f"; done
