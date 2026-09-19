#!/usr/bin/env bash
# One 600-row anthropic_hh set per prompt in ANTHROPIC_HH_GENERATOR_PROMPTS.md, deepseek-v4-pro.
# Five runs at once, 10 calls in flight each, 20 rows per call (10 per label).
# max-tokens 64000: these rows are short, but prompt 4/5 replies plus reasoning outran 32k on
# the toolace batch, and the model allows 393k.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
export OPENROUTER_TIMEOUT_S="${OPENROUTER_TIMEOUT_S:-1800}"
PY=.venv_claude/bin/python
mkdir -p logs data

names=(1:replica 2:twin 3:topicpair 4:redteam 5:long)
for spec in "${names[@]}"; do
    k="${spec%%:*}"; name="${spec##*:}"
    out="data/highstakes_deepseekv4pro_hh5_${name}_600.jsonl"
    [ -s "$out" ] && { echo ">>> SKIP $out"; continue; }
    $PY scripts/generate_prompt_variants.py --prompts-md ANTHROPIC_HH_GENERATOR_PROMPTS.md \
        --split anthropic_hh_balanced --prompt "$k" --n 600 --concurrency 10 \
        --max-tokens 64000 --out "$out" > "logs/gen_hh5_${name}.log" 2>&1 &
done
wait
echo ">>> $(date -Is) all five hh sets finished"
for f in data/highstakes_deepseekv4pro_hh5_*_600.jsonl; do echo "$(wc -l < "$f") $f"; done
