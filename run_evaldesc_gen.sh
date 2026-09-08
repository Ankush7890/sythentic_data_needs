#!/usr/bin/env bash
# Generate the eval-description-steered 600-row sets: 4 generators x 2 concepts.
#
# Same script, same one-shot pair, same 300/300 balance and same models as the unsteered
# *_600.jsonl sets already on this branch — the ONE difference is --eval-description, which
# shows the generator the `eval.data_description` text VERBATIM from that concept's red-team
# configs and points each call at one of its numbered kinds in turn.
#
# batch-size 5 (not the unsteered 10): the steered prompt asks for conversations up to 600
# words with supplied documents and several turns, so ten of them would run past max-tokens
# and lose the tail of the reply.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
mkdir -p logs

gen() {  # concept script_tag model tag concurrency extra_env
    local concept=$1 model=$2 tag=$3 conc=$4
    local out="data/${concept}_${tag}_evaldesc_600.jsonl"
    local log="logs/gen_${concept}_${tag}_evaldesc.log"
    if [ -s "$out" ]; then echo ">>> $(date -Is)  SKIP $out (exists)"; return 0; fi
    echo ">>> $(date -Is)  generating $out with $model (concurrency $conc)"
    $PY "scripts/generate_${concept}_dataset.py" \
        --model "$model" --n-per-label 300 --batch-size 5 --concurrency "$conc" \
        --call-budget-factor 6 --max-tokens 8192 --eval-description \
        --out "$out" > "$log" 2>&1
    echo ">>> $(date -Is)  $out: $(wc -l < "$out" 2>/dev/null || echo 0) rows"
}

for concept in instructions highstakes; do
    gen "$concept" meta-llama/llama-3.3-70b-instruct        llama70b      8
    gen "$concept" openai/gpt-oss-120b                      gptoss        8
    gen "$concept" deepseek/deepseek-v4-pro                 deepseekv4pro 8
    OPENROUTER_TIMEOUT_S=900 gen "$concept" nvidia/nemotron-3-ultra-550b-a55b nemotron 3
done
echo ">>> $(date -Is)  all generation finished."
