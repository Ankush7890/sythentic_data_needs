#!/usr/bin/env bash
# Generate the eval-description-steered 600-row sets for the HUMAN-HARM concept.
#
# The hu_harm counterpart of run_evaldesc_gen.sh, which did instructions + highstakes.
# Same script, same one-shot pair, same 300/300 balance and same four models as the
# unsteered hu_harm_*_600.jsonl sets already on this branch — the ONE difference is
# --eval-description, which shows the generator the `eval.data_description` text VERBATIM
# from the human-harm red-team configs on branch `human_harm_last` (the REWRITTEN text,
# commit d793fe5d — byte-identical in nine of that branch's ten hu_harm configs) and points
# each call at one of its four numbered kinds in turn.
#
# batch-size 5 and max-tokens 8192, as in run_evaldesc_gen.sh: the steered prompt is long
# and asks for a scenario per row, so ten per call would run past max-tokens and lose the
# tail of the reply. Unlike the other two concepts the SHAPE is unchanged (every
# eval_sets/hu_ha row is a two-message exchange), so only the length rule is relaxed.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
mkdir -p logs

gen() {  # model tag concurrency
    local model=$1 tag=$2 conc=$3
    local out="data/hu_harm_${tag}_evaldesc_600.jsonl"
    local log="logs/gen_hu_harm_${tag}_evaldesc.log"
    if [ -s "$out" ]; then echo ">>> $(date -Is)  SKIP $out (exists)"; return 0; fi
    echo ">>> $(date -Is)  generating $out with $model (concurrency $conc)"
    $PY scripts/generate_hu_harm_dataset.py \
        --model "$model" --n-per-label 300 --batch-size 5 --concurrency "$conc" \
        --call-budget-factor 6 --max-tokens 8192 --eval-description \
        --out "$out" > "$log" 2>&1
    echo ">>> $(date -Is)  $out: $(wc -l < "$out" 2>/dev/null || echo 0) rows"
    # Balance, contamination, turn shape, token width and the refusal-shape heuristic.
    # Read the last of these the other way round from the instructions concept: here a
    # refusal in the NEGATIVE class is CORRECT (kind 3's not_harmful_to_human branch is
    # "declines it in one short line"); a refusal opening a harmful_to_human row is the
    # mislabel to look for.
    $PY scripts/inspect_generated_set.py --concept hu_harm "$out" \
        --compare "data/hu_harm_${tag}_50.jsonl" "data/hu_harm_${tag}_600.jsonl" \
        2>&1 | sed 's/^/>>>   /'
}

gen meta-llama/llama-3.3-70b-instruct        llama70b      8
gen openai/gpt-oss-120b                      gptoss        8
gen deepseek/deepseek-v4-pro                 deepseekv4pro 8
OPENROUTER_TIMEOUT_S=900 gen nvidia/nemotron-3-ultra-550b-a55b nemotron 3
echo ">>> $(date -Is)  all hu_harm generation finished."
