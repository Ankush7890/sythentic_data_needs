#!/usr/bin/env bash
# ARM 3 — MINIMAL-PROMPT ABLATION, human-harm concept.
#
# Same script, same model, same 600 rows at 300/300, same batch size as arms 1 and 2. The
# ONLY difference is the prompt: content removed (as in arm 2) AND shape removed — no turn
# counts, no word-length ranges, no register, no voice, no pairing. Specified in
# analysis/split_targeted_prompts_minimal.md; implemented in the _MINIMAL block of the
# generator.
#
# THREE PROMPTS, FOUR SETS.
#   ant_hh and balanced_refusal collapse into ONE prompt once shape is gone (everything that
#   separated them in arm 2 WAS shape), so `request_minimal` is generated TWICE from the same
#   text with no other change. That second draw is not padding: it measures pure generator
#   variance, which is the yardstick the "these two splits are now indistinguishable"
#   prediction has to be read against. One set is fit on eval_ant_hh, the other on
#   eval_balanced_refusal, and the difference between them is a NULL result if the
#   prediction holds.
#
# Every split is UNPAIRED / per-label — a pairing instruction is itself shape.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -f .env2 ]; then set -a; . ./.env2; set +a; fi
PY=.venv_claude/bin/python
MODEL=meta-llama/llama-3.3-70b-instruct
mkdir -p logs

gen () {  # split tag
    local split=$1 tag=$2
    local out="data/hu_harm_llama70b_${tag}_600.jsonl"
    if [ -s "$out" ]; then echo ">>> SKIP $out (exists)"; return 0; fi
    $PY scripts/generate_hu_harm_split_dataset.py --split "$split" --model "$MODEL" \
        --n 600 --batch-size 5 --out "$out" > "logs/gen_${tag}.log" 2>&1
    echo ">>> $(date -Is)  $out: $(wc -l < "$out" 2>/dev/null || echo 0) rows"
}

echo ">>> $(date -Is)  generating 4 minimal sets concurrently"
gen request_minimal         requestMinimal        &
gen request_minimal         requestMinimal2       &
gen ai_dilemmas_minimal     aiDilemmasMinimal     &
gen daily_dilemmas_minimal  dailyDilemmasMinimal  &
wait
echo ">>> $(date -Is)  generation done; inspecting"
# The refusal-opener counts are the point here, not a side check: arm 3 removes the three
# sentences that stopped the negative class degenerating into flat refusals, and this is
# where that shows up.
for tag in requestMinimal requestMinimal2 aiDilemmasMinimal dailyDilemmasMinimal; do
    $PY scripts/inspect_generated_set.py --concept hu_harm \
        "data/hu_harm_llama70b_${tag}_600.jsonl" \
        --compare data/hu_harm_llama70b_50.jsonl data/hu_harm_llama70b_600.jsonl \
        2>&1 | sed 's/^/>>>   /'
done
echo ">>> $(date -Is)  minimal-ablation generation finished."
