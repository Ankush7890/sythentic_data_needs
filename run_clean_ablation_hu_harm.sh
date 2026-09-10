#!/usr/bin/env bash
# CLEAN-PROMPT ABLATION, two arms, human-harm concept.
#
# Same script, same model, same 600 rows, same n and batch size as the split-targeted sets
# it is compared against. The ONLY difference is the prompt: every confounder removed (see
# analysis/split_targeted_prompts_clean.md and the _CLEAN block in the generator).
#
#   ARM A (paired)     ant_hh_clean unpaired + the other three PAIRED, exactly the modes the
#                      original split-targeted sets used, so A-vs-original isolates the
#                      prompt text alone.
#   ARM B (per-label)  the same clean text with every split generated ONE LABEL AT A TIME.
#                      This deliberately breaks the pairing of three splits, so B-vs-A
#                      isolates pairing.
#
# ant_hh is already unpaired, so it is generated ONCE and belongs to both arms.
#
# The seven sets are generated CONCURRENTLY (the generator itself is sequential and has no
# --concurrency flag); that is 7 API streams, fewer than the 8 the earlier arms used.
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

echo ">>> $(date -Is)  generating 7 clean sets concurrently"
gen ant_hh_clean                 antHHclean            &
gen balanced_refusal_clean       refusalClean          &
gen ai_dilemmas_clean            aiDilemmasClean       &
gen daily_dilemmas_clean         dailyDilemmasClean    &
gen balanced_refusal_clean_pl    refusalCleanPL        &
gen ai_dilemmas_clean_pl         aiDilemmasCleanPL     &
gen daily_dilemmas_clean_pl      dailyDilemmasCleanPL  &
wait
echo ">>> $(date -Is)  generation done; inspecting"
for tag in antHHclean refusalClean aiDilemmasClean dailyDilemmasClean \
           refusalCleanPL aiDilemmasCleanPL dailyDilemmasCleanPL; do
    $PY scripts/inspect_generated_set.py --concept hu_harm \
        "data/hu_harm_llama70b_${tag}_600.jsonl" \
        --compare data/hu_harm_llama70b_50.jsonl data/hu_harm_llama70b_600.jsonl \
        2>&1 | sed 's/^/>>>   /'
done
echo ">>> $(date -Is)  clean-ablation generation finished."
