#!/usr/bin/env bash
# One 600-row set per prompt, for four more eval splits:
#   instructions / hc_context_drift, instructions / oig_context_drift
#   hu_ha        / eval_ant_hh,      hu_ha        / eval_ai_dilemmas
# Same protocol as run_toolace_prompts5_gen.sh and run_hh_prompts5_gen.sh: deepseek-v4-pro,
# 20 rows per call (10 per label), 10 calls in flight per set.
#
# Two splits at a time (10 sets, 6 calls in flight each = 60 concurrent), which is the load
# the earlier five-set runs sustained. Instructions first, per the study's brief.
#
# `--dup-key user+reply` on eval_ai_dilemmas: its prompts write each dilemma TWICE with the
# same user turn and opposite decisions, exactly as the split is built, so keying novelty on
# the user turn alone would drop the second half of every pair.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
export OPENROUTER_TIMEOUT_S="${OPENROUTER_TIMEOUT_S:-1800}"
PY=.venv_claude/bin/python
mkdir -p logs data

# concept:split:promptsmd:tag:dupkey  then  k:name per prompt
run_split() {
    local concept="$1" split="$2" md="$3" tag="$4" dupkey="$5"; shift 5
    for spec in "$@"; do
        local k="${spec%%:*}" name="${spec##*:}"
        local out="data/${concept}_deepseekv4pro_${tag}_${name}_600.jsonl"
        [ -s "$out" ] && { echo ">>> SKIP $out"; continue; }
        $PY scripts/generate_prompt_variants.py --concept "$concept" --split "$split" \
            --prompts-md "$md" --prompt "$k" --n 600 --concurrency 6 \
            --max-tokens 64000 --dup-key "$dupkey" --out "$out" \
            > "logs/gen_${tag}_${name}.log" 2>&1 &
    done
}

echo ">>> $(date -Is) stage 1: instructions (hc_context_drift + oig_context_drift)"
run_split instructions hc_context_drift HC_DRIFT_GENERATOR_PROMPTS.md hcd5 user \
    1:replica 2:grid 3:domains 4:verbose 5:failmodes
run_split instructions oig_context_drift OIG_DRIFT_GENERATOR_PROMPTS.md oigd5 user \
    1:replica 2:overlap 3:domains 4:failmodes 5:long
wait
echo ">>> $(date -Is) stage 1 done"

echo ">>> $(date -Is) stage 2: hu_harm (eval_ant_hh + eval_ai_dilemmas)"
run_split hu_harm eval_ant_hh ANT_HH_GENERATOR_PROMPTS.md anthh5 user \
    1:replica 2:intent 3:nearmiss 4:victim 5:longtail
run_split hu_harm eval_ai_dilemmas AI_DILEMMAS_GENERATOR_PROMPTS.md aidil5 user+reply \
    1:replica 2:grid 3:domains 4:verbose 5:nearmiss
wait
echo ">>> $(date -Is) stage 2 done"

for f in data/instructions_deepseekv4pro_hcd5_*_600.jsonl \
         data/instructions_deepseekv4pro_oigd5_*_600.jsonl \
         data/hu_harm_deepseekv4pro_anthh5_*_600.jsonl \
         data/hu_harm_deepseekv4pro_aidil5_*_600.jsonl; do
    [ -e "$f" ] && echo "$(wc -l < "$f") $f"
done
echo ">>> $(date -Is) all twenty sets finished"
