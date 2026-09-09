#!/usr/bin/env bash
# Split-targeted generation + fits for the OTHER TWO hu_ha splits: ai_dilemmas and
# daily_dilemmas. The generator_experiment_1 counterpart of human_harm_last's ant_hh /
# balanced_refusal study (commit 9fcdda21 there).
#
# Llama-3.3-70B only, matching how that study was first run. Both splits are FULLY PAIRED
# (68/68 and 98/98 distinct user turns carrying both labels), so both are generated in
# PAIRED mode — one user turn, two replies — which is the only way to reproduce the property
# that defines them.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
if [ -f .env2 ]; then set -a; . ./.env2; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
# The same OpenRouter id that wrote the generic control set, so the CONTROL differs from the
# targeted sets in the prompt alone.
MODEL=meta-llama/llama-3.3-70b-instruct
mkdir -p logs

gen () {  # split out_tag
    local split=$1 tag=$2
    local out="data/hu_harm_llama70b_${tag}_600.jsonl"
    if [ -s "$out" ]; then echo ">>> $(date -Is)  SKIP $out (exists)"; return 0; fi
    echo ">>> $(date -Is)  generating $out (split $split)"
    $PY scripts/generate_hu_harm_split_dataset.py --split "$split" --model "$MODEL" \
        --n 600 --batch-size 5 --out "$out" > "logs/gen_split_${tag}.log" 2>&1
    echo ">>> $(date -Is)  $out: $(wc -l < "$out" 2>/dev/null || echo 0) rows"
    $PY scripts/inspect_generated_set.py --concept hu_harm "$out" \
        --compare "data/hu_harm_llama70b_50.jsonl" "data/hu_harm_llama70b_600.jsonl" \
        2>&1 | sed 's/^/>>>   /'
}

gen ai_dilemmas    aiDilemmas
gen daily_dilemmas dailyDilemmas
echo ">>> $(date -Is)  generation done; starting fits"
$PY scripts/fit_hu_harm_split_targeted.py --pair dilemmas --draws 4 \
    >> logs/fit_split_dilemmas.log 2>&1
echo ">>> $(date -Is)  split-targeted dilemmas study finished."
tail -12 logs/fit_split_dilemmas.log
