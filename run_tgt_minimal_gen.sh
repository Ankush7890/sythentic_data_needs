#!/usr/bin/env bash
# SHAPE-FREE ("minimal") split-targeted generation — the arm-3 ablation.
#
# Same script and same splits as run_tgt_gen.sh, with --minimal swapping the measured
# description for one that carries the SITUATION and the LABEL BOUNDARY and nothing measured
# off the split file: no turn counts, no character lengths, no pinned system turn, no pairing,
# no topic list, no anchor. So a minimal set differs from a tgtnone set by exactly that, and
# the three arms are directly comparable on the same ten splits.
#
# The same ablation on hu_harm (generator_experiment_1, 5c59c568) INVERTED which split a
# targeted set was good at, which is why it is worth running here. See
# analysis/split_targeted_prompts_minimal.md.
#
# deepseek-v4-pro throughout; 600 rows per set, 300 per label. Every split is unpaired here,
# so one generated item is one row.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
GEN="${GEN:-deepseek/deepseek-v4-pro}"
TAG="${TAG:-deepseekv4pro}"
mkdir -p logs

ARMS="
instructions anthropic_harmless_refusal
instructions bbq_substitution
instructions hc_context_drift
instructions hc_contradiction
instructions mm_substitution
instructions oig_context_drift
highstakes anthropic_hh_balanced
highstakes mt_balanced
highstakes mts_balanced
highstakes toolace_balanced
"

echo "$ARMS" | while read -r concept split; do
    [ -n "${concept:-}" ] || continue
    out="data/${concept}_${TAG}_tgtmin_${split}_600.jsonl"
    if [ -s "$out" ]; then echo ">>> $(date -Is)  SKIP $out"; continue; fi
    echo ">>> $(date -Is)  generating $out"
    $PY scripts/generate_split_targeted.py --concept "$concept" --split "$split" \
        --model "$GEN" --n 600 --minimal --out "$out" \
        > "logs/gen_${concept}_tgtmin_${split}.log" 2>&1
    echo ">>> $(date -Is)  $out: $(wc -l < "$out" 2>/dev/null || echo 0) rows"
    $PY scripts/inspect_generated_set.py --concept "$concept" "$out" \
        --compare "data/${concept}_${TAG}_50.jsonl" 2>&1 | sed 's/^/>>>   /'
done
echo ">>> $(date -Is)  all shape-free split-targeted generation finished."
