#!/usr/bin/env bash
# SPLIT-TARGETED generation, two variants per split: WITH the dev-derived few-shot anchor and
# WITHOUT it. The two prompts differ by that block alone, so the pair measures how much of a
# targeted set's quality comes from the measured DESCRIPTION and how much from seeing two
# real rows.
#
# deepseek-v4-pro throughout; 600 rows per set. Paired splits emit two rows per generated
# item (that IS the pairing), unpaired splits one.
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
    for variant in tgtshot tgtnone; do
        flag=""; [ "$variant" = tgtnone ] && flag="--no-shots"
        out="data/${concept}_${TAG}_${variant}_${split}_600.jsonl"
        if [ -s "$out" ]; then echo ">>> $(date -Is)  SKIP $out"; continue; fi
        echo ">>> $(date -Is)  generating $out"
        $PY scripts/generate_split_targeted.py --concept "$concept" --split "$split" \
            --model "$GEN" --n 600 $flag --out "$out" \
            > "logs/gen_${concept}_${variant}_${split}.log" 2>&1
        echo ">>> $(date -Is)  $out: $(wc -l < "$out" 2>/dev/null || echo 0) rows"
        $PY scripts/inspect_generated_set.py --concept "$concept" "$out" \
            --compare "data/${concept}_${TAG}_50.jsonl" 2>&1 | sed 's/^/>>>   /'
    done
done
echo ">>> $(date -Is)  all split-targeted generation finished."
