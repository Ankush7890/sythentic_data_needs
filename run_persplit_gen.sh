#!/usr/bin/env bash
# PER-SPLIT steered generation: one 600-row set per eval split, all from deepseek-v4-pro.
#
# Same script and same eval-data description as the mixed steered sets, with --kind N
# pinning EVERY call to one numbered kind instead of rotating over all of them. So a
# per-split set differs from the mixed one by exactly that pin, and the arms are directly
# comparable to each other and to the mixed arm already in the two CSVs.
#
# THE KIND -> SPLIT MAPPING. For instructions each kind names its split in parentheses, so
# the mapping is read off the description itself. For high-stakes the kinds do not name
# splits; the correspondence below is positional and was checked against the split contents
# (open-ended chatbot -> anthropic_hh, clinical document -> mt, doctor-patient dialogue ->
# mts, tool-calling agent -> toolace).
#
# oig_omission has NO kind in the verbatim description (the red-team branch removed that
# split before the text was written) and is deliberately left out: it stays the untouched
# control the mixed arm already measured at -0.046.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=.venv_claude/bin/python
# One generator per invocation; the arms are named after it, so two generators' studies
# live side by side in the same CSVs without colliding on the resume key.
GEN="${GEN:-deepseek/deepseek-v4-pro}"
TAG="${TAG:-deepseekv4pro}"
mkdir -p logs

# "<concept> <kind index> <split stem>"
ARMS="
instructions 1 anthropic_harmless_refusal
instructions 2 bbq_substitution
instructions 3 hc_context_drift
instructions 4 hc_contradiction
instructions 5 mm_substitution
instructions 6 oig_context_drift
highstakes 1 anthropic_hh_balanced
highstakes 2 mt_balanced
highstakes 3 mts_balanced
highstakes 4 toolace_balanced
"

echo "$ARMS" | while read -r concept kind split; do
    [ -n "${concept:-}" ] || continue
    out="data/${concept}_${TAG}_${split}_600.jsonl"
    if [ -s "$out" ]; then echo ">>> $(date -Is)  SKIP $out (exists)"; continue; fi
    echo ">>> $(date -Is)  generating $out (kind $kind)"
    # call-budget-factor 8, not 6: 300 rows per label all inside ONE kind is far more
    # dedup pressure than the mixed sets, so more calls come back as near-repeats.
    $PY "scripts/generate_${concept}_dataset.py" \
        --model "$GEN" --n-per-label 300 --batch-size 5 --concurrency 8 \
        --call-budget-factor 8 --max-tokens 8192 --eval-description --kind "$kind" \
        --out "$out" > "logs/gen_${concept}_${split}_persplit.log" 2>&1
    echo ">>> $(date -Is)  $out: $(wc -l < "$out" 2>/dev/null || echo 0) rows"
    $PY scripts/inspect_generated_set.py --concept "$concept" "$out" \
        --compare "data/${concept}_${TAG}_50.jsonl" "data/${concept}_${TAG}_evaldesc_600.jsonl" \
        2>&1 | sed 's/^/>>>   /'
done
echo ">>> $(date -Is)  all per-split generation finished."
