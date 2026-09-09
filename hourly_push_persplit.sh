#!/usr/bin/env bash
# Hourly checkpoint of the per-split steered study to origin/generator_experiment_1: the ten
# per-split sets, the two result CSVs they append to, the generators carrying --kind, and
# the runners. Only files that exist are named — `git commit -o` errors on a pathspec git
# has never seen, and the sets appear one at a time. A commit is retried rather than
# trusted, since another process committing in this worktree can hold index.lock.
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=generator_experiment_1
FILES="scripts/generate_instructions_dataset.py scripts/generate_highstakes_dataset.py
scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv
run_persplit_gen.sh run_persplit_fits.sh hourly_push_persplit.sh watch_persplit.sh"
for t in deepseekv4pro llama70b; do
    for s in anthropic_harmless_refusal bbq_substitution hc_context_drift hc_contradiction \
             mm_substitution oig_context_drift; do
        FILES="$FILES data/instructions_${t}_${s}_600.jsonl"
    done
    for s in anthropic_hh_balanced mt_balanced mts_balanced toolace_balanced; do
        FILES="$FILES data/highstakes_${t}_${s}_600.jsonl"
    done
done

status () {
    local sets fits
    sets=$(ls data/*_600.jsonl 2>/dev/null | grep -cE "_(anthropic_harmless_refusal|bbq_substitution|hc_context_drift|hc_contradiction|mm_substitution|oig_context_drift|anthropic_hh_balanced|mt_balanced|mts_balanced|toolace_balanced)_600")
    fits=$(cat scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv 2>/dev/null \
           | grep -cE "_(anthropic_harmless_refusal|bbq_substitution|hc_context_drift|hc_contradiction|mm_substitution|oig_context_drift|anthropic_hh_balanced|mt_balanced|mts_balanced|toolace_balanced)_600" || true)
    echo "${sets}/20 per-split sets, ${fits} per-split fits"
}

while true; do
    ts="$(date -Is)"
    HAVE=""
    for f in $FILES; do
        if [ -e "$f" ]; then git add -f "$f" 2>/dev/null || true; HAVE="$HAVE $f"; fi
    done
    if [ -z "$HAVE" ] || git diff --cached --quiet -- $HAVE 2>/dev/null; then
        echo "[$ts] nothing new — $(status)"
    else
        ok=0
        for try in 1 2 3; do
            if git commit -q -o $HAVE -m "data(gen): per-split steered generation — $(status)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KPsRsNsnnjqVKxibK2asm2" 2>/dev/null; then ok=1; break; fi
            sleep 20
        done
        if [ "$ok" = 0 ]; then
            echo "[$ts] COMMIT FAILED (index busy?) — will retry next cycle"
        elif git push -q origin "HEAD:$BRANCH" 2>/dev/null; then
            echo "[$ts] pushed $(git rev-parse --short HEAD) — $(status)"
        else
            echo "[$ts] PUSH FAILED for $(git rev-parse --short HEAD) — committed locally"
        fi
    fi
    sleep "$INTERVAL"
done
