#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=per_split_studies
FILES="scripts/split_specs.py scripts/generate_split_targeted.py scripts/inspect_generated_set.py
SPLIT_TARGETED_PROMPTS.txt run_tgt_gen.sh run_tgt_fits.sh watch_tgt.sh hourly_push_tgt.sh
run_generalist_sizecurve.sh run_hu_harm_generalist_curve.sh
data/hu_harm_llama70b_evaldescshape_600.jsonl data/hu_harm_gptoss_evaldescshape_600.jsonl
data/hu_harm_deepseekv4pro_evaldescshape_600.jsonl data/hu_harm_nemotron_evaldescshape_600.jsonl
scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv scripts/hu_harm_gen90.csv"
status () {
    local s f
    s=$(ls data/*_tgt*_600.jsonl 2>/dev/null | wc -l)
    f=$(cat scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv scripts/hu_harm_gen90.csv 2>/dev/null \
        | grep -cE '_(tgtshot|tgtnone)_[a-z_]+_600' || true)
    echo "${s}/20 split-targeted sets, ${f} fits"
}
while true; do
    ts="$(date -Is)"; HAVE=""
    for f in $FILES $(ls data/*_tgt*_600.jsonl 2>/dev/null); do
        [ -e "$f" ] && { git add -f "$f" 2>/dev/null || true; HAVE="$HAVE $f"; }
    done
    if [ -z "$HAVE" ] || git diff --cached --quiet -- $HAVE 2>/dev/null; then
        echo "[$ts] nothing new — $(status)"
    else
        ok=0
        for try in 1 2 3; do
            git commit -q -o $HAVE -m "data(tgt): split-targeted specialists — $(status)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KPsRsNsnnjqVKxibK2asm2" 2>/dev/null && { ok=1; break; }
            sleep 20
        done
        if [ "$ok" = 0 ]; then echo "[$ts] COMMIT FAILED — retry next cycle"
        elif git push -q origin "HEAD:$BRANCH" 2>/dev/null; then
            echo "[$ts] pushed $(git rev-parse --short HEAD) — $(status)"
        else echo "[$ts] PUSH FAILED for $(git rev-parse --short HEAD) — committed locally"; fi
    fi
    sleep "$INTERVAL"
done
