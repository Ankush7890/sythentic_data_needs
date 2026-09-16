#!/usr/bin/env bash
# Hourly checkpoint of the four-split prompt-variant study to origin/toolace_stuff: the
# twenty generated sets and the eight result CSVs, as and when they appear.
# Only files that exist are named — `git commit -o` errors on a pathspec git has never
# seen. A commit is retried rather than trusted, since another process in this worktree
# can hold index.lock.
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=toolace_stuff

globs () {
    ls data/instructions_deepseekv4pro_hcd5_*_600.jsonl \
       data/instructions_deepseekv4pro_oigd5_*_600.jsonl \
       data/hu_harm_deepseekv4pro_anthh5_*_600.jsonl \
       data/hu_harm_deepseekv4pro_aidil5_*_600.jsonl \
       scripts/instructions_hcdrift_prompts5*.csv \
       scripts/instructions_oigdrift_prompts5*.csv \
       scripts/hu_harm_ant_hh_prompts5*.csv \
       scripts/hu_harm_ai_dilemmas_prompts5*.csv \
       hourly_push_4splits.sh 2>/dev/null
}

status () {
    local sets rows fits
    sets=$(ls data/*_deepseekv4pro_{hcd5,oigd5,anthh5,aidil5}_*_600.jsonl 2>/dev/null | wc -l)
    fits=0
    for f in scripts/instructions_hcdrift_prompts5_sizecurve.csv \
             scripts/instructions_oigdrift_prompts5_sizecurve.csv \
             scripts/hu_harm_ant_hh_prompts5_sizecurve.csv \
             scripts/hu_harm_ai_dilemmas_prompts5_sizecurve.csv; do
        [ -e "$f" ] && fits=$(( fits + $(grep -c . "$f") - 1 ))
    done
    rows=0
    for f in $(ls data/*_deepseekv4pro_{hcd5,oigd5,anthh5,aidil5}_*_600.jsonl 2>/dev/null); do
        rows=$(( rows + $(grep -c . "$f") ))
    done
    echo "${sets}/20 sets (${rows} rows), ${fits}/1120 curve fits"
}

while true; do
    ts="$(date -Is)"
    HAVE="$(globs | tr '\n' ' ')"
    for f in $HAVE; do git add -f "$f" 2>/dev/null || true; done
    if [ -z "$HAVE" ] || git diff --cached --quiet -- $HAVE 2>/dev/null; then
        echo "[$ts] nothing new — $(status)"
    else
        ok=0
        for try in 1 2 3; do
            if git commit -q -o $HAVE -m "data(prompts5-4splits): $(status)

Five prompts per split for instructions/hc_context_drift, instructions/oig_context_drift,
hu_ha/eval_ant_hh and hu_ha/eval_ai_dilemmas; deepseek-v4-pro, 600 rows per prompt, no
base data, each arm validated on its split's own dev file and scored on that split alone.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FJ3btYgyQicdSbMSJMmCSm" 2>/dev/null; then ok=1; break; fi
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
