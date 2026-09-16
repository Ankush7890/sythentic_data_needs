#!/usr/bin/env bash
# Hourly checkpoint of the prompt-variant size curve to origin/toolace_stuff: the two curve
# CSVs, the runner, the single-split dev dirs and the --eval-splits change they need.
# Only files that exist are named — `git commit -o` errors on a pathspec git has never seen.
# A commit is retried rather than trusted, since another process in this worktree can hold
# index.lock.
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=toolace_stuff
FILES="scripts/highstakes_toolace_prompts5_sizecurve.csv
scripts/highstakes_anthropic_hh_prompts5_sizecurve.csv
scripts/subsample_curve_concept.py run_prompts5_sizecurve.sh hourly_push_sizecurve.sh
dev_samples/highstakes_500_toolace/toolace_balanced.jsonl
dev_samples/highstakes_500_anthropic_hh/anthropic_hh_balanced.jsonl"

status () {
    local t h
    t=$(( $(grep -c . scripts/highstakes_toolace_prompts5_sizecurve.csv 2>/dev/null || echo 1) - 1 ))
    h=$(( $(grep -c . scripts/highstakes_anthropic_hh_prompts5_sizecurve.csv 2>/dev/null || echo 1) - 1 ))
    echo "${t}/280 toolace fits, ${h}/280 anthropic_hh fits"
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
            if git commit -q -o $HAVE -m "data(prompts5): size curve — $(status)

8 draws at 590/300/150/75/40/20/10, no base data, each study validated on its own split's
125 dev rows and scored on that split alone. n=40/20/10 carry base='none+ga1'.

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
