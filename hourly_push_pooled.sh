#!/usr/bin/env bash
# Hourly checkpoint of the POOLED size curve to origin/per_split_studies.
#
# What this study is: every training row resampled, base included. The pools are NOT
# committed — .pool_work/ is derived deterministically from data/ by build_pooled_sets.py,
# so the script is the artefact and the 650-row files are reproducible from it.
#
# run*.sh is gitignored (see .gitignore:234), so the runners are force-added. Only paths
# that exist are named: `git commit -o` errors on a pathspec git has never seen, and the
# per-concept CSVs appear one at a time. A commit is retried rather than trusted, since
# another process in this worktree can hold index.lock.
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=per_split_studies
FILES="scripts/build_pooled_sets.py scripts/warm_pooled_sets.py scripts/report_pooled_curve.py
scripts/subsample_curve_concept.py
scripts/hu_harm_pooled_size_curve.csv scripts/instructions_pooled_size_curve.csv
scripts/highstakes_pooled_size_curve.csv
run_pooled_sizecurve.sh run_pooled_sizecurve2.sh run_pooled_all.sh hourly_push_pooled.sh"

# Data rows = lines minus the header. The redirect is guarded by -f rather than by 2>/dev/null
# on wc: a `< missing` redirect fails in the SHELL before wc runs, so the message comes from
# the shell and wc's own stderr redirect never sees it.
rows () { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }

status () {
    echo "$(rows scripts/hu_harm_pooled_size_curve.csv) hu_harm + \
$(rows scripts/instructions_pooled_size_curve.csv) instructions + \
$(rows scripts/highstakes_pooled_size_curve.csv) highstakes pooled fits"
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
            if git commit -q -o $HAVE -m "data(pooled): size curve with every training row resampled — $(status)

Base rows are pooled into the set they are drawn from rather than held constant, so the
spread across draws is the spread of a training set of that size. Unsteered arm is on each
generator's OWN 50-row base, not the llama70b base scripts/<concept>_size_curve.csv used.
Automated hourly checkpoint.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HJq7Y4M9yt6z7NpzX8R5at" 2>/dev/null; then ok=1; break; fi
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
