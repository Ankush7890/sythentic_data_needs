#!/usr/bin/env bash
# Hourly checkpoint of the QWEN3-8B pooled size curve to origin/qwen8b.
#
# The pools (.pool_work_qwen8b/) and the probe templates are NOT committed — both are derived
# deterministically by scripts/build_qwen8b_pools.py and scripts/make_qwen8b_probe_templates.py.
#
# run*.sh is gitignored, so runners are force-added. Only paths that exist are named: `git
# commit -o` errors on a pathspec git has never seen, and the per-concept CSVs appear one at a
# time. Commits and pushes only while qwen8b is the checked-out branch, so a checkout of another
# branch in this worktree can never receive these commits.
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=qwen8b
FILES="scripts/qwen8b_hu_harm_pooled_size_curve.csv scripts/qwen8b_instructions_pooled_size_curve.csv
scripts/qwen8b_highstakes_pooled_size_curve.csv
scripts/build_qwen8b_pools.py scripts/make_qwen8b_probe_templates.py scripts/fit_base_plus_concept.py
scripts/build_pooled_sets.py scripts/warm_pooled_sets.py scripts/subsample_curve_concept.py
run_qwen8b_sizecurve.sh hourly_push_qwen8b.sh"

rows () { [ -f "$1" ] && echo $(( $(wc -l < "$1") - 1 )) || echo 0; }

status () {
    echo "$(rows scripts/qwen8b_hu_harm_pooled_size_curve.csv) hu_harm + \
$(rows scripts/qwen8b_instructions_pooled_size_curve.csv) instructions + \
$(rows scripts/qwen8b_highstakes_pooled_size_curve.csv) highstakes fits (targets 896 / 1120 / 448)"
}

while true; do
    ts="$(date -Is)"
    cur="$(git rev-parse --abbrev-ref HEAD)"
    if [ "$cur" != "$BRANCH" ]; then
        echo "[$ts] SKIP — checked-out branch is $cur, not $BRANCH"
        sleep "$INTERVAL"; continue
    fi
    HAVE=""
    for f in $FILES; do
        if [ -e "$f" ]; then git add -f "$f" 2>/dev/null || true; HAVE="$HAVE $f"; fi
    done
    if [ -n "$HAVE" ] && ! git diff --cached --quiet -- $HAVE 2>/dev/null; then
        ok=0
        for try in 1 2 3; do
            if git commit -q -o $HAVE -m "data(qwen8b): pooled size curve on Qwen3-8B L18 — $(status)

52 pools (each 600-row set ∪ its generator's own 50), n = 590 350 170 110 80 30 10, 8 draws (highstakes 4),
--no-base so every training row is resampled. Automated hourly checkpoint.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D85KwJVzJA52BCEPFy8Ggm" 2>/dev/null; then ok=1; break; fi
            sleep 20
        done
        [ "$ok" = 0 ] && echo "[$ts] COMMIT FAILED (index busy?) — will retry next cycle"
    fi
    # Push every cycle, not only after a new commit, so a failed push is retried.
    if git push -q origin "$BRANCH:$BRANCH" 2>/dev/null; then
        echo "[$ts] pushed $(git rev-parse --short "$BRANCH") — $(status)"
    else
        echo "[$ts] PUSH FAILED for $(git rev-parse --short "$BRANCH") — committed locally"
    fi
    sleep "$INTERVAL"
done
