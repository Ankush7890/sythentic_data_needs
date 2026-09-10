#!/usr/bin/env bash
# Hourly checkpoint of the split-targeted size curve to origin/generator_experiment_1.
#
# Commits only the result JSONs (and the two scripts driving them), and only when something
# has actually changed. `git commit -o <path>` errors on a pathspec git has never seen, so
# each path is added only if it exists; the adds are force-added because analysis/ result
# dirs and run*.sh are gitignored. A commit is retried rather than trusted: another process
# in this worktree can hold index.lock.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=generator_experiment_1
FILES="analysis/refit_studies/hu_harm_split_targeted_refusal
analysis/refit_studies/hu_harm_split_targeted_dilemmas
scripts/fit_hu_harm_split_targeted.py
run_split_sizecurve_hu_harm.sh hourly_push_sizecurve.sh"

count () { ls analysis/refit_studies/hu_harm_split_targeted_*/*_f*b_d*.json 2>/dev/null | wc -l; }

while true; do
    ts="$(date -Is)"
    HAVE=""
    for f in $FILES; do
        if [ -e "$f" ]; then git add -f "$f" 2>/dev/null || true; HAVE="$HAVE $f"; fi
    done
    if [ -z "$HAVE" ] || git diff --cached --quiet -- $HAVE 2>/dev/null; then
        echo "[$ts] nothing new — $(count)/224 curve fits"
    else
        for attempt in 1 2 3; do
            if git commit -q -m "data(hu_harm): split-targeted size curve — $(count)/224 fits" \
                   -m "8 class-balanced draws at 50/20/10/5% of each split-targeted set, all four hu_ha splits, generic_600 carried at every fraction as the control. Automated hourly checkpoint." \
                   -o $HAVE 2>/dev/null; then
                break
            fi
            sleep 20
        done
        git push -q origin "$BRANCH" 2>/dev/null \
            && echo "[$ts] pushed — $(count)/224 curve fits" \
            || echo "[$ts] commit ok, PUSH FAILED — $(count)/224 curve fits"
    fi
    sleep "$INTERVAL"
done
