#!/usr/bin/env bash
# Hourly checkpoint of the two generator combination studies to origin/generator_experiment_1.
#
# THE COMMIT NAMES ONLY FILES THAT EXIST. `git commit -o <paths>` errors on a pathspec git
# has never seen, and highstakes_gen_combo_draws.csv does not exist until study ii starts —
# so the path list is rebuilt each cycle from what is actually on disk, not from FILES.
#
# THE ADDS ARE NARROWLY SCOPED — the two CSVs, the script and this runner, named explicitly.
# Another session works in this same worktree on the same branch, so a `git add -A` here
# would sweep its in-progress files into my commit. For the same reason a commit is retried
# rather than trusted: two processes committing in one worktree can collide on index.lock.
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=generator_experiment_1
FILES="scripts/gen_combo_draws.py scripts/instructions_gen_combo_draws.csv scripts/highstakes_gen_combo_draws.csv run_gen_combo_studies.sh hourly_push_gen.sh"

n_fits () {
    cat scripts/instructions_gen_combo_draws.csv scripts/highstakes_gen_combo_draws.csv \
        2>/dev/null | grep -c "^[dgln]\+," || true
}

while true; do
    ts="$(date -Is)"
    HAVE=""
    for f in $FILES; do
        if [ -e "$f" ]; then git add -f "$f" 2>/dev/null || true; HAVE="$HAVE $f"; fi
    done
    if [ -z "$HAVE" ] || git diff --cached --quiet -- $HAVE 2>/dev/null; then
        echo "[$ts] nothing new — $(n_fits) generator-combo fits"
    else
        ok=0
        for try in 1 2 3; do
            if git commit -q -o $HAVE -m "data(gen): generator combination study progress — $(n_fits) fits

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KPsRsNsnnjqVKxibK2asm2" 2>/dev/null; then ok=1; break; fi
            sleep 20
        done
        if [ "$ok" = 0 ]; then
            echo "[$ts] COMMIT FAILED (index busy?) — will retry next cycle"
        elif git push -q origin "HEAD:$BRANCH" 2>/dev/null; then
            echo "[$ts] pushed $(git rev-parse --short HEAD) — $(n_fits) generator-combo fits"
        else
            echo "[$ts] PUSH FAILED for $(git rev-parse --short HEAD) — committed locally"
        fi
    fi
    sleep "$INTERVAL"
done
