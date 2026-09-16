#!/usr/bin/env bash
# Hourly checkpoint of the instructions per-part size-curve study to origin/hello_kitty:
# the two result CSVs the fits append to, the four part assignments, and the scripts.
# Only files that exist are named — `git commit -o` errors on a pathspec git has never
# seen. A commit is retried rather than trusted, since another process committing in this
# worktree can hold index.lock.
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=hello_kitty
FILES="scripts/make_instructions_parts.py scripts/fit_instructions_parts.py
scripts/prep_instructions_parts.py run_instrparts_fits.sh hourly_push_instrparts.sh
scripts/instructions_parts_size_curve.csv scripts/instructions_parts_size_curve_accum1.csv
data/instructions_parts/parts_summary.json
data/instructions_deepseekv4pro_tgtmin_hc_context_drift_600.jsonl
data/instructions_deepseekv4pro_tgtmin_oig_context_drift_600.jsonl
analysis/instructions_parts_size_curve.md"
for s in hc_context_drift hc_contradiction mm_substitution oig_context_drift; do
    FILES="$FILES data/instructions_parts/${s}_parts.jsonl"
done

status () {
    local fits
    fits=$(cat scripts/instructions_parts_size_curve.csv \
                scripts/instructions_parts_size_curve_accum1.csv 2>/dev/null \
           | grep -c "^data/instructions_\|^instructions_" || true)
    echo "${fits}/200 part-curve fits"
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
            if git commit -q -o $HAVE -m "data(instrparts): instructions per-part size curve — $(status)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D85KwJVzJA52BCEPFy8Ggm" 2>/dev/null; then ok=1; break; fi
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
