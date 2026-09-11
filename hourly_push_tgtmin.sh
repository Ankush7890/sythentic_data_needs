#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=per_split_studies2
FILES="scripts/split_specs.py scripts/generate_split_targeted.py scripts/check_minimal_descs.py
SPLIT_TARGETED_PROMPTS_MINIMAL.txt analysis/split_targeted_prompts_minimal.md
run_tgt_minimal_gen.sh run_tgtmin_fits.sh watch_tgtmin.sh hourly_push_tgtmin.sh
scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv
scripts/instructions_tgtmin_size_curve.csv scripts/highstakes_tgtmin_size_curve.csv
run_tgtmin_sizecurve.sh run_tgtnone_backfill.sh
scripts/compare_tgtmin_arms.py analysis/split_targeted_minimal_results.md"
status () {
    local s f c
    s=$(ls data/*_tgtmin_*_600.jsonl 2>/dev/null | wc -l)
    f=$(cat scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv 2>/dev/null \
        | grep -cE '_tgtmin_[a-z_]+_600' || true)
    # `grep -c` on a missing file prints nothing and exits 2, so `|| true` leaves the
    # variable EMPTY and the status line reads "/288". Default it instead.
    c=$(grep -cE '_tgtmin_[a-z_]+_600' scripts/instructions_tgtmin_size_curve.csv 2>/dev/null); c=${c:-0}
    h=$(grep -cE '_tgtmin_[a-z_]+_600' scripts/highstakes_tgtmin_size_curve.csv 2>/dev/null); h=${h:-0}
    echo "${s}/10 shape-free sets, ${f}/90 fits, curve ${c}/288 instr + ${h}/192 hs"
}
while true; do
    ts="$(date -Is)"; HAVE=""
    for f in $FILES $(ls data/*_tgtmin_*_600.jsonl 2>/dev/null); do
        [ -e "$f" ] && { git add -f "$f" 2>/dev/null || true; HAVE="$HAVE $f"; }
    done
    if [ -z "$HAVE" ] || git diff --cached --quiet -- $HAVE 2>/dev/null; then
        echo "[$ts] nothing new — $(status)"
    else
        ok=0
        for try in 1 2 3; do
            git commit -q -o $HAVE -m "data(tgtmin): shape-free split-targeted arm — $(status)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HJq7Y4M9yt6z7NpzX8R5at" 2>/dev/null && { ok=1; break; }
            sleep 20
        done
        if [ "$ok" = 0 ]; then echo "[$ts] COMMIT FAILED — retry next cycle"
        elif git push -q origin "HEAD:$BRANCH" 2>/dev/null; then
            echo "[$ts] pushed $(git rev-parse --short HEAD) — $(status)"
        else echo "[$ts] PUSH FAILED for $(git rev-parse --short HEAD) — committed locally"; fi
    fi
    sleep "$INTERVAL"
done
