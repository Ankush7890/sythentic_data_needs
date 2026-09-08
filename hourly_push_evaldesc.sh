#!/usr/bin/env bash
# Hourly checkpoint of the eval-description-steered generator study to
# origin/generator_experiment_1: the two patched generator scripts, the eight steered
# 600-row sets, the two result CSVs (which also hold the unsteered arms this is read
# against), the 500-row high-stakes dev cut those CSVs validate on, and the two runners.
#
# THE COMMIT NAMES ONLY FILES THAT EXIST — `git commit -o <path>` errors on a pathspec git
# has never seen, and the steered sets appear one at a time as generation finishes. The
# adds are narrowly scoped and force-added (the runners match the .gitignore run*.sh rule,
# as run_gen_combo_studies.sh already does on this branch); a commit is retried rather than
# trusted, since another process committing in this worktree can hold index.lock.
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=generator_experiment_1
FILES="scripts/generate_instructions_dataset.py scripts/generate_highstakes_dataset.py
scripts/inspect_generated_set.py scripts/warm_set_activations.py
scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv
dev_samples/highstakes_500
run_evaldesc_gen.sh run_evaldesc_fits.sh hourly_push_evaldesc.sh watch_evaldesc.sh"
for g in llama70b gptoss deepseekv4pro nemotron; do
    FILES="$FILES data/instructions_${g}_evaldesc_600.jsonl data/highstakes_${g}_evaldesc_600.jsonl"
done

status () {
    local sets fits
    sets=$(ls data/*_evaldesc_600.jsonl 2>/dev/null | wc -l)
    fits=$(cat scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv 2>/dev/null \
           | grep -c "evaldesc_600" || true)
    echo "${sets}/8 steered sets, ${fits} steered fits"
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
            if git commit -q -o $HAVE -m "data(gen): eval-description-steered generation — $(status)

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
