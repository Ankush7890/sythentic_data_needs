#!/usr/bin/env bash
# Hourly checkpoint of the instruction-following combination study to
# origin/experiment_instruction_last.
#
# WHAT IT COMMITS: the study's CSVs, the subset base files the sweep mints on demand, and
# the scripts/runners — nothing else. Deliberately NOT the probe pickles or the activation
# blobs: probes/ins_combined_draws*/ is ~300 single-probe pickles and the shared cache is
# tens of GB, neither of which belongs in the branch (the high-stakes study committed its
# CSVs only, for the same reason).
#
# `results*` and `run*.sh` are gitignored on this branch, so the adds are FORCED — matching
# how the high-stakes results were committed.
#
# One line of stdout per cycle, so it can be watched as a monitor.
cd "$(dirname "${BASH_SOURCE[0]}")"
INTERVAL="${INTERVAL:-3600}"
BRANCH=experiment_instruction_last

n_fits () {   # scored fits so far, across all three CSVs (one row per fit carries dataset=mean)
    cat results_ins_combined_draws/*.csv 2>/dev/null | awk -F, '$5=="mean"' | wc -l
}

while true; do
    git add -f results_ins_combined_draws/*.csv 2>/dev/null || true
    git add data/instructions_base_*.jsonl data/instructions_combined_200.jsonl 2>/dev/null || true
    git add scripts/fit_combined_draws.py scripts/build_subset_base_activations.py 2>/dev/null || true
    git add -f run_ins_*.sh hourly_push_ins.sh 2>/dev/null || true
    ts="$(date -Is)"
    if git diff --cached --quiet; then
        echo "[$ts] nothing new — $(n_fits) scored fits"
    else
        git commit -q -m "data(instructions): combination study progress — $(n_fits) scored fits

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KPsRsNsnnjqVKxibK2asm2"
        if git push -q origin "HEAD:$BRANCH" 2>&1; then
            echo "[$ts] pushed $(git rev-parse --short HEAD) — $(n_fits) scored fits"
        else
            echo "[$ts] PUSH FAILED for $(git rev-parse --short HEAD) — committed locally"
        fi
    fi
    sleep "$INTERVAL"
done
