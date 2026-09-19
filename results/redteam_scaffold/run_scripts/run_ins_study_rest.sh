#!/usr/bin/env bash
set -e
# Stages 2-3 of the instruction-following combination study, chained so the GPU never idles
# between them. Stage 1 (run_ins_combined_draws.sh — the k=4 grid) is expected to be running
# already; this script waits it out first.
#
# THE FIXED-BASE SUBSET SWEEP IS NOT RUN. High-stakes has both designs — every subset on all
# 200 base rows (so the red-team pool is the only thing varying with k) and every subset on
# its own 50*k rows — and they agree at k=4 by construction. Only the subset-base design is
# run here: it is the one that says what a practitioner who had actually run those k
# attackers would have got, and it is the one the high-stakes reading ultimately turned on
# (measuring each cell against its OWN no-red-team reference is what showed red-team data
# LEVELS rather than adds). run_ins_subset_draws.sh is kept on disk and needs nothing built
# that this chain does not already build, so the fixed-base grid can be filled in later.
#
# The order is not arbitrary. The base-only references come SECOND because their four
# singleton fits are what EXTRACT the four per-attacker 50-row base blobs, and
# build_subset_base_activations.py merges the pair/triple blobs out of those — so the
# subset-base sweep has no extraction left to do by the time it starts.
#
# One GPU: every stage is sequential, and each stage's own wait-guard
# (`pgrep -f "fit_combined_draws\.py --combos"`) keeps two fits off the card even if this
# script is restarted while one is mid-flight.
cd "$(dirname "${BASH_SOURCE[0]}")"

echo ">>> $(date -Is)  waiting for the k=4 grid to finish"
while pgrep -f "run_ins_combined_draws\.sh" >/dev/null 2>&1; do sleep 60; done
while pgrep -f "fit_combined_draws\.py --combos" >/dev/null 2>&1; do sleep 60; done

echo ">>> $(date -Is)  STAGE 2: base-only references (also extracts the four 50-row base blobs)"
bash run_ins_baseonly_refs.sh

echo ">>> $(date -Is)  STAGE 3: subset-base subset sweep (k = 1, 2, 3)"
bash run_ins_subsetbase_draws.sh

echo ">>> $(date -Is)  instruction combination study complete."
