#!/usr/bin/env bash
set -e
# COMBINATION analysis for INSTRUCTION-FOLLOWING: pool all four attackers' red-team successes
# within each configuration (memo / +eval-desc / +eval-desc→attacker), refit on the union of
# the four 50-row bases, and resample at 90% — eight draws per configuration.
#
# The high-stakes study (experiment_hs_last, run_hs_combined_draws.sh) run on this concept.
# Same design, same controls, same resume semantics; the arms, the eval/dev sets and the base
# data are instruction-following's.
#
# ONE PROCESS PER (combo, draw): a combined draw holds ~4x the activations of a per-arm one
# and the per-draw leak scales with it. fit_combined_draws.py skips any (combo, draw) already
# in the CSV, so this is also the restart path — re-run and it picks up where it stopped.
#
# PASS 1 primes the caches at fraction 1.0: it extracts the 200-row combined base and the
# 404-row dev set once, fits the base-only reference probe, and mints every contrastive pair
# the pooled filter keeps — so the 24 draws of pass 2 are (almost) pure cache hits and don't
# each pay a gemma-3-27b load. The eval blobs are already on disk.
#
# Writes results_ins_combined_draws/combined_draws.csv (90%) and combined_draws_f100.csv
# (the full-pool fits + the base-only reference).
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python

echo ">>> $(date -Is)  PASS 1: base-only reference + full-pool fits (primes the caches)"
$PY scripts/fit_combined_draws.py --fraction 1.0 --draws 1 --base-only --combos combo_memo
$PY scripts/fit_combined_draws.py --fraction 1.0 --draws 1 --combos combo_desc
$PY scripts/fit_combined_draws.py --fraction 1.0 --draws 1 --combos combo_att

echo ">>> $(date -Is)  PASS 2: 8 draws x 90% per configuration"
for combo in combo_memo combo_desc combo_att; do
    for d in 0 1 2 3 4 5 6 7; do
        echo ">>> $(date -Is)  $combo draw $d"
        $PY scripts/fit_combined_draws.py --combos "$combo" --draws $((d+1)) --fraction 0.9
    done
done
echo ">>> $(date -Is)  all combined draws finished."
$PY scripts/fit_combined_draws.py --combos combo_memo combo_desc combo_att --draws 8 --fraction 0.9
