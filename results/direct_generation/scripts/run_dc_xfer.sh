#!/usr/bin/env bash
# Probe transfer scored on the classifier's own source (docs/dev_coverage_xfer_task.md).
#
#   run_dc_xfer.sh warm <concept>   one extraction load: per-sample cache + 15 blobs
#   run_dc_xfer.sh fit  <concept>   dev own arms, then the generated kind arms
#                                   (instructions / hu_harm: generators PAR at a time;
#                                   highstakes: one at a time, unrestricted eval)
#   run_dc_xfer.sh cells <concept>
#
# Every (concept, generator) writes its own CSVs, so parallel processes never share an
# output file; the harness resumes per fit, so a re-run loses at most one.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
GENS="llama70b gptoss nemotron deepseekv4pro"
PAR="${PAR:-2}"
mkdir -p logs

stage=$1; c=$2
echo ">>> $(date -Is) $stage $c"
case $stage in
    warm)
        $PY scripts/dc_xfer.py --stage warm --concepts "$c" >> "logs/dcxfer_warm_${c}.log" 2>&1
        ;;
    fit)
        $PY scripts/dc_xfer.py --stage fit --kind dev --concepts "$c" \
            >> "logs/dcxfer_fit_${c}_dev.log" 2>&1 || echo ">>> $(date -Is) $c dev FAILED"
        if [ "$c" = highstakes ]; then PAR=1; fi
        set -- $GENS
        while [ $# -gt 0 ]; do
            pids=()
            for _ in $(seq "$PAR"); do
                [ $# -gt 0 ] || break
                g=$1; shift
                $PY scripts/dc_xfer.py --stage fit --kind gen --concepts "$c" --generators "$g" \
                    >> "logs/dcxfer_fit_${c}_${g}.log" 2>&1 &
                pids+=($!)
            done
            for p in "${pids[@]}"; do wait "$p" || echo ">>> $(date -Is) a $c gen fit exited non-zero"; done
        done
        ;;
    cells)
        $PY scripts/dc_xfer.py --stage cells --concepts "$c" >> logs/dcxfer_cells.log 2>&1
        ;;
esac
echo ">>> $(date -Is) done: $stage $c"
