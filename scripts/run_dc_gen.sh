#!/usr/bin/env bash
# The probe transfer statistic on generated samples (docs/dev_coverage_gen_probe_task.md).
#
#   run_dc_gen.sh instructions   kind -> loko, four generators side by side, then the
#                                DeepSeek-validated sensitivity refit, then cells
#   run_dc_gen.sh hu_harm        kind -> loko, four generators side by side, then cells
#   run_dc_gen.sh highstakes     ONE generator at a time (see dc_chain_highstakes.sh for
#                                why), loko first, the unrestricted kind arms last
#
# Every (concept, generator) writes its own CSVs, so the parallel processes never share an
# output file; the harness resumes per (samples, n, draw), so a re-run loses at most one fit.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export AGENTIC_REDTEAM_MAX_MEMORY="${AGENTIC_REDTEAM_MAX_MEMORY:-0=22GiB,cpu=45GiB}"
export MAX_MEMORY="${MAX_MEMORY:-$AGENTIC_REDTEAM_MAX_MEMORY}"
PY=.venv_claude/bin/python
GENS="llama70b gptoss nemotron deepseekv4pro"
mkdir -p logs

parallel_fit() {   # concept, extra args...
    local c=$1; shift
    local pids=()
    for g in $GENS; do
        [ "$*" = "--dsval" ] && [ "$g" = deepseekv4pro ] && continue
        $PY scripts/dc_gen.py --stage fit --concepts "$c" --generators "$g" "$@" \
            >> "logs/dcgen_fit_${c}_${g}${1:+_dsval}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo ">>> $(date -Is) a $c fit process exited non-zero"; done
}

c=$1
echo ">>> $(date -Is) $c"
case $c in
    instructions)
        parallel_fit instructions
        echo ">>> $(date -Is) instructions: DeepSeek-validated kind refits"
        parallel_fit instructions --dsval
        $PY scripts/dc_gen.py --stage cells --concepts instructions --dsval >> logs/dcgen_cells.log 2>&1
        ;;
    hu_harm)
        parallel_fit hu_harm
        ;;
    highstakes)
        for arm in loko kind; do
            for g in $GENS; do
                echo ">>> $(date -Is) highstakes $arm x $g"
                $PY scripts/dc_gen.py --stage fit --concepts highstakes --generators "$g" \
                    --arms "$arm" >> "logs/dcgen_fit_highstakes_${g}.log" 2>&1 \
                    || echo ">>> $(date -Is) FAILED highstakes $arm x $g"
            done
        done
        ;;
esac
$PY scripts/dc_gen.py --stage cells --concepts "$c" >> logs/dcgen_cells.log 2>&1 \
    || echo ">>> $(date -Is) cells for $c FAILED"
echo ">>> $(date -Is) done: $c"
