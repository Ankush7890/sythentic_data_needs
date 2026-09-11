#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"
PAT='_(tgtshot|tgtnone)_[a-z_]+_600'
( tail -F -n0 logs/tgt_gen_driver.log logs/tgt_gen_rev_driver.log logs/tgt_fits_driver.log 2>/dev/null \
  | grep -E --line-buffered "^>>>|Error|Traceback|FAILED|Killed" \
  | grep -v --line-buffered "not yet generated" ) &   # the work-stealing sweep polls every
                                                      # 2 min; the hourly line covers it
while true; do
    sleep 3600
    sets=$(ls data/*_tgt*_600.jsonl 2>/dev/null | wc -l)
    fits=$(cat scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv 2>/dev/null \
           | grep -cE "$PAT"); fits=${fits:-0}
    gen=$(pgrep -fc "[g]enerate_split_targeted.py"); gen=${gen:-0}   # two drivers now
    warm=$(pgrep -fc "[w]arm_set_activations.py"); warm=${warm:-0}
    fit=$(pgrep -fc "[s]ubsample_curve_concept.py"); fit=${fit:-0}
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d '\n')
    newest=$(ls -t logs/*tgt*.log logs/sizecurve_*.log logs/hs_tiny_*.log logs/nobase_*.log logs/hs_nobase*.log logs/generalist_curve*.log 2>/dev/null | head -1)
    age=$(( $(date +%s) - $(stat -c %Y "$newest" 2>/dev/null || date +%s) ))
    flag=ok
    [ "$gen" = 0 ] && [ "$warm" = 0 ] && [ "$fit" = 0 ] && flag=IDLE
    [ "$age" -gt 2400 ] && flag="STALLED(${age}s)"
    echo "[$(date -Is)] $flag — sets ${sets}/20, fits ${fits}/1448 (+generalist curve) | gen=$gen warm=$warm fit=$fit | GPU $gpu"
done
