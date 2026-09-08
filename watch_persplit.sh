#!/usr/bin/env bash
# Stage boundaries + failures from the two per-split drivers, plus an hourly progress line
# (a stalled driver emits no boundary, so the hourly line is what would show it).
cd "$(dirname "${BASH_SOURCE[0]}")"
( tail -F -n0 logs/persplit_gen_driver.log logs/persplit_fits_driver.log 2>/dev/null \
  | grep -E --line-buffered "^>>>|Error|Traceback|FAILED|Killed" ) &
while true; do
    sleep 3600
    sets=$(ls data/*_deepseekv4pro_*_600.jsonl 2>/dev/null | grep -vc evaldesc)
    fits=$(cat scripts/instructions_gen90.csv scripts/highstakes_gen90_dev500.csv 2>/dev/null \
           | grep -c "deepseekv4pro_[a-z_]*_600"); fits=${fits:-0}
    gen=$(pgrep -fc "[g]enerate_(instructions|highstakes)_dataset.py"); gen=${gen:-0}
    warm=$(pgrep -fc "[w]arm_set_activations.py"); warm=${warm:-0}
    fit=$(pgrep -fc "[s]ubsample_curve_concept.py"); fit=${fit:-0}
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d '\n')
    newest=$(ls -t logs/*persplit*.log 2>/dev/null | head -1)
    age=$(( $(date +%s) - $(stat -c %Y "$newest" 2>/dev/null || date +%s) ))
    flag="ok"
    [ "$gen" = 0 ] && [ "$warm" = 0 ] && [ "$fit" = 0 ] && flag="IDLE"
    [ "$age" -gt 2400 ] && flag="STALLED(${age}s since any per-split log write)"
    echo "[$(date -Is)] $flag — sets ${sets}/10, fits ${fits}/90 | gen=$gen warm=$warm fit=$fit | GPU $gpu"
done
