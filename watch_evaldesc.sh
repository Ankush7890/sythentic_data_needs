#!/usr/bin/env bash
# Two event streams in one watch: stage boundaries + failures from the two drivers, and an
# hourly PROGRESS line. The hourly line is what catches a STALL — a driver alive but
# producing nothing — which the boundary stream cannot show, because a stalled run emits no
# boundary either. It names the stage each concept is in and how far through it is, so a
# reader can tell "generating, 140/300" from "idle" without opening a log.
cd "$(dirname "${BASH_SOURCE[0]}")"
GENS="llama70b gptoss deepseekv4pro nemotron"

( tail -F -n0 logs/evaldesc_gen_driver.log logs/evaldesc_fits_driver.log 2>/dev/null \
  | grep -E --line-buffered "^>>>|Error|Traceback|FAILED|Killed|MISSING" \
  | grep -v --line-buffered "waiting on" ) &   # the wait poll fires every 2 min; the hourly line covers it

last_line () {  # file pattern — the newest matching line, trimmed
    [ -f "$1" ] && grep -E "$2" "$1" 2>/dev/null | tail -1 | sed 's/^ *//' || true
}

while true; do
    sleep 3600
    ts="$(date -Is)"
    sets=$(ls data/*_evaldesc_600.jsonl 2>/dev/null | wc -l)
    # 36 steered fits per concept: 4 sets x (one n=600 fit + eight n=540 draws).
    # `grep -c` on a missing file prints nothing and exits 1, and `|| echo 0` would then
    # append a SECOND line to a value that is already "0" — hence the default, not a fallback.
    fi=$(grep -c "evaldesc_600" scripts/instructions_gen90.csv 2>/dev/null); fi=${fi:-0}
    fh=$(grep -c "evaldesc_600" scripts/highstakes_gen90_dev500.csv 2>/dev/null); fh=${fh:-0}
    # Bracketed first character so the pattern cannot match the shell that is evaluating it.
    gen=$(pgrep -fc "[g]enerate_(instructions|highstakes)_dataset.py"); gen=${gen:-0}
    warm=$(pgrep -fc "[w]arm_set_activations.py"); warm=${warm:-0}
    fit=$(pgrep -fc "[s]ubsample_curve_concept.py"); fit=${fit:-0}
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | tr -d '\n')
    newest=$(ls -t logs/*.log 2>/dev/null | head -1)
    age=$(( $(date +%s) - $(stat -c %Y "$newest" 2>/dev/null || date +%s) ))

    genline=""
    for c in instructions highstakes; do for g in $GENS; do
        l=$(last_line "logs/gen_${c}_${g}_evaldesc.log" "after wave")
        [ -n "$l" ] && [ ! -s "data/${c}_${g}_evaldesc_600.jsonl" ] && genline="$c/$g $(echo "$l" | grep -o '[0-9]*/300')"
    done; done
    warmline=$(last_line logs/warm_evaldesc_instructions.log "sample activations")
    [ -n "$(last_line logs/warm_evaldesc_highstakes.log 'sample activations')" ] && \
        warmline=$(last_line logs/warm_evaldesc_highstakes.log "sample activations")

    flag="ok"
    [ "$gen" = 0 ] && [ "$warm" = 0 ] && [ "$fit" = 0 ] && flag="IDLE"
    [ "$age" -gt 2400 ] && flag="STALLED(${age}s since any log write)"
    echo "[$ts] $flag — sets ${sets}/8, fits I=${fi}/36 H=${fh}/36 | gen=$gen warm=$warm fit=$fit | GPU $gpu | ${genline:-gen done} | ${warmline:-no warm yet}"
done
