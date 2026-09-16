#!/usr/bin/env bash
# Fits for the four-split prompt-variant study. One arm per eval split:
#
#   hcd5   instructions / hc_context_drift
#   oigd5  instructions / oig_context_drift
#   anthh5 hu_harm      / eval_ant_hh
#   aidil5 hu_harm      / eval_ai_dilemmas
#
# Per arm, for each of its five 600-row sets, no base data, validating on the SPLIT'S OWN
# dev file and scoring ONLY that split (`--eval-splits`), which is what makes 56 fits per
# set affordable:
#
#   1. n=600, 1 draw  -> scripts/<arm>_prompts5.csv
#      Run first, and first per set: the whole set goes through the 27B model in one load,
#      which warms the per-sample activation cache for every curve fit below.
#   2. n=590 300 150 75, 8 draws  -> scripts/<arm>_prompts5_sizecurve.csv
#   3. n=40 20 10, 8 draws, --grad-accum 1, tagged base='none+ga1'
#      Below ~49 rows the inherited spec (batch 16 x grad-accum 4 = one optimizer step per
#      64 samples) takes ZERO steps and the fit returns the untrained probe. Do not read
#      across that boundary as one line.
#
# Everything is sequential — one GPU. Resume is free: both scripts skip
# (base, samples, n, draw) keys already in their CSV.
#
#   ./run_prompts5_fits_4splits.sh                # all four arms, in order
#   ./run_prompts5_fits_4splits.sh hcd5 oigd5     # just the instructions pair
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
set -a; . ./.env; set +a
PY=.venv_claude/bin/python
mkdir -p logs

arm_spec() {
    case "$1" in
        hcd5)   echo "instructions hc_context_drift  dev_samples/instructions_hc_context_drift  instructions_hcdrift" ;;
        oigd5)  echo "instructions oig_context_drift dev_samples/instructions_oig_context_drift instructions_oigdrift" ;;
        anthh5) echo "hu_harm      eval_ant_hh       dev_samples/hu_ha_ant_hh                   hu_harm_ant_hh" ;;
        aidil5) echo "hu_harm      eval_ai_dilemmas  dev_samples/hu_ha_ai_dilemmas              hu_harm_ai_dilemmas" ;;
        *) echo "" ;;
    esac
}

run_arm() {
    local arm="$1"
    read -r concept split devdir csvname <<<"$(arm_spec "$arm")"
    [ -n "${concept:-}" ] || { echo ">>> unknown arm $arm" >&2; return 1; }

    local prefix="data/${concept}_deepseekv4pro_${arm}_"
    local sets=("${prefix}"*_600.jsonl)
    if [ ! -e "${sets[0]}" ]; then
        echo ">>> $arm: no generated sets at ${prefix}*_600.jsonl — skipping" >&2
        return 0
    fi
    echo ">>> $(date -Is) $arm: ${#sets[@]} sets, concept=$concept split=$split dev=$devdir"

    local common=(--concept "$concept" --no-base --dev-data "$devdir" --eval-splits "$split")

    # n=600 one draw, set by set, so each model load extracts exactly one set.
    for s in "${sets[@]}"; do
        $PY scripts/subsample_curve_concept.py "${common[@]}" --sizes 600 --draws 1 \
            --out "scripts/${csvname}_prompts5.csv" "$s" \
            >> "logs/fits_${arm}_n600.log" 2>&1
    done
    echo ">>> $(date -Is) $arm: n=600 done"

    $PY scripts/subsample_curve_concept.py "${common[@]}" --sizes 590 300 150 75 --draws 8 \
        --out "scripts/${csvname}_prompts5_sizecurve.csv" "${sets[@]}" \
        >> "logs/curve_${arm}.log" 2>&1
    echo ">>> $(date -Is) $arm: sizes 590-75 done"

    $PY scripts/subsample_curve_concept.py "${common[@]}" --sizes 40 20 10 --draws 8 \
        --grad-accum 1 --out "scripts/${csvname}_prompts5_sizecurve.csv" "${sets[@]}" \
        >> "logs/curve_${arm}.log" 2>&1
    echo ">>> $(date -Is) $arm: sizes 40-10 done"
}

for arm in "${@:-hcd5 oigd5 anthh5 aidil5}"; do
    run_arm "$arm"
done
echo ">>> $(date -Is) all requested arms finished"
