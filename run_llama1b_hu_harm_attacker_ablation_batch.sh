#!/usr/bin/env bash
set -e

# BATCH-SUBMISSION attacker ablation on the HARMFUL_TO_HUMAN concept with a
# meta-llama/Llama-3.2-1B-Instruct (L8) probe, base data data/hu_harm_llama70b_50.jsonl.
#
# This is experiment9_cloud's batch setup (run_gemma27b_hs_attacker_ablation_batch.sh) moved
# to the harmful_to_human concept and the llama-1b probe. Both arms carry:
#
#   attacker.batch_submissions: true    (one blind API call per session)
#   attacker.view_limit:        0       (no past-attempts injection either)
#   attacker.capture_prompts:   false   (default, pinned)
#   attacker.cross_iteration_memos: false (default, pinned)
#   judge.hide_opposite_direction: true (default, pinned)
#
# Under batch_submissions each session makes ONE API call, is asked for all `max_turns` (5)
# candidate conversations in that single reply, has every one of them scored, and ends — the
# attacker never sees a probe/judge verdict, and with view_limit: 0 it is shown no past
# attempts either. Its only inputs are the system prompt (probe metadata + the judge's rolling
# round memo) and "submit all N now". Attempt volume is unchanged from every earlier hu_harm
# run: 10 sessions × 5 conversations × 5 rounds ≈ 250 attempts per error type per iteration.
#
# Two arms, IDENTICAL in every knob except attacker.models, run sequentially and fully
# isolated. NEITHER arm uses contrastive label guidance (no preprocessing.concept_description
# / label_guidance) — that is held off in both, so the attacker model is the only variable:
#
#   ARM 1 (gpt-oss-120b):    configs/gptoss120b_hu_harm_llama1b_batch.md
#                            -> results_hu_harm_llama70b50_gptoss120b_batch/
#                               probes/hu_harm_llama1b_gptoss120b_batch
#   ARM 2 (deepseek-v4-pro): configs/deepseekv4pro_hu_harm_llama1b_batch.md
#                            -> results_hu_harm_llama70b50_deepseekv4pro_batch/
#                               probes/hu_harm_llama1b_deepseekv4pro_batch
#
# The judge (openai/gpt-5.1), the preprocessing model (openai/gpt-5.1), the probe
# (Llama-3.2-1B-Instruct L8), the base data and every scheduling knob are held fixed, so any
# delta in the comparison CSVs is attributable to the attacker.
#
# ACTIVATIONS ARE COMPUTED FRESH — unlike the gemma-27b runs there is no `kaggle:` prefetch,
# because llama-1b eval activations are cheap to extract locally. The shared cache dir
# (results_hu_harm_llama70b50_batch_ablation/) starts empty on a clean cloud box; arm 1 fills
# both the base and eval halves and arm 2 hits them, because those blobs depend only on the
# probe model / layer / seed / base data / eval splits / transforms — NOT on the attacker. The
# redteam_acts_* per-conversation cache written into the same dir is content-keyed against a
# frozen LLM, so the two arms' distinct successes get distinct keys. Budget arm 1's wall-clock
# accordingly; arm 2 is cheaper.
#
# A fresh --probe-out-dir per arm matters beyond overwriting:
#   - the old dir holds redteam_done_iter*_*.marker resume markers; reusing it would make the
#     CLI skip red-teaming and just retrain.
#   - it gives a fresh contrastive_cache.jsonl, keeping the two arms' provenance separate.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   mkdir -p logs
#   nohup bash run_llama1b_hu_harm_attacker_ablation_batch.sh > logs/run_llama1b_hu_harm_attacker_ablation_batch.out 2>&1 &
#
# Checkpointing (so a wiped container can --resume): start failsafe_commit.sh alongside it. It
# already defaults to these two arms, in this order, and hands itself off from arm 1 to arm 2
# when arm 1's comparison CSV lands:
#   git checkout -b failsafe/llama1b-hu-harm-attacker-ablation-batch
#   nohup bash failsafe_commit.sh > logs/failsafe_commit.out 2>&1 &
# NOTE the failsafe is NOT usable for the continuation below: stage_finished() is "the
# comparison CSV exists", both arms' CSVs already do, so its startup skip loop would advance
# past both stages and exit immediately. On a local box there is nothing to check point anyway.
#
# ============================================================================================
# CONTINUATION MODE (iterations 3-5, i.e. `--iterations 6`)
# ============================================================================================
# Both arms already completed iterations 0-2 and wrote probe_iter0..3.pkl into their
# --probe-out-dir. This script now RESUMES them for three more cycles rather than starting
# clean, so most of the guards below are inverted relative to the original launch:
#
#   * cli.py's --resume (default on) finds probe_iter3.pkl via _latest_probe_iteration and sets
#     start_iter=3, so `--iterations 6` runs exactly iterations 3, 4, 5 → probe_iter4/5/6.pkl.
#     The stale redteam_done_iter0..2_*.marker files cannot cause a skip: the phase-marker
#     check is gated on `i == start_iter` (=3). rounds_done.jsonl keys include the iteration,
#     so the 15 recorded rounds of iters 0-2 do not false-match either, and the rolling memo
#     correctly restarts empty for iteration 3.
#   * The pre-flight check therefore REQUIRES the per-arm dirs (they are the resume state)
#     instead of refusing to clobber them.
#   * The comparison CSV and the run log are ARCHIVED first. cli.py holds eval_results
#     in-process and rewrites the CSV once at the end, so a resumed run's CSV covers ONLY the
#     iterations that run executed (iter3..iter6) — writing it in place would destroy the
#     iter0..iter2 rows. After each arm the two are merged into *_comparison_full.csv.
#   * The shared activation cache is pre-warmed from results_hu_harm_llama70b50_grok/, which is
#     key-compatible (same probe model+layer, same base data/seed/test_size/transforms → same
#     base-cache hash c209ceed2746d82c; same eval splits+transforms → same <split>-acts_full.pt
#     names). Without it the iteration-3 retrain re-extracts activations for the ENTIRE
#     accumulated red-team set (~906/948 postprocessed samples), not just the new ones.
#
# DO NOT pass --no-resume: it would restart at iteration 0 against the existing JSONLs, whose
# ~700 preloaded rows would dedup away the new rotation and warm-start the success counter.
# ============================================================================================

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first (attacker, judge and preprocessing are all provider: openrouter)}"

SHARED_CACHE="results_hu_harm_llama70b50_batch_ablation"   # shared, arm-independent activation cache
PREWARM_FROM="results_hu_harm_llama70b50_grok"             # key-compatible cache to seed it from
ITERATIONS=6                                               # TOTAL cycles; resume starts at 3 → 3 more

# Require the resume state. (The SHARED cache dir is intentionally NOT in this list — it is
# meant to persist and grow across both arms and across re-runs, and is rebuilt below if
# missing.) A probe dir with no probe_iter*.pkl means resume would silently restart at
# iteration 0, which is never what this script wants.
for p in probes/hu_harm_llama1b_gptoss120b_batch probes/hu_harm_llama1b_deepseekv4pro_batch ; do
    [ -d "$p" ] || { echo "ERROR: $p is missing — this script resumes iterations 3-5 and needs the completed 0-2 state." >&2; exit 1; }
    compgen -G "$p/probe_iter*.pkl" > /dev/null \
        || { echo "ERROR: $p holds no probe_iter*.pkl — nothing to resume from." >&2; exit 1; }
done

mkdir -p "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations"
# Pre-warm from the grok run. -n never clobbers, so re-running this script is a no-op, and the
# arms' own blobs (written during iterations 0-2 on the cloud box and never committed — the
# failsafe hard-excludes **/*.pt) are simply absent. Only the two base_acts_*.pt blobs are
# taken from base_activations/, NOT its redteam_acts_* subdir: those are content-keyed per
# conversation and the grok run's attacker/contrastive model differ, so the hit rate would be
# near zero for ~4 GB of copying.
if [ -d "$PREWARM_FROM" ]; then
    cp -n "$PREWARM_FROM"/base_activations/base_acts_*.pt "$SHARED_CACHE/base_activations/" 2>/dev/null || true
    cp -rn "$PREWARM_FROM"/eval_activations/. "$SHARED_CACHE/eval_activations/" 2>/dev/null || true
    echo ">>> pre-warmed activation cache from $PREWARM_FROM (base blobs + 4 eval splits)"
else
    echo ">>> WARNING: $PREWARM_FROM absent — base and eval activations will be extracted fresh." >&2
fi
echo ">>> activation cache: $SHARED_CACHE"
ls -1 "$SHARED_CACHE/base_activations" "$SHARED_CACHE/eval_activations" 2>/dev/null | sed 's/^/      /'

# --- run one arm ---------------------------------------------------------------------------
# Exit code the CLI uses for "OpenRouter is unusable" (cli.OUTAGE_EXIT_CODE).
OUTAGE_EXIT_CODE=3

# Merge an archived comparison CSV with the resumed run's. The resumed run re-evaluates its
# input probe as "iter3" before red-teaming, so that label appears in BOTH files with
# identical numbers (same probe, same cached eval activations); dedup on (round, dataset)
# keeps the archived copy. Writes a THIRD file — neither input is touched.
merge_csv () {  # $1 = archived csv, $2 = fresh csv, $3 = merged out
    [ -f "$1" ] && [ -f "$2" ] || return 0
    awk -F, 'FNR==1 { if (NR==1) print; next } !seen[$1","$2]++ { print }' "$1" "$2" > "$3"
    echo ">>> merged $1 + $2 -> $3"
}

run_arm () {  # $1 = config, $2 = probe-out-dir, $3 = logfile, $4 = comparison csv
    local cfg="$1" probe_dir="$2" log="$3" csv="$4"
    local archived_csv="${csv%.csv}_iter0_3.csv"
    local merged_csv="${csv%.csv}_full.csv"

    # Archive the completed run's artifacts BEFORE launching. cli.py rewrites the CSV in place
    # at the end covering only the iterations this process ran, so leaving it would lose the
    # iteration 0-2 rows. Both moves are guarded on the ARCHIVE not existing rather than on
    # `mv -n`: on a re-run after a mid-arm crash the archive is already there, and a bare
    # `mv -n` would silently no-op while still reporting success. The log is then appended to,
    # not truncated, so a resumed attempt keeps the crashed one's tail.
    if [ -f "$csv" ] && [ ! -f "$archived_csv" ]; then
        mv "$csv" "$archived_csv"; echo ">>> archived $csv -> $archived_csv"
    fi
    if [ -f "$log" ] && [ ! -f "${log%.log}_iter0_3.log" ]; then
        mv "$log" "${log%.log}_iter0_3.log"; echo ">>> archived $log -> ${log%.log}_iter0_3.log"
    fi

    echo ">>> $(date -Is)  START $cfg  -> $probe_dir   (--iterations $ITERATIONS, resuming at 3; log: $log)"
    local rc=0
    .venv_claude/bin/python scripts/iterative_retrain.py "$cfg" \
        --iterations "$ITERATIONS" \
        --base-training-data data/hu_harm_llama70b_50.jsonl \
        --probe-out-dir "$probe_dir" \
        --eval --eval-dataset-dir eval_dataset_hu_ha \
        >> "$log" 2>&1 || rc=$?
    if [ "$rc" -eq "$OUTAGE_EXIT_CODE" ]; then
        # The circuit breaker stopped the run: OpenRouter is out of credits or the
        # key is dead. Do NOT start the next arm — it would fail identically and
        # write a comparison CSV from probes trained on nothing.
        echo ">>> $(date -Is)  ABORTED $cfg — OpenRouter unusable (exit $rc)." >&2
        tail -n 5 "$log" >&2
        echo ">>> Fix credits/key, then re-run this script to continue the arm (the archive" >&2
        echo "    steps are no-clobber, so a second run picks up where it stopped)." >&2
        exit "$rc"
    elif [ "$rc" -ne 0 ]; then
        echo ">>> $(date -Is)  FAILED  $cfg (exit $rc) — see $log" >&2
        exit "$rc"
    fi
    merge_csv "$archived_csv" "$csv" "$merged_csv"
    echo ">>> $(date -Is)  DONE  $cfg"
}

run_arm configs/gptoss120b_hu_harm_llama1b_batch.md    probes/hu_harm_llama1b_gptoss120b_batch    \
        logs/run_hu_harm_llama1b_gptoss120b_batch.log \
        results_hu_harm_llama70b50_gptoss120b_batch/gptoss120b_comparison.csv
run_arm configs/deepseekv4pro_hu_harm_llama1b_batch.md probes/hu_harm_llama1b_deepseekv4pro_batch \
        logs/run_hu_harm_llama1b_deepseekv4pro_batch.log \
        results_hu_harm_llama70b50_deepseekv4pro_batch/deepseekv4pro_comparison.csv

echo ">>> $(date -Is)  both arms finished (iterations 0-5; probe_iter6.pkl is final)."
echo ">>> full cross-iteration eval tables:"
echo "      results_hu_harm_llama70b50_gptoss120b_batch/gptoss120b_comparison_full.csv"
echo "      results_hu_harm_llama70b50_deepseekv4pro_batch/deepseekv4pro_comparison_full.csv"
