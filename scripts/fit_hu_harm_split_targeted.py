#!/usr/bin/env python3
"""Fit probes on the SPLIT-TARGETED generated sets and score them on those two splits only.

Three conditions, each fit from the generated rows ALONE (no base training data) and each
scored on BOTH target splits, so the table answers three questions at once:

    ant_hh_600     does data written for eval_ant_hh fit eval_ant_hh — and does it transfer
                   to eval_balanced_refusal?
    refusal_600    the mirror image.
    pooled_1200    does the union beat either targeted set on its own split?

WHAT IS HELD FIXED. Generated-only fits (`base_training_data_path=None`), the two-split dev
set below, seed 42, the repo's message transforms, and the ensemble size inherited from the
template probe — so the only thing that varies across the three rows is WHICH generated rows
are in the training set.

THE DEV SET IS RESTRICTED TO THE TWO MATCHING SPLITS (dev_ant_hh 44 + dev_balanced_refusal
134 = 178 rows), assembled into its own directory so `_load_dev_dataset` picks up exactly
those two files. The probe is therefore early-stopped on the same two kinds it is scored on,
which is what "fit and evaluate only on ant_hh and refusal" requires; the usual 290-row
hu_ha dev set would select checkpoints partly on ai_dilemmas and daily_dilemmas, the kinds
this experiment deliberately ignores. Note the dev blob is keyed on the dev FILES' bytes, so
this directory gets its own cached activations rather than being served the full set's.

EVAL IS RESTRICTED THE SAME WAY via `splits=`, so `get_performances` scores only
eval_ant_hh and eval_balanced_refusal. The two eval blobs are already cached (they came from
Kaggle during the arms 9-12 run), so no 27B forward passes are needed for scoring.

Draws: `--draws N` refits each condition on N independent 90% subsamples of its own set,
seeded on (condition, draw), giving the same kind of error bar the rest of this repo's
studies use. `--draws 0` fits each condition once on its whole set.
"""
from __future__ import annotations

import argparse, json, random, shutil, sys, time
from pathlib import Path

REPO = Path("/workspace/probe_auto_improvement")
sys.path.insert(0, str(REPO / "src"))
from agentic_redteam.retrain import retrain_probe          # noqa: E402
from agentic_redteam.evaluation import evaluate_probe      # noqa: E402
from agentic_redteam.persistence import (                  # noqa: E402
    AttemptRecord, Conversation, JsonlStore,
)

POS, NEG = "harmful_to_human", "not_harmful_to_human"

OUT = REPO / "analysis/refit_studies/hu_harm_split_targeted"
WORK = REPO / ".hu_harm_split_work"
CACHE = REPO / "results_hu_harm_gemma27b_batch_ablation/base_activations"
ECACHE = REPO / "results_hu_harm_gemma27b_batch_ablation/eval_activations"
EVAL = REPO / "eval_sets/hu_ha"
SPLITS = ["eval_ant_hh", "eval_balanced_refusal"]
DEV_SRC = REPO / "dev_samples/hu_ha"
DEV_FILES = ["dev_ant_hh.jsonl", "dev_balanced_refusal.jsonl"]
TEMPLATE_PROBE = REPO / "probes/hu_harm_gemma27b_llama70b_l70base_itermemo150/probe_iter0.pkl"

SETS = {
    "ant_hh_600":  [REPO / "data/hu_harm_llama70b_antHH_600.jsonl"],
    "refusal_600": [REPO / "data/hu_harm_llama70b_refusal_600.jsonl"],
    "pooled_1200": [REPO / "data/hu_harm_llama70b_antHH_600.jsonl",
                    REPO / "data/hu_harm_llama70b_refusal_600.jsonl"],
    # The CONTROL: same generator (Llama-3.3-70B), same 600 rows, but written from the
    # GENERIC human-harm prompt with no knowledge of any eval split. This is what the
    # targeted sets must beat to have earned their descriptions; the 50-row base is a
    # weaker comparison because it differs in size AND in prompt.
    "generic_600": [REPO / "data/hu_harm_llama70b_600.jsonl"],
    # refusal_600 re-run with the two verbatim refusal templates removed from the prompt
    # (see REFUSAL_DESC_V2). Its predecessor's 0.9832 was inflated by template overlap:
    # 17/200 eval negatives were reproduced exactly by a generated row.
    "refusal_v2_600": [REPO / "data/hu_harm_llama70b_refusal_v2_600.jsonl"],
}

# The same four conditions for the other three generators. Only the CLEAN (v2) refusal
# description is used: v1's number was inflated by template overlap and is discarded, so
# repeating it under three more generators would buy three more contaminated cells.
for _g in ("gptoss", "nemotron", "deepseekv4pro"):
    SETS[f"{_g}_ant_hh_600"] = [REPO / f"data/hu_harm_{_g}_antHH_600.jsonl"]
    SETS[f"{_g}_refusal_v2_600"] = [REPO / f"data/hu_harm_{_g}_refusal_v2_600.jsonl"]
    SETS[f"{_g}_pooled_1200"] = [REPO / f"data/hu_harm_{_g}_antHH_600.jsonl",
                                 REPO / f"data/hu_harm_{_g}_refusal_v2_600.jsonl"]
    SETS[f"{_g}_generic_600"] = [REPO / f"data/hu_harm_{_g}_600.jsonl"]


def dev_dir() -> Path:
    """The two-split dev set, in its own directory so it gets its own cache key."""
    d = WORK / "dev_two_split"
    d.mkdir(parents=True, exist_ok=True)
    for f in DEV_FILES:
        dst = d / f
        if not dst.exists():
            shutil.copy(DEV_SRC / f, dst)
    return d


def as_attempt_records(name: str) -> Path:
    """Rewrite a generated LabelledDataset set as red-team attempt records.

    WHY, and it is a performance property not a cosmetic one: the base-training path keys
    its activation cache on a hash of the WHOLE FILE, so every 90% draw would be a new key
    and would re-extract its rows from scratch — 12 fits x ~540-1080 rows is about seven
    hours of gemma-3-27b forwards. The red-team path caches PER CONVERSATION instead, so
    the union of all draws is extracted exactly once and every later draw is a pure cache
    hit. The rows are identical either way; only the cache granularity changes.

    The synthetic fields are the ones retrain_probe actually reads: `judge_label` (the
    class the row is trained as), `success` (so iter_successes yields it) and
    `judge_confidence` (compared against min_judge_confidence, which we set to 0). The
    probe-side fields are unused on this path and are filled with neutral values.
    """
    out = WORK / f"{name}_records.jsonl"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    store = JsonlStore(path=out)
    n = 0
    for p in SETS[name]:
        for line in p.open():
            if not line.strip():
                continue
            row = json.loads(line)
            conv = Conversation.from_messages(json.loads(row["inputs"]))
            store.append(AttemptRecord(
                sample=conv, probe_score=0.0, probe_predicts_positive=False,
                judge_label=row["labels"], judge_reason="generated", judge_confidence=10,
                success=True, attacker_model="__generated__", run_id=name,
                round=0, iteration=0, error_type="generated",
                pos_class_label=POS, neg_class_label=NEG))
            n += 1
    print(f"  {name}: wrote {n} rows as attempt records -> {out.name}", flush=True)
    return out


def rows_for(name: str) -> list[str]:
    """The set's attempt-record lines, one per generated conversation."""
    src = as_attempt_records(name)
    return [l if l.endswith("\n") else l + "\n" for l in src.open() if l.strip()]


def run(name: str, draw: int | None, frac: float, dev: Path) -> dict:
    tag = f"{name}_full" if draw is None else f"{name}_f{int(frac * 100)}_d{draw}"
    res_path = OUT / f"{tag}.json"
    if res_path.exists():
        return json.load(res_path.open())
    rows = rows_for(name)
    if draw is None:
        keep = rows
    else:
        k = max(1, round(len(rows) * frac))
        keep = random.Random(f"{name}|{frac}|{draw}").sample(rows, k)
    jl = WORK / f"{tag}.jsonl"
    jl.parent.mkdir(parents=True, exist_ok=True)
    jl.write_text("".join(keep))

    probe_out = WORK / f"{tag}.pkl"
    t0 = time.time()
    print(f"\n===== {tag}: {len(keep)}/{len(rows)} generated rows, no base =====", flush=True)
    # Generated-only: the rows arrive through the RED-TEAM path (per-conversation activation
    # cache, so draws share blobs) and there is no base data underneath.
    retrain_probe(jsonl_path=[jl], base_probe_path=TEMPLATE_PROBE,
                  base_training_data_path=None, new_probe_path=probe_out,
                  preprocessing=None, min_judge_confidence=0,
                  dev_data_path=dev, seed=42, ensemble_size=None,
                  base_activation_cache_dir=CACHE,
                  combine_consecutive_messages=True, convert_tool_to_assistant=True,
                  verbose=True)
    df = evaluate_probe(str(probe_out), str(EVAL), str(ECACHE), splits=SPLITS,
                        max_samples=None, seed=42,
                        combine_consecutive_messages=True, convert_tool_to_assistant=True)
    p = df.set_index("dataset")["auroc"]
    res = dict(condition=name, draw=draw, frac=(1.0 if draw is None else frac),
               n=len(keep), n_all=len(rows),
               ant_hh=round(float(p["eval_ant_hh"]), 4),
               refusal=round(float(p["eval_balanced_refusal"]), 4),
               mean=round(float(p[SPLITS].mean()), 4),
               minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, res_path.open("w"), indent=1)
    print("RESULT " + json.dumps(res), flush=True)
    probe_out.unlink(missing_ok=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=4, help="90%% subsample refits per condition")
    ap.add_argument("--fraction", type=float, default=0.9)
    ap.add_argument("--conditions", nargs="+", default=sorted(SETS))
    ap.add_argument("--full-set", action="store_true", help="also fit each whole set once")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dev = dev_dir()
    for name in args.conditions:
        for p in SETS[name]:
            if not p.exists():
                sys.exit(f"missing generated set: {p}")
        if args.full_set:
            run(name, None, 1.0, dev)
        for d in range(args.draws):
            run(name, d, args.fraction, dev)


if __name__ == "__main__":
    main()
