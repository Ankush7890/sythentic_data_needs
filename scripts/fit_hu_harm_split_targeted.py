#!/usr/bin/env python3
"""Fit probes on SPLIT-TARGETED generated sets and score them on that pair of splits only.

The `generator_experiment_1` port of the study `human_harm_last` ran for the ant_hh /
balanced_refusal pair (commit 9fcdda21 there), extended to the OTHER TWO hu_ha splits:
`eval_ai_dilemmas` and `eval_daily_dilemmas`. Those two are the pair the eval-description
arms could never move — both steered arms on this branch lost ground on them (02623a4c,
c50076c4) — and they are also the two whose shape a generic prompt gets most wrong, so
"describe the target split and nothing else" is the sharpest available test of whether the
description is the binding constraint.

Four conditions per pair, each fit from the generated rows ALONE (no base training data)
and each scored on BOTH splits of the pair, so one table answers three questions:

    <A>_600      does data written for split A fit split A — and does it transfer to B?
    <B>_600      the mirror image.
    pooled_1200  does the union beat either targeted set on its own split?
    generic_600  THE CONTROL: same generator, same 600 rows, written from the GENERIC
                 human-harm prompt with no knowledge of any split. This is what a targeted
                 set must beat to have earned its description. The 50-row base probe is a
                 weaker comparison because it differs in size AND in prompt.

WHAT IS HELD FIXED: generated-only fits (`base_training_data_path=None`), the restricted
dev set below, seed 42, the repo's message transforms, and the ensemble size inherited from
the template probe. The only thing that varies across rows is WHICH generated rows train.

THE DEV SET IS RESTRICTED TO THE PAIR'S OWN DEV SPLITS, assembled into its own directory so
`_load_dev_dataset` picks up exactly those files. The probe is early-stopped on the same two
kinds it is scored on, which is what "fit and evaluate only on this pair" requires; the full
290-row hu_ha dev set would select checkpoints partly on the two kinds this experiment
deliberately ignores. The dev blob is keyed on the dev FILES' bytes, so this directory gets
its own cached activations rather than being served the full set's — 112 rows to extract
once for the dilemmas pair.

EVAL IS RESTRICTED THE SAME WAY via `splits=`. Both blobs are already in
cache_gen_gemma27b_hu_harm/eval_activations, so scoring loads no model.

Draws: `--draws N` refits each condition on N independent 90% subsamples of its own set,
seeded on (condition, draw). `--draws 0` fits each condition once on its whole set.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/fit_hu_harm_split_targeted.py \
        --pair dilemmas --draws 4
"""
from __future__ import annotations

import argparse, json, random, shutil, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from agentic_redteam.retrain import (                      # noqa: E402
    retrain_probe, warm_sample_activation_cache,
)
from agentic_redteam.evaluation import evaluate_probe      # noqa: E402

POS, NEG = "harmful_to_human", "not_harmful_to_human"

OUT_ROOT = REPO / "analysis/refit_studies"
WORK = REPO / ".hu_harm_split_work"
# This branch's own artefacts (human_harm_last's paths do not exist here): the probe trained
# on the llama70b-written 50-row base, and the caches every gen_gemma27b_hu_harm fit uses.
TEMPLATE_PROBE = REPO / "probes/gen_gemma27b_hu_harm/probe_iter0.pkl"
CACHE = REPO / "cache_gen_gemma27b_hu_harm/base_activations"
ECACHE = REPO / "cache_gen_gemma27b_hu_harm/eval_activations"
EVAL = REPO / "eval_sets/hu_ha"
DEV_SRC = REPO / "dev_samples/hu_ha"
GEN = "llama70b"

# fit_base_plus_concept.CONCEPTS["hu_harm"] — reused only for its labels, so `load_rows`
# validates against the same two class strings every other fit on this branch uses.
from fit_base_plus_concept import CONCEPTS, load_rows          # noqa: E402
CONCEPT = CONCEPTS["hu_harm"]

# One entry per split PAIR. `sets` maps a condition name to the files it trains on.
PAIRS = {
    "dilemmas": dict(
        splits=["eval_ai_dilemmas", "eval_daily_dilemmas"],
        dev_files=["dev_ai_dilemmas.jsonl", "dev_daily_dilemmas.jsonl"],
        out=OUT_ROOT / "hu_harm_split_targeted_dilemmas",
        sets={
            "ai_dilemmas_600": [REPO / f"data/hu_harm_{GEN}_aiDilemmas_600.jsonl"],
            "daily_dilemmas_600": [REPO / f"data/hu_harm_{GEN}_dailyDilemmas_600.jsonl"],
            "pooled_1200": [REPO / f"data/hu_harm_{GEN}_aiDilemmas_600.jsonl",
                            REPO / f"data/hu_harm_{GEN}_dailyDilemmas_600.jsonl"],
            "generic_600": [REPO / f"data/hu_harm_{GEN}_600.jsonl"],
        },
    ),
}


def dev_dir(pair: dict) -> Path:
    """The pair's dev splits, in their own directory so they get their own cache key."""
    d = WORK / ("dev_" + "_".join(f.replace("dev_", "").replace(".jsonl", "")
                                  for f in pair["dev_files"]))
    d.mkdir(parents=True, exist_ok=True)
    for f in pair["dev_files"]:
        dst = d / f
        if not dst.exists():
            shutil.copy(DEV_SRC / f, dst)
    return d


def rows_for(pair: dict, name: str) -> list[dict]:
    """The condition's generated rows, as the `{inputs: [...], labels}` dicts retrain wants.

    `human_harm_last`'s version of this script rewrote each set as red-team AttemptRecords,
    because on that branch the only per-conversation-cached path into `retrain_probe` was the
    JSONL red-team one: the base-training path keys its activation cache on a hash of the
    WHOLE FILE, so every 90% draw would have been a new key and would have re-extracted its
    rows from scratch. On THIS branch the in-memory `samples=` argument is that path already
    (it is what fit_base_plus_concept.py and subsample_curve_concept.py use), so the
    workaround is dropped: the union of all draws is extracted once and every draw is a
    cache hit, with no synthetic record fields in between.
    """
    rows: list[dict] = []
    for p in pair["sets"][name]:
        rows.extend(load_rows(p, CONCEPT))
    return rows


def run(pair: dict, name: str, draw: int | None, frac: float, dev: Path) -> dict:
    tag = f"{name}_full" if draw is None else f"{name}_f{int(frac * 100)}_d{draw}"
    res_path = pair["out"] / f"{tag}.json"
    if res_path.exists():
        return json.load(res_path.open())
    rows = rows_for(pair, name)
    if draw is None:
        keep = rows
    else:
        k = max(1, round(len(rows) * frac))
        keep = random.Random(f"{name}|{frac}|{draw}").sample(rows, k)
    probe_out = WORK / f"{tag}.pkl"
    probe_out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\n===== {tag}: {len(keep)}/{len(rows)} generated rows, no base =====", flush=True)
    retrain_probe(samples=keep, base_probe_path=TEMPLATE_PROBE,
                  base_training_data_path=None, new_probe_path=probe_out,
                  dev_data_path=dev, seed=42,
                  base_activation_cache_dir=CACHE,
                  combine_consecutive_messages=True, convert_tool_to_assistant=True,
                  verbose=True)
    df = evaluate_probe(str(probe_out), str(EVAL), str(ECACHE), splits=pair["splits"],
                        max_samples=None, seed=42,
                        combine_consecutive_messages=True, convert_tool_to_assistant=True)
    p = df.set_index("dataset")["auroc"]
    res = dict(condition=name, draw=draw, frac=(1.0 if draw is None else frac),
               n=len(keep), n_all=len(rows),
               **{s.replace("eval_", ""): round(float(p[s]), 4) for s in pair["splits"]},
               mean=round(float(p[pair["splits"]].mean()), 4),
               minutes=round((time.time() - t0) / 60, 1))
    pair["out"].mkdir(parents=True, exist_ok=True)
    json.dump(res, res_path.open("w"), indent=1)
    print("RESULT " + json.dumps(res), flush=True)
    probe_out.unlink(missing_ok=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="dilemmas", choices=sorted(PAIRS))
    ap.add_argument("--draws", type=int, default=4,
                    help="90%% subsample refits per condition (0 = whole-set fit only)")
    ap.add_argument("--frac", type=float, default=0.9)
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()

    pair = PAIRS[args.pair]
    names = args.conditions or list(pair["sets"])
    missing = [str(p) for n in names for p in pair["sets"][n] if not p.exists()]
    if missing:
        raise SystemExit("missing generated sets:\n  " + "\n  ".join(missing))

    dev = dev_dir(pair)
    # One model load for every row any condition will ever fit on, so all 20 fits below are
    # pure cache hits (generic_600's rows are already warm from the earlier arms).
    seen: set[str] = set()
    warm: list[dict] = []
    for name in names:
        for r in rows_for(pair, name):
            k = json.dumps(r["inputs"], sort_keys=True)
            if k not in seen:
                seen.add(k); warm.append(r)
    print(f"warming {len(warm)} distinct generated rows", flush=True)
    warm_sample_activation_cache(warm, base_probe_path=TEMPLATE_PROBE,
                                 base_activation_cache_dir=CACHE,
                                 combine_consecutive_messages=True,
                                 convert_tool_to_assistant=True, verbose=True)

    results = []
    for name in names:
        results.append(run(pair, name, None, 1.0, dev))
        for d in range(args.draws):
            results.append(run(pair, name, d, args.frac, dev))

    a, b = (s.replace("eval_", "") for s in pair["splits"])
    print(f"\n{'condition':22} {'n':>5} {a:>18} {b:>18}")
    for name in names:
        rs = [r for r in results if r["condition"] == name and r["draw"] is not None]
        full = next(r for r in results if r["condition"] == name and r["draw"] is None)
        if rs:
            import statistics as st
            fa = f"{st.mean([r[a] for r in rs]):.4f}+-{st.stdev([r[a] for r in rs]):.4f}" \
                 if len(rs) > 1 else f"{rs[0][a]:.4f}"
            fb = f"{st.mean([r[b] for r in rs]):.4f}+-{st.stdev([r[b] for r in rs]):.4f}" \
                 if len(rs) > 1 else f"{rs[0][b]:.4f}"
        else:
            fa, fb = f"{full[a]:.4f}", f"{full[b]:.4f}"
        print(f"{name:22} {full['n_all']:5d} {fa:>18} {fb:>18}")


if __name__ == "__main__":
    main()
