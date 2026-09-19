#!/usr/bin/env python
"""COMBINATION STUDY: pool one set per eval split and fit on all four at once.

Every fit so far has trained on ONE split-targeted set and asked how far it transfers.
This asks the complementary question: if you take the set written for each of the four
`eval_sets/hu_ha` splits and pool them, does the union beat its own members? `pooled_1200`
answered that for two splits within one pair; this answers it for all four, for three
different prompt arms, at four different per-split sizes.

THREE ARMS, each contributing one set per split:

    arm1     the as-run split-targeted sets (anchors + topic lists + shape)
    cleanPL  arm 2 per-label (content removed, shape kept). ant_hh has no _pl twin — it
             was already unpaired in arm 2 — so ant_hh_clean_600 stands in, which is what
             the arm-2 tables already do.
    minimal  arm 3 per-label (content AND shape removed). Its ant_hh and balanced_refusal
             members are the TWO DRAWS of one identical prompt, so this arm is the only one
             whose four members are not four different descriptions.

FOUR MODES:

    best   per split, the size at which THAT split's own set scored highest on it,
           read off the committed size curve (see BEST below). This is the deliberately
           optimistic combination — every component is taken at its own best point, chosen
           on the eval splits themselves, so it is an UPPER BOUND and not an estimate of
           held-out performance. It is included because it bounds what per-split size
           tuning could buy, not because the number is honest on its own terms.
    n30    30 rows per split  ->  120 total
    n300   300 rows per split -> 1200 total
    n600   600 rows per split -> 2400 total (whole sets, nothing sampled)

Draws: 8 wherever anything is subsampled; n600 is the whole of every set and therefore
deterministic, so it is fit once.

THE DEV SET IS ALL FOUR DEV SPLITS (dev_samples/hu_ha, 290 rows), not a pair's two. A
combined set spans both pairs, so early stopping has to see both; using a pair's dev dir
would select checkpoints on half the concept. This also means combination numbers are NOT
directly comparable with the pair-restricted single-set fits — the probes were early-stopped
against a different validation set. The eval columns are the same four splits either way.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/fit_hu_harm_combination.py --arms arm1 cleanPL minimal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_hu_harm_split_targeted import (  # noqa: E402
    ALL_SPLITS, CACHE, CONCEPT, ECACHE, EVAL, POS, TEMPLATE_PROBE, WORK,
    draw_subset, probe_spec_with_accum,
)
from fit_base_plus_concept import load_rows  # noqa: E402


def rows_of_file(path: Path) -> list[dict]:
    """One generated set, in the `{inputs: [...], labels}` shape retrain_probe wants."""
    return load_rows(path, CONCEPT)
from synthetic_probe_data.evaluation import evaluate_probe            # noqa: E402
from synthetic_probe_data.retrain import retrain_probe, warm_sample_activation_cache  # noqa: E402

OUT = REPO / "analysis/refit_studies/hu_harm_combination"
DEV = REPO / "dev_samples/hu_ha"          # all four dev splits
GEN = "llama70b"
SPLITS = ["ai_dilemmas", "ant_hh", "balanced_refusal", "daily_dilemmas"]

ARMS = {
    "arm1": {
        "ai_dilemmas": f"data/hu_harm_{GEN}_aiDilemmas_600.jsonl",
        "ant_hh": f"data/hu_harm_{GEN}_antHH_600.jsonl",
        "balanced_refusal": f"data/hu_harm_{GEN}_refusal_v2_600.jsonl",
        "daily_dilemmas": f"data/hu_harm_{GEN}_dailyDilemmas_600.jsonl",
    },
    "cleanPL": {
        "ai_dilemmas": f"data/hu_harm_{GEN}_aiDilemmasCleanPL_600.jsonl",
        "ant_hh": f"data/hu_harm_{GEN}_antHHclean_600.jsonl",
        "balanced_refusal": f"data/hu_harm_{GEN}_refusalCleanPL_600.jsonl",
        "daily_dilemmas": f"data/hu_harm_{GEN}_dailyDilemmasCleanPL_600.jsonl",
    },
    "minimal": {
        "ai_dilemmas": f"data/hu_harm_{GEN}_aiDilemmasMinimal_600.jsonl",
        "ant_hh": f"data/hu_harm_{GEN}_requestMinimal_600.jsonl",
        "balanced_refusal": f"data/hu_harm_{GEN}_requestMinimal2_600.jsonl",
        "daily_dilemmas": f"data/hu_harm_{GEN}_dailyDilemmasMinimal_600.jsonl",
    },
}

# Mode "best": the size at which each split's own set scored highest ON THAT SPLIT, taken
# from the committed size curve. Chosen on the eval splits, so this mode is an upper bound.
BEST = {
    "arm1":    {"ant_hh": 600, "balanced_refusal": 60,  "ai_dilemmas": 120, "daily_dilemmas": 30},
    "cleanPL": {"ant_hh": 600, "balanced_refusal": 30,  "ai_dilemmas": 540, "daily_dilemmas": 600},
    "minimal": {"ant_hh": 540, "balanced_refusal": 540, "ai_dilemmas": 540, "daily_dilemmas": 540},
}
FIXED = {"n30": 30, "n300": 300, "n600": 600}


def sizes_for(arm: str, mode: str) -> dict[str, int]:
    return BEST[arm] if mode == "best" else {s: FIXED[mode] for s in SPLITS}


def build(arm: str, mode: str, draw: int) -> tuple[list[dict], dict[str, int]]:
    """The pooled training rows, drawn balanced within EACH split's own set.

    Balancing per component rather than over the pool keeps every split's contribution at
    n/2 per class, so a mode's class ratio cannot drift with which splits it happens to
    draw from.
    """
    sizes, rows = sizes_for(arm, mode), []
    for split in SPLITS:
        allrows = rows_of_file(REPO / ARMS[arm][split])
        k = sizes[split]
        rows += (allrows if k >= len(allrows)
                 else draw_subset(allrows, k / len(allrows), f"{arm}|{mode}|{split}",
                                  draw, True, k))
    return rows, sizes


def run(arm: str, mode: str, draw: int | None, accum: int) -> dict:
    tag = f"{arm}_{mode}" + ("" if draw is None else f"_d{draw}")
    res_path = OUT / f"{tag}.json"
    if res_path.exists():
        return json.load(res_path.open())
    rows, sizes = build(arm, mode, draw or 0)
    probe_out = WORK / f"comb_{tag}.pkl"
    probe_out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\n===== {tag}: {len(rows)} rows, sizes {sizes} =====", flush=True)
    retrain_probe(samples=rows, base_probe_path=TEMPLATE_PROBE,
                  base_training_data_path=None, new_probe_path=probe_out,
                  probe_spec=probe_spec_with_accum(accum),
                  dev_data_path=DEV, seed=42, base_activation_cache_dir=CACHE,
                  combine_consecutive_messages=True, convert_tool_to_assistant=True,
                  verbose=True)
    df = evaluate_probe(str(probe_out), str(EVAL), str(ECACHE), splits=ALL_SPLITS,
                        max_samples=None, seed=42,
                        combine_consecutive_messages=True, convert_tool_to_assistant=True)
    p = df.set_index("dataset")["auroc"]
    npos = sum(1 for r in rows if r["labels"] == POS)
    res = dict(arm=arm, mode=mode, draw=draw, sizes=sizes, grad_accum=accum,
               n=len(rows), n_pos=npos, n_neg=len(rows) - npos,
               **{s.replace("eval_", ""): round(float(p[s]), 4) for s in ALL_SPLITS},
               mean4=round(float(p[ALL_SPLITS].mean()), 4),
               minutes=round((time.time() - t0) / 60, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, res_path.open("w"), indent=1)
    print("RESULT " + json.dumps(res), flush=True)
    probe_out.unlink(missing_ok=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--modes", nargs="+", default=["best", "n30", "n300", "n600"])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    args = ap.parse_args()

    missing = [ARMS[a][s] for a in args.arms for s in SPLITS
               if not (REPO / ARMS[a][s]).exists()]
    if missing:
        raise SystemExit("missing sets:\n  " + "\n  ".join(missing))

    # One model load for every row any condition will fit on. In practice all of these are
    # already cached from the single-set fits; the dev blob for all four splits may not be.
    seen, warm = set(), []
    for a in args.arms:
        for s in SPLITS:
            for r in rows_of_file(REPO / ARMS[a][s]):
                k = json.dumps(r["inputs"], sort_keys=True)
                if k not in seen:
                    seen.add(k); warm.append(r)
    print(f"warming {len(warm)} distinct generated rows", flush=True)
    warm_sample_activation_cache(warm, base_probe_path=TEMPLATE_PROBE,
                                 base_activation_cache_dir=CACHE,
                                 combine_consecutive_messages=True,
                                 convert_tool_to_assistant=True, verbose=True)

    for arm in args.arms:
        for mode in args.modes:
            # n600 takes every row of every set: nothing is sampled, so it is one fit.
            for d in ([None] if mode == "n600" else range(args.draws)):
                run(arm, mode, d, args.accum)


if __name__ == "__main__":
    main()
