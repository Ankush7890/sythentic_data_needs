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
# Every hu_ha eval split, in evaluate_probe's own discovery order. Each fit is scored on all
# four so the non-targeted splits are visible too; `mean` stays over the pair's own two.
ALL_SPLITS = ["eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal", "eval_daily_dilemmas"]
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
            # CLEAN-PROMPT ABLATION. Same script, model, size and mode as the targeted sets
            # above; only the prompt differs (every confounder removed). `_pl` is the second
            # arm: the same clean text generated one label at a time, which breaks pairing.
            "ai_dilemmas_clean_600": [REPO / f"data/hu_harm_{GEN}_aiDilemmasClean_600.jsonl"],
            "daily_dilemmas_clean_600": [REPO / f"data/hu_harm_{GEN}_dailyDilemmasClean_600.jsonl"],
            "ai_dilemmas_cleanPL_600": [REPO / f"data/hu_harm_{GEN}_aiDilemmasCleanPL_600.jsonl"],
            "daily_dilemmas_cleanPL_600": [REPO / f"data/hu_harm_{GEN}_dailyDilemmasCleanPL_600.jsonl"],
            # MINIMAL-PROMPT ABLATION (arm 3). Content removed as in the clean arm, AND
            # shape: no turn counts, no word-length ranges, no register, no voice, no
            # pairing. Per-label for every split, because a pairing instruction is itself
            # shape. See analysis/split_targeted_prompts_minimal.md.
            "ai_dilemmas_minimal_600": [REPO / f"data/hu_harm_{GEN}_aiDilemmasMinimal_600.jsonl"],
            "daily_dilemmas_minimal_600": [REPO / f"data/hu_harm_{GEN}_dailyDilemmasMinimal_600.jsonl"],
        },
    ),
    # The ORIGINAL pair, re-fit here under this branch's protocol. The sets themselves are
    # human_harm_last's (commits 9fcdda21 / e350e258), copied across unchanged; what is new
    # is that they are measured against the same template probe, the same caches and the
    # same conditions as the dilemmas pair, so all four hu_ha splits finally sit in one
    # comparable table instead of two branches' worth of separately-anchored numbers.
    "refusal": dict(
        splits=["eval_ant_hh", "eval_balanced_refusal"],
        dev_files=["dev_ant_hh.jsonl", "dev_balanced_refusal.jsonl"],
        out=OUT_ROOT / "hu_harm_split_targeted_refusal",
        sets={
            "ant_hh_600": [REPO / f"data/hu_harm_{GEN}_antHH_600.jsonl"],
            "refusal_v2_600": [REPO / f"data/hu_harm_{GEN}_refusal_v2_600.jsonl"],
            # v1: the CONTAMINATED refusal set, kept as a condition rather than discarded.
            # Its description quoted two real refusal templates verbatim, and that branch
            # measured the resulting inflation at 0.043 (80dab791). Re-fitting it here says
            # whether that cost reproduces under a different probe and dev set — which is
            # worth more than deleting the file and asserting it.
            "refusal_v1_600_CONTAMINATED": [REPO / f"data/hu_harm_{GEN}_refusal_600.jsonl"],
            "pooled_1200": [REPO / f"data/hu_harm_{GEN}_antHH_600.jsonl",
                            REPO / f"data/hu_harm_{GEN}_refusal_v2_600.jsonl"],
            "generic_600": [REPO / f"data/hu_harm_{GEN}_600.jsonl"],
            # CLEAN-PROMPT ABLATION, this pair's half. ant_hh is already unpaired, so
            # ant_hh_clean_600 is shared by BOTH arms and has no _pl twin.
            "ant_hh_clean_600": [REPO / f"data/hu_harm_{GEN}_antHHclean_600.jsonl"],
            "refusal_clean_600": [REPO / f"data/hu_harm_{GEN}_refusalClean_600.jsonl"],
            "refusal_cleanPL_600": [REPO / f"data/hu_harm_{GEN}_refusalCleanPL_600.jsonl"],
            # MINIMAL-PROMPT ABLATION (arm 3). Once shape is gone, ant_hh and
            # balanced_refusal have the SAME description — everything that separated them
            # was shape — so ONE prompt wrote both sets below. They differ only in the
            # generator's sampling, which makes their gap a direct measure of generator
            # variance and the yardstick for reading the two splits' scores against.
            "request_minimal_600": [REPO / f"data/hu_harm_{GEN}_requestMinimal_600.jsonl"],
            "request_minimal2_600": [REPO / f"data/hu_harm_{GEN}_requestMinimal2_600.jsonl"],
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


def draw_subset(rows: list[dict], frac: float, name: str, draw: int,
                balanced: bool, size: int | None = None) -> list[dict]:
    """One reproducible subsample of ``rows``.

    ``balanced`` draws n/2 per class instead of sampling uniformly. The whole-set fits and
    the earlier 90% draws are UNIFORM (that is what human_harm_last's script did, and those
    rows are already recorded), but a size curve reaching 5% cannot be: 5% of 600 is 30
    rows, and a uniform draw of 30 from a 300/300 set lands anywhere near 15/15 give or take
    five, so the class ratio would vary alongside the size being measured — the same reason
    subsample_curve_concept.py balances its draws. Balanced draws are tagged `f{pct}b` and
    carry `balanced: true`, so they can never be read as points on the same curve as the
    uniform f90 rows.
    """
    # `size` (absolute rows) keys the RNG on itself so the existing frac-keyed draws stay
    # byte-reproducible; without it, 15/600 and 10/600 both round to the same f2b tag.
    rng = random.Random(f"{name}|{size if size is not None else frac}|{draw}"
                        + ("|balanced" if balanced else ""))
    k = size if size is not None else max(1, round(len(rows) * frac))
    if not balanced:
        return rng.sample(rows, k)
    pos = [r for r in rows if r["labels"] == POS]
    neg = [r for r in rows if r["labels"] == NEG]
    half, rest = k // 2, k - k // 2
    if len(pos) < half or len(neg) < rest:
        raise SystemExit(f"cannot draw {half}+{rest} from {len(pos)}/{len(neg)}")
    # Each class under its own stream, so the positive half does not shift when the
    # negative half's size changes.
    out = rng.sample(pos, half) + rng.sample(neg, rest)
    rng.shuffle(out)
    return out


def probe_spec_with_accum(accum: int | None):
    """The template probe's own spec, with gradient_accumulation_steps overridden.

    WHY THIS OVERRIDE EXISTS. tuberlens' trainer steps the optimizer only when
    ``(batch_idx + 1) % gradient_accumulation_steps == 0`` and zeroes the gradient at the TOP
    of each epoch, so a trailing partial accumulation window is discarded rather than
    flushed. At the inherited batch_size=16 / accum=4 that needs >= 4 batches, i.e. >= 49
    training rows, or `optimizer.step()` is NEVER CALLED and the fit returns its seeded
    initialisation — silently, with no error and a perfectly normal-looking loss curve.
    Verified directly: two 30-row fits on entirely different data produced BIT-IDENTICAL
    weights (max|delta| = 0 over all 5376 dimensions) while their training losses differed.

    5% of 600 is 30 rows -> 2 batches -> zero updates. Setting accum=2 there gives 2 batches
    -> exactly ONE optimizer step per epoch, accumulated over all 30 rows: the same number of
    steps per epoch that n=60 gets under accum=4, and an effective batch equal to the whole
    training set. Everything else in the spec is the template probe's own.
    """
    if accum is None:
        return None
    import pickle
    from agentic_redteam.retrain import _infer_probe_spec
    with open(TEMPLATE_PROBE, "rb") as fh:
        spec = _infer_probe_spec(pickle.load(fh))
    hp = dict(spec.hyperparams)
    hp["gradient_accumulation_steps"] = accum
    # ProbeSpec is a pydantic model, not a dataclass.
    return spec.model_copy(update={"hyperparams": hp})


def run(pair: dict, name: str, draw: int | None, frac: float, dev: Path,
        balanced: bool = False, accum: int | None = None,
        size: int | None = None) -> dict:
    suffix = "b" if balanced else ""
    # Absolute sizes are tagged n{N}, not f{pct}: 15 and 10 rows out of 600 both round to
    # the same percentage, so a frac tag would collide and the second fit would be skipped
    # as already-done.
    tag = (f"{name}_full" if draw is None
           else f"{name}_n{size}{suffix}_d{draw}" if size is not None
           else f"{name}_f{int(round(frac * 100))}{suffix}_d{draw}")
    res_path = pair["out"] / f"{tag}.json"
    if res_path.exists():
        return json.load(res_path.open())
    rows = rows_for(pair, name)
    keep = rows if draw is None else draw_subset(rows, frac, name, draw, balanced, size)
    probe_out = WORK / f"{tag}.pkl"
    probe_out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"\n===== {tag}: {len(keep)}/{len(rows)} generated rows, no base =====", flush=True)
    retrain_probe(samples=keep, base_probe_path=TEMPLATE_PROBE,
                  base_training_data_path=None, new_probe_path=probe_out,
                  probe_spec=probe_spec_with_accum(accum),
                  dev_data_path=dev, seed=42,
                  base_activation_cache_dir=CACHE,
                  combine_consecutive_messages=True, convert_tool_to_assistant=True,
                  verbose=True)
    # Scored on ALL FOUR hu_ha splits, not just the pair's own two. The pair still governs
    # what the probe was TRAINED and early-stopped on (its dev dir); the extra columns are a
    # pure cross-evaluation, and every eval blob is already cached so they are nearly free.
    df = evaluate_probe(str(probe_out), str(EVAL), str(ECACHE), splits=ALL_SPLITS,
                        max_samples=None, seed=42,
                        combine_consecutive_messages=True, convert_tool_to_assistant=True)
    p = df.set_index("dataset")["auroc"]
    npos = sum(1 for r in keep if r["labels"] == POS)
    res = dict(condition=name, draw=draw,
               frac=(1.0 if draw is None
                     else round(size / len(rows), 6) if size is not None else frac),
               balanced=bool(draw is not None and balanced),
               grad_accum=(accum if accum is not None else 4),
               n=len(keep), n_all=len(rows), n_pos=npos, n_neg=len(keep) - npos,
               **{s.replace("eval_", ""): round(float(p[s]), 4) for s in ALL_SPLITS},
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
                    help="subsample refits per condition per fraction (0 = whole-set only)")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.9],
                    help="training-set fractions to draw (default: 0.9)")
    ap.add_argument("--balanced", action="store_true",
                    help="draw n/2 per class instead of sampling uniformly")
    ap.add_argument("--accum", type=int, default=None,
                    help="override gradient_accumulation_steps (see probe_spec_with_accum). "
                         "Needed below 49 training rows, where the inherited value of 4 means "
                         "the optimizer never steps at all.")
    ap.add_argument("--skip-full", action="store_true",
                    help="do not fit the whole set (it is already recorded)")
    ap.add_argument("--sizes", type=int, nargs="*", default=None,
                    help="absolute training-set sizes instead of --fracs; tagged n{N} so "
                         "sizes whose percentages round together cannot collide. Below 25 "
                         "rows only --accum 1 takes an optimizer step at all (batch_size 16).")
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
        if not args.skip_full:
            results.append(run(pair, name, None, 1.0, dev))
        if args.sizes:
            for size in args.sizes:
                for d in range(args.draws):
                    results.append(run(pair, name, d, size / 600.0, dev,
                                       balanced=args.balanced, accum=args.accum, size=size))
        else:
            for frac in args.fracs:
                for d in range(args.draws):
                    results.append(run(pair, name, d, frac, dev, balanced=args.balanced,
                                       accum=args.accum))

    import statistics as st
    a, b = (s.replace("eval_", "") for s in pair["splits"])
    print(f"\n{'condition':30} {'frac':>6} {'n':>5} {a:>18} {b:>18}")
    for name in names:
        for frac in args.fracs:
            rs = [r for r in results
                  if r["condition"] == name and r["draw"] is not None and r["frac"] == frac]
            if not rs:
                continue
            fa = (f"{st.mean([r[a] for r in rs]):.4f}+-{st.stdev([r[a] for r in rs]):.4f}"
                  if len(rs) > 1 else f"{rs[0][a]:.4f}")
            fb = (f"{st.mean([r[b] for r in rs]):.4f}+-{st.stdev([r[b] for r in rs]):.4f}"
                  if len(rs) > 1 else f"{rs[0][b]:.4f}")
            print(f"{name:30} {frac:6.2f} {rs[0]['n']:5d} {fa:>18} {fb:>18}")


if __name__ == "__main__":
    main()
