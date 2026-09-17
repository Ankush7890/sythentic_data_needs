#!/usr/bin/env python
"""Size curve for a split-targeted instructions set with NO base data, scored on every
instructions eval split AND on the four parts of each cut split.

The instructions twin of `scripts/fit_toolace_parts.py`. The protocol matches the
`base='none'` rows of the tgtmin no-base curves on per_split_studies2, so those numbers
are directly comparable:

    probe      gemma-3-27b-it L32, linear_then_softmax, single — architecture and metadata
               inherited from probes/gen_gemma27b_instructions/probe_iter0.pkl, whose
               WEIGHTS and base data are never used
    training   the drawn rows ALONE (class-balanced n/2 per class, seeded on (stem, n, draw))
    dev        dev_samples/instructions, used whole (early stopping)
    eval       eval_sets/instructions, FULL splits, activations from Kaggle
    transforms combine_consecutive_messages = convert_tool_to_assistant = True, seed 42

What this adds over `subsample_curve_concept.py`:

- **Per-part AUROC.** Each row of the four cut splits carries a part from
  data/instructions_parts/<split>_parts.jsonl (scripts/make_instructions_parts.py), joined
  on line number. The probe scores the whole split once and `part_<split>_p<k>` is the
  AUROC over that part's rows alone, so the four part columns and `eval_<split>` come from
  exactly the same scores. Every part is class-balanced by construction (the parts are cut
  on PAIRS, and these splits are fully paired), so every part column is a real AUROC.
- **Eval activations are loaded ONCE per process**, not once per fit.

THE 49-ROW FLOOR. The inherited spec is batch_size 16 x gradient_accumulation_steps 4, so
the optimizer steps once per 64 samples and a no-base fit under ~49 rows takes ZERO steps
and returns the probe at INITIALISATION. `--grad-accum 1` steps on every batch, which makes
n=30 trainable; it changes accumulation and NOTHING else (batch_size stays 16), which is
what the published accum1 points on this campaign were run at. Rows carry the setting in
`grad_accum` and the resume key, and the two regimes go to separate CSVs and must not be
pooled — same rule as run_tgtmin_sizecurve_nobase.sh.

AUROC is tuberlens' own `calculate_metrics`, the function `get_performances` uses. `--check`
also runs `evaluate_probe` on the first fit and fails if any split differs, which is what
shows the once-per-process scoring reads the same numbers.

    .venv_claude/bin/python scripts/fit_instructions_parts.py \\
        data/instructions_deepseekv4pro_tgtnone_hc_context_drift_600.jsonl \\
        --sizes 540 300 120 60 --draws 8 --out scripts/instructions_parts_size_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import COMBINE, CONCEPTS, CONVERT, SEED, eval_source, load_rows  # noqa: E402

CONCEPT = CONCEPTS["instructions"]
DEV = CONCEPT.dev_data
PARTS_DIR = REPO / "data/instructions_parts"
# get_performances' default; it only feeds tpr_at_fpr, never the AUROC read here.
FPR = 0.01
SPLITS = sorted(p.stem for p in CONCEPT.eval_dir.glob("*.jsonl"))
DEV_SPLITS = sorted(p.stem for p in DEV.glob("*.jsonl"))
CUT_SPLITS = sorted(p.stem[:-len("_parts")] for p in PARTS_DIR.glob("*_parts.jsonl"))
PART_NAMES = [f"{s}_p{k}" for s in CUT_SPLITS for k in range(4)]
FIELDS = (["samples", "arm", "n", "draw", "grad_accum", "n_pos", "n_neg", "dev_mean"]
          + [f"dev_{s}" for s in DEV_SPLITS] + ["eval_mean"] + [f"eval_{s}" for s in SPLITS]
          + [f"part_{p}" for p in PART_NAMES] + ["seconds"])


def balanced(rows: list[dict], n: int, key: str) -> list[dict]:
    """One class-balanced subset, seeded on `key` = ``<stem>:<n>:<draw>``.

    Deliberately identical to `subsample_curve_concept.draw_subset(..., balanced=True)`:
    same key, and each class sampled under its own stream before one shuffle. So draw d of
    size n from a given set is the SAME rows here as in the published no-base curves, and
    a row here can be read directly against its twin there.
    """
    rng = random.Random(key)
    pos = [r for r in rows if r["labels"] == CONCEPT.pos_label]
    neg = [r for r in rows if r["labels"] == CONCEPT.neg_label]
    half, rest = n // 2, n - n // 2
    if len(pos) < half or len(neg) < rest:
        raise SystemExit(f"{key}: cannot draw {half}+{rest} from {len(pos)}/{len(neg)}")
    out = rng.sample(pos, half) + rng.sample(neg, rest)
    rng.shuffle(out)
    return out


def load_eval():
    """Every eval split with its Kaggle activations attached, loaded once, plus the parts."""
    from tuberlens.interfaces.dataset import LabelledDataset

    from agentic_redteam.evaluation import _assign_cached_activations
    from agentic_redteam.kaggle_activations import prefetch_eval_activations

    datasets = {
        s: LabelledDataset.load_from(
            CONCEPT.eval_dir / f"{s}.jsonl", pos_class_label=CONCEPT.pos_label,
            neg_class_label=CONCEPT.neg_label, combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT)
        for s in SPLITS
    }
    with CONCEPT.base_probe.open("rb") as fh:
        template = pickle.load(fh)
    prefetch_eval_activations(CONCEPT.eval_cache, datasets, eval_source(),
                              model_name=template.model_name, layer=int(template.layer),
                              cache_stem="acts_full.pt")
    _assign_cached_activations(datasets, CONCEPT.eval_cache / "acts_full.pt")
    missing = [s for s, d in datasets.items() if "activations" not in d.other_fields]
    if missing:
        raise SystemExit(f"eval activations missing for {missing}")

    parts = {}
    for s in CUT_SPLITS:
        rows = [json.loads(l) for l in (PARTS_DIR / f"{s}_parts.jsonl").open(encoding="utf-8")
                if l.strip()]
        if len(rows) != len(datasets[s]):
            raise SystemExit(f"{s}_parts.jsonl: {len(rows)} rows, split has {len(datasets[s])}")
        if [r["row"] for r in rows] != list(range(len(rows))):
            raise SystemExit(f"{s}_parts.jsonl is not in file order")
        parts[s] = np.array([r["part"] for r in rows])
    return datasets, parts


def score(probe, datasets, parts) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Per-split and per-part AUROC, plus the per-ROW probabilities they were read off.

    The second return value is what `--row-scores` persists: `probs[split]` is
    `predict_proba` over that split in FILE ORDER, the same vector every AUROC in the first
    return value is computed from. So a per-row statistic derived from it decomposes into
    the CSV's numbers exactly, with no second forward pass and no refit.
    """
    from tuberlens.evaluation import calculate_metrics

    out, probs = {}, {}
    for s, ds in datasets.items():
        y = np.array([label.to_int() for label in ds.labels])
        p = np.asarray(probe.predict_proba(ds))
        probs[s] = p.astype(np.float32)
        out[f"eval_{s}"] = float(calculate_metrics(y, p, fpr=FPR)["auroc"])
        if s in parts:
            for name in [f"{s}_p{k}" for k in range(4)]:
                m = parts[s] == name
                out[f"part_{name}"] = float(calculate_metrics(y[m], p[m], fpr=FPR)["auroc"])
    out["eval_mean"] = float(np.mean([out[f"eval_{s}"] for s in SPLITS]))
    return out, probs


def done_keys(path: Path) -> set[tuple[str, int, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["samples"], int(r["n"]), int(r["draw"]), int(r["grad_accum"]))
                for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", type=Path, nargs="+")
    ap.add_argument("--sizes", type=int, nargs="+", default=[540, 300, 120, 60])
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=0, metavar="K",
                    help="override gradient_accumulation_steps and nothing else. The "
                         "inherited spec is batch_size 16 x accumulation 4, so a no-base fit "
                         "under ~49 rows takes ZERO optimizer steps and returns the UNTRAINED "
                         "probe. K=1 makes n=30 trainable. Write these to their own --out; "
                         "they are not comparable to default-accumulation rows.")
    ap.add_argument("--arm", default=None, help="label for the arm column")
    ap.add_argument("--out", type=Path,
                    default=REPO / "scripts/instructions_parts_size_curve.csv")
    ap.add_argument("--check", action="store_true",
                    help="cross-check the first fit against evaluate_probe")
    ap.add_argument("--row-scores", type=Path, default=None, metavar="DIR",
                    help="also persist the per-ROW probabilities of every fit, as "
                         "DIR/<set stem>/ga<K>_n<n>_d<d>.npz (one float32 array per eval "
                         "split, in file order). These are the SAME scores the CSV's AUROCs "
                         "are computed from, so a per-sample curve built on them decomposes "
                         "into the split and part curves exactly.")
    args = ap.parse_args()

    from agentic_redteam.cli import _free_gpu
    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe, warm_sample_activation_cache

    probe_spec = None
    if args.grad_accum:
        from agentic_redteam.retrain import _infer_probe_spec
        with CONCEPT.base_probe.open("rb") as fh:
            spec = _infer_probe_spec(pickle.load(fh))
        hyper = dict(spec.hyperparams)
        hyper["gradient_accumulation_steps"] = args.grad_accum
        # ProbeSpec is a pydantic model, not a dataclass — dataclasses.replace raises on it.
        probe_spec = spec.model_copy(update={"hyperparams": hyper})
        print(f"probe spec: gradient_accumulation_steps={args.grad_accum} "
              f"(batch_size={hyper.get('batch_size')})", flush=True)

    sources = {p: load_rows(p, CONCEPT) for p in args.sets}
    seen = done_keys(args.out)
    jobs = [(p, n, d) for p in args.sets for n in args.sizes for d in range(args.draws)
            if (p.name, n, d, args.grad_accum) not in seen]
    print(f"{len(jobs)} fits to run ({len(seen)} rows already in {args.out})", flush=True)
    if not jobs:
        return

    # Extract every source row of every set in ONE model load, before any fit — the 27B
    # loads once, not once per set.
    warm_sample_activation_cache([r for rows in sources.values() for r in rows],
                                 base_probe_path=CONCEPT.base_probe,
                                 base_activation_cache_dir=CONCEPT.base_cache,
                                 combine_consecutive_messages=COMBINE,
                                 convert_tool_to_assistant=CONVERT, verbose=True)
    _free_gpu()

    datasets, parts = load_eval()
    fresh = not args.out.exists() or args.out.stat().st_size == 0
    if not fresh:
        with args.out.open(newline="", encoding="utf-8") as _fh:
            header = next(csv.reader(_fh), None)
        if header != FIELDS:
            raise SystemExit(f"{args.out} was written with different columns; use a new --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fh = args.out.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        writer.writeheader()
    scratch = CONCEPT.cache_dir / "instructions_part_probes"
    scratch.mkdir(parents=True, exist_ok=True)

    for i, (p, n, d) in enumerate(jobs, 1):
        subset = balanced(sources[p], n, f"{p.stem}:{n}:{d}")
        npos = sum(r["labels"] == CONCEPT.pos_label for r in subset)
        t0 = time.time()
        pkl = scratch / f"{p.stem}_ga{args.grad_accum or 'd'}_n{n}_d{d}.pkl"
        res = retrain_probe(
            samples=subset, base_probe_path=CONCEPT.base_probe, base_training_data_path=None,
            new_probe_path=pkl, dev_data_path=DEV, seed=SEED, base_data_fraction=1.0,
            base_activation_cache_dir=CONCEPT.base_cache, probe_spec=probe_spec,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        with pkl.open("rb") as pf:
            probe = pickle.load(pf)
        sc, probs = score(probe, datasets, parts)
        if args.row_scores:
            dest = args.row_scores / p.stem
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / f"ga{args.grad_accum or 'd'}_n{n}_d{d}.npz", **probs)
        if args.check and i == 1:
            df = evaluate_probe(pkl, CONCEPT.eval_dir, CONCEPT.eval_cache, max_samples=None,
                                seed=SEED, combine_consecutive_messages=COMBINE,
                                convert_tool_to_assistant=CONVERT, kaggle_source=eval_source())
            for _, r in df.iterrows():
                if r["dataset"] != "mean":
                    ours = sc[f"eval_{r['dataset']}"]
                    if abs(ours - float(r["auroc"])) > 1e-6:
                        raise SystemExit(f"--check: {r['dataset']} {ours} vs {r['auroc']}")
            print("--check: once-per-process scoring matches evaluate_probe", flush=True)
        target = next((s for s in CUT_SPLITS if s in p.stem), None)
        row = {"samples": p.name, "arm": args.arm or p.stem, "n": n, "draw": d,
               "grad_accum": args.grad_accum, "n_pos": npos, "n_neg": len(subset) - npos,
               "dev_mean": round(res.dev_auroc["mean"], 5),
               "seconds": round(time.time() - t0, 1)}
        for s, v in res.dev_auroc.items():
            if s != "mean":
                row[f"dev_{s}"] = round(v, 5)
        row.update({k: round(v, 5) for k, v in sc.items()})
        writer.writerow(row)
        fh.flush()
        on_target = (f"  {target} {row['eval_' + target]:.4f} ["
                     + " ".join(f"{row['part_' + target + '_p' + str(k)]:.3f}" for k in range(4))
                     + "]") if target else ""
        print(f"[{i}/{len(jobs)}] {p.stem} n={n} d={d}:{on_target}"
              f"  eval_mean {row['eval_mean']:.4f}  ({row['seconds']:.0f}s)", flush=True)
        pkl.unlink(missing_ok=True)
        del probe
        _free_gpu()
    fh.close()


if __name__ == "__main__":
    main()
