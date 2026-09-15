#!/usr/bin/env python
"""Fit generated sets with NO base data and score every highstakes eval split, plus toolace's
four parts separately.

hard_split_experiments. The protocol matches the `base='none'` rows of
`scripts/highstakes_gen90_dev500.csv`, so those rows are directly comparable:

    probe      gemma-3-27b-it L32, linear_then_softmax, single — architecture and metadata
               inherited from probes/gen_gemma27b_highstakes/probe_iter0.pkl, whose WEIGHTS
               and base data are never used
    training   the drawn rows ALONE (class-balanced n/2 per class, seeded on (stem, n, draw))
    dev        dev_samples/highstakes_500, used whole (early stopping)
    eval       eval_sets/highstakes, FULL splits, activations from Kaggle
    transforms combine_consecutive_messages = convert_tool_to_assistant = True, seed 42

What this adds over `subsample_curve_concept.py`:

- **Per-part AUROC on toolace.** Each toolace row's part comes from
  data/toolace_parts/toolace_balanced_parts.jsonl (scripts/make_toolace_parts.py), joined on
  line number. The probe scores all 734 rows once; `part_<name>` is the AUROC over that part's
  rows alone, so the four part columns and `eval_toolace_balanced` come from the same scores.
- **Eval activations are loaded ONCE per process**, not once per fit — 46 GB of blobs, which
  is most of what a fit here would otherwise spend.
- **`--union`**: ONE arm over sets of any size; `--sizes` are then PERCENTAGES and a draw takes
  that share of EACH set (class-balanced), so `--union --sizes 100 90 --draws 1 8` is the full
  union plus eight 90% resamples. The `n` column records the total rows actually trained on.
- **`--pool`**: one draw takes n/len(sets) rows from EACH set (each class-balanced), so the
  four specialists can be fit together at the same total size as one of them.

AUROC is tuberlens' own `calculate_metrics`, the function `get_performances` uses. `--check`
also runs `evaluate_probe` on the first fit and fails if any split differs, which is what
shows the once-per-process scoring reads the same numbers.

    .venv_claude/bin/python scripts/fit_toolace_parts.py \\
        data/toolace_parts/highstakes_deepseekv4pro_tgtnone_toolace_ops_600.jsonl \\
        --sizes 600 540 --draws 1 8 --out scripts/highstakes_toolace_parts.csv
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

CONCEPT = CONCEPTS["highstakes"]
DEV = REPO / "dev_samples/highstakes_500"
PARTS_FILE = REPO / "data/toolace_parts/toolace_balanced_subparts.jsonl"
# get_performances' default; it only feeds tpr_at_fpr, never the AUROC read here.
FPR = 0.01
PART_NAMES = ["toolace_ops", "toolace_lookup", "toolace_finance", "toolace_roledef"]
# lookup and finance cut in two (scripts/make_toolace_subparts.py); scored as sub_<name>.
SUBPART_NAMES = ["lookup_media", "lookup_utility", "finance_markets", "finance_money"]
SPLITS = sorted(p.stem for p in CONCEPT.eval_dir.glob("*.jsonl"))
DEV_SPLITS = sorted(p.stem for p in DEV.glob("*.jsonl"))
FIELDS = (["samples", "arm", "n", "draw", "n_pos", "n_neg", "dev_mean"]
          + [f"dev_{s}" for s in DEV_SPLITS] + ["eval_mean"] + [f"eval_{s}" for s in SPLITS]
          + [f"part_{p}" for p in PART_NAMES] + [f"sub_{p}" for p in SUBPART_NAMES] + ["seconds"])


def balanced(rows: list[dict], n: int, key: str) -> list[dict]:
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
    """Every eval split with its Kaggle activations attached, loaded once."""
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
    rows = [json.loads(l) for l in PARTS_FILE.open(encoding="utf-8") if l.strip()]
    if len(rows) != len(datasets["toolace_balanced"]):
        raise SystemExit(f"{PARTS_FILE}: {len(rows)} rows, toolace has "
                         f"{len(datasets['toolace_balanced'])}")
    return datasets, (np.array([r["part"] for r in rows]), np.array([r["subpart"] for r in rows]))


def score(probe, datasets, parts) -> dict[str, float]:
    """`parts` is (part per toolace row, subpart per toolace row)."""
    part, sub = parts
    from tuberlens.evaluation import calculate_metrics

    out = {}
    for s, ds in datasets.items():
        y = np.array([label.to_int() for label in ds.labels])
        p = np.asarray(probe.predict_proba(ds))
        out[f"eval_{s}"] = float(calculate_metrics(y, p, fpr=FPR)["auroc"])
        if s == "toolace_balanced":
            for prefix, names, labels in (("part", PART_NAMES, part), ("sub", SUBPART_NAMES, sub)):
                for name in names:
                    m = labels == name
                    out[f"{prefix}_{name}"] = float(calculate_metrics(y[m], p[m], fpr=FPR)["auroc"])
    out["eval_mean"] = float(np.mean([out[f"eval_{s}"] for s in SPLITS]))
    return out


def done_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["samples"], int(r["n"]), int(r["draw"])) for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", type=Path, nargs="+")
    ap.add_argument("--sizes", type=int, nargs="+", default=[600, 540])
    ap.add_argument("--draws", type=int, nargs="+", default=[1, 8],
                    help="draws per size, aligned with --sizes")
    ap.add_argument("--pool", action="store_true",
                    help="ONE arm: each draw takes n/len(sets) class-balanced rows from every set")
    ap.add_argument("--union", action="store_true",
                    help="ONE arm: --sizes are percentages; each draw takes that share of every "
                         "set, class-balanced")
    ap.add_argument("--arm", default=None, help="label for the arm column")
    ap.add_argument("--out", type=Path, default=REPO / "scripts/highstakes_toolace_parts.csv")
    ap.add_argument("--check", action="store_true",
                    help="cross-check the first fit against evaluate_probe")
    args = ap.parse_args()
    if len(args.draws) != len(args.sizes):
        ap.error("--draws must align with --sizes")
    if args.union and args.pool:
        ap.error("--union and --pool are exclusive")
    if args.union and not all(0 < s <= 100 for s in args.sizes):
        ap.error("--union takes --sizes as percentages in (0, 100]")

    from agentic_redteam.cli import _free_gpu
    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import retrain_probe, warm_sample_activation_cache

    sources = {p: load_rows(p, CONCEPT) for p in args.sets}
    if args.union:
        arms = [(f"union:{len(args.sets)}sets", list(args.sets))]
    elif args.pool:
        arms = [("pool:" + "+".join(p.stem for p in args.sets), list(args.sets))]
    else:
        arms = [(p.name, [p]) for p in args.sets]

    seen = done_keys(args.out)
    def recorded_n(paths, n):   # the `n` a finished row carries: total rows under --union
        if not args.union:
            return n
        return sum(2 * round(len(sources[p]) * n / 200) for p in paths)

    jobs = [(name, paths, n, d) for name, paths in arms
            for n, k in zip(args.sizes, args.draws) for d in range(k)
            if (name, recorded_n(paths, n), d) not in seen]
    print(f"{len(jobs)} fits to run ({len(seen)} rows already in {args.out})", flush=True)
    if not jobs:
        return

    # Extract every source row once, in one model load, before any fit.
    for p, rows in sources.items():
        warm_sample_activation_cache(rows, base_probe_path=CONCEPT.base_probe,
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
    scratch = CONCEPT.cache_dir / "toolace_part_probes"
    scratch.mkdir(parents=True, exist_ok=True)

    for i, (name, paths, n, d) in enumerate(jobs, 1):
        if args.union:   # n is a percentage of each set; keep each share even for balance
            subset = [r for p in paths
                      for r in balanced(sources[p], 2 * round(len(sources[p]) * n / 200),
                                        f"{p.stem}:pct{n}:{d}")]
        else:
            per = n // len(paths)
            subset = [r for p in paths for r in balanced(sources[p], per, f"{p.stem}:{n}:{d}")]
        random.Random(f"{name}:{n}:{d}").shuffle(subset)
        npos = sum(r["labels"] == CONCEPT.pos_label for r in subset)
        t0 = time.time()
        pkl = scratch / f"n{n}_d{d}_{abs(hash(name)) % 10**8}.pkl"
        res = retrain_probe(
            samples=subset, base_probe_path=CONCEPT.base_probe, base_training_data_path=None,
            new_probe_path=pkl, dev_data_path=DEV, seed=SEED, base_data_fraction=1.0,
            base_activation_cache_dir=CONCEPT.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        with pkl.open("rb") as pf:
            probe = pickle.load(pf)
        sc = score(probe, datasets, parts)
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
        arm = args.arm or ("union" if args.union else "pool" if args.pool else "specialist")
        row = {"samples": name, "arm": arm, "n": len(subset) if args.union else n, "draw": d, "n_pos": npos, "n_neg": len(subset) - npos,
               "dev_mean": round(res.dev_auroc["mean"], 5),
               "seconds": round(time.time() - t0, 1)}
        for s, v in res.dev_auroc.items():
            if s != "mean":
                row[f"dev_{s}"] = round(v, 5)
        row.update({k: round(v, 5) for k, v in sc.items()})
        writer.writerow(row)
        fh.flush()
        print(f"[{i}/{len(jobs)}] {name} n={n} d={d}: toolace {row['eval_toolace_balanced']:.4f} "
              + " ".join(f"{p[8:]}={row['part_' + p]:.3f}" for p in PART_NAMES)
              + f"  eval_mean {row['eval_mean']:.4f}  ({row['seconds']:.0f}s)", flush=True)
        pkl.unlink(missing_ok=True)
        del probe
        _free_gpu()
    fh.close()


if __name__ == "__main__":
    main()
