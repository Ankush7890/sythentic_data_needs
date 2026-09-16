#!/usr/bin/env python
"""Fit 20 toolace probes (5 prompt sets x 4 draws at n=590) and score every eval row.

The size curve throws its probes away — `subsample_curve_concept.py` unlinks each candidate
once its CSV row is written — so this refits the draws it needs and KEEPS them, then records
one probability per (probe, eval row) so the rows nothing can classify can be found.

Draws are drawn exactly as the size curve draws them (`draw_subset`, seeded on
`(file stem, n, draw)`, class-balanced), so probe `<set>_d0` here IS the probe behind that
set's n=590 draw-0 row in `scripts/highstakes_toolace_prompts5_sizecurve.csv`.

Writes:
  probes/prompts5_toolace_n590/<set>_d<k>.pkl        the 20 probes
  scripts/prompts5_toolace_eval_scores.csv           734 rows x 20 probe columns, plus the
                                                     row's label and its index in the split

Scoring uses the cached eval activation blob, so no LLM is loaded.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import COMBINE, CONCEPTS, CONVERT, SEED, load_rows  # noqa: E402
from subsample_curve_concept import draw_subset  # noqa: E402

SETS = ["deceptive", "endings", "grid", "longtail", "replica"]
N = 590
DRAWS = 4
SPLIT = "toolace_balanced"
DEV = REPO / "dev_samples/highstakes_500_toolace"
PROBE_DIR = REPO / "probes/prompts5_toolace_n590"
OUT_CSV = REPO / "scripts/prompts5_toolace_eval_scores.csv"


def main() -> None:
    concept = CONCEPTS["highstakes"]
    from agentic_redteam.cli import _free_gpu
    from agentic_redteam.retrain import retrain_probe

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in SETS:
        path = REPO / f"data/highstakes_deepseekv4pro_p5_{name}_600.jsonl"
        rows = load_rows(path, concept)
        for d in range(DRAWS):
            out = PROBE_DIR / f"{name}_d{d}.pkl"
            jobs.append((name, d, path, rows, out))

    for i, (name, d, path, rows, out) in enumerate(jobs, 1):
        if out.exists():
            print(f"[{i}/{len(jobs)}] {out.name} already fit", flush=True)
            continue
        subset = draw_subset(rows, N, path.stem, d, True, concept)
        t0 = time.time()
        res = retrain_probe(
            samples=subset, base_probe_path=concept.base_probe,
            base_training_data_path=None, new_probe_path=out,
            dev_data_path=DEV, seed=SEED, base_data_fraction=1.0,
            base_activation_cache_dir=concept.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        print(f"[{i}/{len(jobs)}] {out.name}: dev {res.dev_auroc['mean']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        _free_gpu()

    # --- score every eval row with every probe, off the cached activation blob ---
    from tuberlens.interfaces.dataset import LabelledDataset

    from agentic_redteam.evaluation import _assign_cached_activations
    from agentic_redteam.retrain import load_probe

    first = load_probe(PROBE_DIR / f"{SETS[0]}_d0.pkl")
    dataset = LabelledDataset.load_from(
        concept.eval_dir / f"{SPLIT}.jsonl",
        pos_class_label=first.pos_class_label, neg_class_label=first.neg_class_label,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
    )
    datasets = {SPLIT: dataset}
    _assign_cached_activations(datasets, concept.eval_cache / "acts_full.pt")
    dataset = datasets[SPLIT]
    if "activations" not in dataset.other_fields:
        raise SystemExit("eval activations are not cached; run the eval prefetch first")

    labels = [lab.to_int() for lab in dataset.labels]
    scores: dict[str, list[float]] = {}
    for name in SETS:
        for d in range(DRAWS):
            probe = load_probe(PROBE_DIR / f"{name}_d{d}.pkl")
            p = probe.predict_proba(dataset)
            scores[f"{name}_d{d}"] = [float(x) for x in p]
            print(f"scored {name}_d{d}", flush=True)
            _free_gpu()

    cols = list(scores)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "label_int", "label"] + cols)
        for i in range(len(labels)):
            w.writerow([i, labels[i], str(dataset.labels[i])]
                       + [round(scores[c][i], 6) for c in cols])
    print(f"wrote {OUT_CSV} ({len(labels)} rows x {len(cols)} probes)", flush=True)


if __name__ == "__main__":
    main()
