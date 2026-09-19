#!/usr/bin/env python
"""Collect `shape_diversity.py --report-json` outputs into one per-concept sets file.

`scripts/<concept>_armfilter_sets.json` is what `report_armfilter.py` reads for each set's
shape and lexical statistics, so the committed artefacts (that file plus the fits CSV) are
self-contained and the multi-gigabyte pools themselves need not be kept.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/build_armfilter_sets.py \\
        --concept instructions --work-dir .arm_filter_work
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ATTACKER_NAME = {"llama70b": "llama70b", "ds": "deepseek", "gptoss": "gpt-oss",
                 "nemo": "nemotron"}
ARM_ORDER = ["general", "desc", "attacker", "gen"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--work-dir", type=Path, default=REPO / ".arm_filter_work")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or REPO / f"scripts/{args.concept}_armfilter_sets.json"
    sets: dict[str, dict] = {}
    for tag, attacker in ATTACKER_NAME.items():
        for arm in ARM_ORDER:
            rep = args.work_dir / f"report_{args.concept}_{tag}_{arm}.json"
            pool = args.work_dir / f"arm_{args.concept}_{tag}_{arm}_orig.jsonl"
            if not rep.exists():
                continue
            r = json.loads(rep.read_text(encoding="utf-8"))
            sh, lx = r["shape"], r["lexical"]
            sets[pool.name] = {
                "attacker": attacker, "arm": arm, "concept": args.concept,
                "n_rows": sh["n_in"], "pool_rows": sh["n_out"], "keep": r["keep"],
                "shape": {k: sh[k] for k in (
                    "signatures_in", "signatures_out", "normalized_entropy_in",
                    "normalized_entropy_out", "entropy_bits_in", "entropy_bits_out",
                    "max_share_in", "max_share_out", "mi_shape_label_bits_in",
                    "mi_shape_label_bits_out", "mean_nn_distance_in", "mean_nn_distance_out")},
                "lexical": {k: lx[k] for k in (
                    "n_features", "train_accuracy", "mean_confidence_in",
                    "mean_confidence_out", "global_cut_would_drop")},
            }
    out.write_text(json.dumps(
        {"_about": (f"Per-set diagnostics for the {args.concept} arm-filter study. Keys are "
                    "the `samples` column of the fits CSV. Built by "
                    "scripts/build_armfilter_sets.py from scripts/shape_diversity.py reports."),
         "sets": sets}, indent=1), encoding="utf-8")
    print(f"wrote {out} — {len(sets)} sets")


if __name__ == "__main__":
    main()
