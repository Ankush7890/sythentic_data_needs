#!/usr/bin/env python
"""Read the instructions per-part size-curve CSVs and print the curves as text tables.

Two tables per arm:

  ON TARGET   the targeted split's own AUROC against n, mean +/- sd over the draws — the
              size curve itself, the same quantity the published no-base curves plot.
  BY PART     the four equal-size parts of that split, each against n, so the on-target
              number above is decomposed into where on the split the data actually lands.

Default-accumulation rows (n >= 60) and accumulation-1 rows (n = 30) live in separate
CSVs and are printed as separate blocks, never pooled — a fit at accumulation 1 takes a
different number of optimizer steps per epoch and is not the same measurement.

    .venv_claude/bin/python scripts/report_instrparts.py
    .venv_claude/bin/python scripts/report_instrparts.py --markdown
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO / "scripts/instructions_parts_size_curve.csv"
ACCUM1 = REPO / "scripts/instructions_parts_size_curve_accum1.csv"
CUT_SPLITS = ["hc_context_drift", "hc_contradiction", "mm_substitution", "oig_context_drift"]


def arm_target(stem: str) -> tuple[str, str]:
    """('+ shape info' | 'shape-free', the split the set was written for)."""
    kind = "+shape" if "_tgtnone_" in stem else "shape-free" if "_tgtmin_" in stem else "?"
    target = next((s for s in CUT_SPLITS if s in stem), "?")
    return kind, target


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def agg(rows: list[dict], col: str) -> tuple[float, float, int]:
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    if not vals:
        return float("nan"), float("nan"), 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def cell(rows: list[dict], col: str, md: bool) -> str:
    m, s, k = agg(rows, col)
    if not k:
        return "     —    "
    return f"{m:.3f}±{s:.3f}" if md else f"{m:.3f}±{s:.3f}"


def block(rows: list[dict], title: str, md: bool) -> None:
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["samples"]].append(r)
    if not by_arm:
        return
    print(f"\n{'#' if md else '='} {title}\n")
    for stem in sorted(by_arm, key=lambda s: (arm_target(s)[0], arm_target(s)[1])):
        arm_rows = by_arm[stem]
        kind, target = arm_target(stem)
        sizes = sorted({int(r["n"]) for r in arm_rows})
        by_n = {n: [r for r in arm_rows if int(r["n"]) == n] for n in sizes}
        ndraws = {n: len(v) for n, v in by_n.items()}
        print(f"{'##' if md else '--'} {kind}  ->  {target}   "
              f"(draws: {', '.join(f'n{n}:{k}' for n, k in ndraws.items())})")
        head = ["row"] + [f"n={n}" for n in sizes]
        lines = [
            ["on target"] + [cell(by_n[n], f"eval_{target}", md) for n in sizes],
            ["eval mean"] + [cell(by_n[n], "eval_mean", md) for n in sizes],
        ]
        for k in range(4):
            lines.append([f"  part p{k}"]
                         + [cell(by_n[n], f"part_{target}_p{k}", md) for n in sizes])
        if md:
            print("| " + " | ".join(head) + " |")
            print("|" + "|".join(["---"] * len(head)) + "|")
            for ln in lines:
                print("| " + " | ".join(ln) + " |")
        else:
            w = max(len(h) for h in head + [ln[0] for ln in lines]) + 2
            print("  " + "".join(h.ljust(w) if i == 0 else h.rjust(13)
                                 for i, h in enumerate(head)))
            for ln in lines:
                print("  " + "".join(c.ljust(w) if i == 0 else c.rjust(13)
                                     for i, c in enumerate(ln)))
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    block(read(MAIN), "Default accumulation (batch 16 x accum 4)", args.markdown)
    block(read(ACCUM1), "Accumulation 1 — n=30 only, NOT comparable to the block above",
          args.markdown)


if __name__ == "__main__":
    main()
