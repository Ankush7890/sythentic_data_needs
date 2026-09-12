#!/usr/bin/env python
"""Render `scripts/highstakes_armfilter.csv` — the two set-level filters, arm by arm.

Four llama70b high-stakes arms (`general`, `+desc`, `+attacker`, `gen`), each in three
pools: the unfiltered set, the lexical-confounder cut and the shape-mix cut
(`scripts/shape_diversity.py`, `--keep 0.8`). Every pool is drawn from at the SAME n —
0.6 x the arm's original row count — 8 class-balanced draws per pool, so the three columns
of an arm differ in which rows were available to draw from and in nothing else.

Error bars are the sd across the 8 draws, as everywhere else in this repo. `Δ` is against
that arm's unfiltered column and carries the standard error of the difference
(`sqrt(sd_a^2 + sd_b^2)/sqrt(8)`) — the draws are independent, not paired, because the
harness seeds each draw on the pool's filename.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/report_armfilter.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARMS = [("general", "arm_general"), ("+desc", "arm_desc"),
        ("+attacker", "arm_attacker"), ("gen", "arm_gen")]
POOLS = [("original", "_orig.jsonl"), ("lexical", "_orig_lex.jsonl"),
         ("shape-mix", "_orig_shape.jsonl")]
SPLITS = ["anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced"]


def cell(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{st.mean(vals):.4f} ±{st.stdev(vals):.4f}"


def delta(a: list[float], b: list[float]) -> str:
    """b − a, with the standard error of the difference."""
    if len(a) < 2 or len(b) < 2:
        return "—"
    se = math.sqrt(st.stdev(a) ** 2 / len(a) + st.stdev(b) ** 2 / len(b))
    d = st.mean(b) - st.mean(a)
    return f"{d:+.4f} ±{se:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=REPO / "scripts/highstakes_armfilter.csv")
    ap.add_argument("--column", default="eval_mean", help="eval_mean | dev_mean")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="", encoding="utf-8")))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["samples"]].append(r)

    def vals(stem: str, suffix: str, col: str) -> list[float]:
        return [float(r[col]) for r in by.get(stem + suffix, []) if r.get(col)]

    out: list[str] = []
    out.append("## Filters against the unfiltered arm — equal n, 8 draws\n")
    for col, title in (("eval_mean", "eval mean (4 high-stakes splits, full)"),
                       ("dev_mean", "dev mean (500-row dev cut)")):
        out.append(f"\n### {title}\n")
        out.append("| arm | n | original | lexical | Δ lex | shape-mix | Δ shape |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, stem in ARMS:
            o = vals(stem, POOLS[0][1], col)
            l = vals(stem, POOLS[1][1], col)
            s = vals(stem, POOLS[2][1], col)
            ns = {int(r["n"]) for r in by.get(stem + POOLS[0][1], [])}
            n = str(sorted(ns)[0]) if ns else "—"
            out.append(f"| {name} | {n} | {cell(o)} | {cell(l)} | {delta(o, l)} | "
                       f"{cell(s)} | {delta(o, s)} |")
        out.append(f"\n(draws per cell: " + ", ".join(
            f"{name} {len(vals(stem, POOLS[0][1], col))}/{len(vals(stem, POOLS[1][1], col))}/"
            f"{len(vals(stem, POOLS[2][1], col))}" for name, stem in ARMS) + ")")

    out.append("\n\n## Per eval split\n")
    for name, stem in ARMS:
        if not by.get(stem + POOLS[0][1]):
            continue
        out.append(f"\n### {name}\n")
        out.append("| pool | " + " | ".join(SPLITS) + " | mean |")
        out.append("| --- | " + " | ".join("---" for _ in SPLITS) + " | --- |")
        for pname, suffix in POOLS:
            cells = [cell(vals(stem, suffix, f"eval_{s}")) for s in SPLITS]
            out.append(f"| {pname} | " + " | ".join(cells) + " | "
                       + cell(vals(stem, suffix, "eval_mean")) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
