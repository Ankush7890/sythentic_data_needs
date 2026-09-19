#!/usr/bin/env python
"""Mean and sd of dev/eval AUROC per (arm, generator, training-set size), pooled curve.

Reads scripts/<concept>_pooled_size_curve.csv — the curve where every training row is
resampled, base included (run_pooled_sizecurve.sh) — and, for context, the fixed-base points
already committed in scripts/<concept>_gen90*.csv. The two are NOT the same protocol and are
printed in separate blocks: the gen90 rows hold each generator's own 50 base rows constant in
every draw, so their sd is the spread of the generated part alone.

`n` in the pooled file is the WHOLE training set, so it is directly comparable to gen90's
`n_training_rows` (540 generated + 50 base = 590) rather than to its `n`.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/report_pooled_curve.py --concept hu_harm
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ATTACKERS = ["llama70b", "gptoss", "nemotron", "deepseekv4pro"]


def arm_of(samples: str) -> str | None:
    """'general' / 'steered' for the two arms this study compares, else None."""
    if "_evaldescshape_" in samples or "_evaldesc_" in samples:
        return "steered"
    if re.search(r"_(general|[a-z0-9]+)_\d+\.jsonl$", samples) and "_tgt" not in samples:
        if "_general_" in samples:
            return "general"
        # gen90 names the unsteered arm `<concept>_<attacker>_600.jsonl`
        if re.match(r"^[a-z_]+_(" + "|".join(ATTACKERS) + r")_600\.jsonl$", samples):
            return "general"
    return None


def attacker_of(samples: str) -> str | None:
    for a in ATTACKERS:
        if f"_{a}_" in samples:
            return a
    return None


def collect(path: Path, size_field: str, tag: str) -> dict:
    out: dict = defaultdict(list)
    if not path.exists():
        return out
    for r in csv.DictReader(path.open(newline="", encoding="utf-8")):
        arm, att = arm_of(r["samples"]), attacker_of(r["samples"])
        if arm is None or att is None:
            continue
        out[(arm, att, int(r[size_field]), tag, r.get("base", ""))].append(
            (float(r["dev_mean"]), float(r["eval_mean"]))
        )
    return out


def show(title: str, data: dict) -> None:
    if not data:
        return
    print(f"\n{title}")
    print(f"  {'arm':8s} {'generator':14s} {'rows':>5s} {'draws':>5s} "
          f"{'dev mean':>9s} {'dev sd':>8s} {'eval mean':>10s} {'eval sd':>8s}   base-tag")
    for key in sorted(data, key=lambda k: (k[0], ATTACKERS.index(k[1]), k[2])):
        arm, att, n, _tag, base = key
        v = data[key]
        dv, ev = [a for a, _ in v], [b for _, b in v]
        sd = lambda x: f"{st.stdev(x):8.4f}" if len(x) > 1 else f"{'  n/a':>8s}"
        print(f"  {arm:8s} {att:14s} {n:5d} {len(v):5d} "
              f"{st.mean(dv):9.4f} {sd(dv)} {st.mean(ev):10.4f} {sd(ev)}   {base}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True)
    args = ap.parse_args()
    c = args.concept

    pooled = collect(REPO / f"scripts/{c}_pooled_size_curve.csv", "n", "pooled")
    gen90_name = "highstakes_gen90_dev500" if c == "highstakes" else f"{c}_gen90"
    fixed = collect(REPO / f"scripts/{gen90_name}.csv", "n_training_rows", "fixed")
    fixed = {k: v for k, v in fixed.items() if not str(k[4]).startswith("none")}

    show(f"{c}: POOLED — every training row resampled, base included", pooled)
    show(f"{c}: FIXED-BASE (existing gen90 rows, own 50-row base constant in every draw)",
         fixed)
    print()


if __name__ == "__main__":
    main()
