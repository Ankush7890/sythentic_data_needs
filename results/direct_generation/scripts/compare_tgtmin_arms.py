#!/usr/bin/env python
"""Compare the shape-free (tgtmin) arm against the arms it ablates, split by split.

The question the arm asks is narrow: a split-targeted set's description was written from
MEASUREMENTS of the split file, so does the set win because of the situation it describes or
because it was told the answer's shape? That is answered ON THE TARGET SPLIT, not on the
seven-split mean, so this reports both — a targeted set that lifts its own split while losing
the rest is the expected shape of the result, and the mean hides it.

Four arms per split, differing by one block each:

    tgtshot   measured description + dev few-shot anchor + topics + shape, split's own mode
    tgtnone   the same, anchor dropped
    tgtmin    situation and label boundary only; no shape, no topics, no anchor, unpaired
    kind-pin  the shared eval-data description pinned to this split's numbered kind

plus the two untargeted controls (unsteered, and the mixed eval-description set).

    ${REPO_ROOT}/.venv_claude/bin/python scripts/compare_tgtmin_arms.py [--n 540] [--concept ...]
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CONCEPTS = {
    "instructions": dict(
        csv=REPO / "scripts/instructions_gen90.csv",
        splits=["anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
                "hc_contradiction", "mm_substitution", "oig_context_drift", "oig_omission"],
        # oig_omission has no targeted arm of any kind: the verbatim eval description never
        # named it, so it is the untouched control every arm is scored against.
        targeted=["anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
                  "hc_contradiction", "mm_substitution", "oig_context_drift"],
    ),
    "highstakes": dict(
        csv=REPO / "scripts/highstakes_gen90_dev500.csv",
        splits=["anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced"],
        targeted=["anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced"],
    ),
}


def summarise(rows: list[dict], target: str, others: list[str]) -> str:
    if not rows:
        return f"{'—':>8}"
    on = [float(r[f"eval_{target}"]) for r in rows]
    off = [st.mean([float(r[f"eval_{s}"]) for s in others]) for r in rows]
    sd = f"±{st.stdev(on):.4f}" if len(on) > 1 else " " * 7
    return f"{st.mean(on):.4f} {sd} / {st.mean(off):.4f}  (x{len(on)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="540", help="draw size to summarise (default 540)")
    ap.add_argument("--tag", default="deepseekv4pro")
    ap.add_argument("--concept", choices=sorted(CONCEPTS), action="append")
    args = ap.parse_args()

    for concept in (args.concept or sorted(CONCEPTS)):
        spec = CONCEPTS[concept]
        if not spec["csv"].exists():
            print(f"{concept}: {spec['csv']} missing"); continue
        rows = [r for r in csv.DictReader(spec["csv"].open()) if r["n"] == args.n]
        print(f"\n=== {concept} (n={args.n}) — ON-TARGET ±sd / off-target mean ===\n")
        arms = [("tgtmin", "{c}_{t}_tgtmin_{s}_600.jsonl"),
                ("tgtshot", "{c}_{t}_tgtshot_{s}_600.jsonl"),
                ("tgtnone", "{c}_{t}_tgtnone_{s}_600.jsonl"),
                ("kind-pin", "{c}_{t}_{s}_600.jsonl")]
        print(f"{'split':28s}" + "".join(f"{a:>32s}" for a, _ in arms))
        for split in spec["targeted"]:
            others = [s for s in spec["splits"] if s != split]
            line = f"{split:28s}"
            for _, pat in arms:
                name = pat.format(c=concept, t=args.tag, s=split)
                line += f"{summarise([r for r in rows if r['samples'] == name], split, others):>32s}"
            print(line)
        for name, label in ((f"{concept}_{args.tag}_600.jsonl", "unsteered"),
                            (f"{concept}_{args.tag}_evaldesc_600.jsonl", "evaldesc (mixed)")):
            got = [r for r in rows if r["samples"] == name]
            if got:
                means = {s: st.mean([float(r[f"eval_{s}"]) for r in got]) for s in spec["splits"]}
                print(f"\n{label:28s} per split: "
                      + "  ".join(f"{s}={v:.4f}" for s, v in means.items()))


if __name__ == "__main__":
    main()
