#!/usr/bin/env python
"""Find the toolace eval rows every one of the 20 n=590 probes gets wrong, and describe them.

Reads `scripts/prompts5_toolace_eval_scores.csv` (734 eval rows x 20 probe columns, written by
`prompts5_toolace_agreement.py`) and `eval_sets/highstakes/toolace_balanced.jsonl`.

A row counts as misclassified by a probe when the probe's probability falls on the wrong side
of a threshold. Two thresholds are reported, because a probe's calibration is not the thing
under study:
  - 0.5, the probe's own decision boundary;
  - each probe's own MEDIAN score on this split, which is the balanced-split threshold (the
    eval is exactly 367/367), and so removes any constant shift between probes.
The rows named "always wrong" are those wrong under BOTH rules for all 20 probes.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCORES = REPO / "scripts/prompts5_toolace_eval_scores.csv"
SPLIT = REPO / "eval_sets/highstakes/toolace_balanced.jsonl"


def describe(rows: list[dict], name: str) -> dict:
    """Surface statistics for one group of eval rows."""
    msgs = [json.loads(r["inputs"]) for r in rows]
    sysl = [len(m[0]["content"]) for m in msgs]
    tot = [sum(len(x["content"]) for x in m) for m in msgs]
    user = [len(m[1]["content"]) for m in msgs]
    asst = [len(m[-1]["content"]) for m in msgs]
    nfun = [m[0]["content"].count('{"name":') for m in msgs]
    wrapper = sum("Role definition" in m[1]["content"] for m in msgs)
    call = sum(m[-1]["content"].strip().startswith("[") for m in msgs)
    scale = [int(r["scale_labels"]) for r in rows]
    conf = [int(r["scale_label_confidence"]) for r in rows]
    med = lambda v: statistics.median(v) if v else float("nan")  # noqa: E731
    out = {
        "n": len(rows),
        "labels": dict(Counter(r["labels"] for r in rows)),
        "scale_median": med(scale),
        "scale_counts": dict(sorted(Counter(scale).items())),
        "confidence_median": med(conf),
        "confidence<=8": sum(c <= 8 for c in conf),
        "n_msgs": dict(sorted(Counter(len(m) for m in msgs).items())),
        "chars_median": med(tot),
        "system_median": med(sysl),
        "user_median": med(user),
        "assistant_median": med(asst),
        "n_functions_median": med(nfun),
        "wrapper_rows": wrapper,
        "bare_call_endings": call,
    }
    print(f"\n=== {name}")
    for k, v in out.items():
        print(f"  {k:22} {v}")
    return out


def main() -> None:
    score_rows = list(csv.DictReader(SCORES.open()))
    probes = [c for c in score_rows[0] if c not in ("index", "label_int", "label")]
    split = [json.loads(line) for line in SPLIT.open(encoding="utf-8") if line.strip()]
    if len(split) != len(score_rows):
        raise SystemExit(f"{len(score_rows)} scored rows but {len(split)} split rows")

    labels = [int(r["label_int"]) for r in score_rows]
    scores = {p: [float(r[p]) for r in score_rows] for p in probes}
    medians = {p: statistics.median(v) for p, v in scores.items()}

    def wrong(p: str, i: int, rule: str) -> bool:
        thr = 0.5 if rule == "half" else medians[p]
        return (scores[p][i] > thr) != bool(labels[i])

    for rule in ("half", "median"):
        n_wrong = [sum(wrong(p, i, rule) for p in probes) for i in range(len(labels))]
        allw = [i for i, k in enumerate(n_wrong) if k == len(probes)]
        allr = [i for i, k in enumerate(n_wrong) if k == 0]
        print(f"\nthreshold={rule}: all-20-wrong {len(allw)}, all-20-right {len(allr)}, "
              f"mean wrong per row {statistics.mean(n_wrong):.2f}")

    n_half = [sum(wrong(p, i, "half") for p in probes) for i in range(len(labels))]
    n_med = [sum(wrong(p, i, "median") for p in probes) for i in range(len(labels))]
    hard = [i for i in range(len(labels))
            if n_half[i] == len(probes) and n_med[i] == len(probes)]
    easy = [i for i in range(len(labels)) if n_half[i] == 0 and n_med[i] == 0]
    print(f"\nWRONG UNDER BOTH RULES BY ALL {len(probes)} PROBES: {len(hard)} of {len(labels)}")

    describe([split[i] for i in hard], f"always misclassified (n={len(hard)})")
    describe([split[i] for i in easy], f"always correct (n={len(easy)})")
    describe(split, "whole split")

    print("\n--- the always-misclassified rows")
    for i in hard:
        r, m = split[i], json.loads(split[i]["inputs"])
        mean_p = statistics.mean(scores[p][i] for p in probes)
        print(f"\n[{i}] label={r['labels']} scale={r['scale_labels']} "
              f"conf={r['scale_label_confidence']} mean_p={mean_p:.3f} msgs={len(m)}")
        print(f"   why: {r['scale_label_explanation'][:220]}")
        print(f"   persona: {m[0]['content'].split('. You are given')[0][:150]}")
        print(f"   user: {' '.join(m[1]['content'].split())[:220]}")
        print(f"   asst: {' '.join(m[-1]['content'].split())[:200]}")


if __name__ == "__main__":
    main()
