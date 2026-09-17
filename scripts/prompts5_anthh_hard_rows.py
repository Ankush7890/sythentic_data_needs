#!/usr/bin/env python
"""Which eval_ant_hh rows do the 40 n=590 probes get wrong, and how often?

Reads `scripts/prompts5_anthh_eval_scores.csv` (134 eval rows x 40 probe columns, written
by `prompts5_anthh_agreement.py`) and `eval_sets/hu_ha/eval_ant_hh.jsonl`.

A row counts as misclassified by a probe when the probe's probability falls on the wrong
side of a threshold. Two thresholds, because a probe's calibration is not the thing under
study:
  - **0.5**, the probe's own decision boundary;
  - **each probe's median score on this split**, the balanced-split threshold (the eval is
    exactly 67/67), which removes any constant shift between probes.

The toolace run of this analysis found the 0.5 rule produced a badly one-sided picture —
241 high-stakes errors against 6 low-stakes — that the median rule showed to be a
calibration artefact (65/74). Both are reported here for the same reason.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCORES = REPO / "scripts/prompts5_anthh_eval_scores.csv"
SPLIT = REPO / "eval_sets/hu_ha/eval_ant_hh.jsonl"
NEG = re.compile(r"\b(no|not|don't|won't|can't|sorry|refuse|rather not|i'm not sure)\b")


def describe(rows: list[dict], name: str) -> None:
    msgs = [json.loads(r["inputs"]) for r in rows]
    user = [m[0]["content"] for m in msgs]
    asst = [m[-1]["content"] for m in msgs]
    med = lambda v: statistics.median(v) if v else float("nan")  # noqa: E731
    print(f"\n=== {name}  (n={len(rows)})")
    print(f"  labels               {dict(Counter(r['labels'] for r in rows))}")
    print(f"  user chars median    {med([len(x) for x in user]):.0f}")
    print(f"  asst chars median    {med([len(x) for x in asst]):.0f}")
    print(f"  refusal/negation     {sum(bool(NEG.search(a.lower())) for a in asst)}"
          f" ({sum(bool(NEG.search(a.lower())) for a in asst) / max(1, len(rows)):.0%})")
    print(f"  question to user     {sum(a.strip().endswith('?') for a in asst)}")
    expl = [r.get("harm_explanation", "") or "" for r in rows]
    print(f"  why mentions 'user'  {sum('user' in e.lower() for e in expl)}")
    print(f"  why mentions assistant {sum('assistant' in e.lower() for e in expl)}")


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

    print(f"{len(probes)} probes x {len(labels)} eval rows\n")
    rates = {}
    for rule in ("half", "median"):
        n_wrong = [sum(wrong(p, i, rule) for p in probes) for i in range(len(labels))]
        rates[rule] = [k / len(probes) for k in n_wrong]
        over = [i for i, k in enumerate(n_wrong) if k > len(probes) / 2]
        allw = [i for i, k in enumerate(n_wrong) if k == len(probes)]
        never = [i for i, k in enumerate(n_wrong) if k == 0]
        by_label = Counter(split[i]["labels"] for i in over)
        print(f"threshold={rule:6}  >50% misclassified: {len(over):3} of {len(labels)}"
              f"   all-40-wrong: {len(allw):3}   never wrong: {len(never):3}")
        print(f"{'':22}of those >50%: {dict(by_label)}")

    over_both = [i for i in range(len(labels))
                 if rates["half"][i] > 0.5 and rates["median"][i] > 0.5]
    print(f"\n>50% UNDER BOTH RULES: {len(over_both)} of {len(labels)}")

    describe([split[i] for i in over_both], "misclassified >50% under both rules")
    rest = [i for i in range(len(labels)) if i not in over_both]
    describe([split[i] for i in rest], "the rest of the split")

    print("\n--- the rows, worst first")
    for i in sorted(over_both, key=lambda j: -rates["median"][j]):
        r, m = split[i], json.loads(split[i]["inputs"])
        mean_p = statistics.mean(scores[p][i] for p in probes)
        print(f"\n[{i}] label={r['labels']} rate(0.5)={rates['half'][i]:.0%} "
              f"rate(median)={rates['median'][i]:.0%} mean_p={mean_p:.3f}")
        print(f"   U: {' '.join(m[0]['content'].split())[:160]}")
        print(f"   A: {' '.join(m[-1]['content'].split())[:200]}")
        print(f"   why: {' '.join((r.get('harm_explanation') or '').split())[:180]}")

    out = REPO / "scripts/prompts5_anthh_row_rates.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "label", "rate_half", "rate_median", "mean_p", "user", "assistant"])
        for i in range(len(labels)):
            m = json.loads(split[i]["inputs"])
            w.writerow([i, split[i]["labels"], round(rates["half"][i], 4),
                        round(rates["median"][i], 4),
                        round(statistics.mean(scores[p][i] for p in probes), 4),
                        " ".join(m[0]["content"].split())[:300],
                        " ".join(m[-1]["content"].split())[:300]])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
