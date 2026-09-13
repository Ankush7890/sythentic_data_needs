#!/usr/bin/env python
"""Render `scripts/highstakes_armfilter.csv` — two set-level filters, 16 arms, 4 attackers.

Four attackers (llama70b, deepseek-v4-pro, gpt-oss-120b, nemotron) x four arms
(`general`, `+desc`, `+attacker`, `gen`), each in three pools: the unfiltered set, the
lexical-confounder cut and the shape-mix cut (`scripts/shape_diversity.py`, `--keep 0.8`).
Every pool is drawn from at the SAME n — 0.6 x the arm's row count — 8 class-balanced draws
per pool, each arm fitted on ITS OWN attacker's 50-row base, so the three columns of an arm
differ in which rows were available to draw from and in nothing else.

Error bars are the sd across the 8 draws. `Δ` is against that arm's unfiltered column and
carries the standard error of the difference (`sqrt(sd_a²/8 + sd_b²/8)`) — the draws are
independent, not paired, because the harness seeds each draw on the pool's filename.

Per-set shape and lexical statistics come from `scripts/highstakes_armfilter_sets.json`, so
this script needs nothing but the two committed files.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/report_armfilter.py
    ${REPO_ROOT}/.venv_claude/bin/python scripts/report_armfilter.py --column dev_mean
    ${REPO_ROOT}/.venv_claude/bin/python scripts/report_armfilter.py --splits llama70b/gen
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARMS = ["general", "desc", "attacker", "gen"]
ARM_LABEL = {"general": "general", "desc": "+desc", "attacker": "+attacker", "gen": "gen"}
SPLITS = ["anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced"]
POOLS = [("original", "_orig.jsonl"), ("lexical", "_orig_lex.jsonl"),
         ("shape-mix", "_orig_shape.jsonl")]


def cell(v: list[float]) -> str:
    if not v:
        return "—"
    return f"{v[0]:.4f}" if len(v) == 1 else f"{st.mean(v):.4f} ±{st.stdev(v):.4f}"


def delta(a: list[float], b: list[float]) -> tuple[float | None, float | None]:
    if len(a) < 2 or len(b) < 2:
        return None, None
    return st.mean(b) - st.mean(a), math.sqrt(st.stdev(a) ** 2 / len(a) + st.stdev(b) ** 2 / len(b))


def fmt_delta(a: list[float], b: list[float]) -> str:
    d, se = delta(a, b)
    return "—" if d is None else f"{d:+.4f} ±{se:.4f}"


def corr(x: list[float], y: list[float]) -> float:
    mx, my = st.mean(x), st.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=REPO / "scripts/highstakes_armfilter.csv")
    ap.add_argument("--sets", type=Path, default=REPO / "scripts/highstakes_armfilter_sets.json")
    ap.add_argument("--column", default="eval_mean", help="eval_mean | dev_mean")
    ap.add_argument("--splits", default=None,
                    help="also print the per-eval-split table for one arm, e.g. llama70b/gen")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="", encoding="utf-8")))
    meta = json.loads(args.sets.read_text(encoding="utf-8"))["sets"]

    def vals(stem: str, col: str) -> list[float]:
        return [float(r[col]) for r in rows if r["samples"] == stem and r.get(col)]

    order = list(dict.fromkeys(m["attacker"] for m in meta.values()))
    print(f"## {args.column} — per arm, 8 draws per pool\n")
    print("| attacker | arm | N | n | H_norm | top shape % | original | Δ lex | Δ shape |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    pts = []
    for attacker in order:
        for arm in ARMS:
            stem = next((k for k, m in meta.items()
                         if m["attacker"] == attacker and m["arm"] == arm), None)
            if stem is None:
                continue
            m = meta[stem]
            o = vals(stem, args.column)
            l = vals(stem.replace("_orig.jsonl", "_orig_lex.jsonl"), args.column)
            s = vals(stem.replace("_orig.jsonl", "_orig_shape.jsonl"), args.column)
            if not o:
                continue
            n = sorted({int(r["n"]) for r in rows if r["samples"] == stem})[0]
            print(f"| {attacker} | {ARM_LABEL[arm]} | {m['n_rows']} | {n} | "
                  f"{m['shape']['normalized_entropy_in']:.3f} | "
                  f"{m['shape']['max_share_in'] * 100:.1f} | {cell(o)} | "
                  f"{fmt_delta(o, l)} | {fmt_delta(o, s)} |")
            dl, _ = delta(o, l)
            ds, _ = delta(o, s)
            if dl is not None and ds is not None:
                pts.append((m, st.mean(o), dl, ds))

    if len(pts) > 2:
        print(f"\n## Across the {len(pts)} arms\n")
        print("| filter | mean Δ | median | negative | range | t |")
        print("| --- | --- | --- | --- | --- | --- |")
        for name, idx in (("lexical", 2), ("shape-mix", 3)):
            v = [p[idx] for p in pts]
            mean, se = st.mean(v), st.stdev(v) / math.sqrt(len(v))
            print(f"| {name} | {mean:+.4f} ±{se:.4f} | {st.median(v):+.4f} | "
                  f"{sum(1 for x in v if x < 0)}/{len(v)} | "
                  f"{min(v):+.4f} … {max(v):+.4f} | {mean / se:+.2f} |")

        print("\n| candidate predictor of Δ shape-mix | correlation |")
        print("| --- | --- |")
        for label, f in (
            ("shape entropy of the set (H_norm)", lambda p: p[0]["shape"]["normalized_entropy_in"]),
            ("dominant-shape share", lambda p: p[0]["shape"]["max_share_in"]),
            ("distinct shapes in the set", lambda p: len(p[0]["shape"]["signatures_in"])),
            ("rows in the set", lambda p: p[0]["n_rows"]),
            ("unfiltered AUROC of the arm", lambda p: p[1]),
            ("bag-of-words train accuracy", lambda p: p[0]["lexical"]["train_accuracy"]),
        ):
            print(f"| {label} | {corr([f(p) for p in pts], [p[3] for p in pts]):+.3f} |")

    if args.splits:
        attacker, arm = args.splits.split("/")
        stem = next(k for k, m in meta.items()
                    if m["attacker"] == attacker and m["arm"] == arm)
        print(f"\n## {attacker} / {ARM_LABEL[arm]} — per eval split\n")
        print("| pool | " + " | ".join(SPLITS) + " | mean |")
        print("| --- | " + " | ".join("---" for _ in SPLITS) + " | --- |")
        for pname, suffix in POOLS:
            s = stem.replace("_orig.jsonl", suffix)
            print(f"| {pname} | " + " | ".join(cell(vals(s, f"eval_{x}")) for x in SPLITS)
                  + " | " + cell(vals(s, "eval_mean")) + " |")


if __name__ == "__main__":
    main()
