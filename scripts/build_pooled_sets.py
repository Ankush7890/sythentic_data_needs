#!/usr/bin/env python
"""Pool each generator's own 50-row base with its 600 generated rows into one 650-row set.

WHY. Every size curve in this repo so far fits `fixed 50-row base + n drawn generated
rows`, so at the small end most of the training set is the SAME 50 rows in every draw and
the spread across draws understates the real run-to-run variation. These pools let
`subsample_curve_concept.py --no-base` draw all n rows from base ∪ generated, so no row is
held constant across draws.

The pool is 325/325 (base 25/25 + generated 300/300), so a class-balanced draw of any even
n <= 650 is available. Rows are deduplicated on the canonical conversation text: a base row
that also appears in the generated set would otherwise be drawable twice.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/build_pooled_sets.py --concept instructions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import CONCEPTS  # noqa: E402

OUT_DIR = REPO / ".pool_work"
ATTACKERS = ["llama70b", "gptoss", "nemotron", "deepseekv4pro"]
# The steered arm's filename token differs by concept: instructions and highstakes carry
# `_evaldesc_`, hu_harm only ever had the SHAPED variant written for it (and that is the
# one scripts/hu_harm_gen90.csv actually curves), so the token is `_evaldescshape_`.
STEERED_TOKEN = {"instructions": "evaldesc", "highstakes": "evaldesc",
                 "hu_harm": "evaldescshape"}


def key(row: dict) -> str:
    """Canonical text of one conversation, for dedup."""
    msgs = json.loads(row["inputs"]) if isinstance(row["inputs"], str) else row["inputs"]
    return json.dumps([[m.get("role"), m.get("content")] for m in msgs], sort_keys=True)


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def build(concept_name: str, attacker: str, arm: str, src: Path, base: Path) -> Path | None:
    if not src.exists():
        print(f"  SKIP {src.name} (missing)")
        return None
    concept = CONCEPTS[concept_name]
    rows, seen, dropped = [], set(), 0
    for r in read(base) + read(src):          # base first, so a collision keeps the base row
        k = key(r)
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        rows.append(r)
    pos = sum(1 for r in rows if r["labels"] == concept.pos_label)
    neg = sum(1 for r in rows if r["labels"] == concept.neg_label)
    if pos + neg != len(rows):
        raise SystemExit(f"{src.name}: {len(rows) - pos - neg} rows carry an unknown label")
    out = OUT_DIR / f"pool_{concept_name}_{attacker}_{arm}_{len(rows)}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  {out.name:52s} {len(rows):4d} rows  {pos}/{neg}  (dup dropped: {dropped})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    tok = STEERED_TOKEN[args.concept]
    print(f"{args.concept}:")
    for a in ATTACKERS:
        base = REPO / f"data/{args.concept}_{a}_50.jsonl"
        if not base.exists():
            print(f"  SKIP {a} (no own base)")
            continue
        build(args.concept, a, "general", REPO / f"data/{args.concept}_{a}_600.jsonl", base)
        build(args.concept, a, tok, REPO / f"data/{args.concept}_{a}_{tok}_600.jsonl", base)


if __name__ == "__main__":
    main()
