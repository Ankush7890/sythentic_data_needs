#!/usr/bin/env python
"""Split two toolace parts in two: toolace_lookup and toolace_finance.

Clustering these small parts is stable only where it is useless: k=2 on the embeddings
re-derives at ARI 1.00 for lookup and 0.97 for finance, but both cuts isolate a tiny,
nearly single-label corner (lookup's sports rows: 42 rows, 4 high-stakes; finance's crypto
rows: 19 rows, 4 low-stakes), on which no AUROC can be read. Every balanced embedding cut is
unstable (ARI 0.36-0.55). So each part is cut by CONTENT, with a keyword rule over the system
persona and the user turn, and the rule is checked against the embeddings rather than derived
from them:

  toolace_lookup  -> lookup_media    sports, music, video, social media, news, games
                     lookup_utility  everything else: documents, web and domains, location and
                                     property, drug and health information, business data
                     (agrees with a content-merged k=5 embedding clustering on 91% of rows)
  toolace_finance -> finance_money   crypto, forex and currency, loans, mortgages, retirement,
                                     tax, banking and credit
                     finance_markets everything else: stocks, earnings, ratings, market news
                     (agrees with embedding k=2 on the persona on 81% of rows)

ops and roledef keep their part as their subpart. Writes
data/toolace_parts/toolace_balanced_subparts.jsonl: {ids, row, part, subpart, labels}.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "eval_sets/highstakes/toolace_balanced.jsonl"
PARTS = REPO / "data/toolace_parts/toolace_balanced_parts.jsonl"
OUT = REPO / "data/toolace_parts/toolace_balanced_subparts.jsonl"

MEDIA = re.compile(
    r"\b(sport|football|soccer|cricket|basketball|league|match|team|music|song|video|instagram|"
    r"tiktok|twitter|tweet|social media|movie|game|gaming|entertainment|news|media|anime|"
    r"celebrit|playlist|youtube)", re.I)
MONEY = re.compile(
    r"\b(crypto|bitcoin|ethereum|dogecoin|blockchain|nft|token|wallet|defi|forex|currenc|"
    r"exchange rate|loan|mortgage|retire|bank|credit|tax|insurance|annuit|pension|budget|payment)",
    re.I)


def persona_and_user(msgs: list[dict]) -> str:
    return msgs[0]["content"].split("You are given a question")[0] + " " + msgs[1]["content"]


def subpart(part: str, msgs: list[dict]) -> str:
    text = persona_and_user(msgs)
    if part == "toolace_lookup":
        return "lookup_media" if MEDIA.search(text) else "lookup_utility"
    if part == "toolace_finance":
        return "finance_money" if MONEY.search(text) else "finance_markets"
    return part


def main() -> None:
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    parts = [json.loads(l) for l in PARTS.open(encoding="utf-8") if l.strip()]
    assert [p["row"] for p in parts] == list(range(len(rows)))
    counts = collections.Counter()
    with OUT.open("w", encoding="utf-8") as fh:
        for r, p in zip(rows, parts):
            sp = subpart(p["part"], json.loads(r["inputs"]))
            counts[(sp, r["labels"])] += 1
            fh.write(json.dumps({"ids": r["ids"], "row": p["row"], "part": p["part"],
                                 "subpart": sp, "labels": r["labels"]}) + "\n")
    for sp in sorted({k[0] for k in counts}):
        print(f"{sp:18s} {counts[(sp, 'high-stakes')]:4d} high  {counts[(sp, 'low-stakes')]:4d} low")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
