#!/usr/bin/env python
"""Cut `eval_sets/highstakes/toolace_balanced.jsonl` into four parts of similar rows.

toolace has no well-separated clusters (cosine silhouette <= 0.15 at every k from 2 to 20),
so this is a partition chosen for being STABLE and INTERPRETABLE, not a discovered structure:

  toolace_roledef  every row whose user turn is the "Role definition: ... Historical dialog
                   data is as follows" wrapper — a crisp format rule, independent of topic.
  the other rows   k-means k=3 on bge-base-en-v1.5 CLS embeddings of
                   (system persona + first 8 function names/descriptions + conversation).
                   Re-clustering 20 random 80% subsamples agrees at mean ARI 0.90.
                   The three clusters are named after their content: toolace_ops,
                   toolace_lookup, toolace_finance.

Writes data/toolace_parts/toolace_balanced_parts.jsonl: one {ids, row, part, labels} per eval
row, in file order (`row` is the 0-based line number, which is what joins it to the eval
activations blob).

    .venv_claude/bin/python scripts/make_toolace_parts.py
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "eval_sets/highstakes/toolace_balanced.jsonl"
OUT = REPO / "data/toolace_parts/toolace_balanced_parts.jsonl"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def row_text(msgs: list[dict]) -> str:
    s = msgs[0]["content"]
    persona = s.split("You are given a question")[0].strip()
    fns = re.findall(r'"name": "([^"]+)", "description": "([^"]{0,150})', s)
    fn_text = "; ".join(f"{a}: {b}" for a, b in fns[:8])
    convo = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in msgs[1:])
    return f"{persona}\nFunctions: {fn_text}\n{convo}"


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(EMBED_MODEL).to(dev).eval()
    out = []
    for i in range(0, len(texts), 32):
        b = tok(texts[i:i + 32], padding=True, truncation=True, max_length=512,
                return_tensors="pt").to(dev)
        with torch.no_grad():
            h = model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    return np.concatenate(out)


def name_clusters(labels: np.ndarray, msgs: list[list[dict]], idx: np.ndarray) -> dict[int, str]:
    """Name the three k-means clusters by content, so the names survive a label permutation."""
    finance = re.compile(r"\b(stock|financ|invest|crypto|forex|trading|market)", re.I)
    rapid = re.compile(r'"name": "[^"]* [^"]*"')
    score = {}
    for c in set(labels):
        rows = [msgs[i] for i, l in zip(idx, labels) if l == c]
        score[c] = (
            np.mean([bool(finance.search(m[0]["content"][:300])) for m in rows]),
            np.mean([bool(rapid.search(m[0]["content"])) for m in rows]),
        )
    fin = max(score, key=lambda c: score[c][0])
    rest = [c for c in score if c != fin]
    lookup = max(rest, key=lambda c: score[c][1])
    ops = next(c for c in rest if c != lookup)
    return {fin: "toolace_finance", lookup: "toolace_lookup", ops: "toolace_ops"}


def main() -> None:
    from sklearn.cluster import KMeans

    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    msgs = [json.loads(r["inputs"]) for r in rows]
    roledef = np.array(["Role definition" in m[1]["content"] for m in msgs])
    rest = np.where(~roledef)[0]
    X = embed([row_text(msgs[i]) for i in rest])
    km = KMeans(3, n_init=20, random_state=0).fit(X)
    names = name_clusters(km.labels_, msgs, rest)

    part = ["toolace_roledef"] * len(rows)
    for i, l in zip(rest, km.labels_):
        part[i] = names[l]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for i, (r, p) in enumerate(zip(rows, part)):
            fh.write(json.dumps({"ids": r["ids"], "row": i, "part": p, "labels": r["labels"]}) + "\n")
    counts = collections.Counter((p, r["labels"]) for p, r in zip(part, rows))
    for p in sorted(set(part)):
        print(f"{p:18s} {counts[(p, 'high-stakes')]:4d} high  {counts[(p, 'low-stakes')]:4d} low")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
