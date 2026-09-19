#!/usr/bin/env python
"""Cut four `eval_sets/instructions/*.jsonl` splits into four parts of similar rows each.

Same recipe as `scripts/make_toolace_parts.py` — k-means on bge-base-en-v1.5 CLS
embeddings — with one structural difference that matters here.

**Every one of these four splits is fully class-paired**: the two rows of a pair share
every message but the last, one carrying each label (verified: each prefix occurs exactly
twice; mm_substitution has 8 prefixes written four times, which is two pairs of pairs).
So the clustering runs on the PAIR, embedding the shared prefix alone, and both rows of a
pair land in the same part. That is what keeps every part exactly class-balanced — a part
with one class in it has no AUROC at all, and clustering rows individually would drift
that way, since the final assistant turn is exactly what the label is a property of.

Embedding the prefix also keeps the partition a statement about *what the conversation is
about*, independent of the thing being scored.

The four splits are cut at k=4:

    hc_context_drift    user+doc, answer, new doc, answer      194 rows /  97 pairs
    hc_contradiction    user+doc, answer                       200 rows / 100 pairs
    mm_substitution     user+context, answer                   200 rows / 100 pairs
    oig_context_drift   user, answer, user, answer             194 rows /  97 pairs

**The four parts are held to EQUAL SIZE**, which plain k-means does not do: run free, it
cuts oig_context_drift into 108 / 42 / 38 / 6 rows and mm_substitution into 78 / 58 / 54 /
10, and an AUROC over three rows per class is not a measurement. So the assignment step is
balanced — Lloyd's algorithm with the nearest-centroid step replaced by a minimum-cost
assignment of pairs to equal-sized slots (`scipy.optimize.linear_sum_assignment`, which is
exact here: ~100 pairs). Every part then holds a quarter of the pairs — ~48 rows, ~24 per
class — so the four per-part AUROCs are on equal footing and move for the same reason.
`--free` runs the unconstrained k-means instead.

Parts are named `<split>_p0..p3` and each is described in the sidecar by its top
distinguishing terms and a sample prefix. Like toolace, these are not well-separated
clusters — the printed silhouette says how weak — so STABILITY is what is reported: 20
re-clusterings of random 80% subsamples, scored by mean adjusted Rand index against the
full-data labels.

Writes, per split, `data/instructions_parts/<split>_parts.jsonl`: one
`{row, pair, part, labels}` per eval row in FILE ORDER (`row` is the 0-based line number,
which is what joins it to the eval activations blob), plus
`data/instructions_parts/parts_summary.json` with the descriptions and diagnostics.

    .venv_claude/bin/python scripts/make_instructions_parts.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "eval_sets/instructions"
OUT_DIR = REPO / "data/instructions_parts"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
SPLITS = ["hc_context_drift", "hc_contradiction", "mm_substitution", "oig_context_drift"]
K = 4
# Re-clustering subsample fraction and count, matching make_toolace_parts.py's report.
STABILITY_FRAC, STABILITY_N = 0.8, 20


def prefix_text(msgs: list[dict]) -> str:
    """The conversation up to (not including) the final assistant turn.

    Truncated per message so a long retrieved document cannot crowd the user's actual
    question out of the 512-token window — the question is what the parts should be about.
    """
    return "\n".join(f"{m['role']}: {m['content'][:700]}" for m in msgs[:-1])


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
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def top_terms(texts: list[str], labels: np.ndarray, cluster: int, n: int = 12) -> list[str]:
    """Terms that distinguish one cluster from the rest of the split (log-odds on counts)."""
    import re

    stop = set("""the a an and or of to in on for is are was were be been it its this that
        with as at by from what how why when who which do does did not no you your i my we
        they he she his her their them there here if then than so but about into over under
        please context following text answer question based use using can could should would
        will may might have has had also more most some any all one two new same other
        """.split())
    tok = lambda t: [w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in stop]  # noqa: E731
    inside, outside = collections.Counter(), collections.Counter()
    for t, l in zip(texts, labels):
        (inside if l == cluster else outside).update(set(tok(t)))
    n_in = max(int((labels == cluster).sum()), 1)
    n_out = max(len(labels) - n_in, 1)
    score = {
        w: (inside[w] + 1) / (n_in + 2) / ((outside[w] + 1) / (n_out + 2))
        for w in inside if inside[w] >= max(3, 0.08 * n_in)
    }
    return [w for w, _ in sorted(score.items(), key=lambda kv: -kv[1])[:n]]


def balanced_kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """k-means whose assignment step is constrained to equal-sized clusters.

    Lloyd's algorithm with nearest-centroid replaced by a minimum-cost assignment of the
    n points to n slots, `ceil(n/k)` or `floor(n/k)` of them per cluster. The assignment
    is solved exactly (Hungarian, O(n^3) — n is ~100 pairs here), so each iteration is
    the best balanced assignment to the current centroids and the objective cannot rise.
    Seeded from a free k-means, which is a better start than random centroids.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans

    n = len(X)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    slot_cluster = np.repeat(np.arange(k), sizes)          # length n
    labels = KMeans(k, n_init=20, random_state=seed).fit_predict(X)
    for _ in range(iters):
        cent = np.stack([X[labels == c].mean(0) if (labels == c).any() else X[c]
                         for c in range(k)])
        cost = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)   # n x k
        _, slot = linear_sum_assignment(cost[:, slot_cluster])     # n x n
        new = slot_cluster[slot]
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def stability(X: np.ndarray, labels: np.ndarray, k: int, equal: bool) -> float:
    """Mean ARI of `STABILITY_N` re-clusterings of random subsamples against `labels`."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    aris = []
    for _ in range(STABILITY_N):
        idx = rng.choice(len(X), size=int(STABILITY_FRAC * len(X)), replace=False)
        sub = (balanced_kmeans(X[idx], k) if equal
               else KMeans(k, n_init=10, random_state=0).fit_predict(X[idx]))
        aris.append(adjusted_rand_score(labels[idx], sub))
    return float(np.mean(aris))


def cut_split(split: str, equal: bool = True) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rows = [json.loads(l) for l in (EVAL_DIR / f"{split}.jsonl").open(encoding="utf-8")
            if l.strip()]
    msgs = [json.loads(r["inputs"]) for r in rows]

    # Group rows into pairs by their shared prefix, in first-appearance order.
    pair_of_row, prefixes = [], {}
    for m in msgs:
        key = json.dumps(m[:-1], sort_keys=True)
        pair_of_row.append(prefixes.setdefault(key, len(prefixes)))
    pair_of_row = np.array(pair_of_row)
    pair_text = [""] * len(prefixes)
    for m, p in zip(msgs, pair_of_row):
        pair_text[p] = prefix_text(m)

    X = embed(pair_text)
    raw = (balanced_kmeans(X, K) if equal
           else KMeans(K, n_init=20, random_state=0).fit_predict(X))
    # Under --free, order parts by descending size so p0 is always the largest; when the
    # parts are equal-sized, order by the first pair each contains. Either way the
    # numbering is a property of the data, not of k-means' own label permutation.
    order = ([c for c, _ in collections.Counter(raw).most_common()] if not equal
             else sorted(set(raw), key=lambda c: int(np.argmax(raw == c))))
    remap = {c: i for i, c in enumerate(order)}
    pair_part = np.array([remap[c] for c in raw])
    part_of_row = pair_part[pair_of_row]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{split}_parts.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"row": i, "pair": int(pair_of_row[i]),
                                 "part": f"{split}_p{part_of_row[i]}",
                                 "labels": r["labels"]}) + "\n")

    counts = collections.Counter(zip(part_of_row, (r["labels"] for r in rows)))
    labels_seen = sorted({r["labels"] for r in rows})
    info = {
        "split": split, "n_rows": len(rows), "n_pairs": len(prefixes),
        "equal_size": equal,
        "silhouette": round(float(silhouette_score(X, pair_part)), 4),
        "stability_ari": round(stability(X, pair_part, K, equal), 4),
        "parts": [],
    }
    for p in range(K):
        info["parts"].append({
            "part": f"{split}_p{p}",
            "n_rows": int((part_of_row == p).sum()),
            "n_pairs": int((pair_part == p).sum()),
            "per_label": {l: counts[(p, l)] for l in labels_seen},
            "top_terms": top_terms(pair_text, pair_part, p),
            "example": pair_text[int(np.argmax(pair_part == p))][:400],
        })
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--free", action="store_true",
                    help="unconstrained k-means; parts come out very unequal (see the "
                         "module docstring) and small parts have no usable AUROC")
    args = ap.parse_args()

    summary = {}
    for split in args.splits:
        info = cut_split(split, equal=not args.free)
        summary[split] = info
        print(f"\n== {split}  {info['n_rows']} rows / {info['n_pairs']} pairs  "
              f"silhouette {info['silhouette']:+.3f}  stability ARI {info['stability_ari']:.2f}")
        for p in info["parts"]:
            per = "  ".join(f"{v} {k.split('_')[1][:4]}" for k, v in p["per_label"].items())
            print(f"   {p['part']:24s} {p['n_rows']:4d} rows ({per})  "
                  f"{', '.join(p['top_terms'][:8])}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "parts_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}/<split>_parts.jsonl and parts_summary.json")


if __name__ == "__main__":
    main()
