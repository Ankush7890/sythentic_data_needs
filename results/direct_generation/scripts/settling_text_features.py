#!/usr/bin/env python
"""A wider search for what distinguishes the settling / takeoff populations.

`explain_settling_classes.py` tested the obvious handles — length, turn count, refusal
shape, label, k-means part, the base probe's prior, prefix embeddings — and found nothing
that survived correction. This widens the search on the text side, and adds the
representation side, which is cheap because the activations are already cached.

    surface      characters, words, sentences, type-token ratio, digits
    form         does the ending open with Yes / No, is it refusal-shaped, how many
                 hedges ("may", "might", "appears") and negations it carries
    grounding    lexical overlap between the ending and the prefix, and the cosine between
                 their embeddings. These splits are about whether the answer follows the
                 supplied source, so "how much of the source does this answer reuse" is the
                 feature most likely to carry the concept, if anything does.
    prefix       length, sentence count, and whether the question is polar (Is/Can/Does)
                 or open (What/Why/How) — pair-level, since the prefix is shared
    content      cross-validated logistic regression on bge embeddings of the ending
    geometry     mean-pooled activation norm, the crude centroid margin, and local density
                 (mean cosine distance to the 10 nearest rows). Rows sitting in a dense,
                 well-separated region of the representation are the ones a linear probe
                 should pick up first — this is the natural mechanism for "learned early",
                 and it costs one pass over a blob already on disk.

THE LEVEL OF ANALYSIS. As in `explain_settling_classes.py`: these splits are class-paired,
so pair-level features are tested per PAIR against a pair-permutation null, and row-level
features get a within-pair sign test whose null is a coin flip per pair, which removes
every between-pair confound by construction. Every family gets Benjamini-Hochberg q values,
and every null is reported with the smallest effect it could have detected.

    .venv_claude/bin/python scripts/settling_text_features.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import analyze_row_settling as ARS  # noqa: E402
from analyze_row_settling import EVAL_DIR, REFUSAL, load_split  # noqa: E402
from explain_settling_classes import auroc, bh, cv_auroc, perm_p  # noqa: E402

SETTLING = REPO / "scripts/instructions_row_settling.csv"
SMALL = REPO / "scripts/instructions_hcdrift_small_settling.csv"
SMALL_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
PERM = 2000
SEED = 0

HEDGE = re.compile(r"\b(may|might|could|possibly|generally|typically|appears|suggests|"
                   r"some evidence|not always|unclear|however)\b", re.I)
NEG = re.compile(r"\b(no|not|cannot|can't|don't|doesn't|never|without)\b", re.I)
WORD = re.compile(r"[a-z]{3,}")
STOP = set("""the and for that with this from you your are was were has have had but not
    can could would should will may might about into over under they them their his her
    its our out who which what when where how why answer question based use using""".split())


def content_words(t: str) -> set[str]:
    return {w for w in WORD.findall(t.lower()) if w not in STOP}


def report(title: str, names, stats, ps, note: str = "") -> None:
    qs = bh(ps)
    print(f"\n  {title:<38} {'AUROC':>7} {'p':>7} {'BH q':>7}")
    for i, n in enumerate(names):
        mark = "  <-" if qs[i] < 0.05 else ("  (p only)" if ps[i] < 0.05 else "")
        s = f"{stats[i]:>7.3f}" if not np.isnan(stats[i]) else f"{'':>7}"
        print(f"  {n:<38} {s} {ps[i]:>7.3f} {qs[i]:>7.3f}{mark}")
    if note:
        print(f"    {note}")


def geometry(split: str, y: np.ndarray):
    """Mean-pooled activation norm, centroid margin and local density, per row."""
    from fit_instructions_parts import load_eval
    datasets, _ = load_eval()
    acts = datasets[split].other_fields["activations"]
    X = np.stack([a.float().mean(0).cpu().numpy() for a in acts])
    tokens = np.array([float(a.shape[0]) for a in acts])
    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cp, cn = Z[y].mean(0), Z[~y].mean(0)
    cp, cn = cp / np.linalg.norm(cp), cn / np.linalg.norm(cn)
    # Signed toward the row's OWN class, so a high value means "far on the right side".
    margin = np.where(y, Z @ cp - Z @ cn, Z @ cn - Z @ cp)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    density = np.sort(S, axis=1)[:, -10:].mean(1)
    return {"activation norm (mean-pooled)": np.linalg.norm(X, axis=1),
            "centroid margin, own class": margin,
            "local density (10-NN cosine)": density,
            "conversation tokens": tokens}


def analyse(split: str, arm: str, target: np.ndarray, label: str, rng) -> None:
    """`target` is 1 for the slow population, 0 for the fast one, -1 to drop."""
    print(f"\n\n## {split} — {arm}   [{label}]")
    labels, prefixes, members, part = load_split(split)
    raw = [json.loads(l) for l in (EVAL_DIR / f"{split}.jsonl").open(encoding="utf-8")
           if l.strip()]
    end = [" ".join(json.loads(r["inputs"])[-1]["content"].split()) for r in raw]
    msgs = [json.loads(r["inputs"]) for r in raw]
    pre = ["\n".join(m["content"] for m in mm[:-1]) for mm in msgs]
    geo = geometry(split, labels)

    rowf = {
        "ending characters": np.array([len(e) for e in end], float),
        "ending words": np.array([len(e.split()) for e in end], float),
        "ending sentences": np.array([e.count(".") + e.count("?") + 1.0 for e in end]),
        "ending type-token ratio": np.array(
            [len(set(e.lower().split())) / max(len(e.split()), 1) for e in end]),
        "ending has digits": np.array([float(any(ch.isdigit() for ch in e)) for e in end]),
        "ending opens 'Yes'": np.array([float(e[:4].lower().startswith("yes")) for e in end]),
        "ending opens 'No'": np.array([float(e[:3].lower().startswith("no")) for e in end]),
        "ending is refusal-shaped": np.array(
            [float(any(k in e.lower()[:60] for k in REFUSAL)) for e in end]),
        "hedge words in ending": np.array([float(len(HEDGE.findall(e))) for e in end]),
        "negations in ending": np.array([float(len(NEG.findall(e))) for e in end]),
        "ending/prefix word overlap": np.array(
            [len(content_words(e) & content_words(p)) / max(len(content_words(e)), 1)
             for e, p in zip(end, pre)]),
        **geo,
    }

    # -------------------------------------------------------------- pair level
    ok, yp, gi = [], [], []
    for g, (pos, neg) in enumerate(members):
        st = target[pos + neg]
        if (st < 0).any():
            continue
        ok.append(g)
        yp.append(int(st.max() == 1))
    yp = np.array(yp)
    gi = np.array(ok)
    print(f"\n  {len(yp)} usable pairs — {(yp == 0).sum()} fast, {(yp == 1).sum()} slow")
    if len(set(yp.tolist())) < 2:
        print("  (one class empty; skipped)")
        return

    first = [members[g][0][0] for g in gi]
    pairf = {
        "prefix characters": np.array([len(pre[i]) for i in first], float),
        "prefix sentences": np.array([pre[i].count(".") + 1.0 for i in first]),
        "longest source message": np.array(
            [max(len(m["content"]) for m in msgs[i][:-1]) for i in first], float),
        "polar question (Is/Can/Does/Do)": np.array(
            [float(bool(re.match(r"\s*(is|can|does|do|are|was|were|will|should)\b",
                                 pre[i], re.I))) for i in first]),
        "|ending length gap| in pair": np.array(
            [abs(np.mean([len(end[j]) for j in members[g][0]])
                 - np.mean([len(end[j]) for j in members[g][1]])) for g in gi]),
        "|overlap gap| in pair": np.array(
            [abs(np.mean([rowf["ending/prefix word overlap"][j] for j in members[g][0]])
                 - np.mean([rowf["ending/prefix word overlap"][j] for j in members[g][1]]))
             for g in gi]),
        "pair mean local density": np.array(
            [geo["local density (10-NN cosine)"][members[g][0] + members[g][1]].mean()
             for g in gi]),
    }
    names, stats, ps = [], [], []
    for n, x in pairf.items():
        a = auroc(x, yp)
        names.append(n)
        stats.append(a)
        ps.append(perm_p(x, yp, a, rng))
    report("pair-level feature", names, stats, ps)

    # -------------------------------------------------------------- within pair
    sp = [(pos[0], neg[0]) for pos, neg in members
          if len(pos) == 1 and len(neg) == 1
          and set(target[[pos[0], neg[0]]].tolist()) == {0, 1}]
    print(f"\n  {len(sp)} pairs split one fast / one slow")
    if len(sp) >= 10:
        names, stats, ps = [], [], []
        for n, x in rowf.items():
            if np.ptp(x) == 0:
                continue
            wins = np.array([float(x[a_] > x[b_]) + 0.5 * float(x[a_] == x[b_])
                             for a_, b_ in ((p_[0], p_[1]) if target[p_[0]] == 1
                                            else (p_[1], p_[0]) for p_ in sp)])
            obs = wins.mean()
            nd = np.where(rng.random((PERM, len(wins))) < 0.5, wins, 1 - wins).mean(1)
            names.append(f"the slow row has the higher {n}")
            stats.append(obs)
            ps.append(float((np.abs(nd - 0.5) >= abs(obs - 0.5)).mean()))
        mde = 0.5 + 2.80 * 0.5 / np.sqrt(len(sp))
        report("within-pair sign test", names, stats, ps,
               note=f"a null here is a null only down to AUROC {mde:.2f} "
                    f"({len(sp)} pairs)")

    # -------------------------------------------------------------- omnibus
    from make_instructions_parts import embed
    Xe = embed([end[members[g][0][0]] + " || " + end[members[g][1][0]] for g in gi])
    Xg = np.stack([np.concatenate([
        [geo[k][members[g][0] + members[g][1]].mean()] for k in geo]).ravel() for g in gi])
    print(f"\n  {'omnibus model (5-fold CV, pair level)':<38} {'AUROC':>7} {'p':>7}")
    for n, X in (("both endings, bge embedding", Xe), ("geometry features", Xg)):
        a = cv_auroc(X, yp)
        null = np.array([cv_auroc(X, rng.permutation(yp), seed=s) for s in range(20)])
        p = float((np.abs(null - 0.5) >= abs(a - 0.5)).mean())
        print(f"  {n:<38} {a:>7.3f} {p:>7.3f}   (null {null.mean():.3f} +- {null.std():.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    rng = np.random.default_rng(SEED)

    rows = list(csv.DictReader(SETTLING.open(encoding="utf-8")))
    for split, arm in (("oig_context_drift", "shape-free"),
                       ("mm_substitution", "+ shape info")):
        k = {int(r["row"]): r["klass"] for r in rows if r["split"] == split}
        n = max(k) + 1
        t = np.array([1 if k.get(i) == "unsettled" else
                      0 if k.get(i) in {"free", "early", "late"} else -1 for i in range(n)])
        analyse(split, arm, t, "still moving at 540 vs settled", rng)

    # hc_drift: the sub-60 grid, where the population that replicates is takeoff.
    ARS.SIZES, ARS.GA = SMALL_SIZES, "1"
    srows = list(csv.DictReader(SMALL.open(encoding="utf-8")))
    labels, _, _, _ = load_split("hc_context_drift")
    u = np.array([[float(r[f"u{n}"]) for n in SMALL_SIZES] for r in srows])
    d = np.array([float(r["delta"]) for r in srows])
    tk = np.array([ARS.takeoff(u[i], d[i]) for i in range(len(srows))])
    tk = np.where(tk == 0, 130, tk)
    med = np.median(tk)
    t = np.where(tk > med, 1, 0)
    analyse("hc_context_drift", "shape-free", t, f"takeoff later than n={med:.0f}", rng)


if __name__ == "__main__":
    main()
