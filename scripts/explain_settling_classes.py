#!/usr/bin/env python
"""Is there anything about a row, other than its curve, that says when it will settle?

`analyze_row_settling.py --validate` shows the settled/unsettled split is real on
`oig_context_drift` and `mm_substitution` — a row called settled moves 4-20x less on draws
that had no say in labelling it. This asks the next question: given only the conversation,
could you have predicted which side it lands on?

Curve-derived features are excluded on purpose. `u(60)` would win every test here and mean
nothing, because the settling label is computed from a curve that contains it.

THE LEVEL OF ANALYSIS IS THE POINT. These splits are class-paired, so a row is not an
independent observation: the two rows of a pair share a prefix, a topic and a length. Two
kinds of question have to be asked separately, with different nulls.

  between pairs   Does this CONVERSATION settle late? Target is the pair, features are the
                  shared prefix and the pair's average. The null permutes pairs. Anything
                  pair-level tested per row instead would double every observation and halve
                  every p-value — and in the embedding case would put two identical feature
                  vectors carrying opposite labels in different CV folds, which is enough on
                  its own to drive an out-of-fold AUROC below chance.

  within pairs    Given a pair that splits — one row settles, the other does not — is it the
                  LONGER ending that fails to settle? Only features that vary inside a pair
                  can be asked this, and the null is a coin flip per pair, which removes
                  every between-pair confound by construction. This is the sharper test, and
                  the only one that can attribute anything to the ending itself.

Both report against label-permutation nulls, and every family of tests gets a
Benjamini-Hochberg q alongside its p, because roughly nine features per arm will hand you a
p < 0.05 whether or not anything is there.

    .venv_claude/bin/python scripts/explain_settling_classes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_row_settling import EVAL_DIR, REFUSAL, load_split, placement  # noqa: E402

SETTLING = REPO / "scripts/instructions_row_settling.csv"
# The two arms whose settling labels survived the out-of-sample check.
ARMS = {"oig_context_drift": "shape-free", "mm_substitution": "+ shape info"}
SETTLED = {"free", "early", "late"}
UNSETTLED = {"unsettled"}
PERM = 2000
SEED = 0


def auroc(x: np.ndarray, y: np.ndarray) -> float:
    """AUROC of a single feature for separating y == 1 from y == 0, ties at 0.5."""
    a, b = x[y == 1], x[y == 0]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).astype(float) + 0.5 * (a[:, None] == b[None, :])
    return float(gt.mean())


def perm_p(x: np.ndarray, y: np.ndarray, obs: float, rng) -> float:
    null = np.array([auroc(x, rng.permutation(y)) for _ in range(PERM)])
    return float((np.abs(null - 0.5) >= abs(obs - 0.5)).mean())


def bh(ps: list[float]) -> list[float]:
    """Benjamini-Hochberg q values, in the order the p values were given."""
    p = np.asarray(ps, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    running = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        running = min(running, p[i] * len(p) / rank)
        q[i] = running
    return list(q)


def cv_auroc(X: np.ndarray, y: np.ndarray, seed: int = SEED, folds: int = 5) -> float:
    """Out-of-fold AUROC of a logistic regression — never an in-sample fit.

    On ~100 observations and 768 dimensions an in-sample model separates anything; only the
    held-out score says whether the content carries the label.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return auroc(oof, y)


def base_probe_placement(split: str, y: np.ndarray) -> np.ndarray:
    """Placement values of the campaign's BASE probe — where a row sat before any of this."""
    import pickle

    from fit_base_plus_concept import CONCEPTS
    from fit_instructions_parts import load_eval

    datasets, _ = load_eval()
    concept = CONCEPTS["instructions"]
    with concept.base_probe.open("rb") as fh:
        probe = pickle.load(fh)
    u, _ = placement(np.asarray(probe.predict_proba(datasets[split])), y)
    return u


def report(title: str, names: list[str], stats: list[float], ps: list[float],
           extra: list[str] | None = None) -> None:
    qs = bh(ps)
    print(f"\n  {title:<34} {'AUROC':>7} {'p':>7} {'BH q':>7}")
    for i, n in enumerate(names):
        mark = "  <-" if qs[i] < 0.05 else ("  (p only)" if ps[i] < 0.05 else "")
        s = f"{stats[i]:>7.3f}" if not np.isnan(stats[i]) else f"{'':>7}"
        tail = f"   {extra[i]}" if extra else ""
        print(f"  {n:<34} {s} {ps[i]:>7.3f} {qs[i]:>7.3f}{mark}{tail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--no-base", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(SETTLING.open(encoding="utf-8")))
    rng = np.random.default_rng(SEED)

    for split, arm in ARMS.items():
        print(f"\n\n## {split} — {arm}\n")
        labels, prefixes, members, part = load_split(split)
        raw = [json.loads(l) for l in (EVAL_DIR / f"{split}.jsonl").open(encoding="utf-8")
               if l.strip()]
        end = [" ".join(json.loads(r["inputs"])[-1]["content"].split()) for r in raw]
        msgs = [json.loads(r["inputs"]) for r in raw]

        klass = {int(r["row"]): r["klass"] for r in rows if r["split"] == split}
        # 1 = still moving at the last size, 0 = settled, -1 = neither class (dropped)
        state = np.array([1 if klass.get(i) in UNSETTLED else
                          0 if klass.get(i) in SETTLED else -1 for i in range(len(raw))])
        u0 = None if args.no_base else base_probe_placement(split, labels)

        # ---------------------------------------------------------- between pairs
        pmask, ptarget = [], []
        for pos, neg in members:
            st = state[pos + neg]
            if (st < 0).any():
                pmask.append(False)
                ptarget.append(0)
                continue
            pmask.append(True)
            ptarget.append(int(st.max() == 1))     # any row of the pair still moving
        pmask = np.array(pmask)
        yp = np.array(ptarget)[pmask]
        gi = np.where(pmask)[0]
        print(f"  between pairs: {len(yp)} usable pairs — {(yp == 0).sum()} settle wholly, "
              f"{(yp == 1).sum()} carry a still-moving row")

        feats = {
            "prefix characters": np.array(
                [sum(len(m["content"]) for m in msgs[members[g][0][0]][:-1]) for g in gi], float),
            "number of turns": np.array([len(msgs[members[g][0][0]]) for g in gi], float),
            "mean ending length": np.array(
                [np.mean([len(end[j]) for j in members[g][0] + members[g][1]]) for g in gi], float),
            "|length gap| inside the pair": np.array(
                [abs(np.mean([len(end[j]) for j in members[g][0]])
                     - np.mean([len(end[j]) for j in members[g][1]])) for g in gi], float),
            "non-compliant ending is a refusal": np.array(
                [any(any(k in end[j].lower()[:60] for k in REFUSAL) for j in members[g][1])
                 for g in gi], float),
        }
        if u0 is not None:
            feats["base probe placement (pair mean)"] = np.array(
                [u0[members[g][0] + members[g][1]].mean() for g in gi])
        names, stats, ps = [], [], []
        for n, x in feats.items():
            a = auroc(x, yp)
            names.append(n)
            stats.append(a)
            ps.append(perm_p(x, yp, a, rng))
        if part is not None:
            from scipy.stats import chi2_contingency
            pp = np.array([part[members[g][0][0]] for g in gi])
            tab = np.array([[int(((pp == v) & (yp == k)).sum()) for v in sorted(set(part))]
                            for k in (0, 1)])
            names.append("k-means part (chi-square)")
            stats.append(float("nan"))
            ps.append(float(chi2_contingency(tab).pvalue))
        report("feature (pair level)", names, stats, ps)

        if not args.no_embed:
            from make_instructions_parts import embed
            Xp = embed([prefixes[g] for g in gi])
            a = cv_auroc(Xp, yp)
            null = np.array([cv_auroc(Xp, rng.permutation(yp), seed=s) for s in range(20)])
            p = float((np.abs(null - 0.5) >= abs(a - 0.5)).mean())
            print(f"\n  {'prefix embedding, 5-fold CV':<34} {a:>7.3f} {p:>7.3f}"
                  f"          (null {null.mean():.3f} +- {null.std():.3f})")

        # ---------------------------------------------------------- within pairs
        # Only pairs that SPLIT can answer this, and the null is a coin flip per pair, so
        # every between-pair confound — topic, prefix length, difficulty — is gone.
        split_pairs = [(pos + neg) for pos, neg in members
                       if len(pos) == 1 and len(neg) == 1
                       and set(state[pos + neg].tolist()) == {0, 1}]
        print(f"\n  within pairs: {len(split_pairs)} pairs split one settled / one not")
        if len(split_pairs) >= 8:
            names, stats, ps = [], [], []
            rowfeats = {
                "the unsettled row is the longer": lambda i: float(len(end[i])),
                "the unsettled row is the 'follows' one": lambda i: float(labels[i]),
            }
            if u0 is not None:
                rowfeats["the unsettled row had a lower prior"] = lambda i: -float(u0[i])
            for n, f in rowfeats.items():
                wins = np.array([float(f(a_) > f(b_)) + 0.5 * float(f(a_) == f(b_))
                                 for a_, b_ in ((p_[0], p_[1]) if state[p_[0]] == 1
                                                else (p_[1], p_[0]) for p_ in split_pairs)])
                obs = wins.mean()
                null = rng.random((PERM, len(wins))) < 0.5
                nd = np.where(null, wins, 1 - wins).mean(axis=1)
                names.append(n)
                stats.append(obs)
                ps.append(float((np.abs(nd - 0.5) >= abs(obs - 0.5)).mean()))
            report("within-pair sign test", names, stats, ps,
                   extra=[f"({len(split_pairs)} pairs)"] * len(names))
            # What a null result here is worth: the smallest effect this many pairs could
            # have caught. Two-sided alpha 0.05, 80% power, SE = 0.5/sqrt(n).
            mde = 0.5 + 2.80 * 0.5 / np.sqrt(len(split_pairs))
            print(f"    a null above is a null only down to AUROC {mde:.2f} — "
                  f"{len(split_pairs)} pairs cannot see a smaller effect than that")


if __name__ == "__main__":
    main()
