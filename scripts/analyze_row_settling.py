#!/usr/bin/env python
"""Which eval rows stop improving as the training set grows, and which never settle.

A split's AUROC curve is an average, and an average that flattens can hide rows still
climbing, rows that peaked and are now sliding back, and rows that were never learned at
all. This reads the curve one row at a time, off the per-ROW probabilities
`fit_instructions_parts.py --row-scores` persists.

THE UNIT. A single sample has no AUROC, so two per-row statistics stand in, and they answer
different questions:

  placement value u_i   for a positive row, the fraction of the split's negative rows it
                        outranks (ties at 0.5); for a negative row, the fraction of
                        positives it sits below. The mean over positives IS the split's
                        AUROC, exactly — so the per-row curves average back into the
                        published split curve with nothing left over. Asserted, not assumed.

  pair solve rate c_i   every one of these splits is fully class-paired: the two rows of a
                        pair share every message but the last. Comparing a pair's two rows
                        holds topic, length and register exactly constant, which u_i does
                        not — u_i is dominated by how a prefix ranks against the whole
                        split, so an easy topic reads as an early settler when it was never
                        hard. c_i is the fraction of draws in which the pair is ordered
                        correctly, and m_i the same comparison in rank units.

Both are computed on RANKS within each fit, never on raw probabilities: these are ~200
separately-fit probes whose output scales are not calibrated to one another, and only a
rank statistic is comparable across n.

THE THRESHOLD. Per-row curves are noisy, so "improves negligibly" has to be read in units
of the row's own across-draw spread, not against a fixed 0.01. Each row gets

    delta_i = max(2 * se_i, FLOOR)      se_i = pooled across-draw SE of u_i, over sizes

and settles at the smallest n from which it never again departs from its n=540 value by
more than delta_i. A row whose delta_i is large is not thereby "settled" — it is
unmeasurable at this draw count, and is reported as such rather than folded into a class.

THE CLASSES. Flatness alone cannot separate "solved before training started" from "never
learned", so the taxonomy crosses the settling point with the final level, and adds the two
shapes a mean curve hides:

    free        settles at 60 AND high at 60      — geometry already separates these
    early/late  rises, then flat by 120 / by 300  — genuine learners
    unsettled   still above delta_i at the last step — still paying for rows
    declining   peaked at some n, then fell by more than delta_i
    never       flat AND low — more rows will not move these
    noisy       delta_i too wide to call

Then the question of whether a class is a COLLECTION or just a bucket: the classes are
projected back onto the bge-base-en-v1.5 prefix embeddings from make_instructions_parts.py
and scored for coherence against a label-permutation null. A class that is coherent can be
named; one that is not says settling is row-idiosyncratic, which is itself the finding.

n=30 is absent by construction — it ran at accumulation 1, a different optimizer regime,
and run_instrparts_fits.sh documents why it must not sit on the same curve.

    .venv_claude/bin/python scripts/analyze_row_settling.py --repro
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

EVAL_DIR = REPO / "eval_sets/instructions"
SCORES_DIR = REPO / "data/instructions_row_scores"
PARTS_DIR = REPO / "data/instructions_parts"
PUBLISHED = REPO / "scripts/instructions_parts_size_curve.csv"
REFIT = REPO / "scripts/instructions_rowscores_size_curve.csv"
OUT_CSV = REPO / "scripts/instructions_row_settling.csv"

POS = "assistant_follows_the_instruction"
SIZES = [60, 120, 300, 540]
FLOOR = 0.03          # no settling call finer than this, however tight the draws look
HIGH = 0.90           # "already high" for the free/never split, in placement-value units
LOW = 0.65

ARMS = {
    "instructions_deepseekv4pro_tgtmin_hc_context_drift_600":
        ("hc_context_drift", "shape-free"),
    "instructions_deepseekv4pro_tgtmin_oig_context_drift_600":
        ("oig_context_drift", "shape-free"),
    "instructions_deepseekv4pro_tgtnone_mm_substitution_600":
        ("mm_substitution", "+ shape info"),
}


# ---------------------------------------------------------------- the split, as rows

def load_split(split: str):
    """Labels, part ids and pair groups for one eval split, in FILE ORDER.

    File order is what joins every artefact here: the activation blob, the persisted
    score vectors and `<split>_parts.jsonl` are all indexed by line number.
    """
    rows = [json.loads(l) for l in (EVAL_DIR / f"{split}.jsonl").open(encoding="utf-8")
            if l.strip()]
    y = np.array([r["labels"] == POS for r in rows])

    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        msgs = json.loads(r["inputs"])
        groups["\n".join(f"{m['role']}: {m['content']}" for m in msgs[:-1])].append(i)

    prefixes, members = [], []
    for text, idx in groups.items():
        pos = [i for i in idx if y[i]]
        neg = [i for i in idx if not y[i]]
        if not pos or not neg:
            raise SystemExit(f"{split}: prefix group of {len(idx)} rows is not class-paired")
        prefixes.append(text)
        members.append((pos, neg))

    part = None
    pf = PARTS_DIR / f"{split}_parts.jsonl"
    if pf.exists():
        pr = [json.loads(l) for l in pf.open(encoding="utf-8") if l.strip()]
        if [r["row"] for r in pr] != list(range(len(rows))):
            raise SystemExit(f"{pf} is not in file order")
        part = np.array([r["part"] for r in pr])
    return y, prefixes, members, part


# ------------------------------------------------------------ the per-row statistics

def placement(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-row placement value, and the AUROC it averages to.

    u_i for a positive row is the fraction of negatives it outranks, ties counting half;
    for a negative row, the fraction of positives above it. mean(u[pos]) == mean(u[neg])
    == AUROC identically, which is the property that makes these curves decompose the
    published one.
    """
    pos, neg = p[y], p[~y]
    u = np.empty(len(p))
    # ~200 rows; the direct comparison is clearer than a rank trick and costs nothing.
    gt = (pos[:, None] > neg[None, :]).astype(float)
    eq = (pos[:, None] == neg[None, :]).astype(float)
    cmp_ = gt + 0.5 * eq
    u[y] = cmp_.mean(axis=1)
    u[~y] = cmp_.mean(axis=0)   # fraction of positives ABOVE this negative — high is good
    return u, float(cmp_.mean())


def pair_stats(p: np.ndarray, members) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair solve indicator and rank-unit margin, averaged over a group's comparisons.

    Most groups are one positive and one negative row. mm_substitution has eight prefixes
    written four times (two pairs of pairs); those average over all pos x neg comparisons
    inside the group, so a group contributes one number whatever its size.
    """
    r = np.argsort(np.argsort(p)) / (len(p) - 1)
    solved = np.empty(len(members))
    margin = np.empty(len(members))
    for g, (pos, neg) in enumerate(members):
        d = r[np.array(pos)][:, None] - r[np.array(neg)][None, :]
        solved[g] = float((d > 0).mean() + 0.5 * (d == 0).mean())
        margin[g] = float(d.mean())
    return solved, margin


# ------------------------------------------------------------------- settling

def settle(curve: np.ndarray, delta: float) -> int:
    """Smallest size from which the curve never again departs from its last value by > delta."""
    final = curve[-1]
    for k, n in enumerate(SIZES):
        if np.all(np.abs(curve[k:] - final) <= delta):
            return n
    return SIZES[-1]


def classify(curve: np.ndarray, delta: float, se: float) -> str:
    if delta > 0.25:
        return "noisy"
    peak = float(curve.max())
    if peak - curve[-1] > delta:
        return "declining"
    if curve[-1] - curve[-2] > delta:
        return "unsettled"
    n = settle(curve, delta)
    if n == 60:
        if curve[0] >= HIGH:
            return "free"
        if curve[-1] <= LOW:
            return "never"
        return "flat-mid"
    return "early" if n <= 120 else "late"


# ------------------------------------------------------------------- loading scores

def load_scores(stem: str, split: str):
    """[size, draw, row] probabilities for one arm's target split."""
    per_size = []
    for n in SIZES:
        files = sorted((SCORES_DIR / stem).glob(f"gad_n{n}_d*.npz"),
                       key=lambda f: int(f.stem.split("_d")[-1]))
        if not files:
            raise SystemExit(f"no score files for {stem} n={n} — run run_instrparts_rowscores.sh")
        per_size.append(np.stack([np.load(f)[split] for f in files]))
    draws = min(len(a) for a in per_size)
    if len({len(a) for a in per_size}) > 1:
        print(f"  note: uneven draw counts {[len(a) for a in per_size]}; using {draws}")
    return np.stack([a[:draws] for a in per_size]), draws


# ------------------------------------------------------------------- coherence

def coherence(X: np.ndarray, labels: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """Mean within-class cosine similarity minus the label-permutation null, as a z score.

    Says whether a settling class is a describable COLLECTION of conversations or a bucket
    of unrelated rows that happen to share a curve shape. X is row-normalised, so a dot
    product is a cosine.
    """
    def stat(lab):
        tot, cnt = 0.0, 0
        for c in np.unique(lab):
            m = lab == c
            if m.sum() < 2:
                continue
            S = X[m] @ X[m].T
            k = m.sum()
            tot += (S.sum() - k) / (k * (k - 1))
            cnt += 1
        return tot / max(cnt, 1)

    obs = stat(labels)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(labels)) for _ in range(500)])
    z = (obs - null.mean()) / (null.std() + 1e-12)
    return obs, float(z)


# ------------------------------------------------------------------- reproduction

def repro_check() -> None:
    """Draws 0-7 of the refit are the same row subsets as the published curve.

    They are different PROCESSES though, and a fit is not bit-reproducible across processes
    (the probe returned is the last epoch and tuberlens early-stops on validation AUROC),
    so this prints the drift rather than asserting equality. It is the floor under every
    settling claim below: no per-row movement smaller than this is real.
    """
    def read(path):
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    if not PUBLISHED.exists() or not REFIT.exists():
        print("  (missing a CSV; skipped)")
        return
    pub = {(r["samples"], int(r["n"]), int(r["draw"])): r for r in read(PUBLISHED)}
    print(f"  {'arm':<28} {'n':>4}  {'published':>9} {'refit':>8} {'delta':>7}   worst draw")
    for stem, (split, _) in ARMS.items():
        rows = [r for r in read(REFIT) if r["samples"].startswith(stem[:40])
                or Path(r["samples"]).stem == stem]
        by_n = collections.defaultdict(list)
        for r in rows:
            by_n[int(r["n"])].append(r)
        for n in SIZES:
            pairs = []
            for r in by_n.get(n, []):
                key = (r["samples"], n, int(r["draw"]))
                if key in pub:
                    pairs.append((float(pub[key][f"eval_{split}"]), float(r[f"eval_{split}"])))
            if not pairs:
                continue
            a = np.array([x for x, _ in pairs])
            b = np.array([y for _, y in pairs])
            w = int(np.argmax(np.abs(a - b)))
            print(f"  {split + ' n':<28} {n:>4}  {a.mean():>9.4f} {b.mean():>8.4f} "
                  f"{b.mean() - a.mean():>+7.4f}   {abs(a[w] - b[w]):.4f} ({len(pairs)} draws)")


# ------------------------------------------------------------------- report

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repro", action="store_true",
                    help="also print the refit-vs-published drift on the shared draws")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip the bge coherence test (no GPU / no download)")
    ap.add_argument("--only", nargs="+", default=None, metavar="SPLIT",
                    help="restrict to these target splits (useful while a run is still going)")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    arms = {k: v for k, v in ARMS.items() if not args.only or v[0] in args.only}

    if args.repro:
        print("\n## Refit vs published, on the 8 shared draws\n")
        repro_check()

    fields = ["arm", "split", "row", "pair", "part", "label", "klass", "settles_at",
              "delta", "se"] + [f"u{n}" for n in SIZES] + [f"sd{n}" for n in SIZES] \
        + [f"c{n}" for n in SIZES]
    out_rows = []

    for stem, (split, arm_label) in arms.items():
        print(f"\n\n## {split} — {arm_label}\n")
        y, prefixes, members, part = load_split(split)
        P, draws = load_scores(stem, split)          # [size, draw, row]

        U = np.empty(P.shape[:2] + (len(y),))
        C = np.empty(P.shape[:2] + (len(members),))
        A = np.empty(P.shape[:2])
        for i in range(P.shape[0]):
            for d in range(P.shape[1]):
                U[i, d], A[i, d] = placement(P[i, d], y)
                C[i, d], _ = pair_stats(P[i, d], members)

        # The decomposition, asserted rather than assumed: the mean of the per-row
        # placement values IS the AUROC every published number on this split came from.
        assert np.allclose(U[:, :, y].mean(axis=2), A, atol=1e-9), "placement != AUROC (pos)"
        assert np.allclose(U[:, :, ~y].mean(axis=2), A, atol=1e-9), "placement != AUROC (neg)"

        u, sd = U.mean(axis=1), U.std(axis=1, ddof=1)
        c = C.mean(axis=1)
        se = sd.mean(axis=0) / np.sqrt(draws)
        delta = np.maximum(2 * se, FLOOR)

        print(f"  {draws} draws, {len(y)} rows, {len(members)} pairs.  "
              f"split AUROC {' '.join(f'n={n}:{A[k].mean():.3f}' for k, n in enumerate(SIZES))}")
        print(f"  per-row across-draw SE: median {np.median(se):.3f}, "
              f"90th pct {np.quantile(se, 0.9):.3f}  ->  median delta {np.median(delta):.3f}")

        klass = np.array([classify(u[:, i], delta[i], se[i]) for i in range(len(y))])
        settles = np.array([settle(u[:, i], delta[i]) for i in range(len(y))])

        print(f"\n  settling point (smallest n from which the row never moves by > delta):")
        for n in SIZES:
            m = settles == n
            tag = "  <- still moving at the last step" if n == SIZES[-1] and m.any() else ""
            print(f"    n={n:<4} {m.sum():>4} rows {100 * m.mean():>4.0f}%   "
                  f"final level {u[-1, m].mean() if m.any() else float('nan'):.3f}{tag}")

        order = ["free", "early", "late", "unsettled", "declining", "never", "flat-mid", "noisy"]
        print(f"\n  {'class':<11} {'rows':>5} {'%':>5}  "
              + "  ".join(f"u({n})" for n in SIZES) + "   gain 60->540")
        for k in order:
            m = klass == k
            if not m.any():
                continue
            cu = u[:, m].mean(axis=1)
            print(f"  {k:<11} {m.sum():>5} {100 * m.mean():>4.0f}%  "
                  + "  ".join(f"{v:.3f}" for v in cu) + f"   {cu[-1] - cu[0]:+.3f}")

        # What the classes cost the split: how much of the n=60 -> n=540 gain each carries.
        gain = u[-1] - u[0]
        tot = gain.sum()
        print(f"\n  share of the split's total 60->540 gain ({tot / len(y):+.4f} AUROC):")
        for k in order:
            m = klass == k
            if m.any() and abs(tot) > 1e-9:
                print(f"    {k:<11} {100 * gain[m].sum() / tot:>5.0f}%")

        if part is not None:
            print("\n  class x k-means part (the published panels' cut):")
            names = sorted(set(part))
            print("    " + " " * 11 + "".join(f"{p.split('_')[-1]:>6}" for p in names))
            for k in order:
                m = klass == k
                if m.any():
                    print(f"    {k:<11}" + "".join(f"{int((part[m] == p).sum()):>6}" for p in names))

        pair_klass = np.array([collections.Counter(klass[pos + neg]).most_common(1)[0][0]
                               for pos, neg in members])

        # The prefix-controlled view: within a pair, topic and length are held exactly
        # constant, so this is the same question with the between-prefix variance removed.
        # Where c saturates but u does not, what is still being learned above that size is
        # how to rank prefixes against each other, not how to tell the two endings apart.
        print(f"\n  pair solve rate (the prefix's two endings ordered correctly), by class:")
        print(f"  {'class':<11} {'pairs':>5}  " + "  ".join(f"c({n})" for n in SIZES))
        for k in order:
            m = pair_klass == k
            if m.any():
                print(f"  {k:<11} {m.sum():>5}  "
                      + "  ".join(f"{v:.3f}" for v in c[:, m].mean(axis=1)))

        c_se = C.std(axis=1, ddof=1).mean(axis=0) / np.sqrt(draws)
        c_delta = np.maximum(2 * c_se, FLOOR)
        c_klass = np.array([classify(c[:, g], c_delta[g], c_se[g]) for g in range(len(members))])
        print(f"\n  and the SAME taxonomy on the pair statistic ({len(members)} pairs):")
        for k in order:
            m = c_klass == k
            if m.any():
                cc = c[:, m].mean(axis=1)
                print(f"    {k:<11} {m.sum():>4} {100 * m.mean():>4.0f}%  "
                      + "  ".join(f"{v:.3f}" for v in cc))
        stuck = np.where(c[-1] < 0.75)[0]
        print(f"    pairs the probe still gets wrong at n=540 in >1/4 of draws: {len(stuck)}"
              + (f" (pairs {', '.join(map(str, stuck[:12]))})" if len(stuck) else ""))

        if not args.no_embed:
            from make_instructions_parts import embed, top_terms
            X = embed(prefixes)
            obs, z = coherence(X, pair_klass)
            print(f"\n  coherence of the settling classes in prefix-embedding space: "
                  f"mean within-class cosine {obs:.4f}, z vs permutation null {z:+.2f}")
            print("    " + ("classes separate in content space — they are describable "
                            "collections, named below" if z > 2 else
                            "classes do NOT separate in content space: settling is "
                            "row-idiosyncratic, not topical, and the names below are decoration"))
            print("\n  what each class is about (log-odds terms vs the rest of the split):")
            for k in order:
                m = pair_klass == k
                if m.sum() >= 3:
                    print(f"    {k:<11} {', '.join(top_terms(prefixes, pair_klass, k, 10))}")

        # The rows that end lowest, whatever their class — the split's hard core.
        worst = np.argsort(u[-1])[:5]
        print("\n  lowest final placement values (the rows 540 rows still do not buy):")
        for i in worst:
            g = next(gi for gi, (p_, n_) in enumerate(members) if i in p_ + n_)
            snippet = " ".join(prefixes[g].split())[:96]
            print(f"    row {i:>3} {klass[i]:<10} u: "
                  + " ".join(f"{u[k, i]:.2f}" for k in range(len(SIZES)))
                  + f"  | {snippet}")

        pair_of = {}
        for g, (pos, neg) in enumerate(members):
            for i in pos + neg:
                pair_of[i] = g
        for i in range(len(y)):
            out_rows.append({
                "arm": arm_label, "split": split, "row": i, "pair": pair_of[i],
                "part": part[i] if part is not None else "", "label": int(y[i]),
                "klass": klass[i], "settles_at": settles[i],
                "delta": round(float(delta[i]), 4), "se": round(float(se[i]), 4),
                **{f"u{n}": round(float(u[k, i]), 4) for k, n in enumerate(SIZES)},
                **{f"sd{n}": round(float(sd[k, i]), 4) for k, n in enumerate(SIZES)},
                **{f"c{n}": round(float(c[k, pair_of[i]]), 4) for k, n in enumerate(SIZES)},
            })

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n\nwrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
