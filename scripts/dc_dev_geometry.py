#!/usr/bin/env python
"""The difference-of-means transfer statistic on DEV samples.

Brief: ``docs/dev_coverage_dom_task.md``. ``direction_count.geometry_for_set`` computes,
on a generated set, one unit difference-of-means direction per LLM-tagged kind in the
set's own standardised mean-pooled feature space, and scores it as a linear projection on
the other kinds' samples (``T_ij``) and on the evaluation splits (``E_is``). No training.
``dc_dev.py`` answered the same transfer question on the dev samples with a *trained
probe*. This script computes the generated-set statistic, unchanged, with the dev split
in the role of the kind, so that the real-data row can be reported with the same
classifier as the generated one.

Stages:

    pool      mean-pool every dev sample of the concept from the per-sample activation
              cache into scripts/dc_pooled/dcdev_<concept>_<split>_own_{mean,labels,
              ntokens}.npy (``direction_count.pool_set``, reading the ``own`` arm files,
              which carry the assistant-first fix the cache key was computed over)
    geometry  scripts/dc_dev_geometry.csv, dc_dev_neff.csv,
              dc_dev_dom_transfer_<concept>_{dev,eval}.csv, dc_dev_dom_cells.csv

**Standardiser.** Fit once per concept on the stacked dev splits, and the eval splits are
pushed through it — never standardised on their own statistics — exactly as the
generated study pushes them through the generated set's. The size-matched re-run keeps
that full-dev standardiser (the brief re-runs steps 2-4, not step 1), so only the
directions change between the full-size and the size-matched numbers.

**In-sample entries.** The diagonal of ``T`` is five-fold held out, as in the generated
study. ``d_all`` scored on a split's dev samples is fit on those samples too, so the
``all`` row of the dev transfer matrix and the denominator of ``G_dom_dev`` are
in-sample; ``G_dom_dev_ho`` replaces that denominator with a five-fold held-out one
(``d_all`` refit without each fold of s) to show how much that matters. ``d_-s`` never
sees s, so its entries are out-of-sample as they stand.

``oig_omission`` is excluded everywhere (``dc_dev.dev_files`` drops it). Nothing in
``direction_count.py`` or any existing CSV is edited; the helpers are imported.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
for p in (SCRIPTS, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import dc_dev  # noqa: E402
from direction_count import (  # noqa: E402
    GEOM_FIELDS,
    MIN_KIND_PER_CLASS,
    N_GEOM_FOLDS,
    POOLED,
    SEED,
    _auroc,
    _direction,
    _load_eval_features,
    _standardiser,
    pool_set,
)

N_MATCH = 100          # the size-matched arm: each split class-balanced to min(bal, 100)
N_MATCH_DRAWS = 20


def dev_stem(name: str, split: str) -> str:
    return f"dcdev_{name}_{split}_own"


def dev_splits(name: str) -> list[str]:
    """The concept's dev splits, named by the eval split each is the dev set of."""
    return [dc_dev.eval_stem(p.stem) for p in dc_dev.dev_files(name)]


# --------------------------------------------------------------------------- #
# stage: pool
# --------------------------------------------------------------------------- #
def stage_pool(args) -> None:
    concepts = dc_dev._concepts()
    for name in args.concepts:
        c = concepts[name]
        missing = [s for s in dev_splits(name)
                   if not dc_dev.arm_file(name, s, "own").exists()]
        if missing:
            # build_arms, not stage_arms: the latter rewrites dc_dev_arms.csv, which this
            # task must not touch. The files it writes are the same bytes either way.
            print(f"[pool] {name}: own arm files missing for {missing}, rebuilding",
                  flush=True)
            dc_dev.build_arms(name)
        for s in dev_splits(name):
            # pool_set does not read its spec argument; the output name is the file stem.
            pool_set(dc_dev.arm_file(name, s, "own"), c, None)


# --------------------------------------------------------------------------- #
# the statistic
# --------------------------------------------------------------------------- #
def load_dev(name: str):
    """Stacked dev features, labels and split name per row, in dev_splits order."""
    Xs, ys, gs = [], [], []
    for s in dev_splits(name):
        stem = dev_stem(name, s)
        f = POOLED / f"{stem}_mean.npy"
        if not f.exists():
            raise SystemExit(f"{f} missing — run --stage pool first")
        X, y = np.load(f), np.load(POOLED / f"{stem}_labels.npy")
        Xs.append(X)
        ys.append(y.astype(bool))
        gs.append(np.array([s] * len(y), dtype=object))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def _held_out(Z, y, sel: np.ndarray, rng, fit_rows: np.ndarray | None = None) -> float:
    """``direction_count``'s held-out diagonal: five folds of ``sel``, same RNG use.

    With ``fit_rows`` None the direction is fit on the other folds of ``sel`` (the T
    diagonal). Otherwise it is fit on ``fit_rows`` minus the held-out fold — used for
    ``d_all`` scored on one split without that split's scored rows.
    """
    idx = np.where(sel)[0]
    order = rng.permutation(len(idx))
    folds = np.array_split(order, N_GEOM_FOLDS)
    aurocs = []
    for f in folds:
        te = idx[f]
        if fit_rows is None:
            tr = idx[np.setdiff1d(order, f, assume_unique=False)]
        else:
            keep = fit_rows.copy()
            keep[te] = False
            tr = np.where(keep)[0]
        d = _direction(Z[tr], y[tr])
        if d is None or y[te].sum() == 0 or (~y[te]).sum() == 0:
            continue
        aurocs.append(_auroc(y[te], Z[te] @ d))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def geometry(Z, y, groups, kinds: list[str], evals: dict, rng, use=None,
             with_minus: bool = True):
    """The generated study's per-kind geometry on standardised features ``Z``.

    ``groups[r]`` is row r's kind (here: its dev split), ``kinds`` their order, ``evals``
    ``{eval split: (Ze, ye)}`` already pushed through the same standardiser, and ``use``
    an optional row mask (the size-matched draws). Returns the long rows, the directions
    and the split-level matrices the summaries are read off.
    """
    if use is None:
        use = np.ones(len(y), bool)
    sel = {k: (groups == k) & use for k in kinds}
    npos = {k: int((sel[k] & y).sum()) for k in kinds}
    nneg = {k: int((sel[k] & ~y).sum()) for k in kinds}
    dirs = {k: _direction(Z[sel[k]], y[sel[k]]) for k in kinds
            if npos[k] >= MIN_KIND_PER_CLASS and nneg[k] >= MIN_KIND_PER_CLASS}
    d_all = _direction(Z[use], y[use])

    T, cos = {}, {}
    for ki in kinds:                       # ki outer, kj inner: the RNG order of the
        di = dirs.get(ki)                  # generated study's diagonal calls
        for kj in kinds:
            dj = dirs.get(kj)
            cos[ki, kj] = None if di is None or dj is None else float(di @ dj)
            if di is None or sel[kj].sum() == 0:
                T[ki, kj] = None
            elif ki == kj:
                T[ki, kj] = _held_out(Z, y, sel[ki], rng)
            else:
                T[ki, kj] = _auroc(y[sel[kj]], Z[sel[kj]] @ di)
    E = {}
    for ki in kinds:
        for s, (Ze, ye) in evals.items():
            di = dirs.get(ki)
            E[ki, s] = None if di is None else _auroc(ye, Ze @ di)
    for s, (Ze, ye) in evals.items():
        E["all", s] = None if d_all is None else _auroc(ye, Ze @ d_all)

    # Step 5: set-minus-split directions, and d_all on each split's own dev rows.
    minus = {}
    for k in kinds:
        if not with_minus:
            break
        rest = use & (groups != k)
        dm = _direction(Z[rest], y[rest])
        rec = {"n_pos": int((rest & y).sum()), "n_neg": int((rest & ~y).sum())}
        rec["t"] = None if dm is None else _auroc(y[sel[k]], Z[sel[k]] @ dm)
        rec["e"] = None if dm is None or k not in evals else _auroc(evals[k][1],
                                                                    evals[k][0] @ dm)
        rec["t_all"] = None if d_all is None else _auroc(y[sel[k]], Z[sel[k]] @ d_all)
        rec["t_all_ho"] = _held_out(Z, y, sel[k], rng, fit_rows=use)
        rec["e_all"] = E.get(("all", k))
        minus[k] = rec

    # n_eff: participation ratio of the Gram matrix of the unit split directions.
    present = [k for k in kinds if k in dirs]
    if len(present) >= 2:
        D = np.stack([dirs[k] for k in present])
        G = D @ D.T
        lam = np.clip(np.linalg.eigvalsh(G), 0.0, None)
        n_eff = float(lam.sum() ** 2 / (lam ** 2).sum())
        off = G[~np.eye(len(present), dtype=bool)]
        mean_off, max_off = float(off.mean()), float(off.max())
    else:
        n_eff = mean_off = max_off = float("nan")
    return {
        "kinds": kinds, "present": present, "dirs": dirs, "d_all": d_all,
        "npos": npos, "nneg": nneg, "T": T, "cos": cos, "E": E, "minus": minus,
        "n_eff": n_eff, "mean_off_cos": mean_off, "max_off_cos": max_off,
        "n_all_pos": int((use & y).sum()), "n_all_neg": int((use & ~y).sum()),
    }


def _m(v):
    v = [x for x in v if x is not None and x == x]
    return float(np.mean(v)) if v else float("nan")


def split_stats(g: dict) -> dict[str, dict]:
    """Per split s: the numbers ``dc_dev_cells.csv`` has for the probe, for d_s."""
    kinds, T, E, minus = g["kinds"], g["T"], g["E"], g["minus"]
    out = {}
    for s in kinds:
        t_own, e_own = T[s, s], E.get((s, s))
        t_oth = _m([T[o, s] for o in kinds if o != s])
        e_oth = _m([E.get((o, s)) for o in kinds if o != s])
        rec = {"t_own_dom": t_own, "t_other_dom": t_oth,
               "e_own_dom": e_own, "e_other_dom": e_oth,
               "t_gap_dom": None if t_own is None else t_own - t_oth,
               "e_gap_dom": None if e_own is None else e_own - e_oth}
        if s in minus:
            m = minus[s]

            def ratio(a, b):
                return None if a is None or b is None or b == 0.5 else (a - .5) / (b - .5)

            rec.update({
                "auc_minus_dev": m["t"], "auc_all_dev": m["t_all"],
                "auc_all_dev_ho": m["t_all_ho"],
                "auc_minus_eval": m["e"], "auc_all_eval": m["e_all"],
                "G_dom_dev": ratio(m["t"], m["t_all"]),
                "G_dom_dev_ho": ratio(m["t"], m["t_all_ho"]),
                "G_dom_eval": ratio(m["e"], m["e_all"]),
            })
        out[s] = rec
    return out


def concept_summary(g: dict) -> dict:
    """Set-level means, as ``dc_neff.csv`` defines them."""
    kinds, T, E = g["kinds"], g["T"], g["E"]
    return {
        "t_own": _m([T[k, k] for k in kinds]),
        "t_other": _m([T[a, b] for a in kinds for b in kinds if a != b]),
        "e_own": _m([E.get((k, k)) for k in kinds]),
        "e_other": _m([E.get((a, b)) for a in kinds for b in kinds if a != b]),
        "e_all": _m([E.get(("all", k)) for k in kinds]),
        "n_eff": g["n_eff"], "mean_off_cos": g["mean_off_cos"],
        "max_off_cos": g["max_off_cos"],
    }


def matched_mask(y, groups, kinds, rng) -> np.ndarray:
    """Every split class-balanced to min(its balanced size, N_MATCH)."""
    use = np.zeros(len(y), bool)
    for k in kinds:
        pos = np.where((groups == k) & y)[0]
        neg = np.where((groups == k) & ~y)[0]
        n = min(len(pos), len(neg), N_MATCH // 2)
        use[rng.choice(pos, n, replace=False)] = True
        use[rng.choice(neg, n, replace=False)] = True
    return use


# --------------------------------------------------------------------------- #
# stage: geometry
# --------------------------------------------------------------------------- #
def _r(x, nd=5):
    return "" if x is None or x != x else round(float(x), nd)


def _kind_index(name: str) -> dict[str, int]:
    """Split -> the kind number direction_count gives it (for the cos_all_k<i> column)."""
    from direction_count import _concept_specs

    return {k.split: k.index for k in _concept_specs()[name].kinds}


def long_rows(name: str, g: dict) -> list[dict]:
    """``dc_geometry.csv``'s layout, plus ``minus_<split>`` rows (step 5)."""
    base = {"concept": name, "gen": "dev", "set": f"dcdev_{name}"}
    kinds, T, E, cos = g["kinds"], g["T"], g["E"], g["cos"]
    rows = []
    for ki in kinds:
        for kj in kinds:
            rows.append({**base, "kind_i": ki, "kind_j": kj, "split_i": ki, "split_j": kj,
                         "n_i_pos": g["npos"][ki], "n_i_neg": g["nneg"][ki],
                         "cos_ij": _r(cos[ki, kj]), "T_ij": _r(T[ki, kj]), "E_is": ""})
    for ki in kinds:
        for s in kinds:
            rows.append({**base, "kind_i": ki, "kind_j": "eval", "split_i": ki,
                         "split_j": s, "n_i_pos": g["npos"][ki], "n_i_neg": g["nneg"][ki],
                         "cos_ij": "", "T_ij": "", "E_is": _r(E.get((ki, s)))})
    for s in kinds:
        rows.append({**base, "kind_i": "all", "kind_j": "eval", "split_i": "all",
                     "split_j": s, "n_i_pos": g["n_all_pos"], "n_i_neg": g["n_all_neg"],
                     "cos_ij": "", "T_ij": "", "E_is": _r(E.get(("all", s)))})
    for s, m in g["minus"].items():
        common = {**base, "kind_i": f"minus_{s}", "split_i": f"minus_{s}",
                  "n_i_pos": m["n_pos"], "n_i_neg": m["n_neg"], "cos_ij": ""}
        rows.append({**common, "kind_j": s, "split_j": s, "T_ij": _r(m["t"]), "E_is": ""})
        rows.append({**common, "kind_j": "eval", "split_j": s, "T_ij": "",
                     "E_is": _r(m["e"])})
    return rows


def transfer_tables(name: str, g: dict) -> tuple[list[dict], list[dict]]:
    """Rows: direction; columns: scored split. ``dc_dev_transfer_<concept>.csv``'s format.

    The ``all`` row on dev is in-sample (d_all saw those rows); a ``minus_<s>`` row is
    filled only in column s, the one split it never saw.
    """
    kinds, T, E = g["kinds"], g["T"], g["E"]
    dev, ev = [], []
    for k in kinds:
        dev.append({"train_split": k, "n": g["npos"][k] + g["nneg"][k],
                    **{t: _r(T[k, t]) for t in kinds}})
        ev.append({"train_split": k, "n": g["npos"][k] + g["nneg"][k],
                   **{t: _r(E.get((k, t))) for t in kinds}})
    n_all = g["n_all_pos"] + g["n_all_neg"]
    dev.append({"train_split": "all", "n": n_all,
                **{t: _r(g["minus"][t]["t_all"]) if t in g["minus"] else "" for t in kinds}})
    ev.append({"train_split": "all", "n": n_all,
               **{t: _r(E.get(("all", t))) for t in kinds}})
    for s, m in g["minus"].items():
        n = m["n_pos"] + m["n_neg"]
        dev.append({"train_split": f"minus_{s}", "n": n,
                    **{t: (_r(m["t"]) if t == s else "") for t in kinds}})
        ev.append({"train_split": f"minus_{s}", "n": n,
                   **{t: (_r(m["e"]) if t == s else "") for t in kinds}})
    return dev, ev


NEFF_BASE = ["concept", "gen", "set", "n_kinds", "n_kinds_used", "t_own", "t_other",
             "e_own", "e_other", "e_all", "n_eff", "mean_off_cos", "max_off_cos",
             "n_tagged", "n_rows"]
MATCHED_STATS = ("t_own", "t_other", "e_own", "e_other", "e_all", "n_eff",
                 "mean_off_cos", "max_off_cos")
CELL_STATS = ("t_other_dom", "t_own_dom", "e_own_dom", "e_other_dom", "G_dom_dev",
              "G_dom_dev_ho", "G_dom_eval", "t_gap_dom", "e_gap_dom")


def _neff_row(name: str, gen: str, g: dict, summ: dict, gaps: dict) -> dict:
    kidx = _kind_index(name)
    rec = {"concept": name, "gen": gen, "set": f"dcdev_{name}",
           "n_kinds": len(g["kinds"]), "n_kinds_used": len(g["present"]),
           **{k: _r(v, 4) for k, v in summ.items()},
           "n_tagged": g["n_all_pos"] + g["n_all_neg"],
           "n_rows": g["n_all_pos"] + g["n_all_neg"]}
    for s in g["kinds"]:
        i = kidx[s]
        d = g["dirs"].get(s)
        rec[f"cos_all_k{i}"] = ("" if d is None or g["d_all"] is None
                                else round(float(d @ g["d_all"]), 4))
        rec[f"n_k{i}"] = g["npos"][s] + g["nneg"][s]
        rec[f"e_gap_{s}"] = _r(gaps[s]["e_gap"], 4)
        rec[f"t_gap_{s}"] = _r(gaps[s]["t_gap"], 4)
    return rec


def _paper_per_split() -> dict[str, dict]:
    """Per split, the generated-set t_other and e_gap the link stage used.

    ``dc_link_stats.csv`` holds only the correlation results; the per-split values behind
    them are ``direction_count._link_to_paper_m``'s medians over the generators' rows of
    ``dc_ratios.csv``, recomputed here the same way.
    """
    out: dict[str, dict] = {}
    by: dict[str, dict[str, list]] = {}
    for r in dc_dev._read_csv(SCRIPTS / "dc_ratios.csv"):
        d = by.setdefault(r["split"], {"t_other": [], "e_gap": []})
        for k in d:
            if r.get(k, "") != "":
                d[k].append(float(r[k]))
    for s, d in by.items():
        out[s] = {k: (statistics.median(v) if v else None) for k, v in d.items()}
    return out


def stage_geometry(args) -> None:
    geom_rows, neff_rows, cell_rows = [], [], []
    probe_cells = {(r["concept"], r["split"]): r
                   for r in dc_dev._read_csv(SCRIPTS / "dc_dev_cells.csv")}
    paper = _paper_per_split()
    for name in args.concepts:
        X, y, groups = load_dev(name)
        kinds = dev_splits(name)
        mu, sd = _standardiser(X)
        Z = (X - mu) / sd
        evals = {}
        for s in kinds:
            Xe, ye = _load_eval_features(s)
            evals[s] = ((Xe - mu) / sd, ye.astype(bool))

        g = geometry(Z, y, groups, kinds, evals, np.random.default_rng(SEED))
        stats = split_stats(g)
        summ = concept_summary(g)
        gaps = {s: {"e_gap": stats[s]["e_gap_dom"], "t_gap": stats[s]["t_gap_dom"]}
                for s in kinds}
        neff_rows.append(_neff_row(name, "dev", g, summ, gaps))
        geom_rows.extend(long_rows(name, g))
        dev_t, ev_t = transfer_tables(name, g)
        fields = ["train_split", "n", *kinds]
        dc_dev._write_csv(SCRIPTS / f"dc_dev_dom_transfer_{name}_dev.csv", dev_t, fields)
        dc_dev._write_csv(SCRIPTS / f"dc_dev_dom_transfer_{name}_eval.csv", ev_t, fields)

        # Size check: each split class-balanced to min(bal, 100), N_MATCH_DRAWS draws.
        draws_s, draws_c = [], []
        for d in range(N_MATCH_DRAWS):
            use = matched_mask(y, groups, kinds, np.random.default_rng([SEED, d]))
            gd = geometry(Z, y, groups, kinds, evals,
                          np.random.default_rng([SEED, d, 1]), use=use)
            draws_s.append(concept_summary(gd))
            draws_c.append(split_stats(gd))
        m_summ = {k: _m([s[k] for s in draws_s]) for k in MATCHED_STATS}
        sd_summ = {k: statistics.stdev([s[k] for s in draws_s]) for k in MATCHED_STATS}
        m_gaps = {s: {"e_gap": _m([c[s]["e_gap_dom"] for c in draws_c]),
                      "t_gap": _m([c[s]["t_gap_dom"] for c in draws_c])} for s in kinds}
        # gd is the last draw: its per-split counts are the same in every draw.
        rec = _neff_row(name, "dev_n100", gd, m_summ, m_gaps)
        for k in MATCHED_STATS:
            rec[f"sd_{k}"] = _r(sd_summ[k], 4)
        rec["n_draws"] = N_MATCH_DRAWS
        neff_rows.append(rec)

        for s in kinds:
            pc = probe_cells.get((name, s), {})
            pp = paper.get(s, {})
            cell = {"concept": name, "split": s, "knee": dc_dev.knee_key(s),
                    "n_pos": g["npos"][s], "n_neg": g["nneg"][s],
                    "n_per_class_n100": min(g["npos"][s], g["nneg"][s], N_MATCH // 2)}
            for k in CELL_STATS:
                cell[k] = _r(stats[s].get(k))
                vals = [c[s].get(k) for c in draws_c]
                vals = [v for v in vals if v is not None and v == v]
                cell[f"{k}_n100"] = _r(_m(vals))
                cell[f"sd_{k}_n100"] = _r(statistics.stdev(vals)) if len(vals) > 1 else ""
            for k in ("auc_minus_dev", "auc_all_dev", "auc_all_dev_ho", "auc_minus_eval",
                      "auc_all_eval"):
                cell[k] = _r(stats[s].get(k))
            cell["t_other_dev"] = pc.get("t_other_dev", "")
            cell["G_dev"] = pc.get("G_dev", "")
            cell["t_other_gen"] = _r(pp.get("t_other"), 4)
            cell["e_gap_gen"] = _r(pp.get("e_gap"), 4)
            cell_rows.append(cell)

        print(f"[geometry] {name}: t_own {summ['t_own']:.3f} t_other {summ['t_other']:.3f} "
              f"e_own {summ['e_own']:.3f} e_other {summ['e_other']:.3f} "
              f"e_all {summ['e_all']:.3f} n_eff {summ['n_eff']:.2f} | n100: t_other "
              f"{m_summ['t_other']:.3f}±{sd_summ['t_other']:.3f} e_other "
              f"{m_summ['e_other']:.3f}±{sd_summ['e_other']:.3f}", flush=True)

    # Keep other concepts' rows when only some are recomputed.
    done = set(args.concepts)
    geom_rows = [r for r in dc_dev._read_csv(SCRIPTS / "dc_dev_geometry.csv")
                 if r["concept"] not in done] + geom_rows
    neff_rows = [r for r in dc_dev._read_csv(SCRIPTS / "dc_dev_neff.csv")
                 if r["concept"] not in done] + neff_rows
    cell_rows = [r for r in dc_dev._read_csv(SCRIPTS / "dc_dev_dom_cells.csv")
                 if r["concept"] not in done] + cell_rows
    order = dc_dev.CONCEPT_ORDER
    geom_rows.sort(key=lambda r: order.index(r["concept"]))
    neff_rows.sort(key=lambda r: (order.index(r["concept"]), r["gen"]))
    cell_rows.sort(key=lambda r: order.index(r["concept"]))
    dc_dev._write_csv(SCRIPTS / "dc_dev_geometry.csv", geom_rows, GEOM_FIELDS)
    with (SCRIPTS / "dc_neff.csv").open(newline="", encoding="utf-8") as fh:
        neff_fields = next(csv.reader(fh))
    extra = [k for r in neff_rows for k in r if k not in neff_fields]
    dc_dev._write_csv(SCRIPTS / "dc_dev_neff.csv", neff_rows,
                      neff_fields + list(dict.fromkeys(extra)))
    dc_dev._write_csv(SCRIPTS / "dc_dev_dom_cells.csv", cell_rows,
                      list(dict.fromkeys(k for r in cell_rows for k in r)))
    print("[geometry] wrote dc_dev_geometry.csv, dc_dev_neff.csv, dc_dev_dom_cells.csv "
          "and the dc_dev_dom_transfer_*.csv tables")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("pool", "geometry"))
    ap.add_argument("--concepts", nargs="+", default=list(dc_dev.CONCEPT_ORDER),
                    choices=dc_dev.CONCEPT_ORDER)
    args = ap.parse_args(argv)
    {"pool": stage_pool, "geometry": stage_geometry}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
