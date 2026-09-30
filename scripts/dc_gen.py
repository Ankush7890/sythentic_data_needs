#!/usr/bin/env python
"""The probe transfer statistic on GENERATED samples, the way Part A ran it on dev samples.

Brief: ``docs/dev_coverage_gen_probe_task.md``. Companion of ``dc_dev.py`` (Part A: one
probe per dev split at full size, read as levels) and of ``direction_count.py`` (the
generated-set study, whose kind arms are size ladders and whose ``t_other`` is a
difference-of-means direction). Here the generated sets get Part A's treatment:

- ``kind``  (Part A's ``own``)    — every tagged sample of the kind, fit once at
              ``2 * min(n_pos, n_neg)``, eight draws, scored UNRESTRICTED on the concept's
              eval splits (minus oig_omission), so one fit is one row of the transfer
              matrix. Kinds under ``MIN_KIND_TAGGED`` are skipped, as in the study.
- ``loko``  (Part A's ``others``) — the set minus the kind, at
              ``min(2 * min(n_pos, n_neg), 590)``, scored on the kind's split alone
              (``dc_run_curve.py --eval-split``).
- ``mixed`` (Part A's ``all``)    — NOT refit: the eight n = 590 draws already in
              ``dc_curves_<concept>[__<gen>].csv``.

The arm files are ``direction_count.build_arms``'s, from the committed kind tags. The
manifest is ``scripts/dc_gen_arms.csv``; ``dc_arms.csv`` is never written.

**Validation set.** The concept's dev samples, exactly as every generated-set curve used
them: the harness default (``dev_samples/instructions``, ``dev_samples/hu_ha``) and
``dev_samples/highstakes_500`` for high-stakes. ``--dsval`` (instruction ``kind`` arms of
llama70b / gptoss / nemotron only) early-stops on the DeepSeek-V4-Pro detailed set
instead, ``dc_dev.val_dir("instructions")``, the file Part A used; those rows go to their
own ``..._own_dsval.csv``.

Stages:

    arms     write .dc_work/dc_<concept>_<gen>_*.jsonl and scripts/dc_gen_arms.csv
    fit      kind then loko arms for --concepts x --generators (one process per
             (concept, generator): every pair has its own output files)
    cells    scripts/dc_gen_cells.csv, scripts/dc_gen_transfer_<concept>__<gen>.csv
    summary  scripts/dc_gen_summary.csv and scripts/dc_gen_link_stats.csv

This script edits none of ``subsample_curve_concept.py``, ``dc_run_curve.py``,
``direction_count.py``, ``knee_predictor.py`` or ``fit_base_plus_concept.py``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
for p in (SCRIPTS, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dc_dev  # noqa: E402
from dc_dev import (BATCH, CONCEPT_ORDER, DRAWS, EXCLUDED_SPLITS, N_CAP, WORK,  # noqa: E402
                    _mean, _read_csv, _sd, _write_csv, eval_column, knee_key)

GENERATORS = ("llama70b", "gptoss", "nemotron", "deepseekv4pro")
MIN_KIND_TAGGED = 60             # direction_count.MIN_KIND_TAGGED
HIGHSTAKES_DEV = REPO / "dev_samples/highstakes_500"
DSVAL_GENS = ("llama70b", "gptoss", "nemotron")   # DeepSeek's detailed set is its source
ARMS_CSV = SCRIPTS / "dc_gen_arms.csv"
ARM_FIELDS = ["concept", "gen", "arm", "split", "file", "n", "n_pos", "n_neg",
              "balanced", "sizes", "skipped"]


# --------------------------------------------------------------------------- #
# stage: arms
# --------------------------------------------------------------------------- #
def stage_arms(args) -> None:
    import direction_count as dc
    from fit_base_plus_concept import CONCEPTS

    specs = dc._concept_specs()
    rows = []
    for name in CONCEPT_ORDER:
        for gen in GENERATORS:
            for r in dc.build_arms(specs[name], gen, CONCEPTS[name]):
                r["skipped"] = ""
                if r["arm"] == "kind" and r["n"] < MIN_KIND_TAGGED:
                    r["skipped"] = f"only {r['n']} tagged (< {MIN_KIND_TAGGED})"
                    r["sizes"] = ""
                elif r["arm"] == "kind":
                    r["sizes"] = str(r["balanced"])
                else:                               # loko, and mixed (read, not refit)
                    r["sizes"] = str(min(r["balanced"], N_CAP))
                rows.append(r)
                print(f"[arms] {r['file']:56s} n={r['n']:3d} ({r['n_pos']}/{r['n_neg']}) "
                      f"size {r['sizes'] or '-'} {r['skipped']}", flush=True)
    _write_csv(ARMS_CSV, rows, ARM_FIELDS)
    fit = collections.Counter((r["concept"], r["arm"]) for r in rows
                              if r["sizes"] and r["arm"] != "mixed")
    print(f"[arms] wrote {ARMS_CSV.name} ({len(rows)} arms); to fit: {dict(fit)}")


def _arms() -> list[dict]:
    if not ARMS_CSV.exists():
        raise SystemExit(f"{ARMS_CSV.name} missing — run --stage arms first")
    rows = _read_csv(ARMS_CSV)
    missing = [r["file"] for r in rows if not (WORK / r["file"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} arm files missing from .dc_work — run --stage arms")
    return rows


# --------------------------------------------------------------------------- #
# stage: fit
# --------------------------------------------------------------------------- #
def out_csv(name: str, gen: str, arm: str, split: str = "", dsval: bool = False) -> Path:
    if arm == "loko":
        return SCRIPTS / f"dc_gen_{name}__{gen}_others_{split}.csv"
    return SCRIPTS / f"dc_gen_{name}__{gen}_own{'_dsval' if dsval else ''}.csv"


def dev_args(name: str, dsval: bool) -> list[str]:
    if dsval:
        return ["--dev-data", str(dc_dev.val_dir(name))]
    if name == "highstakes":
        return ["--dev-data", str(HIGHSTAKES_DEV)]
    return []                    # the harness default: the concept's dev samples


def fit_jobs(concepts, gens, arms_wanted, dsval=False) -> list[dict]:
    order = {"kind": 0, "loko": 1}
    jobs = [r for r in _arms() if r["concept"] in concepts and r["gen"] in gens
            and r["arm"] in arms_wanted and r["sizes"]]
    if dsval:
        jobs = [r for r in jobs if r["concept"] == "instructions" and r["arm"] == "kind"
                and r["gen"] in DSVAL_GENS]
    jobs.sort(key=lambda r: (CONCEPT_ORDER.index(r["concept"]), GENERATORS.index(r["gen"]),
                             order[r["arm"]], r["split"]))
    return jobs


def stage_fit(args) -> None:
    """One harness call per arm file at its single size, eight draws (Part A's regime).

    ``kind`` through ``dc_dev.py --stage harness`` (unrestricted eval minus oig_omission),
    ``loko`` through ``dc_run_curve.py --eval-split``: accumulation ceil(n/16) at batch 16,
    one optimiser step per epoch, rows tagged ``none+ga<K>bs16``. The harness resumes on
    (samples, n, draw) in each output file.
    """
    jobs = fit_jobs(args.concepts, args.generators, args.arms, args.dsval)
    print(f"[fit] {len(jobs)} arm files x {args.draws} draws"
          f"{' (DeepSeek-V4-Pro validation)' if args.dsval else ''}", flush=True)
    t0 = time.time()
    for i, r in enumerate(jobs, 1):
        name, gen, arm, split, n = r["concept"], r["gen"], r["arm"], r["split"], int(r["sizes"])
        common = [
            "--concept", name, str(WORK / r["file"]), "--no-base",
            "--grad-accum", str(math.ceil(n / BATCH)), "--batch-size", str(BATCH),
            "--sizes", str(n), "--draws", str(args.draws),
            *dev_args(name, args.dsval),
            "--out", str(out_csv(name, gen, arm, split, args.dsval)),
        ]
        if arm == "loko":
            cmd = [sys.executable, str(SCRIPTS / "dc_run_curve.py"), "--eval-split", split] + common
        else:
            cmd = [sys.executable, str(SCRIPTS / "dc_dev.py"), "--stage", "harness", "--"] + common
        print(f"[fit] [{i}/{len(jobs)}] {r['file']} n={n} "
              f"({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc != 0:
            print(f"[fit] FAILED {r['file']} (exit {rc})", file=sys.stderr, flush=True)
            if args.stop_on_error:
                raise SystemExit(rc)


# --------------------------------------------------------------------------- #
# stage: cells
# --------------------------------------------------------------------------- #
def _fit_rows(path: Path, fname: str, n: int) -> list[dict]:
    if not path.exists():
        return []
    return [r for r in _read_csv(path)
            if r["samples"] == fname and int(r["n"]) == n and r["base"].startswith("none+ga")]


def _by_draw(rows, col) -> dict[int, float]:
    return {int(r["draw"]): float(r[col]) for r in rows if r.get(col, "") != ""}


def mixed_rows(name: str, fname: str) -> list[dict]:
    """The n = 590 mixed draws, wherever direction_count wrote them (unrestricted files)."""
    out = []
    for p in sorted(glob.glob(str(SCRIPTS / f"dc_curves_{name}*.csv"))):
        out += _fit_rows(Path(p), fname, N_CAP)
    return out


def gen_cells(name: str, gen: str, arms: list[dict], partial: bool = False,
              dsval: bool = False):
    arms = [a for a in arms if a["concept"] == name and a["gen"] == gen]
    by = {(a["arm"], a["split"]): a for a in arms}
    splits = [a["split"] for a in arms if a["arm"] == "loko"]          # every kind
    kinds = [a["split"] for a in arms if a["arm"] == "kind" and a["sizes"]]

    def rows_for(arm, split):
        a = by[(arm, split)]
        rs = _fit_rows(out_csv(name, gen, arm, split, dsval), a["file"], int(a["sizes"]))
        if len(rs) < DRAWS and not partial:
            raise SystemExit(f"{name} {gen} {arm} {split}: {len(rs)}/{DRAWS} draws fitted")
        return rs

    kind = {k: {t: _by_draw(rows_for("kind", k), eval_column(t)) for t in splits}
            for k in kinds}
    mix = mixed_rows(name, by[("mixed", "")]["file"])
    if len(mix) < DRAWS and not partial:
        raise SystemExit(f"{name} {gen} mixed: {len(mix)}/{DRAWS} n=590 draws found")
    mix_d = {t: _by_draw(mix, eval_column(t)) for t in splits}

    transfer = []
    for k in kinds:
        transfer.append({"train": k, "n": by[("kind", k)]["sizes"],
                         **{t: round(_mean(list(kind[k][t].values())), 5) for t in splits}})
    transfer.append({"train": "mixed", "n": N_CAP,
                     **{t: round(_mean(list(mix_d[t].values())), 5) for t in splits}})

    cells = []
    for s in splits:
        own_d = kind[s][s] if s in kind else {}
        oth_d = {} if dsval else _by_draw(rows_for("loko", s), eval_column(s))
        all_d = mix_d[s]
        others = [k for k in kinds if k != s]
        t_d = {d: _mean([kind[o][s][d] for o in others if d in kind[o][s]])
               for d in range(DRAWS) if any(d in kind[o][s] for o in others)}
        g_d = {d: (oth_d[d] - 0.5) / (all_d[d] - 0.5)
               for d in oth_d if d in all_d and all_d[d] != 0.5}
        a_own, a_oth, a_all = (_mean(list(x.values())) for x in (own_d, oth_d, all_d))
        t_oth = _mean([_mean(list(kind[o][s].values())) for o in others])
        r = lambda v: round(v, 5) if math.isfinite(v) else ""          # noqa: E731
        cells.append({
            "concept": name, "gen": gen, "split": s, "knee": knee_key(s),
            "n_own": by[("kind", s)]["sizes"], "n_others": by[("loko", s)]["sizes"],
            "n_all": N_CAP,
            "a_own": r(a_own), "a_others": r(a_oth), "a_all": r(a_all),
            "G_gen": r((a_oth - 0.5) / (a_all - 0.5)),
            "gap_all_others": r(a_all - a_oth),
            "t_other_gen": r(t_oth), "t_own_gen": r(a_own),
            "sd_a_own": r(_sd(list(own_d.values()))),
            "sd_a_others": r(_sd(list(oth_d.values()))),
            "sd_a_all": r(_sd(list(all_d.values()))),
            "sd_G_gen": r(_sd(list(g_d.values()))),
            "sd_t_other_gen": r(_sd(list(t_d.values()))),
            "sd_t_own_gen": r(_sd(list(own_d.values()))),
            "n_draws_own": len(own_d), "n_draws_others": len(oth_d),
            "n_draws_all": len(all_d), "n_kinds_other": len(others),
        })
    return cells, transfer, splits


CELL_FIELDS = ["concept", "gen", "split", "knee", "n_own", "n_others", "n_all", "a_own",
               "a_others", "a_all", "G_gen", "gap_all_others", "t_other_gen", "t_own_gen",
               "sd_a_own", "sd_a_others", "sd_a_all", "sd_G_gen", "sd_t_other_gen",
               "sd_t_own_gen", "n_draws_own", "n_draws_others", "n_draws_all",
               "n_kinds_other"]


def stage_cells(args) -> None:
    arms = _arms()
    path = SCRIPTS / ("dc_gen_cells_dsval.csv" if args.dsval else "dc_gen_cells.csv")
    keep = [r for r in (_read_csv(path) if path.exists() else [])
            if (r["concept"], r["gen"]) not in
            {(c, g) for c in args.concepts for g in args.generators}]
    new = []
    for name in args.concepts:
        for gen in args.generators:
            if args.dsval and not (name == "instructions" and gen in DSVAL_GENS):
                continue
            cells, transfer, splits = gen_cells(name, gen, arms, args.allow_partial, args.dsval)
            new += cells
            if not args.dsval:
                _write_csv(SCRIPTS / f"dc_gen_transfer_{name}__{gen}.csv", transfer,
                           ["train", "n"] + splits)
            print(f"\n[cells] {name} x {gen}: transfer (rows train, cols eval)")
            print("  " + " " * 26 + " ".join(f"{knee_key(t)[:9]:>9s}" for t in splits))
            for t in transfer:
                print(f"  {t['train'][:26]:26s} " + " ".join(f"{t[s]:9.3f}" for s in splits))
            for c in cells:
                print(f"  {c['split'][:26]:26s} own {c['a_own']!s:7} others "
                      f"{c['a_others']!s:7} all {c['a_all']!s:7} G {c['G_gen']!s:7} "
                      f"t_other {c['t_other_gen']}")
    rows = sorted(keep + new, key=lambda r: (CONCEPT_ORDER.index(r["concept"]),
                                             GENERATORS.index(r["gen"]), r["split"]))
    _write_csv(path, rows, CELL_FIELDS)
    print(f"[cells] wrote {path.name} ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# stage: summary — medians over generators, beside the real-sample and direction numbers
# --------------------------------------------------------------------------- #
STATS = ("a_own", "a_others", "a_all", "G_gen", "t_other_gen", "gap_all_others")
SUMMARY_FIELDS = (["level", "concept", "split", "knee", "n_gens"]
                  + [f"{s}{x}" for s in STATS for x in ("", "_min", "_max")]
                  + ["t_other_dev", "G_dev", "a_own_dev", "a_others_dev", "a_all_dev",
                     "t_other_dir", "e_other_dir"])


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 5) if xs else ""


def per_split_summary() -> list[dict]:
    cells = _read_csv(SCRIPTS / "dc_gen_cells.csv")
    dev = {(r["concept"], r["knee"]): r for r in _read_csv(SCRIPTS / "dc_dev_cells.csv")}
    neff = [r for r in _read_csv(SCRIPTS / "dc_neff.csv") if r["gen"] in GENERATORS]
    by = collections.defaultdict(list)
    for c in cells:
        by[(c["concept"], c["split"])].append(c)
    out = []
    for (name, split), rs in sorted(by.items(), key=lambda kv: (
            CONCEPT_ORDER.index(kv[0][0]), kv[0][1])):
        rec = {"level": "split", "concept": name, "split": split, "knee": knee_key(split),
               "n_gens": len(rs)}
        for s in STATS:
            vals = [v for v in (_num(r[s]) for r in rs) if v is not None]
            rec[s] = _med(vals)
            rec[f"{s}_min"] = round(min(vals), 5) if vals else ""
            rec[f"{s}_max"] = round(max(vals), 5) if vals else ""
        d = dev.get((name, knee_key(split)), {})
        rec.update({"t_other_dev": d.get("t_other_dev", ""), "G_dev": d.get("G_dev", ""),
                    "a_own_dev": d.get("a_own", ""), "a_others_dev": d.get("a_others", ""),
                    "a_all_dev": d.get("a_all", "")})
        out.append(rec)
    # per concept: median over splits of the per-split medians; the direction numbers
    # (dc_neff.csv, one row per generated set) as the median over generators
    for name in CONCEPT_ORDER:
        rs = [r for r in out if r["concept"] == name]
        if not rs:
            continue
        rec = {"level": "concept", "concept": name, "split": "", "knee": "",
               "n_gens": max(r["n_gens"] for r in rs)}
        for s in STATS:
            rec[s] = _med([_num(r[s]) for r in rs])
            rec[f"{s}_min"] = _med([_num(r[f"{s}_min"]) for r in rs])
            rec[f"{s}_max"] = _med([_num(r[f"{s}_max"]) for r in rs])
        for k in ("t_other_dev", "G_dev", "a_own_dev", "a_others_dev", "a_all_dev"):
            rec[k] = _med([_num(r[k]) for r in rs])
        nf = [r for r in neff if r["concept"] == name]
        rec["t_other_dir"] = _med([_num(r["t_other"]) for r in nf])
        rec["e_other_dir"] = _med([_num(r["e_other"]) for r in nf])
        out.append(rec)
    return out


LINK_PREDICTORS = ("t_other_gen", "G_gen")


def link_stats(summary: list[dict]) -> list[dict]:
    """``dc_dev.py --stage link``'s computation, for the per-split medians over generators."""
    import numpy as np

    import knee_predictor as kp

    targets = kp.load_targets()
    per_split = collections.defaultdict(list)
    for c in kp.load_curves():
        if not c["flat"]:
            per_split[(c["split"], c["recipe"])].append(c)
    rows_split = [r for r in summary if r["level"] == "split"]
    concept_of = {r["knee"]: r["concept"] for r in rows_split}
    out = []
    for name in LINK_PREDICTORS:
        rows = sorted((r["knee"], _num(r[name])) for r in rows_split
                      if _num(r[name]) is not None
                      and targets.get(r["knee"], {}).get(kp.PRIMARY_TARGET) is not None)
        if len(rows) < 5:
            out.append({"predictor": name, "n_splits": len(rows)})
            continue
        x = np.array([v for _, v in rows])
        y = np.array([targets[s][kp.PRIMARY_TARGET] for s, _ in rows])
        cs = [concept_of[s] for s, _ in rows]
        rho = kp._spearman(x, y)
        rec = {
            "predictor": name, "n_splits": len(rows), "rho": round(rho, 4),
            "p_perm": round(kp._perm_p(x, y, rho, np.random.default_rng(dc_dev.LINK_SEED)), 5),
            "loo_rmse_pred": round(kp._loo_rmse(x, y, cs, True, False), 4),
            "loo_rmse_concept": round(kp._loo_rmse(x, y, cs, False, True), 4),
            "loo_rmse_grand": round(kp._loo_rmse(x, y, cs, False, False), 4),
            "loo_rmse_pred_concept": round(kp._loo_rmse(x, y, cs, True, True), 4),
        }
        rec["beats_concept"] = int(rec["loo_rmse_pred"] < rec["loo_rmse_concept"])
        lo, hi = kp._boot_ci(x, [s for s, _ in rows], "detailed", per_split,
                             np.random.default_rng(dc_dev.LINK_SEED + 1))
        rec["ci_lo"], rec["ci_hi"] = round(lo, 4), round(hi, 4)
        sel = np.array([c == "instructions" for c in cs])
        if sel.sum() >= 5:
            r_in = kp._spearman(x[sel], y[sel])
            rec["rho_instructions"] = round(r_in, 4)
            rec["p_instructions"] = round(kp._perm_p(
                x[sel], y[sel], r_in, np.random.default_rng(dc_dev.LINK_SEED + 2)), 5)
        out.append(rec)
    return out


def stage_summary(args) -> None:
    summary = per_split_summary()
    _write_csv(SCRIPTS / "dc_gen_summary.csv", summary, SUMMARY_FIELDS)
    print("[summary] per concept (generated probe | real probe | direction):")
    for r in summary:
        if r["level"] == "concept":
            print(f"  {r['concept']:13s} a_own {r['a_own']} a_others {r['a_others']} "
                  f"a_all {r['a_all']} G {r['G_gen']} t_other {r['t_other_gen']} | "
                  f"dev a_own {r['a_own_dev']} a_others {r['a_others_dev']} a_all "
                  f"{r['a_all_dev']} G {r['G_dev']} t_other {r['t_other_dev']} | "
                  f"dir t_other {r['t_other_dir']} e_other {r['e_other_dir']}")
    stats = link_stats(summary)
    _write_csv(SCRIPTS / "dc_gen_link_stats.csv", stats, dc_dev.LINK_FIELDS)
    for r in stats:
        print(f"[link] {r}")
    print("[summary] wrote dc_gen_summary.csv, dc_gen_link_stats.csv")


# --------------------------------------------------------------------------- #
STAGES = ("arms", "fit", "cells", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--concepts", nargs="+", default=list(CONCEPT_ORDER), choices=CONCEPT_ORDER)
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS), choices=GENERATORS)
    ap.add_argument("--arms", nargs="+", default=["kind", "loko"], choices=["kind", "loko"])
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--dsval", action="store_true",
                    help="fit/cells: the instruction kind arms of llama70b/gptoss/nemotron, "
                         "early-stopped on the DeepSeek-V4-Pro detailed set")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args(argv)
    args.concepts = [c for c in CONCEPT_ORDER if c in args.concepts]
    assert not EXCLUDED_SPLITS & set(dc_dev.eval_splits("instructions"))
    {"arms": stage_arms, "fit": stage_fit, "cells": stage_cells,
     "summary": stage_summary}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
