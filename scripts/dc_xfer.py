#!/usr/bin/env python
"""Probe transfer scored on the classifier's own source: dev split -> dev split, generated
kind -> generated kind.

Brief: ``docs/dev_coverage_xfer_task.md``. Every probe transfer number so far is scored on
the eval splits (Part A's ``t_other_dev``, ``dc_gen.py``'s ``t_other_gen``); the direction
statistic has been scored on its own source (``t_other`` on generated kinds,
``t_other_dom`` on dev splits). This fills the two missing probe cells:

- **dev to dev** — a probe trained on every dev sample of split s (Part A's ``own`` arm),
  scored on the dev samples of every split of the concept (``xfer_dev_<concept>/``).
- **generated to generated** — a probe trained on the tagged samples of kind k of one
  generated set (``dc_gen.py``'s ``kind`` arm), scored on every kind of the same set
  (``xfer_gen_<concept>_<gen>/``).

**Scoring outside the harness.** ``subsample_curve_concept.py`` deletes each probe pickle
before the next draw, so the transfer score has to be taken while it exists. This script
does not edit the harness: ``--stage harness`` imports it and calls its ``main()`` with
the concept's ``eval_dir`` replaced (restricted to the arm's own eval split, or the
concept's eval dir minus ``oig_omission``), and with ``agentic_redteam.evaluation.
evaluate_probe`` replaced by a wrapper that first scores the pickle on the transfer
directory through ``retrain.score_probe_on_dev`` and appends one row to a side CSV, then
calls the original. ``score_probe_on_dev`` keys a directory's activation blob on its
files' names and bytes, so each scoring directory is one blob, read by every fit.

**Validation sets.** Dev arms early-stop on the concept's DeepSeek-V4-Pro detailed set
(``dc_dev.val_dir``, as in Part A) and are scored on dev samples; generated arms
early-stop on the concept's dev samples (the harness default, ``dev_samples/highstakes_500``
for high-stakes, as every generated curve did) and are scored on generated samples. No
fit early-stops on its scoring directory, an eval split or its training arm.

Stages:

    dirs     .dc_work/xfer_dev_<concept>/, .dc_work/xfer_gen_<concept>_<gen>/ and the
             manifest scripts/dc_xfer_arms.csv (the high-stakes dev cut: 150 per split)
    warm     per concept, ONE extraction-model load for every sample any fit or blob
             needs, then every validation and scoring blob (assembled from the
             per-sample cache, see ``assemble_blob``), and the base probe's AUROC on
             each scoring directory as the sanity check
    fit      --kind dev|gen: the 14 dev own arms / the 51 generated kind arms, 8 draws
    cells    the matrices, scripts/dc_xfer_cells.csv, scripts/dc_xfer_summary.csv and
             scripts/dc_xfer_link_stats.csv

Edits none of ``subsample_curve_concept.py``, ``dc_run_curve.py``, ``direction_count.py``,
``dc_dev.py``, ``dc_dev_geometry.py``, ``knee_predictor.py`` or ``fit_base_plus_concept.py``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import math
import os
import random
import re
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
import dc_gen  # noqa: E402
from dc_dev import (BATCH, CONCEPT_ORDER, DRAWS, EXCLUDED_SPLITS, WORK, _mean,  # noqa: E402
                    _read_csv, _sd, _write_csv, eval_column, knee_key)
from dc_gen import GENERATORS, MIN_KIND_TAGGED  # noqa: E402

CUT_SEED = getattr(dc_dev, "SEED", 20260917)
HS_CUT = 150                         # rows per high-stakes dev split, class-balanced
ARMS_CSV = SCRIPTS / "dc_xfer_arms.csv"
ARM_FIELDS = dc_dev.ARM_FIELDS + ["xfer_dir"]


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
def dev_dir(name: str) -> Path:
    return WORK / f"xfer_dev_{name}"


def gen_dir(name: str, gen: str) -> Path:
    return WORK / f"xfer_gen_{name}_{gen}"


def side_csv(kind: str, name: str, gen: str = "") -> Path:
    if kind == "dev":
        return SCRIPTS / f"dc_xfer_dev_{name}_scores.csv"
    return SCRIPTS / f"dc_xfer_gen_{name}__{gen}_scores.csv"


def harness_csv(kind: str, name: str, gen: str = "", split: str = "") -> Path:
    """The harness's own rows. A restricted-eval run carries only its split's eval column,
    and the harness refuses to append under a header naming another split, so each
    training split gets its own file (as dc_dev.py's ``others`` arms do)."""
    if kind == "dev":
        return SCRIPTS / f"dc_xfer_dev_{name}_own_{split}.csv"
    if name == "highstakes":
        # docs/dev_coverage_gen_probe_task.md had not run on high-stakes: its kind arms are
        # these arms with unrestricted eval, so one set of fits serves both briefs and the
        # harness rows go where dc_gen.py --stage cells reads them.
        return dc_gen.out_csv(name, gen, "kind")
    return SCRIPTS / f"dc_xfer_gen_{name}__{gen}_own_{split}.csv"


def unrestricted(kind: str, name: str) -> bool:
    return kind == "gen" and name == "highstakes"


def _link(d: Path, name: str, src: Path) -> None:
    link = d / name
    if link.is_symlink() and link.resolve() == src.resolve():
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(src.resolve())


def _prune(d: Path, keep: set[str]) -> None:
    for stale in d.glob("*.jsonl"):
        if stale.name not in keep:
            stale.unlink()


# --------------------------------------------------------------------------- #
# stage: dirs
# --------------------------------------------------------------------------- #
def _cut_balanced(lines: list[str], pos: str, n: int, split: str) -> list[str]:
    """n/2 per class, seeded on (CUT_SEED, split), original row order kept."""
    rng = random.Random(f"{CUT_SEED}:{split}")
    idx_pos = [i for i, ln in enumerate(lines) if json.loads(ln)["labels"] == pos]
    idx_neg = [i for i, ln in enumerate(lines) if json.loads(ln)["labels"] != pos]
    keep = set(rng.sample(idx_pos, n // 2) + rng.sample(idx_neg, n - n // 2))
    return [ln for i, ln in enumerate(lines) if i in keep]


def build_dev_dir(name: str) -> list[dict]:
    c = dc_dev._concepts()[name]
    d = dev_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    arms = [a for a in dc_dev._arms() if a["concept"] == name and a["arm"] == "own"]
    rows, keep = [], set()
    for a in arms:
        src = WORK / a["file"]
        fname = f"{a['split']}.jsonl"
        keep.add(fname)
        rec = dict(a, xfer_dir=str(d.relative_to(REPO)))
        rows.append(rec)
        if name != "highstakes":
            _link(d, fname, src)
            continue
        # A 1,908-row blob is ~21 GB, resident at every scoring call: cut each split to
        # 150 class-balanced rows (600 in all, the generated sets' size).
        lines = dc_dev._raw_lines(src)
        cut = _cut_balanced(lines, c.pos_label, HS_CUT, a["split"])
        out = d / fname
        if out.is_symlink():
            out.unlink()
        text = "\n".join(cut) + "\n"
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            out.write_text(text, encoding="utf-8")
        npos, nneg = dc_dev._count(cut, c.pos_label)
        rows.append({
            "concept": name, "gen": "dev", "arm": "xfer_cut", "split": a["split"],
            "file": f"{d.name}/{fname}", "n": len(cut), "n_pos": npos, "n_neg": nneg,
            "balanced": 2 * min(npos, nneg), "sizes": "",
            "skipped": f"scoring cut of {a['file']} ({a['n']} rows): {HS_CUT} class-balanced, "
                       f"seeded '{CUT_SEED}:<split>', source order kept",
            "xfer_dir": str(d.relative_to(REPO)),
        })
    _prune(d, keep)
    return rows


def build_gen_dir(name: str, gen: str) -> list[dict]:
    d = gen_dir(name, gen)
    d.mkdir(parents=True, exist_ok=True)
    arms = [a for a in dc_gen._arms() if a["concept"] == name and a["gen"] == gen
            and a["arm"] == "kind"]
    keep = set()
    rows = []
    for a in arms:           # every kind, including the ones too small to train on
        fname = f"{a['split']}.jsonl"
        keep.add(fname)
        _link(d, fname, WORK / a["file"])
        rows.append(dict(a, xfer_dir=str(d.relative_to(REPO))))
    _prune(d, keep)
    return rows


def stage_dirs(args) -> None:
    rows = []
    for name in CONCEPT_ORDER:
        rows += build_dev_dir(name)
        for gen in GENERATORS:
            rows += build_gen_dir(name, gen)
    for r in rows:
        if r["arm"] in ("own", "kind") and r["sizes"]:
            assert r["split"] not in EXCLUDED_SPLITS
    _write_csv(ARMS_CSV, rows, ARM_FIELDS)
    fit = collections.Counter((r["concept"], r["arm"]) for r in rows
                              if r["arm"] in ("own", "kind") and r["sizes"])
    print(f"[dirs] wrote {ARMS_CSV.name} ({len(rows)} rows); to fit: {dict(fit)}")
    for d in sorted(WORK.glob("xfer_*")):
        print(f"[dirs] {d.name}: " + " ".join(sorted(p.stem for p in d.glob('*.jsonl'))))


def _xfer_arms() -> list[dict]:
    rows = _read_csv(ARMS_CSV)
    if not rows:
        raise SystemExit(f"{ARMS_CSV.name} missing — run --stage dirs first")
    return rows


def scoring_dirs(name: str) -> list[Path]:
    return [dev_dir(name)] + [gen_dir(name, g) for g in GENERATORS]


def validation_dirs(name: str) -> list[Path]:
    c = dc_dev._concepts()[name]
    return [dc_dev.val_dir(name), dc_gen.HIGHSTAKES_DEV if name == "highstakes" else c.dev_data]


# --------------------------------------------------------------------------- #
# stage: warm
# --------------------------------------------------------------------------- #
def _dir_rows(d: Path, concept) -> list[dict]:
    """The rows of a scoring/validation dir, in the representation load_from gives them."""
    from fit_base_plus_concept import load_rows

    out = []
    for f in sorted(d.glob("*.jsonl")):
        lines = [dc_dev.fix_assistant_first(ln)[0] for ln in dc_dev._raw_lines(f)]
        tmp = WORK / "_xfer_tmp.jsonl"
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out += load_rows(tmp, concept)
        tmp.unlink()
    return out


def assemble_blob(d: Path, probe_path: Path, cache_dir: Path, verbose: bool = True) -> str:
    """Write ``d``'s dev-activation blob from the per-sample cache, if every row hits.

    ``score_probe_on_dev`` would otherwise forward every row again: ``get_activations``
    over the directory at tuberlens' ``BATCH_SIZE`` 1 is one unpadded forward per row,
    right-padded with zeros to the longest row (capped at 1024) and concatenated —
    exactly what the per-sample cache already holds row by row. So the blob is built
    from those rows, in ``_load_dev_dataset``'s order, under the path and in the format
    ``get_activations(save_path=...)`` writes; ``score_probe_on_dev`` then reads it as a
    cache hit. Returns "exists", "assembled" or "missing <k>" (nothing written; the
    caller lets ``score_probe_on_dev`` extract).
    """
    import torch

    from fit_base_plus_concept import COMBINE, CONVERT

    from agentic_redteam.retrain import (_dev_activation_cache_path, _load_dev_dataset,
                                         _sample_activation_cache_path, load_probe)

    if int(os.environ.get("BATCH_SIZE", "1") or 1) != 1:
        return "missing (BATCH_SIZE != 1: a batched extraction pads within the batch)"
    probe = load_probe(probe_path)
    model_name, layer = str(probe.model_name), int(probe.layer)
    ds, files, _ = _load_dev_dataset(d, probe.pos_class_label, probe.neg_class_label,
                                     COMBINE, CONVERT, verbose=False)
    out = _dev_activation_cache_path(cache_dir, files, model_name, layer, COMBINE, CONVERT)
    if out.exists():
        return "exists"
    paths = [_sample_activation_cache_path(cache_dir, m, model_name, layer, COMBINE, CONVERT)
             for m in ds.inputs]
    miss = sum(1 for p in paths if not p.exists())
    if miss:
        return f"missing {miss}/{len(paths)}"
    parts = [torch.load(p, map_location="cpu") for p in paths]
    width = max(x["activations"].shape[1] for x in parts)

    def pad(t):
        if t.shape[1] >= width:
            return t[:, :width]
        z = torch.zeros(t.shape[0], width - t.shape[1], *t.shape[2:], dtype=t.dtype)
        return torch.cat([t, z], dim=1)

    blob = {k: torch.cat([pad(x[k]) for x in parts], dim=0)
            for k in ("activations", "attention_mask", "input_ids")}
    blob["layer"] = parts[0].get("layer", layer)
    blob["model_name"] = parts[0].get("model_name", model_name)
    del parts
    tmp = out.with_suffix(".tmp")
    torch.save(blob, tmp)
    tmp.rename(out)
    if verbose:
        print(f"[warm] assembled {out.name} for {d.name}: {tuple(blob['activations'].shape)}",
              flush=True)
    return "assembled"


def stage_warm(args) -> None:
    from fit_base_plus_concept import COMBINE, CONVERT, load_rows

    from agentic_redteam.retrain import score_probe_on_dev, warm_sample_activation_cache

    for name in args.concepts:
        c = dc_dev._concepts()[name]
        t0 = time.time()
        # Training rows exactly as the harness loads them (the per-sample keys every fit
        # hits), then every row of every blob in load_from's representation.
        rows = []
        for a in dc_dev._arms():
            if a["concept"] == name and a["arm"] == "own":
                rows += load_rows(WORK / a["file"], c)
        for a in dc_gen._arms():
            if a["concept"] == name and a["arm"] == "kind":
                rows += load_rows(WORK / a["file"], c)
        for d in validation_dirs(name) + scoring_dirs(name):
            rows += _dir_rows(d, c)
        # The same conversation sits in a training arm, a scoring dir and possibly a
        # validation set; the warm checks the cache before extracting, so a repeat would
        # be forwarded once per occurrence. Dedupe on the (untransformed) content.
        seen, uniq = set(), []
        for r in rows:
            k = json.dumps(r["inputs"], sort_keys=True, ensure_ascii=False)
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        print(f"[warm] {name}: {len(uniq)} distinct rows ({len(rows)} with repeats)",
              flush=True)
        rows = uniq
        n = warm_sample_activation_cache(
            rows, base_probe_path=c.base_probe, base_activation_cache_dir=c.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=True,
        )
        t1 = time.time()
        print(f"[warm] {name}: {n} newly extracted in {(t1 - t0) / 60:.1f} min"
              f"{f' ({(t1 - t0) / n:.2f} s/sample incl. load)' if n else ''}", flush=True)
        for d in validation_dirs(name) + scoring_dirs(name):
            ta = time.time()
            status = assemble_blob(d, c.base_probe, c.base_cache)
            auc = score_probe_on_dev(c.base_probe, d, c.base_cache,
                                     combine_consecutive_messages=COMBINE,
                                     convert_tool_to_assistant=CONVERT, verbose=False)
            print(f"[warm] {name} {d.name}: blob {status} ({time.time() - ta:.0f}s); base probe "
                  + " ".join(f"{k} {v:.3f}" for k, v in auc.items()), flush=True)


# --------------------------------------------------------------------------- #
# stage: harness — the unmodified harness, with a scoring hook on evaluate_probe
# --------------------------------------------------------------------------- #
SIDE_KEY = ("samples", "base", "n", "draw")


def _side_fields(xfer: Path) -> list[str]:
    return list(SIDE_KEY) + sorted(p.stem for p in xfer.glob("*.jsonl")) + ["xfer_mean",
                                                                           "seconds"]


def _keys(path: Path) -> set[tuple]:
    return {(r["samples"], r["base"], int(r["n"]), int(r["draw"]))
            for r in _read_csv(path) if r.get("samples")}


def reconcile(out: Path, side: Path) -> None:
    """Make "done" mean "in both files".

    The side row is written before the harness row, so a kill between them leaves a
    side row alone (handled by the hook, which does not append a second one). A harness
    row with no side row would be skipped by the harness forever; drop it so the fit is
    redone. Only this script's own output files are ever rewritten.
    """
    if not out.exists():
        return
    have = _keys(side)
    with out.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fields, rows = rd.fieldnames, list(rd)
    keep = [r for r in rows if not r.get("samples")
            or (r["samples"], r["base"], int(r["n"]), int(r["draw"])) in have]
    if len(keep) != len(rows):
        print(f"[harness] {out.name}: {len(rows) - len(keep)} fit(s) with no transfer row "
              f"— refitting", flush=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(keep)


def stage_harness(argv: list[str]) -> None:
    from fit_base_plus_concept import COMBINE, CONCEPTS, CONVERT

    import dc_run_curve
    import subsample_curve_concept as harness

    import agentic_redteam.evaluation as evaluation
    from agentic_redteam.retrain import score_probe_on_dev

    argv = list(argv)

    def pop(flag):
        i = argv.index(flag)
        v = argv[i + 1]
        del argv[i: i + 2]
        return v

    xfer = Path(pop("--xfer-dir"))
    side = Path(pop("--xfer-out"))
    split = pop("--eval-split") if "--eval-split" in argv else None
    name = argv[argv.index("--concept") + 1]
    samples = Path(next(a for a in argv if a.endswith(".jsonl"))).name
    base = "none" + (f"+ga{argv[argv.index('--grad-accum') + 1]}" if "--grad-accum" in argv
                     else "") + (f"bs{argv[argv.index('--batch-size') + 1]}"
                                 if "--batch-size" in argv else "")
    assert "--no-base" in argv
    out = Path(argv[argv.index("--out") + 1])

    concept = CONCEPTS[name]
    ev_dir = (dc_run_curve.restricted_eval_dir(concept, split) if split
              else dc_dev.eval_dir(name))
    CONCEPTS[name] = dataclasses.replace(concept, eval_dir=ev_dir)

    reconcile(out, side)
    fields = _side_fields(xfer)
    if side.exists() and side.stat().st_size:
        with side.open(newline="", encoding="utf-8") as fh:
            head = next(csv.reader(fh))
        if head != fields:
            raise SystemExit(f"{side} header {head} != {fields}")
    else:
        _write_csv(side, [], fields)

    original = evaluation.evaluate_probe

    def hooked(probe_path, *a, **kw):
        m = re.search(r"_n(\d+)_d(\d+)\.pkl$", str(probe_path))
        key = (samples, base, int(m.group(1)), int(m.group(2)))
        if key not in _keys(side):
            t0 = time.time()
            auc = score_probe_on_dev(probe_path, xfer, concept.base_cache,
                                     combine_consecutive_messages=COMBINE,
                                     convert_tool_to_assistant=CONVERT, verbose=False)
            row = dict(zip(SIDE_KEY, key))
            for k, v in auc.items():
                col = "xfer_mean" if k == "mean" else k
                row[col] = "" if math.isnan(v) else round(v, 5)
            row["seconds"] = round(time.time() - t0, 1)
            with side.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fields).writerow(row)
            print(f"[xfer] {samples} n={key[2]} d={key[3]}: " + " ".join(
                f"{k[:12]} {row[k]}" for k in fields[4:-1]) + f" ({row['seconds']}s)",
                flush=True)
        return original(probe_path, *a, **kw)

    evaluation.evaluate_probe = hooked
    sys.argv = [str(SCRIPTS / "subsample_curve_concept.py")] + argv
    harness.main()


# --------------------------------------------------------------------------- #
# stage: fit
# --------------------------------------------------------------------------- #
def fit_jobs(kind: str, concepts, gens) -> list[dict]:
    arms = _xfer_arms()
    if kind == "dev":
        jobs = [a for a in arms if a["gen"] == "dev" and a["arm"] == "own"
                and a["concept"] in concepts]
    else:
        jobs = [a for a in arms if a["arm"] == "kind" and a["sizes"]
                and a["concept"] in concepts and a["gen"] in gens]
    jobs.sort(key=lambda r: (CONCEPT_ORDER.index(r["concept"]),
                             (["dev"] + list(GENERATORS)).index(r["gen"]), r["split"]))
    return jobs


def stage_fit(args) -> None:
    jobs = fit_jobs(args.kind, args.concepts, args.generators)
    print(f"[fit] {args.kind}: {len(jobs)} arm files x {args.draws} draws", flush=True)
    t0 = time.time()
    for i, r in enumerate(jobs, 1):
        name, split, n = r["concept"], r["split"], int(r["sizes"])
        gen = "" if args.kind == "dev" else r["gen"]
        if args.kind == "dev":
            dev = ["--dev-data", str(dc_dev.val_dir(name))]
        else:
            dev = dc_gen.dev_args(name, False)
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--stage", "harness", "--",
            "--concept", name, str(WORK / r["file"]), "--no-base",
            "--grad-accum", str(math.ceil(n / BATCH)), "--batch-size", str(BATCH),
            "--sizes", str(n), "--draws", str(args.draws), *dev,
            "--out", str(harness_csv(args.kind, name, gen, split)),
            "--xfer-dir", str(REPO / r["xfer_dir"]),
            "--xfer-out", str(side_csv(args.kind, name, gen)),
        ]
        if not unrestricted(args.kind, name):
            cmd += ["--eval-split", split]
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
def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _r(v):
    return round(v, 5) if v is not None and math.isfinite(v) else ""


def _med(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return round(statistics.median(xs), 5) if xs else ""


def matrix(side: Path, arms: list[dict], cols: list[str], partial: bool):
    """{train split: {col: {draw: AUROC}}} from the side CSV, arms' own n and base."""
    rows = _read_csv(side)
    out = {}
    for a in arms:
        n = int(a["sizes"])
        rs = [r for r in rows if r["samples"] == a["file"] and int(r["n"]) == n
              and r["base"] == f"none+ga{math.ceil(n / BATCH)}bs{BATCH}"]
        if len(rs) < DRAWS and not partial:
            raise SystemExit(f"{side.name} {a['split']}: {len(rs)}/{DRAWS} draws scored")
        out[a["split"]] = {c: {int(r["draw"]): float(r[c]) for r in rs if r.get(c, "") != ""}
                           for c in cols}
    return out


def transfer_stats(mat: dict, cols: list[str]):
    """Per scoring split: off-diagonal column mean (+ across-draw sd) and the diagonal."""
    stats = {}
    for s in cols:
        others = [o for o in mat if o != s]
        vals = [_mean(list(mat[o][s].values())) for o in others if mat[o][s]]
        per_draw = {d: _mean([mat[o][s][d] for o in others if d in mat[o][s]])
                    for d in range(DRAWS) if any(d in mat[o][s] for o in others)}
        diag = mat.get(s, {}).get(s, {})
        stats[s] = {"t": _mean(vals) if vals else float("nan"),
                    "sd": _sd(list(per_draw.values())), "n_train": len(vals),
                    "diag": _mean(list(diag.values())) if diag else float("nan")}
    return stats


def _write_matrix(path: Path, mat: dict, cols: list[str], arms: list[dict]) -> None:
    by = {a["split"]: a for a in arms}
    rows = []
    for s in mat:
        rec = {"train": s, "n": by[s]["sizes"]}
        for c in cols:
            v = list(mat[s][c].values())
            rec[c] = _r(_mean(v)) if v else ""
        rec["diag_in_sample"] = rec.get(s, "")
        rows.append(rec)
    _write_csv(path, rows, ["train", "n"] + cols + ["diag_in_sample"])


def _print_matrix(title, mat, cols):
    print(f"\n[cells] {title} (rows train, cols scored; diagonal in-sample)")
    print("  " + " " * 24 + " ".join(f"{knee_key(c)[:9]:>9s}" for c in cols))
    for s in mat:
        cells = []
        for c in cols:
            v = list(mat[s][c].values())
            txt = f"{_mean(v):.3f}" if v else "-"
            cells.append(f"{('[' + txt + ']') if c == s else txt:>9s}")
        print(f"  {knee_key(s)[:24]:24s} " + " ".join(cells))


CELL_FIELDS = ["concept", "split", "knee", "t_dev2dev", "sd_t_dev2dev", "n_train_dev",
               "diag_dev", "t_gen2gen", "t_gen2gen_min", "t_gen2gen_max", "n_gens",
               "diag_gen", "t_other_dev", "t_other_dom", "e_other_dom", "t_other",
               "e_gap", "t_other_gen"]


def stage_cells(args) -> None:
    arms = _xfer_arms()
    cells = []
    for name in args.concepts:
        # dev to dev
        d_arms = [a for a in arms if a["concept"] == name and a["gen"] == "dev"
                  and a["arm"] == "own"]
        cols = sorted(p.stem for p in dev_dir(name).glob("*.jsonl"))
        dev_stats = {}
        if side_csv("dev", name).exists():
            mat = matrix(side_csv("dev", name), d_arms, cols, args.allow_partial)
            _write_matrix(SCRIPTS / f"dc_xfer_dev_{name}.csv", mat, cols, d_arms)
            _print_matrix(f"{name} dev to dev", mat, cols)
            dev_stats = transfer_stats(mat, cols)
        # generated to generated, per generator
        gen_stats = collections.defaultdict(dict)
        for gen in GENERATORS:
            g_arms = [a for a in arms if a["concept"] == name and a["gen"] == gen
                      and a["arm"] == "kind" and a["sizes"]]
            side = side_csv("gen", name, gen)
            if not side.exists():
                continue
            gcols = sorted(p.stem for p in gen_dir(name, gen).glob("*.jsonl"))
            mat = matrix(side, g_arms, gcols, args.allow_partial)
            _write_matrix(SCRIPTS / f"dc_xfer_gen_{name}__{gen}.csv", mat, gcols, g_arms)
            _print_matrix(f"{name} x {gen} generated to generated", mat, gcols)
            for s, st in transfer_stats(mat, gcols).items():
                gen_stats[s][gen] = st
        cells += split_cells(name, dev_stats, gen_stats)
    path = SCRIPTS / "dc_xfer_cells.csv"
    keep = [r for r in _read_csv(path) if r["concept"] not in args.concepts]
    rows = sorted(keep + cells, key=lambda r: (CONCEPT_ORDER.index(r["concept"]), r["split"]))
    _write_csv(path, rows, CELL_FIELDS)
    print(f"[cells] wrote {path.name} ({len(rows)} rows)")
    summary = write_summary(rows)
    link = link_stats(rows)
    print("\n[summary] per concept:")
    for r in summary:
        print("  " + " ".join(f"{k} {v}" for k, v in r.items()))
    for r in link:
        print(f"[link] {r}")


def _existing() -> dict:
    """The existing per-split numbers, keyed (concept, knee)."""
    ex = collections.defaultdict(dict)
    for r in _read_csv(SCRIPTS / "dc_dev_cells.csv"):
        ex[(r["concept"], r["knee"])]["t_other_dev"] = _num(r["t_other_dev"])
    for r in _read_csv(SCRIPTS / "dc_dev_dom_cells.csv"):
        ex[(r["concept"], r["knee"])].update(t_other_dom=_num(r["t_other_dom"]),
                                             e_other_dom=_num(r["e_other_dom"]))
    # direction on generated samples, per (split, generator): dc_scatter.csv is the
    # per-split table dc_link_stats.csv is computed from
    sc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in _read_csv(SCRIPTS / "dc_scatter.csv"):
        if r["gen"] in GENERATORS:
            for k in ("t_other", "e_gap"):
                sc[(r["concept"], r["split"])][k].append(_num(r[k]))
    for key, d in sc.items():
        ex[key].update({k: _med(v) for k, v in d.items()})
    gc = collections.defaultdict(list)
    for r in _read_csv(SCRIPTS / "dc_gen_cells.csv"):
        gc[(r["concept"], r["knee"])].append(_num(r["t_other_gen"]))
    for key, v in gc.items():
        ex[key]["t_other_gen"] = _med(v)
    # A concept dc_gen.py --stage cells has not covered (high-stakes: its loko arms were
    # never run) still has t_other_gen if its unrestricted kind rows exist — this run's
    # harness rows for high-stakes are exactly those. Computed here, as dc_gen.gen_cells
    # computes it, rather than written into dc_gen_cells.csv as partial rows.
    done = {r["concept"] for r in _read_csv(SCRIPTS / "dc_gen_cells.csv")}
    arms = _xfer_arms()
    for name in CONCEPT_ORDER:
        if name in done:
            continue
        per = collections.defaultdict(list)
        for gen in GENERATORS:
            kinds = [a for a in arms if a["concept"] == name and a["gen"] == gen
                     and a["arm"] == "kind" and a["sizes"]]
            path = dc_gen.out_csv(name, gen, "kind")
            if not path.exists() or not kinds:
                continue
            rows = _read_csv(path)
            col_means = {}
            for a in kinds:
                rs = dc_gen._fit_rows(path, a["file"], int(a["sizes"]))
                if len(rs) < DRAWS:
                    break
                col_means[a["split"]] = {s: _mean([float(r[eval_column(s)]) for r in rs])
                                         for s in dc_dev.eval_splits(name)}
            else:
                for s in dc_dev.eval_splits(name):
                    others = [k for k in col_means if k != s]
                    if others:
                        per[s].append(_mean([col_means[k][s] for k in others]))
            del rows
        for s, v in per.items():
            ex[(name, knee_key(s))]["t_other_gen"] = _med(v)
    return ex


def split_cells(name: str, dev_stats: dict, gen_stats: dict) -> list[dict]:
    ex = _existing()
    splits = sorted(set(dev_stats) | set(gen_stats))
    out = []
    for s in splits:
        k = knee_key(s)
        ds = dev_stats.get(s, {})
        gs = gen_stats.get(s, {})
        tg = [st["t"] for st in gs.values() if math.isfinite(st["t"])]
        e = ex.get((name, k), {})
        out.append({
            "concept": name, "split": s, "knee": k,
            "t_dev2dev": _r(ds.get("t")), "sd_t_dev2dev": _r(ds.get("sd")),
            "n_train_dev": ds.get("n_train", ""), "diag_dev": _r(ds.get("diag")),
            "t_gen2gen": _med(tg), "t_gen2gen_min": _r(min(tg)) if tg else "",
            "t_gen2gen_max": _r(max(tg)) if tg else "", "n_gens": len(tg),
            "diag_gen": _med([st["diag"] for st in gs.values()]),
            **{c: ("" if e.get(c) in (None, "") else e[c]) for c in
               ("t_other_dev", "t_other_dom", "e_other_dom", "t_other", "e_gap",
                "t_other_gen")},
        })
    return out


SUMMARY_COLS = ("t_dev2dev", "t_gen2gen", "t_other_dev", "t_other_gen", "t_other_dom",
                "e_other_dom", "t_other", "e_gap")


def write_summary(cells: list[dict]) -> list[dict]:
    """Per concept, median over splits: two classifiers x two scoring sets x two sources.

    probe, dev source:        t_dev2dev (own source)      t_other_dev (eval)
    probe, generated source:  t_gen2gen (own source)      t_other_gen (eval)
    direction, dev source:    t_other_dom (own source)    e_other_dom (eval)
    direction, generated:     t_other (own source)        e_gap (eval side, as committed)
    """
    out = []
    for name in CONCEPT_ORDER:
        rs = [r for r in cells if r["concept"] == name]
        if not rs:
            continue
        rec = {"concept": name, "n_splits": len(rs)}
        for c in SUMMARY_COLS:
            rec[c] = _med([_num(r[c]) for r in rs])
        rec["t_gen2gen_min"] = _med([_num(r["t_gen2gen_min"]) for r in rs])
        rec["t_gen2gen_max"] = _med([_num(r["t_gen2gen_max"]) for r in rs])
        rec["sd_splits_t_dev2dev"] = _r(_sd([x for x in (_num(r["t_dev2dev"]) for r in rs)
                                             if x is not None]))
        rec["sd_splits_t_gen2gen"] = _r(_sd([x for x in (_num(r["t_gen2gen"]) for r in rs)
                                             if x is not None]))
        out.append(rec)
    _write_csv(SCRIPTS / "dc_xfer_summary.csv", out,
               ["concept", "n_splits", *SUMMARY_COLS, "t_gen2gen_min", "t_gen2gen_max",
                "sd_splits_t_dev2dev", "sd_splits_t_gen2gen"])
    return out


def link_stats(cells: list[dict]) -> list[dict]:
    """``dc_dev.py --stage link``'s computation for t_dev2dev and t_gen2gen."""
    import numpy as np

    import knee_predictor as kp

    targets = kp.load_targets()
    per_split = collections.defaultdict(list)
    for c in kp.load_curves():
        if not c["flat"]:
            per_split[(c["split"], c["recipe"])].append(c)
    concept_of = {r["knee"]: r["concept"] for r in cells}
    out = []
    for name in ("t_dev2dev", "t_gen2gen"):
        rows = sorted((r["knee"], _num(r[name])) for r in cells if _num(r[name]) is not None
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
    _write_csv(SCRIPTS / "dc_xfer_link_stats.csv", out, dc_dev.LINK_FIELDS)
    return out


# --------------------------------------------------------------------------- #
STAGES = ("dirs", "warm", "fit", "harness", "cells")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:2] == ["--stage", "harness"]:
        rest = argv[2:]
        stage_harness(rest[1:] if rest[:1] == ["--"] else rest)
        return 0
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--kind", choices=["dev", "gen"], default="dev")
    ap.add_argument("--concepts", nargs="+", default=list(CONCEPT_ORDER), choices=CONCEPT_ORDER)
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS), choices=GENERATORS)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args(argv)
    args.concepts = [c for c in CONCEPT_ORDER if c in args.concepts]
    {"dirs": stage_dirs, "warm": stage_warm, "fit": stage_fit,
     "cells": stage_cells}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
