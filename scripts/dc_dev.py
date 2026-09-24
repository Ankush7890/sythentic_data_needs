#!/usr/bin/env python
"""The coverage test on DEV samples, and whether it predicts the synthetic half-gain size.

Brief: ``docs/dev_coverage_task.md``. Companion to ``direction_count.py``, whose
generated-set study cuts each detailed set into kind-only / mixed / set-minus-kind arms
by LLM-tagged kind. Here the kind is exact — it is the dev split a sample comes from —
and the samples are real and in-distribution, so the question is whether the
non-substitution that study found under *instruction* is a property of the concept or of
the generator's rendering of the kinds.

Part A needs no curve: every arm is fit ONCE, at its full class-balanced size
``n_arm = min(2 * min(n_pos, n_neg), 590)``, eight draws, and read as a level.

- ``own``    — every dev sample of split s. Scored on every eval split of the concept, so
               one ``own`` fit is one ROW of the within-concept transfer matrix.
- ``others`` — every dev sample of the concept's other splits (set-minus-kind with exact
               labels). Scored on s alone (``dc_run_curve.py --eval-split``).
- ``all``    — the concept's whole dev set (the mixed arm, G's denominator).

Stages (each resumable; ``fit`` resumes per fit through the harness's own key):

    arms      write .dc_work/dcdev_<concept>_<split>_{own,others,all}.jsonl and
              scripts/dc_dev_arms.csv
    prefetch  pull the concept's eval activation blobs from Kaggle (network only)
    warm      extract every dev sample into the per-sample cache (one model load per
              concept) and build the validation set's dev blob
    fit       the 248 fits, instruction -> harmful -> high-stakes, own -> others -> all
    cells     scripts/dc_dev_cells.csv and scripts/dc_dev_transfer_<concept>.csv
    link      Part B: scripts/dc_dev_link_stats.csv and scripts/dc_dev_scatter.csv

**Validation set.** The dev samples are the TRAINING data here, so the fit early-stops on
the concept's DeepSeek-V4-Pro detailed generated set (off-distribution to the target,
disjoint from every eval split), the same file for all three arms. It is passed to the
harness as a directory holding one symlink to that file, not as the file itself: the
harness names its CSV columns from ``dev_data.glob('*.jsonl')``, which is empty for a
file path, and would then refuse the ``dev_<stem>`` column the fit returns. The dev blob
is keyed on the file's NAME and bytes, both unchanged by the symlink, so the activation
cache is the one the file itself would get.

**oig_omission** is excluded everywhere, as in the paper: from the arms (it is in
``dev_samples/instructions``) and from the eval — the unrestricted *instruction* fits are
scored on a directory of symlinks to the six other eval splits, so the harness never
reads it. Nothing in ``eval_sets/`` is touched.

This script edits none of ``subsample_curve_concept.py``, ``dc_run_curve.py``,
``direction_count.py``, ``knee_predictor.py`` or ``fit_base_plus_concept.py``; it imports
them.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
for p in (SCRIPTS, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WORK = REPO / ".dc_work"

CONCEPT_ORDER = ("instructions", "hu_harm", "highstakes")
EXCLUDED_SPLITS = {"oig_omission"}
N_CAP = 590                      # the direction_count mixed arm's largest size
DRAWS = 8
BATCH = 16

# The concept's DeepSeek-V4-Pro detailed set: the early-stopping set for every fit here.
VAL_FILES = {
    "instructions": REPO / "data/instructions_deepseekv4pro_evaldesc_600.jsonl",
    "hu_harm": REPO / "data/hu_harm_deepseekv4pro_evaldescshape_600.jsonl",
    "highstakes": REPO / "data/highstakes_deepseekv4pro_evaldesc_600.jsonl",
}

ARM_FIELDS = ["concept", "gen", "arm", "split", "file", "n", "n_pos", "n_neg",
              "balanced", "sizes", "skipped"]


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
def _concepts():
    from fit_base_plus_concept import CONCEPTS
    return CONCEPTS


def dev_files(name: str) -> list[Path]:
    c = _concepts()[name]
    return [p for p in sorted(c.dev_data.glob("*.jsonl")) if p.stem not in EXCLUDED_SPLITS]


def eval_stem(dev_stem: str) -> str:
    """Dev split stem -> the eval split it is the dev set of (hu_ha: dev_X -> eval_X)."""
    return "eval_" + dev_stem[4:] if dev_stem.startswith("dev_") else dev_stem


def knee_key(stem: str) -> str:
    """The split's key in knee_predictor's targets (eval_/dev_ prefix stripped)."""
    for pre in ("eval_", "dev_"):
        if stem.startswith(pre):
            return stem[len(pre):]
    return stem


def eval_column(split: str) -> str:
    """The harness's per-split eval AUROC column for eval split ``split``."""
    return split if split.startswith("eval_") else f"eval_{split}"


def arm_file(name: str, split: str, arm: str) -> Path:
    if arm == "all":                     # per concept, not per split
        return WORK / f"dcdev_{name}_all.jsonl"
    return WORK / f"dcdev_{name}_{split}_{arm}.jsonl"


def val_dir(name: str) -> Path:
    """A directory holding one symlink to the concept's validation file (see docstring)."""
    src = VAL_FILES[name]
    d = WORK / f"val_{name}"
    d.mkdir(parents=True, exist_ok=True)
    link = d / src.name
    if not link.exists():
        link.symlink_to(src.resolve())
    for stale in d.glob("*.jsonl"):
        if stale.name != src.name:
            stale.unlink()
    return d


def eval_dir(name: str) -> Path:
    """The concept's eval dir with the excluded splits removed (symlinks, never copies)."""
    c = _concepts()[name]
    files = sorted(c.eval_dir.glob("*.jsonl"))
    keep = [f for f in files if f.stem not in EXCLUDED_SPLITS]
    if len(keep) == len(files):
        return c.eval_dir
    d = WORK / f"eval_{name}_dev_coverage"
    d.mkdir(parents=True, exist_ok=True)
    for f in keep:
        link = d / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    for stale in d.glob("*.jsonl"):
        if stale.stem in EXCLUDED_SPLITS or stale.name not in {f.name for f in keep}:
            stale.unlink()
    return d


def eval_splits(name: str) -> list[str]:
    return sorted(p.stem for p in eval_dir(name).glob("*.jsonl"))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# stage: arms
# --------------------------------------------------------------------------- #
def _raw_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def fix_assistant_first(line: str) -> tuple[str, bool]:
    """tuberlens' ``_fix_assistant_first``, applied to a raw JSONL row.

    ``LabelledDataset.load_from`` — the loader behind every eval and dev split — ALWAYS
    inserts a blank user turn before a conversation whose first non-system message is
    the assistant's (after tool->assistant conversion, before combining). The harness
    trains through ``retrain_probe(samples=...)``, whose transforms do not, and gemma's
    chat template refuses such a conversation outright. 263 of the 274 high-stakes
    ``mts_balanced`` dev rows open that way (system, assistant, user, ...). Writing the
    row in the representation ``load_from`` gives it makes the dev sample, as training
    data, the same token sequence it is as validation/eval data; every other row is
    copied through untouched.
    """
    rec = json.loads(line)
    raw = rec["inputs"]
    msgs = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(msgs, list):
        return line, False
    for i, m in enumerate(msgs):
        role = m["role"].lower()
        if role == "system":
            continue
        if role in ("assistant", "tool"):   # tool counts: it is converted first
            msgs = msgs[:i] + [{"role": "user", "content": ""}] + msgs[i:]
            rec["inputs"] = json.dumps(msgs, ensure_ascii=False) if isinstance(raw, str) else msgs
            return json.dumps(rec, ensure_ascii=False), True
        break
    return line, False


def _count(lines: list[str], pos_label: str) -> tuple[int, int]:
    labels = [json.loads(ln)["labels"] for ln in lines]
    npos = sum(1 for lab in labels if lab == pos_label)
    return npos, len(labels) - npos


def build_arms(name: str) -> list[dict]:
    """Write the own / others / all JSONLs for one concept, rows copied verbatim."""
    c = _concepts()[name]
    WORK.mkdir(parents=True, exist_ok=True)
    per_split = {}
    for p in dev_files(name):
        fixed = [fix_assistant_first(ln) for ln in _raw_lines(p)]
        per_split[p.stem] = [ln for ln, _ in fixed]
        n_fixed = sum(f for _, f in fixed)
        if n_fixed:
            print(f"[arms] {name}/{p.stem}: blank user turn inserted in {n_fixed} "
                  f"assistant-first rows (tuberlens load_from representation)", flush=True)
    for stem, lines in per_split.items():
        bad = {json.loads(ln)["labels"] for ln in lines} - {c.pos_label, c.neg_label}
        if bad:
            raise SystemExit(f"{name}/{stem}: labels {bad} are not the probe's own")
    out = []

    def write(arm: str, split: str, lines: list[str]) -> None:
        p = arm_file(name, split, arm)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        npos, nneg = _count(lines, c.pos_label)
        bal = 2 * min(npos, nneg)
        n_arm = min(bal, N_CAP)
        out.append({
            "concept": name, "gen": "dev", "arm": arm, "split": split, "file": p.name,
            "n": len(lines), "n_pos": npos, "n_neg": nneg, "balanced": bal,
            "sizes": str(n_arm),
            "skipped": f"capped at {N_CAP}" if bal > N_CAP else "",
        })

    for stem, lines in per_split.items():
        split = eval_stem(stem)
        write("own", split, lines)
        write("others", split,
              [ln for s, ls in per_split.items() if s != stem for ln in ls])
    write("all", "", [ln for ls in per_split.values() for ln in ls])
    return out


def stage_arms(args) -> None:
    rows = []
    for name in args.concepts:
        for r in build_arms(name):
            rows.append(r)
            print(f"[arms] {r['file']:58s} n={r['n']:4d} ({r['n_pos']}/{r['n_neg']}) "
                  f"n_arm={r['sizes']} {r['skipped']}", flush=True)
    # Keep other concepts' rows when only some are rebuilt.
    path = SCRIPTS / "dc_dev_arms.csv"
    keep = [r for r in _read_csv(path) if r["concept"] not in args.concepts]
    rows = sorted(keep + rows, key=lambda r: (CONCEPT_ORDER.index(r["concept"]),
                                              ("own", "others", "all").index(r["arm"]),
                                              r["split"]))
    _write_csv(path, rows, ARM_FIELDS)
    print(f"[arms] wrote {path.name} ({len(rows)} arms)")


def _arms() -> list[dict]:
    rows = _read_csv(SCRIPTS / "dc_dev_arms.csv")
    if not rows:
        raise SystemExit("dc_dev_arms.csv missing — run --stage arms first")
    return rows


# --------------------------------------------------------------------------- #
# stage: prefetch / warm
# --------------------------------------------------------------------------- #
def stage_prefetch(args) -> None:
    """Eval blobs from Kaggle (network only), and the base probe's eval as a check."""
    from fit_base_plus_concept import COMBINE, CONVERT, SEED, eval_source

    from agentic_redteam.evaluation import evaluate_probe

    for name in args.concepts:
        c = _concepts()[name]
        t0 = time.time()
        if name == "highstakes":
            # Download only: the base-probe eval would hold all 47 GB of blobs in RAM,
            # which must not overlap the extraction model in --stage warm.
            _prefetch_only(name)
            print(f"[prefetch] {name}: eval blobs ready ({time.time() - t0:.0f}s)",
                  flush=True)
            continue
        df = evaluate_probe(
            c.base_probe, eval_dir(name), c.eval_cache, max_samples=None, seed=SEED,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            kaggle_source=eval_source(),
        )
        print(f"[prefetch] {name}: eval blobs ready ({time.time() - t0:.0f}s); "
              f"base probe eval:", flush=True)
        print(df[["dataset", "auroc"]].to_string(index=False), flush=True)


def _prefetch_only(name: str) -> None:
    """What ``evaluate_probe`` does before scoring: load the splits, fetch their blobs."""
    import pickle

    from fit_base_plus_concept import COMBINE, CONVERT, eval_source
    from tuberlens.interfaces.dataset import LabelledDataset

    from agentic_redteam.kaggle_activations import prefetch_eval_activations

    c = _concepts()[name]
    with c.base_probe.open("rb") as fh:
        probe = pickle.load(fh)
    datasets = {
        s: LabelledDataset.load_from(
            eval_dir(name) / f"{s}.jsonl", pos_class_label=probe.pos_class_label,
            neg_class_label=probe.neg_class_label, combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT)
        for s in eval_splits(name)
    }
    c.eval_cache.mkdir(parents=True, exist_ok=True)
    prefetch_eval_activations(c.eval_cache, datasets, eval_source(),
                              model_name=probe.model_name, layer=int(probe.layer),
                              cache_stem="acts_full.pt")


def stage_warm(args) -> None:
    """Every dev sample into the per-sample cache; the validation set into its dev blob."""
    from fit_base_plus_concept import COMBINE, CONVERT, load_rows

    from agentic_redteam.retrain import score_probe_on_dev, warm_sample_activation_cache

    for name in args.concepts:
        c = _concepts()[name]
        # The arm files, not dev_samples/: they carry the assistant-first fix.
        rows = [r for p in dev_files(name)
                for r in load_rows(arm_file(name, eval_stem(p.stem), "own"), c)]
        print(f"[warm] {name}: {len(rows)} dev samples", flush=True)
        n = warm_sample_activation_cache(
            rows, base_probe_path=c.base_probe, base_activation_cache_dir=c.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=True,
        )
        print(f"[warm] {name}: {n} newly extracted", flush=True)
        # Builds (or loads) exactly the blob every fit will read, and reports the base
        # probe's AUROC on it as a sanity check.
        auc = score_probe_on_dev(
            c.base_probe, val_dir(name), c.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=True,
        )
        print(f"[warm] {name}: validation blob ready; base probe {auc}", flush=True)


# --------------------------------------------------------------------------- #
# stage: fit
# --------------------------------------------------------------------------- #
def out_csv(name: str, arm: str, split: str) -> Path:
    if arm == "others":
        return SCRIPTS / f"dc_dev_{name}_others_{split}.csv"
    return SCRIPTS / f"dc_dev_{name}_{arm}.csv"


def fit_jobs(concepts: list[str]) -> list[dict]:
    arms = _arms()
    order = {"own": 0, "others": 1, "all": 2}
    jobs = [r for r in arms if r["concept"] in concepts]
    jobs.sort(key=lambda r: (CONCEPT_ORDER.index(r["concept"]), order[r["arm"]], r["split"]))
    return jobs


def stage_fit(args) -> None:
    """One harness call per arm file at its single size, eight draws.

    ``own`` and ``all`` go through ``--stage harness`` (unrestricted eval, minus the
    excluded splits); ``others`` through ``dc_run_curve.py --eval-split``. The optimiser
    regime is ``direction_count.py --stage fit``'s: accumulation ceil(n/16) at batch 16,
    one step per epoch, rows tagged ``none+ga<K>bs16``.
    """
    jobs = fit_jobs(args.concepts)
    print(f"[fit] {len(jobs)} arm files x {args.draws} draws", flush=True)
    t0 = time.time()
    for i, r in enumerate(jobs, 1):
        name, arm, split, n = r["concept"], r["arm"], r["split"], int(r["sizes"])
        common = [
            "--concept", name, str(WORK / r["file"]), "--no-base",
            "--grad-accum", str(math.ceil(n / BATCH)), "--batch-size", str(BATCH),
            "--sizes", str(n), "--draws", str(args.draws),
            "--dev-data", str(val_dir(name)), "--out", str(out_csv(name, arm, split)),
        ]
        if arm == "others":
            cmd = [sys.executable, str(SCRIPTS / "dc_run_curve.py"),
                   "--eval-split", split] + common
        else:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "harness",
                   "--"] + common
        print(f"[fit] [{i}/{len(jobs)}] {r['file']} n={n} "
              f"({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc != 0:
            print(f"[fit] FAILED {r['file']} (exit {rc})", file=sys.stderr, flush=True)
            if args.stop_on_error:
                raise SystemExit(rc)


def stage_harness(argv: list[str]) -> None:
    """The unmodified harness, with the concept's eval dir minus the excluded splits."""
    from fit_base_plus_concept import CONCEPTS

    import subsample_curve_concept as harness

    name = argv[argv.index("--concept") + 1]
    CONCEPTS[name] = dataclasses.replace(CONCEPTS[name], eval_dir=eval_dir(name))
    sys.argv = [str(SCRIPTS / "subsample_curve_concept.py")] + argv
    harness.main()


# --------------------------------------------------------------------------- #
# stage: cells — the Part A statistics
# --------------------------------------------------------------------------- #
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _by_draw(rows: list[dict], col: str) -> dict[int, float]:
    return {int(r["draw"]): float(r[col]) for r in rows if r.get(col, "") != ""}


def _fit_rows(path: Path, fname: str, n: int) -> list[dict]:
    return [r for r in _read_csv(path)
            if r["samples"] == fname and int(r["n"]) == n and r["base"].startswith("none+ga")]


def concept_cells(name: str, arms: list[dict], partial: bool = False) -> tuple[list[dict], list[dict]]:
    arms = [a for a in arms if a["concept"] == name]
    splits = [a["split"] for a in arms if a["arm"] == "own"]
    ev = eval_splits(name)
    if sorted(splits) != sorted(ev):
        raise SystemExit(f"{name}: own arms {splits} do not match eval splits {ev}")
    arm_of = {(a["arm"], a["split"]): a for a in arms}

    def rows_for(arm: str, split: str) -> list[dict]:
        a = arm_of[(arm, split)]
        rs = _fit_rows(out_csv(name, arm, split), a["file"], int(a["sizes"]))
        if len(rs) < DRAWS and not partial:
            raise SystemExit(f"{name} {arm} {split}: {len(rs)}/{DRAWS} draws fitted")
        return rs

    # own[s][s'] -> {draw: AUROC of the probe trained on s, scored on s'}
    own = {s: {t: _by_draw(rows_for("own", s), eval_column(t)) for t in splits}
           for s in splits}
    all_rows = rows_for("all", "")
    transfer = []
    for s in splits:
        rec = {"train_split": s, "n": arm_of[("own", s)]["sizes"]}
        for t in splits:
            rec[t] = round(_mean(list(own[s][t].values())), 5)
        transfer.append(rec)

    cells = []
    for s in splits:
        a_own_d = own[s][s]
        a_oth_d = _by_draw(rows_for("others", s), eval_column(s))
        a_all_d = _by_draw(all_rows, eval_column(s))
        # t_other_dev per draw index: mean over the other splits' own probes. The draws
        # of different arm files are unrelated fits, so pairing them by index is only a
        # device for a spread; the mean is what the statistic is.
        t_oth_d = {d: _mean([own[o][s][d] for o in splits if o != s and d in own[o][s]])
                   for d in range(DRAWS)
                   if any(d in own[o][s] for o in splits if o != s)}
        g_d = {d: (a_oth_d[d] - 0.5) / (a_all_d[d] - 0.5)
               for d in a_oth_d if d in a_all_d and a_all_d[d] != 0.5}
        a_own, a_oth, a_all = (_mean(list(x.values())) for x in (a_own_d, a_oth_d, a_all_d))
        t_oth = _mean([_mean(list(own[o][s].values())) for o in splits if o != s])
        cells.append({
            "concept": name, "split": s, "knee": knee_key(s),
            "n_own": arm_of[("own", s)]["sizes"], "n_others": arm_of[("others", s)]["sizes"],
            "n_all": arm_of[("all", "")]["sizes"],
            "a_own": round(a_own, 5), "a_others": round(a_oth, 5), "a_all": round(a_all, 5),
            "G_dev": round((a_oth - 0.5) / (a_all - 0.5), 5),
            "gap_all_others": round(a_all - a_oth, 5),
            "t_other_dev": round(t_oth, 5), "t_own_dev": round(a_own, 5),
            "sd_a_own": round(_sd(list(a_own_d.values())), 5),
            "sd_a_others": round(_sd(list(a_oth_d.values())), 5),
            "sd_a_all": round(_sd(list(a_all_d.values())), 5),
            "sd_G_dev": round(_sd(list(g_d.values())), 5),
            "sd_t_other_dev": round(_sd(list(t_oth_d.values())), 5),
            "sd_t_own_dev": round(_sd(list(a_own_d.values())), 5),
            "n_draws_own": len(a_own_d), "n_draws_others": len(a_oth_d),
            "n_draws_all": len(a_all_d),
        })
    return cells, transfer


CELL_FIELDS = ["concept", "split", "knee", "n_own", "n_others", "n_all", "a_own",
               "a_others", "a_all", "G_dev", "gap_all_others", "t_other_dev", "t_own_dev",
               "sd_a_own", "sd_a_others", "sd_a_all", "sd_G_dev", "sd_t_other_dev",
               "sd_t_own_dev", "n_draws_own", "n_draws_others", "n_draws_all"]


def stage_cells(args) -> None:
    arms = _arms()
    path = SCRIPTS / "dc_dev_cells.csv"
    keep = [r for r in _read_csv(path) if r["concept"] not in args.concepts]
    new = []
    for name in args.concepts:
        cells, transfer = concept_cells(name, arms, partial=args.allow_partial)
        new += cells
        splits = [t["train_split"] for t in transfer]
        _write_csv(SCRIPTS / f"dc_dev_transfer_{name}.csv", transfer,
                   ["train_split", "n"] + splits)
        print(f"\n[cells] {name}: transfer matrix (rows train, cols eval)")
        print("  " + " " * 28 + " ".join(f"{knee_key(t)[:9]:>9s}" for t in splits))
        for t in transfer:
            print(f"  {t['train_split'][:28]:28s} " + " ".join(f"{t[s]:9.3f}" for s in splits))
        print(f"[cells] {name}: per split")
        for c in cells:
            print(f"  {c['split'][:28]:28s} own {c['a_own']:.3f} others {c['a_others']:.3f} "
                  f"all {c['a_all']:.3f} G {c['G_dev']:.3f} t_other {c['t_other_dev']:.3f}")
    rows = sorted(keep + new, key=lambda r: (CONCEPT_ORDER.index(r["concept"]), r["split"]))
    _write_csv(path, rows, CELL_FIELDS)
    print(f"[cells] wrote {path.name} ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# stage: link — Part B
# --------------------------------------------------------------------------- #
PREDICTORS = ("G_dev", "t_other_dev", "a_others", "a_own", "gap_all_others")
LINK_FIELDS = ["predictor", "n_splits", "rho", "p_perm", "loo_rmse_pred",
               "loo_rmse_concept", "loo_rmse_grand", "loo_rmse_pred_concept",
               "beats_concept", "ci_lo", "ci_hi", "rho_instructions", "p_instructions"]
LINK_SEED = 20260918             # direction_count.SEED: the same permutation streams


def stage_link(args) -> None:
    """Exactly what ``direction_count._link_to_paper_m`` computes, for the dev predictors."""
    import numpy as np

    import knee_predictor as kp

    cells = _read_csv(SCRIPTS / "dc_dev_cells.csv")
    extra = {}
    rdev = SCRIPTS / "dc_dev_partc_ratios.csv"
    for r in _read_csv(rdev):          # Part C, if it ran
        if r.get("R_dev", "") != "":
            extra[r["knee"]] = float(r["R_dev"])
    targets = kp.load_targets()
    per_split = collections.defaultdict(list)
    pool = collections.defaultdict(list)
    for c in kp.load_curves():
        if not c["flat"]:
            per_split[(c["split"], c["recipe"])].append(c)
            if c["recipe"] == "detailed":
                pool[c["split"]].append(c["lm"])

    concept_of = {c["knee"]: c["concept"] for c in cells}
    value = {name: {c["knee"]: float(c[name]) for c in cells} for name in PREDICTORS}
    predictors = list(PREDICTORS)
    if extra:
        value["R_dev"] = extra
        predictors.append("R_dev")

    out = []
    for name in predictors:
        rows = sorted((s, v) for s, v in value[name].items()
                      if targets.get(s, {}).get(kp.PRIMARY_TARGET) is not None
                      and math.isfinite(v))
        if len(rows) < 5:
            out.append({"predictor": name, "n_splits": len(rows)})
            continue
        x = np.array([v for _, v in rows])
        y = np.array([targets[s][kp.PRIMARY_TARGET] for s, _ in rows])
        cs = [concept_of[s] for s, _ in rows]
        rho = kp._spearman(x, y)
        rec = {
            "predictor": name, "n_splits": len(rows), "rho": round(rho, 4),
            "p_perm": round(kp._perm_p(x, y, rho, np.random.default_rng(LINK_SEED)), 5),
            "loo_rmse_pred": round(kp._loo_rmse(x, y, cs, True, False), 4),
            "loo_rmse_concept": round(kp._loo_rmse(x, y, cs, False, True), 4),
            "loo_rmse_grand": round(kp._loo_rmse(x, y, cs, False, False), 4),
            "loo_rmse_pred_concept": round(kp._loo_rmse(x, y, cs, True, True), 4),
        }
        rec["beats_concept"] = int(rec["loo_rmse_pred"] < rec["loo_rmse_concept"])
        lo, hi = kp._boot_ci(x, [s for s, _ in rows], "detailed", per_split,
                             np.random.default_rng(LINK_SEED + 1))
        rec["ci_lo"], rec["ci_hi"] = round(lo, 4), round(hi, 4)
        sel = np.array([c == "instructions" for c in cs])
        if sel.sum() >= 5:
            r_in = kp._spearman(x[sel], y[sel])
            rec["rho_instructions"] = round(r_in, 4)
            rec["p_instructions"] = round(
                kp._perm_p(x[sel], y[sel], r_in, np.random.default_rng(LINK_SEED + 2)), 5)
        out.append(rec)
    _write_csv(SCRIPTS / "dc_dev_link_stats.csv", out, LINK_FIELDS)

    print("[link] dev-sample predictors of the paper's per-split log10 m "
          f"({kp.PRIMARY_TARGET}):")
    for r in out:
        if "rho" not in r:
            print(f"  {r['predictor']}: only {r['n_splits']} splits")
            continue
        bars = (abs(r["rho"]) >= 0.6 and r["p_perm"] < 0.05,
                r["beats_concept"] == 1,
                r.get("rho_instructions", float("nan")) * r["rho"] > 0)
        print(f"  {r['predictor']:15s} n={r['n_splits']} rho={r['rho']:+.3f} "
              f"p={r['p_perm']:.4f} CI [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] "
              f"LOO {r['loo_rmse_pred']:.3f} vs concept {r['loo_rmse_concept']:.3f} "
              f"(grand {r['loo_rmse_grand']:.3f}, pred+concept "
              f"{r['loo_rmse_pred_concept']:.3f})  within-instr "
              f"rho={r.get('rho_instructions', float('nan')):+.3f} "
              f"p={r.get('p_instructions', float('nan')):.3f}  bars {bars}")

    scatter = []
    for c in cells:
        k = c["knee"]
        lms = sorted(pool.get(k, []))
        scatter.append({
            "split": k, "concept": c["concept"], "gen": "dev",
            "R": extra.get(k, ""), "usable": int(k in extra),
            **{p: c[p] for p in PREDICTORS},
            "a_all": c["a_all"], "t_own_dev": c["t_own_dev"],
            "log_m": targets.get(k, {}).get(kp.PRIMARY_TARGET, ""),
            "log_m_lo": f"{lms[0]:.4f}" if lms else "",
            "log_m_hi": f"{lms[-1]:.4f}" if lms else "",
            "n_curves": len(lms),
        })
    _write_csv(SCRIPTS / "dc_dev_scatter.csv", scatter,
               ["split", "concept", "gen", "R", "usable", *PREDICTORS, "a_all",
                "t_own_dev", "log_m", "log_m_lo", "log_m_hi", "n_curves"])
    print("[link] wrote dc_dev_link_stats.csv, dc_dev_scatter.csv")


# --------------------------------------------------------------------------- #
STAGES = ("arms", "prefetch", "warm", "fit", "cells", "link", "harness")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:2] == ["--stage", "harness"]:
        rest = argv[2:]
        stage_harness(rest[1:] if rest[:1] == ["--"] else rest)
        return 0
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--concepts", nargs="+", default=list(CONCEPT_ORDER),
                    choices=CONCEPT_ORDER)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="cells: tolerate arms with fewer than 8 draws fitted")
    args = ap.parse_args(argv)
    args.concepts = [c for c in CONCEPT_ORDER if c in args.concepts]
    {"arms": stage_arms, "prefetch": stage_prefetch, "warm": stage_warm,
     "fit": stage_fit, "cells": stage_cells, "link": stage_link}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
