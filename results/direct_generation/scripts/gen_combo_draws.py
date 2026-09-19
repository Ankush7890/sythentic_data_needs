#!/usr/bin/env python
"""COMBINATION study over the GENERATORS: pool k generators' 600-row sets, resample, refit.

The design of the red-teaming combination study (experiment_hs_last /
experiment_instruction_last, `scripts/fit_combined_draws.py`) applied to this branch's
generated data. There the pool was a configuration's red-team successes and the axis was
the four attackers; here the pool is the generated training rows and the axis is the four
GENERATORS that wrote them. Everything structural is carried over: every subset of the
generators, the pool drawn ONCE at `--fraction`, several seeded draws per subset, resume
on (combo, fraction, draw), and a full-pool identity fit as each subset's reference.

WHAT IT MEASURES. `mix_curve_concept.py` already answers "is a blend of two generators
worth more than either alone AT THE SAME n" — it takes n rows from EACH set, so a k-way
mix at n is k*n rows and the generators are held to equal weight by construction. This
asks the combination question instead: take each generator's set WHOLE (600 rows), pool
the subset, and resample the pool at 90% several times. So the k-curve here is a curve in
how many generators you ran, with each contributing everything it wrote — which is what a
practitioner who ran k generators would actually have had — and the spread across draws is
the run-to-run noise at that k.

Two consequences of drawing the pool once rather than per generator-and-class, both
inherited deliberately from the red-team study: a draw can shift the generator ratio and
the class ratio slightly, and it is *meant* to — those ratios are properties of the
mixture under test, not of the draw, so a draw that rebalanced them would be measuring a
different mixture. At `--fraction 1.0` the draw is the identity, so one fit is run and the
rest skipped.

NO BASE TRAINING DATA IS USED AT ALL — `retrain_probe(base_training_data_path=None)`, as
in `mix_curve_concept.py`. The fit is the generated rows alone, so what is measured is
what they are worth by themselves with no 50 real rows underneath. That is what "only
generator based" means, and it is why these numbers are NOT comparable with
`<concept>_size_curve.csv` or `<concept>_gen90.csv`, every point of which sits on a base.

ONE CSV, NOT ONE PER FRACTION. The high-stakes script gave each `--fraction` its own file
because its resume key was (combo, draw) and carried no fraction; here the key carries it,
so the full-pool references and the 90% grid live in one table and can be differenced
without a join.

NO ACTIVATIONS ARE EXTRACTED provided each 600-row set has been through
`fit_base_plus_concept.py` (they all have — the mix curves ran on them), so every fit is
probe-head work on cached activations plus a cached dev and eval.

Examples:
    ${REPO_ROOT}/.venv_claude/bin/python scripts/gen_combo_draws.py --concept instructions
    ${REPO_ROOT}/.venv_claude/bin/python scripts/gen_combo_draws.py --concept highstakes \\
        --dev-data dev_samples/highstakes_500
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import (  # noqa: E402
    COMBINE, CONCEPTS, CONVERT, SEED, Concept, eval_source, load_rows,
)
from subsample_curve_concept import fields_for, split_column  # noqa: E402

SCRATCH_SUBDIR = "combo_probes"

# The four generators, and the one-letter codes a subset is named by. The ORDER IS FIXED
# and is the order a subset's pool is concatenated in, so `gd` and `dg` are the same combo
# and the same rows — a subset's name and its content are both independent of how it was
# enumerated.
GENERATORS = {"d": "deepseekv4pro", "g": "gptoss", "l": "llama70b", "n": "nemotron"}
ORDER = sorted(GENERATORS)                                     # d, g, l, n

IDENT = ["combo", "gens", "k", "frac", "draw", "n_pool",
         "n_pos", "n_neg", "n_training_rows", "dev_mean", "eval_mean"]


def set_path(concept: Concept, code: str) -> Path:
    return REPO / f"data/{concept.name}_{GENERATORS[code]}_600.jsonl"


def draw_pool(codes: str, loaded: dict, frac: float, draw: int) -> list[dict]:
    """`frac` of the subset's pooled rows, drawn ONCE over the pool.

    Seeded on (combo, frac, draw) so a row of the output is reproducible on its own and
    adding draws or subsets never moves the existing ones. Not seeded per generator: the
    point of pooling is that the draw sees one bag, so the generator ratio is free to move
    exactly as much as sampling makes it move."""
    pool = [r for c in codes for r in loaded[c]]
    if frac >= 1.0:
        return pool
    k = max(1, round(len(pool) * frac))
    return random.Random(f"{codes}:{frac}:{draw}").sample(pool, k)


def done_keys(csv_path: Path) -> set[tuple[str, str, int]]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {(r["combo"], r["frac"], int(r["draw"]))
                for r in csv.DictReader(fh) if r.get("combo")}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4],
                    choices=[1, 2, 3, 4], help="generator-subset sizes to enumerate")
    ap.add_argument("--combos", nargs="+", default=None,
                    help="subset codes to run (e.g. gd gln); default: every subset of --sizes")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=0.9)
    ap.add_argument("--full-pool", action="store_true",
                    help="also fit each subset's WHOLE pool (fraction 1.0, one identity "
                         "fit) — the reference its draws are read against")
    ap.add_argument("--dev-data", type=Path, default=None,
                    help="override the concept's dev dir. The 1908-row highstakes dev set "
                         "does not fit the card and makes a fit on that concept ~20x the "
                         "others; dev_samples/highstakes_500 is the subsample to use there.")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: scripts/<concept>_gen_combo_draws.csv")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    concept = CONCEPTS[args.concept]
    if args.dev_data:
        # dataclasses.replace, not mutation: Concept is frozen, and the dev blob is keyed
        # on the dev FILES' bytes, so pointing at a subsample dir picks up its own cached
        # blob rather than being served the full set's.
        dev = args.dev_data if args.dev_data.is_absolute() else REPO / args.dev_data
        if not dev.is_dir():
            ap.error(f"--dev-data {dev} is not a directory")
        concept = dataclasses.replace(concept, dev_data=dev.resolve())
    out_csv = args.out or REPO / f"scripts/{concept.name}_gen_combo_draws.csv"

    codes_all = ["".join(c) for k in sorted(set(args.sizes))
                 for c in itertools.combinations(ORDER, k)]
    if args.combos:
        wanted = ["".join(sorted(c)) for c in args.combos]
        unknown = [c for c in wanted if c not in
                   ["".join(x) for k in (1, 2, 3, 4) for x in itertools.combinations(ORDER, k)]]
        if unknown:
            ap.error(f"unknown combos {unknown}")
    else:
        wanted = codes_all

    from synthetic_probe_data.cli import _free_gpu
    from synthetic_probe_data.evaluation import evaluate_probe
    from synthetic_probe_data.retrain import retrain_probe

    scratch = concept.cache_dir / SCRATCH_SUBDIR
    scratch.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    seen = set() if args.no_resume else done_keys(out_csv)

    fields = IDENT + [f for f in fields_for(concept)
                      if f not in ("samples", "base", "n", *IDENT)]
    fresh = not out_csv.exists() or out_csv.stat().st_size == 0
    fh = out_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if fresh:
        writer.writeheader()
        fh.flush()

    loaded = {c: load_rows(set_path(concept, c), concept) for c in ORDER}
    for c in ORDER:
        print(f"  {GENERATORS[c]:14} {len(loaded[c]):4d} rows  {set_path(concept, c).name}")

    jobs = []
    for codes in wanted:
        # The full-pool fit FIRST for each subset, so an interrupted run still leaves every
        # subset's reference line rather than draws with nothing to read them against.
        if args.full_pool and (codes, "1.0", 0) not in seen:
            jobs.append((codes, 1.0, 0))
        for d in range(args.draws):
            if (codes, str(args.fraction), d) not in seen:
                jobs.append((codes, args.fraction, d))
    print(f"\n{len(jobs)} fits to run ({len(seen)} already in {out_csv.name}) | "
          f"dev {concept.dev_data.name} | no base data\n", flush=True)

    for i, (codes, frac, d) in enumerate(jobs, 1):
        rows = draw_pool(codes, loaded, frac, d)
        npos = sum(1 for r in rows if r["labels"] == concept.pos_label)
        gens = "+".join(GENERATORS[c] for c in codes)
        t0 = time.time()
        out_pkl = scratch / f"{codes}_f{round(frac * 100)}_d{d}.pkl"
        res = retrain_probe(
            samples=rows, base_probe_path=concept.base_probe,
            base_training_data_path=None,      # generated rows alone — see the docstring
            new_probe_path=out_pkl,
            dev_data_path=concept.dev_data, seed=SEED, base_data_fraction=1.0,
            base_activation_cache_dir=concept.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
        df = evaluate_probe(
            out_pkl, concept.eval_dir, concept.eval_cache, max_samples=None, seed=SEED,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            kaggle_source=eval_source(),
        )
        ev = {r["dataset"]: float(r["auroc"]) for _, r in df.iterrows()}
        row = {
            "combo": codes, "gens": gens, "k": len(codes), "frac": frac, "draw": d,
            "n_pool": 600 * len(codes),
            "n_pos": npos, "n_neg": len(rows) - npos,
            "n_training_rows": res.n_training_samples_total,
            "dev_mean": round(res.dev_auroc["mean"], 5),
            "eval_mean": round(ev["mean"], 5),
            "seconds": round(time.time() - t0, 1),
        }
        for split, v in res.dev_auroc.items():
            if split != "mean":
                row[split_column("dev", split)] = round(v, 5)
        for split, v in ev.items():
            if split != "mean":
                row[split_column("eval", split)] = round(v, 5)
        writer.writerow(row)
        fh.flush()
        print(f"[{i}/{len(jobs)}] {codes} (k={len(codes)}) frac={frac} draw={d}: "
              f"{len(rows)} rows  dev {row['dev_mean']:.5f}  eval {row['eval_mean']:.5f}  "
              f"({row['seconds']:.0f}s)", flush=True)
        out_pkl.unlink(missing_ok=True)
        _free_gpu()

    fh.close()


if __name__ == "__main__":
    main()
