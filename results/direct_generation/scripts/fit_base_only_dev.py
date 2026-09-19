#!/usr/bin/env python
"""Fit the 50-row base ALONE under the size-curve protocol — the n=0 point.

The size curve's small-n points are not "the targeted set at 5 rows": the fit still trains
on the 50-row base and still early-stops against the concept's dev set (500 rows for
highstakes, 436 for instructions), so at n=5 the validation data outnumbers the generated
data 100 to 1. Without this control the curve has no floor and a flat tail cannot be told
apart from "the base and the dev set are doing all the work".

Same base, same dev, same seed and transforms as every arm in the two CSVs; scored on all
splits. One fit per concept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import COMBINE, CONCEPTS, CONVERT, SEED, eval_source  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    ap.add_argument("--base-data", type=Path, required=True)
    ap.add_argument("--dev-data", type=Path, default=None)
    args = ap.parse_args()

    import dataclasses

    from synthetic_probe_data.evaluation import evaluate_probe
    from synthetic_probe_data.retrain import retrain_probe

    concept = CONCEPTS[args.concept]
    if args.dev_data:
        concept = dataclasses.replace(concept, dev_data=args.dev_data.resolve())

    out = concept.cache_dir / "subsample_probes" / f"baseonly_{args.base_data.stem}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = retrain_probe(
        samples=[], base_probe_path=concept.base_probe,
        base_training_data_path=args.base_data, new_probe_path=out,
        dev_data_path=concept.dev_data, seed=SEED, base_data_fraction=1.0,
        base_activation_cache_dir=concept.base_cache,
        combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
        verbose=False,
    )
    df = evaluate_probe(out, concept.eval_dir, concept.eval_cache, max_samples=None,
                        seed=SEED, combine_consecutive_messages=COMBINE,
                        convert_tool_to_assistant=CONVERT, kaggle_source=eval_source())
    print(f"\n=== {args.concept}: BASE ONLY ({args.base_data.name}, "
          f"{res.n_training_samples_total} training rows, dev {concept.dev_data.name}) ===")
    print(f"  dev  mean {res.dev_auroc['mean']:.5f}")
    for _, r in df.iterrows():
        print(f"  {'eval ' + r['dataset']:<34} {r['auroc']:.5f}")
    out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
