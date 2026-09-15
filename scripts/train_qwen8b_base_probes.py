#!/usr/bin/env python
"""Train `probes/qwen8b_<concept>/probe_iter0.pkl`: the Qwen/Qwen3-8B (layer 18) base probe.

Every fit in `subsample_curve_concept.py` inherits model, layer, labels, architecture and
hyperparameters from the concept's base probe, so this probe is what points the curve at
Qwen. It copies the gemma probe's spec exactly (linear_then_softmax, batch 16, accum 4, lr
5e-3, 200 epochs, patience 50) and its class labels; only the model, the layer and the
description change. It is fit on the concept's llama70b 50-row base, as the gemma
probe_iter0 was, and scored on dev + eval as the base-only reference.

Run with PROBE_PROFILE=qwen8b. The first run also extracts and caches the dev blob and the
eval splits, which every later fit reuses.

    PROBE_PROFILE=qwen8b ${REPO_ROOT}/.venv_claude/bin/python \\
        scripts/train_qwen8b_base_probes.py --concept hu_harm [--dev-data dev_samples/highstakes_500]
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import (  # noqa: E402
    COMBINE,
    CONCEPTS,
    CONVERT,
    LAYER,
    MODEL_NAME,
    PROFILE,
    SEED,
    eval_source,
    report,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    ap.add_argument("--dev-data", type=Path, default=None,
                    help="override the concept's dev directory (as subsample_curve_concept.py)")
    ap.add_argument("--force", action="store_true", help="retrain even if the probe exists")
    args = ap.parse_args()
    if PROFILE != "qwen8b":
        raise SystemExit("run with PROBE_PROFILE=qwen8b")

    concept = CONCEPTS[args.concept]
    if args.dev_data:
        concept = dataclasses.replace(concept, dev_data=args.dev_data.resolve())

    from agentic_redteam.cli import _free_gpu
    from agentic_redteam.evaluation import evaluate_probe
    from agentic_redteam.retrain import (
        _cpu_unpickle,
        _infer_probe_spec,
        score_probe_on_dev,
        train_initial_probe,
    )

    gemma_probe = REPO / f"probes/gen_gemma27b_{concept.name}/probe_iter0.pkl"
    with gemma_probe.open("rb") as fh:
        ref = _cpu_unpickle(fh)
    spec = _infer_probe_spec(ref)
    description = str(ref.description).replace("google/gemma-3-27b-it", MODEL_NAME)
    print(f"{concept.name}: {MODEL_NAME} layer {LAYER}, spec {spec}", flush=True)

    concept.probe_dir.mkdir(parents=True, exist_ok=True)
    concept.base_cache.mkdir(parents=True, exist_ok=True)
    dev = None
    if args.force or not concept.base_probe.exists():
        res = train_initial_probe(
            base_training_data_path=concept.base_data, model_name=MODEL_NAME, layer=LAYER,
            new_probe_path=concept.base_probe,
            pos_class_label=ref.pos_class_label, neg_class_label=ref.neg_class_label,
            probe_description=description, probe_spec=spec,
            dev_data_path=concept.dev_data, seed=SEED, base_data_fraction=1.0,
            base_activation_cache_dir=concept.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=True,
        )
        dev = res.dev_auroc
        _free_gpu()
    else:
        print(f"{concept.base_probe} exists; scoring it", flush=True)
        dev = score_probe_on_dev(
            concept.base_probe, concept.dev_data, concept.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=False,
        )
    with concept.base_probe.open("rb") as fh:
        got = pickle.load(fh)
    assert (str(got.model_name), int(got.layer)) == (MODEL_NAME, LAYER), (got.model_name, got.layer)

    df = evaluate_probe(
        concept.base_probe, concept.eval_dir, concept.eval_cache, max_samples=None,
        seed=SEED, combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
        kaggle_source=eval_source(),
    )
    report(f"{concept.name}: qwen8b base only ({concept.base_data.name})", 50, dev, df)
    _free_gpu()


if __name__ == "__main__":
    main()
