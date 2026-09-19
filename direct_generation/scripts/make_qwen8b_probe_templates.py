#!/usr/bin/env python
"""Write `probes/qwen8b_<concept>/probe_template.pkl`: the metadata the qwen8b curve fits inherit.

Every fit in `subsample_curve_concept.py` runs `retrain_probe(base_probe_path=...)`, which reads
exactly five things off that pickle — model_name, layer, the two class labels, and the
architecture + hyperparameters (`_infer_probe_spec`) — and never its weights. The curve runs
with --no-base, so no base training set is needed either. So instead of training a Qwen probe
on some 50-row base, this copies the gemma probe_iter0 (linear_then_softmax, batch 16, accum 4,
lr 5e-3, 200 epochs, patience 50) and relabels it Qwen/Qwen3-8B, layer 18. No model load, no
extraction, no training data.

The template still carries the gemma classifier's weights (hidden size 5376). It is NOT a usable
probe: scoring it on Qwen activations (hidden size 4096) fails on the shape mismatch rather than
producing a number.

    PROBE_PROFILE=qwen8b ${REPO_ROOT}/.venv_claude/bin/python scripts/make_qwen8b_probe_templates.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import CONCEPTS, LAYER, MODEL_NAME, PROFILE  # noqa: E402

GEMMA_MODEL = "google/gemma-3-27b-it"


def main() -> None:
    if PROFILE != "qwen8b":
        raise SystemExit("run with PROBE_PROFILE=qwen8b")
    from synthetic_probe_data.retrain import _cpu_unpickle, _infer_probe_spec, read_probe_metadata

    for concept in CONCEPTS.values():
        src = REPO / f"probes/gen_gemma27b_{concept.name}/probe_iter0.pkl"
        with src.open("rb") as fh:
            probe = _cpu_unpickle(fh)
        assert probe.model_name == GEMMA_MODEL, probe.model_name
        probe.model_name = MODEL_NAME
        probe.layer = LAYER
        probe.description = str(probe.description).replace(GEMMA_MODEL, MODEL_NAME)
        concept.probe_dir.mkdir(parents=True, exist_ok=True)
        with concept.base_probe.open("wb") as fh:
            pickle.dump(probe, fh)
        meta = read_probe_metadata(concept.base_probe)
        assert (meta["model_name"], meta["layer"]) == (MODEL_NAME, LAYER), meta
        with concept.base_probe.open("rb") as fh:
            spec = _infer_probe_spec(_cpu_unpickle(fh))
        print(f"{concept.base_probe.relative_to(REPO)}: {meta['model_name']} L{meta['layer']} "
              f"{meta['pos_class_label']}/{meta['neg_class_label']}  {spec}")


if __name__ == "__main__":
    main()
