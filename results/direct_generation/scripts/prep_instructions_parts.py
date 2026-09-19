#!/usr/bin/env python
"""Warm everything the instructions part-curve fits read, before the first fit needs it.

1. The eval activations for all seven instructions splits, from Kaggle (no extraction).
2. The dev_samples/instructions activation blob, also from Kaggle, assembled into the
   single content-hashed blob the fit looks for.

probe_iter0.pkl is used only as the TEMPLATE the fits inherit (model, layer, labels,
architecture) — no base data enters any fit in this experiment.

    .venv_claude/bin/python scripts/prep_instructions_parts.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from fit_base_plus_concept import (  # noqa: E402
    COMBINE,
    CONCEPTS,
    CONVERT,
    SEED,
    eval_source,
    prefetch_dev,
)

c = CONCEPTS["instructions"]
from synthetic_probe_data.cli import _free_gpu  # noqa: E402
from synthetic_probe_data.evaluation import evaluate_probe  # noqa: E402

df = evaluate_probe(c.base_probe, c.eval_dir, c.eval_cache, max_samples=None, seed=SEED,
                    combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
                    kaggle_source=eval_source())
print(df, flush=True)
_free_gpu()
prefetch_dev(c)
print("dev blob ready", flush=True)
