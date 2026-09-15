#!/usr/bin/env python
"""Warm everything the toolace-part fits read, before the first fit needs it.

1. The eval activations for all four highstakes splits, from Kaggle (no extraction).
2. The dev_samples/highstakes_500 activation blob — there is no Kaggle blob for this 500-row
   cut, so it is extracted once on gemma-3-27b L32 into the content-keyed dev cache.

probe_iter0.pkl is used only as the TEMPLATE the fits inherit (model, layer, labels,
architecture) — no base data enters any fit in this experiment.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from fit_base_plus_concept import COMBINE, CONCEPTS, CONVERT, SEED, eval_source  # noqa: E402

c = CONCEPTS["highstakes"]
from agentic_redteam.evaluation import evaluate_probe  # noqa: E402
from agentic_redteam.retrain import score_probe_on_dev  # noqa: E402

df = evaluate_probe(c.base_probe, c.eval_dir, c.eval_cache, max_samples=None, seed=SEED,
                    combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
                    kaggle_source=eval_source())
print(df, flush=True)
from agentic_redteam.cli import _free_gpu  # noqa: E402
_free_gpu()
dev = score_probe_on_dev(c.base_probe, REPO / "dev_samples/highstakes_500", c.base_cache,
                         combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
                         verbose=True)
print(dev, flush=True)
