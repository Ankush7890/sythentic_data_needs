#!/usr/bin/env python
"""Pull the eval activation blobs from Kaggle into the local cache.

This workspace starts with no `cache_gen_gemma27b_highstakes/`, so the first eval would
otherwise extract all 4408 eval rows locally on gemma-3-27b. Scoring probe_iter0 is what
fetches them, and it gives the base-probe reference line for free.

The DEV blobs are deliberately not fetched. What is published is one blob per *full* dev
split (anthropic_hh_balanced is 1028 rows), while the fits here validate on
`dev_samples/highstakes_500`, whose splits are 125-row cuts. `prefetch_dev_activations`
checks the row count against the split file, so those blobs are rejected — after a ~20 GB
download. 500 rows extract locally in one model load instead, which the first fit does.
"""

from __future__ import annotations

import dataclasses
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
)

concept = dataclasses.replace(
    CONCEPTS["highstakes"], dev_data=REPO / "dev_samples/highstakes_500"
)

from agentic_redteam.evaluation import evaluate_probe  # noqa: E402

print("scoring probe_iter0 on the eval splits (fetches the eval blobs) ...", flush=True)
df = evaluate_probe(
    concept.base_probe, concept.eval_dir, concept.eval_cache, max_samples=None, seed=SEED,
    combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
    kaggle_source=eval_source(),
)
print(df.to_string(), flush=True)
