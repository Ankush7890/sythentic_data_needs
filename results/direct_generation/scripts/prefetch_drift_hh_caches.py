#!/usr/bin/env python
"""Pull the eval and dev activation blobs for the four prompt-variant target splits.

This workspace has only `cache_gen_gemma27b_highstakes/`, so the `hu_harm` and
`instructions` studies would otherwise extract every eval and dev row locally on
gemma-3-27b. Both blobs are published, and both are fetchable here — unlike the
highstakes prompt study, where the dev sets were 125-row *cuts* and the published blobs
are full splits (see `prefetch_highstakes_caches.py`). Here each arm validates on the
split's own dev file **unchanged**, so `prefetch_dev_activations`' row-count check passes.

Eval is fetched by scoring `probe_iter0` on just the target splits, which also prints the
base-probe reference line every generated set is measured against.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/prefetch_drift_hh_caches.py
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
    DEV_FILE,
    DEV_SLUG,
    KAGGLE_OWNER,
    LAYER,
    MODEL_NAME,
    SEED,
    eval_source,
)

# (concept, eval split stem, the single-split dev dir that arm validates on)
ARMS = [
    ("instructions", "hc_context_drift", REPO / "dev_samples/instructions_hc_context_drift"),
    ("instructions", "oig_context_drift", REPO / "dev_samples/instructions_oig_context_drift"),
    ("hu_harm", "eval_ant_hh", REPO / "dev_samples/hu_ha_ant_hh"),
    ("hu_harm", "eval_ai_dilemmas", REPO / "dev_samples/hu_ha_ai_dilemmas"),
]


def prefetch_dev(dev_dir: Path, base_cache: Path) -> None:
    from synthetic_probe_data.kaggle_activations import (
        KaggleActivationSource,
        prefetch_dev_activations,
    )
    from synthetic_probe_data.retrain import _dev_activation_cache_path

    dev_files = sorted(dev_dir.glob("*.jsonl"))
    if not dev_files:
        raise SystemExit(f"{dev_dir} holds no *.jsonl splits")
    base_cache.mkdir(parents=True, exist_ok=True)
    path = _dev_activation_cache_path(base_cache, dev_files, MODEL_NAME, LAYER, COMBINE, CONVERT)
    if path.exists():
        print(f"  dev blob already cached: {path.name}", flush=True)
        return
    prefetch_dev_activations(
        path, dev_files,
        KaggleActivationSource(KAGGLE_OWNER, DEV_SLUG, DEV_FILE),
        model_name=MODEL_NAME, layer=LAYER, verbose=True,
    )


def main() -> None:
    from synthetic_probe_data.evaluation import evaluate_probe

    for concept_name in ("instructions", "hu_harm"):
        concept = CONCEPTS[concept_name]
        splits = [s for c, s, _ in ARMS if c == concept_name]
        print(f"\n=== {concept_name}: scoring probe_iter0 on {splits} "
              f"(fetches the eval blobs)", flush=True)
        df = evaluate_probe(
            concept.base_probe, concept.eval_dir, concept.eval_cache, splits=splits,
            max_samples=None, seed=SEED, combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT, kaggle_source=eval_source(),
        )
        print(df.to_string(), flush=True)

    for concept_name, split, dev_dir in ARMS:
        concept = dataclasses.replace(CONCEPTS[concept_name], dev_data=dev_dir)
        print(f"\n=== dev blob for {split} ({dev_dir.name})", flush=True)
        prefetch_dev(dev_dir, concept.base_cache)


if __name__ == "__main__":
    main()
