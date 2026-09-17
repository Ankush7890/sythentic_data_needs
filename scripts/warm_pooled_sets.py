#!/usr/bin/env python
"""Extract every uncached activation across ALL of one concept's pooled sets, in ONE load.

`warm_set_activations.py` takes a set at a time and calls `warm_sample_activation_cache`
once per file, which means one 27B load per file. The pooled sets share rows — the same
50-row own base sits in both that generator's `_general_` and steered pool — and there are
eight files per concept, so doing it per file pays for up to eight loads and re-extracts the
shared base rows. This collects the union of uncached conversations across every pool first
and extracts them in a single pass.

Deduplication is on the per-sample cache PATH, i.e. on the transformed conversation content
that keys the cache, not on the raw row, so two rows that differ only in provenance columns
collapse to one extraction exactly as the cache itself would.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/warm_pooled_sets.py --concept hu_harm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import COMBINE, CONCEPTS, CONVERT, load_rows  # noqa: E402

POOL_DIR = REPO / ".pool_work"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    ap.add_argument("--dry-run", action="store_true", help="report the count and exit")
    args = ap.parse_args()

    from agentic_redteam.cli import _free_gpu
    from agentic_redteam.retrain import (
        _apply_message_transforms,
        _cpu_unpickle,
        _sample_activation_cache_path,
        samples_to_dataset,
        warm_sample_activation_cache,
    )

    concept = CONCEPTS[args.concept]
    with concept.base_probe.open("rb") as fh:
        probe = _cpu_unpickle(fh)
    model_name, layer = str(probe.model_name), int(probe.layer)
    pos, neg = probe.pos_class_label, probe.neg_class_label

    pools = sorted(POOL_DIR.glob(f"pool_{args.concept}_*.jsonl"))
    if not pools:
        raise SystemExit(f"no pooled sets for {args.concept} in {POOL_DIR}")

    # Union the uncached rows across every pool, keyed on the cache path so a row shared
    # between two pools (every own-base row is) is extracted once.
    todo: dict[str, dict] = {}
    for path in pools:
        rows = load_rows(path, concept)
        ds = _apply_message_transforms(samples_to_dataset(rows, pos, neg), COMBINE, CONVERT)
        n_miss = 0
        for row, msgs in zip(rows, ds.inputs):
            key = str(_sample_activation_cache_path(
                concept.base_cache, msgs, model_name, layer, COMBINE, CONVERT))
            if Path(key).exists():
                continue
            n_miss += 1
            todo.setdefault(key, row)
        print(f"  {path.name:52s} {len(rows):4d} rows, {n_miss:4d} uncached", flush=True)

    print(f"{args.concept}: {len(todo)} unique conversations to extract", flush=True)
    if args.dry_run or not todo:
        return

    t0 = time.time()
    warm_sample_activation_cache(
        list(todo.values()),
        base_probe_path=concept.base_probe,
        base_activation_cache_dir=concept.base_cache,
        combine_consecutive_messages=COMBINE,
        convert_tool_to_assistant=CONVERT,
        verbose=True,
    )
    print(f"{args.concept}: warm in {time.time() - t0:.0f}s", flush=True)
    _free_gpu()


if __name__ == "__main__":
    main()
