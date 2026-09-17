#!/usr/bin/env python
"""Extract and cache the gemma-3-27b activations of one generated set, without fitting.

`subsample_curve_concept.py` extracts lazily — the first draw that meets an uncached row
pays for it — which is fine when the set is already warm and wasteful when it is not: the
extraction is the long pole (~2-3 s/sample for a 600-row set) and it would sit inside the
first fit, serialised behind generation. This does the same work as the
`warm_sample_activation_cache` call inside `fit_base_plus_concept.py`, for one set, so a
set can be extracted the moment it is written while later sets are still being generated.

The cache is per conversation and content-keyed, so a set extracted here is a pure hit for
every later fit, and re-running this on a warm set is a no-op.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/warm_set_activations.py \\
        --concept instructions data/instructions_llama70b_evaldesc_600.jsonl
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    ap.add_argument("samples", type=Path, nargs="+")
    args = ap.parse_args()

    from agentic_redteam.cli import _free_gpu
    from agentic_redteam.retrain import warm_sample_activation_cache

    concept = CONCEPTS[args.concept]
    for path in args.samples:
        rows = load_rows(path, concept)
        t0 = time.time()
        print(f"warming {len(rows)} rows from {path.name}", flush=True)
        warm_sample_activation_cache(
            rows,
            base_probe_path=concept.base_probe,
            base_activation_cache_dir=concept.base_cache,
            combine_consecutive_messages=COMBINE,
            convert_tool_to_assistant=CONVERT,
            verbose=True,
        )
        print(f"{path.name}: warm in {time.time() - t0:.0f}s", flush=True)
        _free_gpu()


if __name__ == "__main__":
    main()
