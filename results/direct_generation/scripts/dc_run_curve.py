#!/usr/bin/env python
"""Run ``subsample_curve_concept.py`` with the eval scored on ONE split.

WHY. ``evaluate_probe`` auto-discovers every ``*.jsonl`` in the concept's eval dir, so
every fit scores every split of the concept. That is right for the mixed arm — one fit,
read by all of them — and pure waste for the kind-only and leave-one-kind-out arms, which
are fit per split and only ever read their own column. On high-stakes it is the whole
cost: the four eval blobs are 47 GB and a fit that needs one of them still reads all four,
which is why a high-stakes fit measures ~240 s against ~12 s for the other two concepts.

WHAT IT DOES NOT DO. It does not edit, copy or reimplement the harness: it imports
``subsample_curve_concept`` and calls its ``main()``, having replaced the concept's
``eval_dir`` with a directory holding a symlink to the one split. Same fit, same draws,
same seeds, same optimiser regime, same activation caches; only the set of columns the
CSV carries is smaller. Because the harness refuses to append rows under a header it
cannot fill, a restricted run writes its own ``--out`` file, and
``direction_count.py --stage analyse`` reads every ``dc_curves_<concept>*.csv``.

    python scripts/dc_run_curve.py --concept highstakes --eval-split mts_balanced \\
        .dc_work/dc_highstakes_gptoss_mts_balanced_kind.jsonl --no-base --sizes 30 ...
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

WORK = REPO / ".dc_work"


def restricted_eval_dir(concept, split: str) -> Path:
    """A directory holding exactly one split, symlinked to the real file.

    A symlink rather than a copy so the split JSONL stays single-sourced — a copy could
    drift from ``eval_sets/`` and no downstream number would show it.
    """
    src = concept.eval_dir / f"{split}.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} does not exist")
    d = WORK / f"eval_{concept.name}_{split}"
    d.mkdir(parents=True, exist_ok=True)
    link = d / src.name
    if not link.exists():
        link.symlink_to(src.resolve())
    for stale in d.glob("*.jsonl"):
        if stale.name != src.name:
            stale.unlink()
    return d


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--eval-split" not in argv:
        raise SystemExit("--eval-split is required (otherwise call the harness directly)")
    i = argv.index("--eval-split")
    split = argv[i + 1]
    del argv[i : i + 2]

    from fit_base_plus_concept import CONCEPTS
    import subsample_curve_concept as harness

    name = argv[argv.index("--concept") + 1]
    concept = CONCEPTS[name]
    CONCEPTS[name] = dataclasses.replace(
        concept, eval_dir=restricted_eval_dir(concept, split))
    sys.argv = [str(REPO / "scripts" / "subsample_curve_concept.py")] + argv
    harness.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
