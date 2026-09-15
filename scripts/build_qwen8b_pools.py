#!/usr/bin/env python
"""Build the 52 pooled sets the qwen8b size curve is drawn from, into .pool_work_qwen8b/.

Each pool is one 600-row generated set ∪ the 50-row base written by the SAME generator
(deduplicated, base row kept on a collision), so `subsample_curve_concept.py --no-base`
resamples every training row — nothing is held fixed across draws. Same construction as
`build_pooled_sets.py`, over a wider manifest:

    arm            highstakes                    instructions                  hu_harm
    general        4 generators                  4 generators                  4 generators
    eval desc      _evaldesc_ x4                 _evaldesc_ x4                 _evaldescshape_ x4
    split, shape   tgtnone x4 splits (deepseek)  tgtnone x6 splits (deepseek)  full x4 splits (llama70b)
    split, minimal tgtmin  x4 splits (deepseek)  tgtmin  x6 splits (deepseek)  minimal x4 splits (llama70b)

`_evaldescshape_` exists only for hu_harm; the other two concepts' `_evaldesc_` prompt already
carried the conversation shape, so it is their counterpart. `tgtnone` is the measured-shape
split prompt WITHOUT the dev few-shot anchor (`tgtshot` is excluded on purpose). hu_harm's
"full" arm is arm 1 as run (`refusal_v2`, not the contaminated v1); its minimal arm has one
prompt for ant_hh and balanced_refusal, drawn twice (`requestMinimal`, `requestMinimal2`).

The pool filename's arm token is `<family>-<split>` for the per-split arms, e.g.
`pool_hu_harm_llama70b_minimal-ant_hh_650.jsonl`.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/build_qwen8b_pools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from build_pooled_sets import ATTACKERS, build  # noqa: E402

OUT_DIR = REPO / ".pool_work_qwen8b"

EVALDESC_TOKEN = {"highstakes": "evaldesc", "instructions": "evaldesc",
                  "hu_harm": "evaldescshape"}
SPLITS = {
    "highstakes": ["anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced"],
    "instructions": ["anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
                     "hc_contradiction", "mm_substitution", "oig_context_drift"],
}
# hu_harm split -> (full-detail file tag, minimal file tag); all llama70b.
HU_HARM_SPLIT_TAGS = {
    "ant_hh": ("antHH", "requestMinimal"),
    "balanced_refusal": ("refusal_v2", "requestMinimal2"),
    "ai_dilemmas": ("aiDilemmas", "aiDilemmasMinimal"),
    "daily_dilemmas": ("dailyDilemmas", "dailyDilemmasMinimal"),
}


def manifest() -> list[tuple[str, str, str, Path]]:
    """(concept, generator, arm, source file) for every pool."""
    out = []
    for concept, tok in EVALDESC_TOKEN.items():
        for gen in ATTACKERS:
            out.append((concept, gen, "general", REPO / f"data/{concept}_{gen}_600.jsonl"))
            out.append((concept, gen, tok, REPO / f"data/{concept}_{gen}_{tok}_600.jsonl"))
    for concept, splits in SPLITS.items():
        for family in ("tgtnone", "tgtmin"):
            for split in splits:
                out.append((concept, "deepseekv4pro", f"{family}-{split}",
                            REPO / f"data/{concept}_deepseekv4pro_{family}_{split}_600.jsonl"))
    for split, (full, minimal) in HU_HARM_SPLIT_TAGS.items():
        for family, tag in (("full", full), ("minimal", minimal)):
            out.append(("hu_harm", "llama70b", f"{family}-{split}",
                        REPO / f"data/hu_harm_llama70b_{tag}_600.jsonl"))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where the pools go (default .pool_work_qwen8b; the llama1b curve "
                         "uses .pool_work_llama1b — same pools, the model plays no part)")
    out_dir = ap.parse_args().out_dir
    out_dir.mkdir(exist_ok=True)
    entries = manifest()
    missing = [src for *_, src in entries if not src.exists()]
    if missing:
        raise SystemExit("missing source sets:\n  " + "\n  ".join(map(str, missing)))
    for concept, gen, arm, src in entries:
        build(concept, gen, arm, src, REPO / f"data/{concept}_{gen}_50.jsonl", out_dir=out_dir)
    print(f"{len(entries)} pools in {out_dir}")


if __name__ == "__main__":
    main()
