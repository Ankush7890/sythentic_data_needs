#!/usr/bin/env python
"""Render the per-split `--kind` generation prompts, verbatim, without calling anything.

`generate_split_targeted.py` already carries `--dump-prompts`; the OTHER split-targeted
family — `generate_{concept}_dataset.py --eval-description --kind N`, which wrote every
`data/{concept}_{tag}_{split}_600.jsonl` set — did not. This is that dump, so both
families' prompts can be read side by side.

For each concept it walks the numbered kinds of that script's own EVAL_DATA_DESCRIPTION
(the same `eval_kinds()` split the generator uses, so the focus lines here cannot drift
from the ones that were sent) and renders `build_prompt` once per label.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/dump_persplit_prompts.py \\
        --out PER_SPLIT_KIND_PROMPTS.txt

Batch size is shown as 5 and the avoid-block as a placeholder; a real call carries up to
20 sampled openings. Nothing else differs from what was sent.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# kind index -> the eval split it was pinned to. For instructions each kind names its split
# in parentheses, so the mapping is read off the description itself; for high-stakes the
# kinds do not name splits and the correspondence is positional (checked against the split
# contents: open-ended chatbot -> anthropic_hh, clinical document -> mt, doctor-patient
# dialogue -> mts, tool-calling agent -> toolace). Both are exactly the mapping
# `run_persplit_gen.sh` used.
KIND_TO_SPLIT = {
    "instructions": [
        "anthropic_harmless_refusal",
        "bbq_substitution",
        "hc_context_drift",
        "hc_contradiction",
        "mm_substitution",
        "oig_context_drift",
    ],
    "highstakes": [
        "anthropic_hh_balanced",
        "mt_balanced",
        "mts_balanced",
        "toolace_balanced",
    ],
}

BATCH = 5
AVOID_PLACEHOLDER = ["<an already-used instruction/scenario>"]

HEADER = """\
PER-SPLIT `--kind` GENERATION PROMPTS — WHAT WAS ACTUALLY SENT
=============================================================================================

Twenty prompts: one per (eval split x class label) for the six steered instruction splits and
the four high-stakes splits. These wrote

    data/instructions_<tag>_<split>_600.jsonl
    data/highstakes_<tag>_<split>_600.jsonl

via `run_persplit_gen.sh`, i.e.

    .venv_claude/bin/python scripts/generate_<concept>_dataset.py \\
        --model <generator> --n-per-label 300 --batch-size 5 --concurrency 8 \\
        --call-budget-factor 8 --max-tokens 8192 --eval-description --kind <N> --out <file>

Each is sent as a SINGLE user message; there is no system prompt. Batch size is shown as 5
and the avoid-block as a placeholder; a real call carries up to 20 sampled openings, and the
generator alternates labels and calls until 300 rows per label are collected. Nothing else
differs from what was sent.

WHAT `--kind N` DOES. Without it the eval-data description is shown in full and calls
round-robin over its numbered kinds. With it EVERY call is pinned to kind N, so the whole
600-row set sits inside the one kind that corresponds to a single eval split. The description
itself is shown in full either way — that is the `eval_block` below — and the pin is the
`For THIS batch write every example in ONE kind only:` block near the end.

WHAT IT DOES NOT DO. The description names a split's SUBJECT MATTER and says nothing about
its SHAPE: no turn count, no pairing, no measured lengths. That is the gap
`scripts/generate_split_targeted.py` was written to close, and its prompts are in
SPLIT_TARGETED_PROMPTS.txt.

`oig_omission` has no kind in the instructions description (the red-team branch removed that
split before the text was written) and so has no prompt here; it stays the untouched control.

Rendered by:

    .venv_claude/bin/python scripts/dump_persplit_prompts.py --out PER_SPLIT_KIND_PROMPTS.txt

"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    args = ap.parse_args()

    with args.out.open("w", encoding="utf-8") as fh:
        fh.write(HEADER + "\n")
        for concept, splits in KIND_TO_SPLIT.items():
            mod = importlib.import_module(f"generate_{concept}_dataset")
            desc = mod.EVAL_DATA_DESCRIPTION
            kinds = mod.eval_kinds(desc)
            if len(kinds) != len(splits):
                raise SystemExit(
                    f"{concept}: {len(kinds)} kinds in EVAL_DATA_DESCRIPTION but "
                    f"{len(splits)} splits mapped")
            for i, (kind, split) in enumerate(zip(kinds, splits), start=1):
                for label in (mod.POS_LABEL, mod.NEG_LABEL):
                    fh.write("=" * 100 + f"\n{concept} / {split}   [--kind {i}]"
                             f"   [label: {label}]\n" + "=" * 100 + "\n")
                    fh.write(mod.build_prompt(label, args.batch_size,
                                              avoid=AVOID_PLACEHOLDER,
                                              eval_description=desc, focus_kind=kind))
                    fh.write("\n\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
