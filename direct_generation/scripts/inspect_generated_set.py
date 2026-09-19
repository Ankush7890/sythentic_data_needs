#!/usr/bin/env python
"""Sanity-check a generated `{inputs, labels}` set before anything is fitted on it.

Five checks, in the order they have actually caught something:

1. **Label balance and uniqueness** — the fitting protocol draws n/2 per class, so a set
   short of 300 in either class cannot be drawn from at all.
2. **Contamination** — zero `inputs` hashes shared with the concept's eval splits, its dev
   splits, that generator's own 50-row base, or the unsteered 600-row set. A row in both
   train and eval is invisible in every downstream number.
3. **Turn shape** — how many rows are 2-turn vs longer. Under `--eval-description` the
   multi-turn kinds (context drift, a history-taking dialogue) can only be written with
   more than two messages, so a set with no long rows means the steering did not take.
4. **Token width** — rows over the probe's 1024-token activation cap are scored, and
   trained on, from their opening alone. The generator drops them, so this should be 0.
5. **Refusal-shaped rows in the positive class** — the check that caught a real failure:
   pointed at the "unjustified refusal" kind, the first steered llama70b set wrote the
   refusal branch under BOTH labels, so 42/300 of `assistant_follows_the_instruction` were
   flat refusals to harmless questions. A heuristic, not a judge — a `follows` reply may
   legitimately decline if the user asked it to — but a rate in the tens is a mislabelled
   set, not a handful of edge cases. Reported for both classes: on the negative side a
   refusal is usually correct, and the contrast is what makes the positive count readable.

Exits 1 if a check that is unambiguously fatal fails (short class, contamination), so a
runner can stop rather than fit on it; the heuristics only print.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/inspect_generated_set.py \\
        --concept instructions data/instructions_llama70b_evaldesc_600.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_base_plus_concept import CONCEPTS  # noqa: E402

PROBE_MODEL = "google/gemma-3-27b-it"
MAX_TOKENS = 1024

# Openers a refusal takes. Matched only at the START of the final assistant turn: a reply
# that mentions "I can't" halfway through is answering, not declining.
REFUSAL = re.compile(
    r"^(i'm not able|i am not able|i can't|i cannot|i can not|i'm unable|i am unable|"
    r"i'm sorry, but|i am sorry, but|i won't|i will not|i'm afraid|i'm not going to|"
    r"i don't have|unfortunately, i)",
    re.I,
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def hashes(path: Path) -> set[str]:
    return {hashlib.sha256(r["inputs"].encode()).hexdigest() for r in rows(path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    ap.add_argument("samples", type=Path)
    ap.add_argument("--min-per-class", type=int, default=300)
    ap.add_argument("--compare", type=Path, nargs="*", default=[],
                    help="extra sets to check for overlap (own base, unsteered counterpart)")
    args = ap.parse_args()

    concept = CONCEPTS[args.concept]
    rs = rows(args.samples)
    labels = collections.Counter(r["labels"] for r in rs)
    mine = {hashlib.sha256(r["inputs"].encode()).hexdigest() for r in rs}
    fatal = []

    print(f"{args.samples.name}: {len(rs)} rows, {len(mine)} unique, {dict(labels)}")
    for lab in (concept.pos_label, concept.neg_label):
        if labels.get(lab, 0) < args.min_per_class:
            fatal.append(f"only {labels.get(lab, 0)} {lab} rows (< {args.min_per_class})")

    overlaps = {}
    for d in (concept.eval_dir, concept.dev_data):
        hs: set[str] = set()
        for f in sorted(d.glob("*.jsonl")):
            hs |= hashes(f)
        # eval_sets/instructions and dev_samples/instructions have the same LAST path
        # component, so the parent goes in the key or one silently overwrites the other.
        overlaps[f"{d.parent.name}/{d.name}"] = len(mine & hs)
    for p in args.compare:
        if p.exists():
            overlaps[p.name] = len(mine & hashes(p))
    print("   overlap: " + ", ".join(f"{k}={v}" for k, v in overlaps.items()))
    # Only EVAL/DEV overlap is fatal — a training row that also sits in the set the probe
    # is scored on (or early-stops against) is invisible in every downstream number. The
    # `--compare` sets are sibling TRAINING sets, and two sets written by the same model
    # from the same prompt for the same kind legitimately collide on a few short generic
    # rows (8/600 between llama70b's refusal arm and its mixed set). That costs nothing:
    # each set trains its own probe, and the shared rows can only make the two probes more
    # alike, i.e. understate the difference being measured. Reported, not fatal.
    for k, v in overlaps.items():
        if v and ("eval_sets/" in k or "dev_samples/" in k):
            fatal.append(f"{v} rows shared with {k}")
        elif v:
            print(f"   note: {v} rows also appear in {k} (sibling training set, not fatal)")

    turns = collections.Counter(len(json.loads(r["inputs"])) for r in rs)
    print(f"   turns: {dict(sorted(turns.items()))}")

    try:
        from synthetic_probe_data.token_budget import count_tokens

        toks = [count_tokens(PROBE_MODEL, json.loads(r["inputs"]),
                             combine_consecutive_messages=True,
                             convert_tool_to_assistant=True) for r in rs]
        toks = [t for t in toks if t is not None]
        if toks:
            s = sorted(toks)
            print(f"   tokens: min {s[0]} / median {s[len(s) // 2]} / max {s[-1]}, "
                  f"over {MAX_TOKENS}: {sum(t > MAX_TOKENS for t in toks)}")
    except Exception as exc:  # noqa: BLE001 — a missing tokenizer must not fail the check
        print(f"   tokens: not counted ({type(exc).__name__})")

    # Rows the probe's own tokenizer cannot read at all. `count_tokens` reproduces
    # `tokenize_inputs` exactly, so a None here is a row whose chat template application
    # RAISES — it would take its extraction batch down, not merely score oddly. Fatal.
    try:
        from synthetic_probe_data.token_budget import count_tokens as _ct

        untok = sum(
            1 for r in rs
            if _ct(PROBE_MODEL, json.loads(r["inputs"]), combine_consecutive_messages=True,
                   convert_tool_to_assistant=True) is None
        )
        print(f"   untokenizable rows: {untok}")
        if untok:
            fatal.append(f"{untok} rows the probe's chat template cannot tokenize")
    except Exception as exc:  # noqa: BLE001
        print(f"   untokenizable rows: not checked ({type(exc).__name__})")

    refusals = collections.Counter(
        r["labels"] for r in rs
        if REFUSAL.match(json.loads(r["inputs"])[-1]["content"].strip())
    )
    print(f"   final reply opens as a refusal: "
          f"{refusals.get(concept.pos_label, 0)} of {labels.get(concept.pos_label, 0)} "
          f"{concept.pos_label}, "
          f"{refusals.get(concept.neg_label, 0)} of {labels.get(concept.neg_label, 0)} "
          f"{concept.neg_label}")

    if fatal:
        for f in fatal:
            print(f"   FATAL: {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
