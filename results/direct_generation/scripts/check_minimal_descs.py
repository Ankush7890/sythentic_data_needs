#!/usr/bin/env python
"""Guard the shape-free arm: no minimal description may transmit SHAPE.

The arm's whole claim is that its prompts carry the SITUATION and the LABEL BOUNDARY and
nothing measured off the eval split. A hand-written description drifts back toward shape one
clause at a time ("a short question", "two turns", "about 50 characters"), so the rule is
checked mechanically rather than by reading.

Two checks, both against `split_specs`:

1. BANNED VOCABULARY — no digit, no length word, no turn/role word, no pairing word appears
   in any minimal description. Split names and label names are exempt (they are identity, not
   shape) and are blanked before the scan.
2. WHAT WAS DROPPED — every line of the full description that has no counterpart in the
   minimal one is printed, so the removals are reviewable rather than assumed.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/check_minimal_descs.py [--verbose]

Exit status is non-zero if any banned term survives.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from split_specs import MINIMAL_DESCS, SPLIT_SPECS  # noqa: E402

# Every term here names a property of a row's FORM rather than of the situation or of what a
# reply does. `character`/`word`/`sentence` catch the measured lengths, `turn`/`message`
# the role sequence, `paired` the pairing statement, `system` the copied
# system text. `\d` catches every count and median outright.
BANNED = [
    r"\d",
    r"\bcharacters?\b", r"\bwords?\b", r"\bsentences?\b", r"\bparagraphs?\b",
    # "turns" the noun, not the verb: "nothing much turns on it" is the concept itself.
    r"\bturns?\b(?!\s+on\b)", r"\bmessages?\b", r"\brows?\b", r"\bsystem\b",
    r"\bpaired?\b", r"\bpairing\b", r"\btwice\b", r"\bboth replies\b",
    r"\bshort\b", r"\blong\b", r"\bmedium\b", r"\bbrief\b",
    r"\bmedian\b", r"\baverag\w*\b", r"\bmeasured\b", r"\btypical\b",
    r"\bregister\b", r"\bformat\b",
]


def scan(text: str, exempt: list[str]) -> list[str]:
    """Banned terms surviving in `text`, with the exempt identity strings blanked first."""
    probe = text
    for term in exempt:
        probe = probe.replace(term, " ")
    hits = []
    for pat in BANNED:
        for m in re.finditer(pat, probe, flags=re.IGNORECASE):
            hits.append(f"{pat} -> {probe[max(0, m.start() - 30):m.end() + 30]!r}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="also print the dropped lines")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    from fit_base_plus_concept import CONCEPTS  # noqa: E402  (heavier import, only if needed)

    failures = 0
    for concept, splits in MINIMAL_DESCS.items():
        c = CONCEPTS[concept]
        for split, minimal in splits.items():
            exempt = [c.pos_label, c.neg_label, split, "THE SPLIT YOU ARE WRITING FOR"]
            hits = scan(minimal, exempt)
            status = "ok " if not hits else "FAIL"
            print(f"[{status}] {concept}/{split}: {len(minimal.split())} words, "
                  f"{len(hits)} banned term(s)")
            for h in hits:
                print(f"         {h}")
            failures += len(hits)
            if args.verbose:
                full = SPLIT_SPECS[concept][split]["desc"]
                kept = set(minimal.lower().split())
                for line in full.splitlines():
                    words = set(line.lower().split())
                    if line.strip() and not words <= kept:
                        print(f"         dropped: {line.strip()}")
    print(f"\n{failures} banned term(s) across {sum(len(v) for v in MINIMAL_DESCS.values())} "
          f"minimal descriptions")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
