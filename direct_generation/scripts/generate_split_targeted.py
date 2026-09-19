#!/usr/bin/env python
"""Generate a training set aimed at ONE eval split, from that split's measured shape.

The method is the one used for hu_harm on `human_harm_last`, applied to the instructions and
highstakes splits: a per-split description written from measurements of the split file, a
per-split few-shot anchor taken from `dev_samples/` (never `eval_sets/`), and a per-split
generation MODE — paired where the split is paired, unpaired where it is not.

WHAT THIS DOES THAT `--kind` DID NOT. The `--kind` arms pinned the shared eval-data
description to one of its numbered kinds. That description names a split's subject matter and
says nothing about its shape, so those sets reproduced the topic and missed the structure —
most importantly the pairing: every instruction split is a paired corpus (the same user turn
answered once each way) and every `--kind` arm was generated unpaired.

SHAPE IS ENFORCED, NOT REQUESTED. Each spec fixes the turn count, whether a system turn is
part of the row, and whether the row ends on the assistant or the user. A generated row that
does not match is dropped rather than repaired, so an arm's data has the split's shape by
construction and the log says how much was thrown away.

Output is the ordinary `{inputs, labels}` JSONL every other set on this branch uses.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/generate_split_targeted.py \\
        --concept instructions --split bbq_substitution --model deepseek/deepseek-v4-pro \\
        --n 600 --out data/instructions_deepseekv4pro_tgt_bbq_substitution_600.jsonl

    ... --dump-prompts prompts.txt     # render every split's prompt and exit, no API calls
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from synthetic_probe_data.json_extract import extract_json_values  # noqa: E402
from synthetic_probe_data.openrouter_client import (  # noqa: E402
    extract_openrouter_error,
    make_sync_client,
)
from fit_base_plus_concept import CONCEPTS  # noqa: E402
from split_specs import SPLIT_SPECS, minimal_spec  # noqa: E402

PROBE_MODEL = "google/gemma-3-27b-it"
AVOID_SAMPLE = 20

HEAD = {
    "instructions": (
        "You are helping build a labelled dataset for a classifier that detects whether an "
        "AI assistant's reply follows the user's instruction. The label is a property of the "
        "ASSISTANT's reply, never of the topic or of how reasonable the request was."
    ),
    "highstakes": (
        "You are helping build a labelled dataset for a classifier that detects whether a "
        "conversation with an AI assistant is high-stakes. The label is a property of the "
        "SITUATION — what is at risk if it goes wrong — never of whether the assistant "
        "refused, complied or was polite."
    ),
}


MINIMAL_DUMP_HEADER = """\
SHAPE-FREE ("MINIMAL") SPLIT-TARGETED PROMPTS — THE ARM-3 ABLATION
=============================================================================================

Twenty prompts: one per (eval split x class label) for the six instruction splits and the four
high-stakes splits. Every one carries the split's SITUATION and its LABEL BOUNDARY and nothing
measured off the split file. Removed, relative to SPLIT_TARGETED_PROMPTS.txt:

  * every character count, median, SHORT/LONG/MEDIUM and sentence count;
  * every turn count and role sequence, and with them the whole SHAPE (exact) block;
  * every pinned system turn -- which is not merely shape but a string copied verbatim out of
    the eval rows;
  * the pairing statement, and with it the paired MODE: a pairing instruction is shape read
    off the split, so every split here generates ONE LABEL PER CALL and the two labels never
    share an opening. That makes the removal structural rather than verbal;
  * the topic list, which is subject matter read off the split;
  * the dev-derived few-shot anchor (the tgtnone arm already dropped this one).

WHY. The same ablation on hu_harm (generator_experiment_1, 5c59c568) found that stripping
shape INVERTED which split a targeted set was good at -- the best score on the hardest split
came from the wrong prompt entirely. That says the shape cues, not the situation, were
carrying the targeted results. This is that ablation for instructions and high-stakes.

THE IRREDUCIBLE RESIDUAL. The output schema still asks for {"messages": [{"role", ...}]} and
the ask still says the row belongs to one class; that implies a conversation with roles ending
on an assistant turn. It applies to the unsteered generic control too, so it does not
differentiate the arms -- but it does mean mt_balanced, whose real rows carry NO assistant
turn at all, cannot be reproduced by this arm even in principle.

Shape is no longer ENFORCED either (`split_specs.minimal_spec` sets enforce_shape=False): the
drop filter would otherwise discard nearly every row, since the prompt no longer says what
shape to write. What survives is the tokenizability rule alone -- an optional leading system
turn, then user/assistant alternation, ending on the assistant.

Batch size is shown as 5 and the avoid-block as a placeholder; a real call carries up to 20
sampled openings. Nothing else differs from what is sent. Guarded by
`scripts/check_minimal_descs.py` (0 banned shape terms here against 205 in the full
descriptions). Rendered by:

    .venv_claude/bin/python scripts/generate_split_targeted.py --minimal \\
        --dump-prompts SPLIT_TARGETED_PROMPTS_MINIMAL.txt

"""


def anchors(concept: str, split: str, spec: dict) -> list[dict]:
    """Few-shot rows for this split, read from `dev_samples/`.

    Never `eval_sets/`: these rows go into the prompt and the generated data becomes training
    data, so anchoring on eval would copy test content into the training set. Dev is the
    validation set, a far milder dependency, and is disjoint from eval.
    """
    c = CONCEPTS[concept]
    path = c.dev_data / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"no dev counterpart for {split} at {path}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    for r in rows:
        r["msgs"] = json.loads(r["inputs"])
    if spec["mode"] == "paired":
        # A dev user turn carrying BOTH labels is a ready-made paired anchor.
        by_user: dict[str, list[dict]] = {}
        for r in rows:
            u = next((m["content"] for m in r["msgs"] if m["role"] == "user"), "")
            by_user.setdefault(u, []).append(r)
        for pair in by_user.values():
            if len({r["labels"] for r in pair}) == 2:
                return sorted(pair, key=lambda r: r["labels"] != c.pos_label)[:2]
        raise SystemExit(f"{split}: dev holds no paired anchor")
    out = []
    for lab in (c.pos_label, c.neg_label):
        match = next((r for r in rows if r["labels"] == lab), None)
        if match is None:
            raise SystemExit(f"{split}: dev holds no {lab} row")
        out.append(match)
    return out


def _render(msgs: list[dict]) -> str:
    return json.dumps([{"role": m["role"], "content": m["content"]} for m in msgs],
                      ensure_ascii=False)


def build_prompt(concept: str, split: str, spec: dict, batch: int, avoid: list[str],
                 label: str | None, shots: list[dict]) -> str:
    """The prompt for one call. `label` is None in paired mode (both labels come at once)."""
    c = CONCEPTS[concept]
    # The shape-free arm carries no SHAPE block at all: turn count, system turn and
    # ending turn are each measured off the eval split, which is exactly what it ablates.
    shape_block = ""
    if spec.get("enforce_shape", True):
        shape = [f"- The row is {spec['turns']} messages"
                 + (" including the system turn" if spec["system"] else "")
                 + f", ending on the {spec['ends_on']} turn."]
        if spec["system"] and spec.get("system_text"):
            shape.append(f'- The system turn is exactly: "{spec["system_text"]}"')
        elif spec["system"]:
            shape.append("- Write the system turn yourself; it is part of the row.")
        shape_block = "SHAPE (exact):\n" + "\n".join(shape) + "\n\n"
    # A topic list is subject matter read off the split, so the shape-free arm has none.
    topic_block = (f"Vary the subject matter widely across: {spec['topics']}.\n\n"
                   if spec["topics"] else "Vary the subject matter widely.\n\n")

    # The no-shot variant drops the anchor block entirely and leans on the measured
    # description alone. It is the control for "how much of a targeted set's quality is the
    # DESCRIPTION and how much is seeing two real rows" — the two differ by this block only.
    shot_block = ""
    if shots:
        ex = "\n".join(f'{r["labels"]}: {_render(r["msgs"])}' for r in shots)
        shot_block = (
            f"Here {'are two real rows' if len(shots) == 2 else 'is a real row'} from the "
            f"validation half of this split, for register and shape only — do not reuse "
            f"their content:\n{ex}\n\n"
        )
    avoid_block = ""
    if avoid:
        avoid_block = ("\n\nDo NOT reuse or lightly reword these already-used openings; pick "
                       "clearly different ones:\n" + "\n".join(f"- {a}" for a in avoid))

    if spec["mode"] == "paired":
        ask = (
            f"Write {batch} NEW and DIVERSE items. Each item is ONE conversation written "
            f"TWICE — the shared opening, then the two possible final assistant turns:\n"
            f'{{"prefix": [{{"role": "user", "content": "..."}}, ...], '
            f'"follows": "...", "not_follows": "..."}}\n'
            f'"prefix" holds every message BEFORE the final assistant turn '
            f'({spec["turns"] - 1} of them). "follows" is the final assistant turn for the '
            f'{c.pos_label} row and "not_follows" the final assistant turn for the '
            f'{c.neg_label} row. Everything before it is shared, which is what makes the '
            f'pair a pair.'
        )
    else:
        ask = (
            f"Write {batch} NEW and DIVERSE '{label}' rows. Each is one conversation:\n"
            f'{{"messages": [{{"role": "...", "content": "..."}}, ...]}}\n'
            f"Every row must belong to the '{label}' class."
        )

    return (
        f"{HEAD[concept]}\n\n{spec['desc']}\n\n{shape_block}{topic_block}"
        f"{shot_block}{ask}{avoid_block}\n\n"
        f"Respond with ONLY a JSON array of those objects. No prose, no markdown fences."
    )


def _accept(value):
    def ok(d):
        return isinstance(d, dict) and (
            ("prefix" in d and "follows" in d and "not_follows" in d)
            or isinstance(d.get("messages"), list)
        )
    if isinstance(value, dict):
        if ok(value):
            return [value]
        inner = value.get("items") or value.get("examples") or value.get("conversations")
        return _accept(inner) if isinstance(inner, list) else None
    if isinstance(value, list):
        got = [v for v in value if ok(v)]
        return got or None
    return None


def to_rows(item: dict, spec: dict, concept: str, label: str | None) -> list[dict]:
    """One generated item -> the row(s) it yields, or [] if it does not match the shape."""
    c = CONCEPTS[concept]

    def clean(msgs):
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                return None
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", "")).strip()
            if role not in ("system", "user", "assistant") or not content:
                return None
            out.append({"role": role, "content": content})
        return out

    def shaped(msgs) -> bool:
        # The shape IS the split, so a row that misses it is dropped, not repaired. Also the
        # gemma chat template raises on anything but user/assistant alternation after an
        # optional leading system turn, so this doubles as the extractability check.
        if msgs is None or not msgs:
            return False
        if any(m["role"] == "system" for m in msgs[1:]):
            return False
        has_system = msgs[0]["role"] == "system"
        body = msgs[1:] if has_system else msgs
        if not body or body[0]["role"] != "user":
            return False
        expect = ["user", "assistant"] * len(body)
        if [m["role"] for m in body] != expect[:len(body)]:
            return False
        if not spec.get("enforce_shape", True):
            # The shape-free arm never told the model what shape to write, so the only rule
            # left is the one every row must satisfy to be tokenized at all (checked above)
            # plus the concept's own requirement that the row end on the assistant turn.
            # Enforcing turn counts here would discard nearly everything the arm generates.
            return msgs[-1]["role"] == "assistant"
        if len(msgs) != spec["turns"]:
            return False
        if spec["system"] != has_system:
            return False
        return msgs[-1]["role"] == spec["ends_on"]

    if spec["mode"] == "paired":
        prefix = clean(item.get("prefix") or [])
        if prefix is None:
            return []
        rows = []
        for key, lab in (("follows", c.pos_label), ("not_follows", c.neg_label)):
            final = str(item.get(key, "")).strip()
            if not final:
                return []
            msgs = prefix + [{"role": "assistant", "content": final}]
            if not shaped(msgs):
                return []
            rows.append({"inputs": json.dumps(msgs, ensure_ascii=False), "labels": lab})
        return rows

    msgs = clean(item.get("messages") or [])
    if not shaped(msgs):
        return []
    return [{"inputs": json.dumps(msgs, ensure_ascii=False), "labels": label}]


def opening(row: dict) -> str:
    msgs = json.loads(row["inputs"])
    return next((m["content"] for m in msgs if m["role"] == "user"), "")[:200].lower()


def _call(client, model, prompt, temperature, max_tokens, tag, tries=4):
    delay = 5.0
    for attempt in range(1, tries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 — one dead call must not kill the run
            print(f"  [warn] {tag} attempt {attempt}: {type(exc).__name__}: {str(exc)[:160]}",
                  file=sys.stderr)
            if attempt == tries:
                return []
            time.sleep(delay)
            delay *= 2
            continue
        if not getattr(resp, "choices", None):
            print(f"  [warn] {tag}: {extract_openrouter_error(resp) or 'no choices'}",
                  file=sys.stderr)
            return []
        text = resp.choices[0].message.content or ""
        found = extract_json_values(text, _accept)
        items = [i for g in found for i in g]
        if items:
            return items
        print(f"  [warn] {tag}: no items parsed from {text[:140]!r}", file=sys.stderr)
        return []
    return []


def generate(client, model, concept, split, spec, n, batch, concurrency, temperature,
             max_tokens, budget, seed, budget_factor, use_shots=True):
    c = CONCEPTS[concept]
    shots = anchors(concept, split, spec) if use_shots else []
    rows: list[dict] = []
    seen: set[str] = set()
    counts = {c.pos_label: 0, c.neg_label: 0}
    lock = threading.Lock()
    rng = random.Random(seed)
    dropped = {"shape": 0, "long": 0, "dup": 0}
    labels = [None] if spec["mode"] == "paired" else [c.pos_label, c.neg_label]
    per_label = n if spec["mode"] == "paired" else n // 2
    calls = 0
    max_calls = budget_factor * (n // max(batch, 1) + 2)

    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        while len(rows) < n and calls < max_calls:
            want = [lab for lab in labels
                    if (len(rows) if lab is None else counts[lab]) < per_label]
            if not want:
                break
            futures = []
            for j in range(min(concurrency, max_calls - calls)):
                lab = want[j % len(want)]
                pool_avoid = sorted(seen)
                sample = (rng.sample(pool_avoid, AVOID_SAMPLE)
                          if len(pool_avoid) > AVOID_SAMPLE else pool_avoid)
                prompt = build_prompt(concept, split, spec, batch, sorted(sample), lab, shots)
                futures.append((lab, pool.submit(_call, client, model, prompt, temperature,
                                                 max_tokens, f"{split} call{calls + j}")))
            calls += len(futures)
            for lab, fut in futures:
                for item in fut.result():
                    got = to_rows(item, spec, concept, lab)
                    if not got:
                        dropped["shape"] += 1
                        continue
                    if budget is not None and any(
                            budget.overage(json.loads(r["inputs"])) is not None for r in got):
                        dropped["long"] += 1
                        continue
                    key = opening(got[0])
                    with lock:
                        if not key or key in seen or len(rows) >= n:
                            dropped["dup"] += 1
                            continue
                        if any(counts[r["labels"]] >= per_label for r in got) and lab is not None:
                            continue
                        seen.add(key)
                        for r in got:
                            counts[r["labels"]] += 1
                        rows.extend(got)
            print(f"  {split}: {len(rows)}/{n} rows "
                  f"({counts[c.pos_label]}/{counts[c.neg_label]}) after {calls} calls; "
                  f"dropped shape={dropped['shape']} long={dropped['long']} dup={dropped['dup']}",
                  file=sys.stderr)
    return rows[:n], dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concept", choices=sorted(SPLIT_SPECS))
    ap.add_argument("--split")
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--call-budget-factor", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-sample-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-shots", action="store_true",
                    help="omit the dev-derived few-shot anchor; the measured description is "
                         "then the only thing describing the split")
    ap.add_argument("--minimal", action="store_true",
                    help="the shape-free arm: the split's MINIMAL description (situation plus "
                         "one sentence per label), no anchor, no topic list, no shape block, "
                         "and one label per call. See split_specs.minimal_spec.")
    ap.add_argument("--dump-prompts", type=Path,
                    help="render every split's prompt to this file and exit (no API calls); "
                         "with --minimal, renders the shape-free arm instead")
    args = ap.parse_args()

    if args.dump_prompts:
        with args.dump_prompts.open("w", encoding="utf-8") as fh:
            if args.minimal:
                fh.write(MINIMAL_DUMP_HEADER + "\n")
            for concept, splits in SPLIT_SPECS.items():
                c = CONCEPTS[concept]
                for split in splits:
                    if args.minimal:
                        spec = minimal_spec(concept, split)
                        variants = [("MINIMAL: no shape, no anchor, no topics", [])]
                    else:
                        spec = splits[split]
                        variants = [("WITH few-shot anchor", anchors(concept, split, spec)),
                                    ("NO few-shot anchor", [])]
                    labs = [None] if spec["mode"] == "paired" else [c.pos_label, c.neg_label]
                    for variant, sh in variants:
                        for lab in labs:
                            fh.write("=" * 100 + f"\n{concept} / {split}   [{variant}]"
                                     + (f"   [label: {lab}]" if lab else "   [paired]")
                                     + "\n" + "=" * 100 + "\n")
                            fh.write(build_prompt(concept, split, spec, 5,
                                                  ["<an already-used opening>"], lab, sh))
                            fh.write("\n\n")
        print(f"wrote {args.dump_prompts}")
        return

    if not (args.concept and args.split and args.out):
        ap.error("--concept, --split and --out are required unless --dump-prompts")
    if args.split not in SPLIT_SPECS[args.concept]:
        ap.error(f"no spec for {args.concept}/{args.split}")
    spec = (minimal_spec(args.concept, args.split) if args.minimal
            else SPLIT_SPECS[args.concept][args.split])

    budget = None
    if args.max_sample_tokens > 0:
        from synthetic_probe_data.token_budget import TokenBudget
        budget = TokenBudget(PROBE_MODEL, args.max_sample_tokens,
                             combine_consecutive_messages=True, convert_tool_to_assistant=True)
        budget.warmup()

    rows, dropped = generate(make_sync_client(), args.model, args.concept, args.split, spec,
                             args.n, args.batch_size, args.concurrency, args.temperature,
                             args.max_tokens, budget, args.seed, args.call_budget_factor,
                             use_shots=not (args.no_shots or args.minimal))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}  (dropped {dropped})", file=sys.stderr)


if __name__ == "__main__":
    main()
