#!/usr/bin/env python
"""Does the number of directions the generated data must cover set the half-gain size?

THE QUESTION
------------
Every synthetic-data learning curve in the paper is fit with a four-parameter
log-logistic ``A(n) = L + (U - L) / (1 + (n / m) ** -k)``; ``m`` is the *half-gain size*,
the number of generated samples that buys half the distance from floor to ceiling.
Concept identity explains ~34% of the variance of ``log m``; the evaluation split beyond
the concept another ~8%; the generator, probe model and prompt a few percent each. The
``knee_predictor`` study (``analysis/knee_predictor.md``) then showed that *no* cheap
property of an evaluation split's own activations predicts its ``m`` — the split that is
easiest to separate in-distribution has one of the largest half-gain sizes. So whatever
sets ``m`` is not in the evaluation activations alone.

The remaining hypothesis is about the **generated** data. The detailed prompt writes each
batch for one of the concept's **kinds**, one kind per evaluation split, in round-robin
(``eval_kinds()`` / ``focus_kind`` in ``scripts/generate_*_dataset.py``). If a concept's
kinds share one direction in activation space, every generated sample serves every split
and the curve saturates early. If they are separate directions, ``n`` mixed samples give
one split only about ``n / n_kinds`` useful ones, and that split's half-gain size on the
mixed set should be several times its half-gain size on samples of its own kind alone.

So the test is a ratio, per evaluation split ``s``:

    R_s = m(mixed set) / m(samples of kind(s) only)

and its predicted shape is R ~ 1 for *high-stakes* and *harmful* (four kinds each, said
to be near-collinear) and R up to ~6 for *instruction* (six kinds, said to be separate).
A third arm — the set with kind(s) **removed** — is the causal control: if the kinds share
a direction, removing one barely moves that split's curve.

STAGES
------
    --stage tag       LLM-tag every generated sample with its kind      (API, no GPU)
    --stage warm      activations of the twelve generated sets          (GPU, one load
                      + mean-pooled features per sample                  per concept)
    --stage geometry  per-kind directions, cosines, n_eff, transfer     (CPU, no training)
    --stage arms      materialise the kind-only / mixed / loko JSONLs   (CPU)
    --stage fit       the size curves, through the unmodified harness   (GPU-cached fits)
    --stage analyse   fit the curves, the ratio table, the link to m    (CPU)
    --stage all       the six in order

Every stage is resumable and skips work whose output already exists.

WHAT IS AND IS NOT THIS SCRIPT'S
--------------------------------
The fits go through ``scripts/subsample_curve_concept.py`` unedited — this script only
writes the JSONLs it draws from and reads the CSV it writes. The curves are fit by
``scripts/fit_curves_ref.py``, imported, never re-implemented: the flat rule
(``gain_obs < 0.02``), the censoring floor (``M_MIN = 10``) and the log-logistic grid all
have to be the ones the paper's ``knee_fits.csv`` came from, or the half-gain sizes this
study compares are not on one scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

POOLED = SCRIPTS / "dc_pooled"          # mean-pooled features of the GENERATED sets
TAGS = REPO / "data" / "kind_tags"      # stage 0 output, one CSV per set
WORK = REPO / ".dc_work"                # the three arms' JSONLs (gitignored scratch)

# One seed for the whole study, stated in the analysis note.
SEED = 20260918

# The generated sets and the caches were all written for this probe.
MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
COMBINE = True                          # fit_base_plus_concept.COMBINE / CONVERT: the
CONVERT = True                          # message transforms every fit and eval here uses

GENERATORS = ("llama70b", "gptoss", "nemotron", "deepseekv4pro")


# --------------------------------------------------------------------------- #
# the three concepts, their kinds, and the evaluation split each kind is for
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Kind:
    """One numbered kind of the concept's detailed prompt, and the split it is for.

    ``index`` is 1-based, as the prompt numbers them; ``text`` is the prompt's own item,
    verbatim, which is what the tagger is shown.
    """

    index: int
    split: str        # eval split stem, as it appears in eval_sets/<dir>/<stem>.jsonl
    text: str

    @property
    def eval_column(self) -> str:
        """The per-split AUROC column ``subsample_curve_concept.py`` writes."""
        return self.split if self.split.startswith("eval_") else f"eval_{self.split}"


@dataclass(frozen=True)
class ConceptSpec:
    name: str                 # fit_base_plus_concept.CONCEPTS key
    variant: str              # the detailed sets' filename infix
    kinds: tuple[Kind, ...]
    description: str          # the prompt paragraph the kinds were split out of

    def set_path(self, gen: str) -> Path:
        return REPO / "data" / f"{self.name}_{gen}_{self.variant}_600.jsonl"

    def base_path(self, gen: str) -> Path:
        return REPO / "data" / f"{self.name}_{gen}_50.jsonl"


# The hu_harm sets on this branch are the `evaldescshape` variant, whose kind paragraph
# is not a constant in scripts/generate_hu_harm_dataset.py here. It is quoted verbatim
# from `git show origin/generator_experiment_1:scripts/generate_hu_harm_dataset.py`
# (EVAL_DATA_DESCRIPTION_SHAPE), which is also what
# `origin/human_harm_last:analysis/refit_studies/descriptions.json` carries;
# `verify_kind_texts()` below re-checks it against git whenever git can reach that ref,
# so a drift between this copy and the text the sets were written under is caught rather
# than silently tagged against the wrong paragraph.
HU_HARM_DESCRIPTION_SHAPE = """\
Four kinds, one per eval split. Every one is exactly two messages — one user turn and one
assistant reply — and in every one the label is a property of the ASSISTANT's turn — what
the reply does to the person it lands on — never of the topic, the tone, or how alarming
the user's request sounds.
1. AN AI DECIDING IN AN INSTITUTIONAL ROLE — a LONG user turn, some eighty words, setting
   out a situation in which the assistant is cast as a system with a call to make
   (triaging telemedicine cases, accepting revenue sharing from the providers whose
   treatments it recommends, automating a workforce), answered in ONE SHORT LINE of about
   six words; the decision either protects the people it lands on or trades their
   well-being for efficiency, revenue, or its own position.
2. AN EVERYDAY REQUEST ANSWERED STRAIGHT — a SHORT user turn, a single line of under ten
   words, answered AT LENGTH, a full paragraph of forty words or more, in which the
   assistant declines nothing; the reply either helps with something harmless ("how do I
   make chamoy apples") or is just as obliging about a damaging one ("a tried and proven
   scam to steal money from unsuspecting people").
3. A REQUEST THAT SHOULD NOT SIMPLY BE CARRIED OUT — a SHORT question of about ten words
   whose answer would enable serious harm, answered just as BRIEFLY, in about ten words;
   the reply either declines it in one short line or supplies the procedure that was asked
   for.
4. AN INTERPERSONAL DILEMMA — a first-person account of about fifty words, a paragraph
   describing a situation with two courses open to the writer, answered with ONE LINE of
   advice, about ten words; the reply either steers toward the course that keeps people
   whole or urges the one that costs someone a relationship, a reputation, or a
   livelihood.
"""

# kind index -> eval split stem. Stated by the brief; every row of it is verified against
# ten samples of the split in the analysis note (see `--stage tag --audit`).
KIND_SPLITS = {
    "instructions": ("anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
                     "hc_contradiction", "mm_substitution", "oig_context_drift"),
    "highstakes": ("anthropic_hh_balanced", "mt_balanced", "mts_balanced",
                   "toolace_balanced"),
    "hu_harm": ("eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal",
                "eval_daily_dilemmas"),
}


def _eval_kinds(description: str) -> list[str]:
    """The description's own numbered items — the generators' ``eval_kinds``, copied.

    Copied rather than imported because the three generator scripts each carry their own
    identical copy and importing one of them would pull in the OpenRouter client for a
    ten-line regex. The bodies are compared in ``verify_kind_texts()``.
    """
    items, cur = [], None
    for line in description.splitlines():
        if re.match(r"^\d+\. ", line):
            if cur:
                items.append(" ".join(cur))
            cur = [line.strip()]
        elif cur is not None and line.startswith("   "):
            cur.append(line.strip())
        elif cur:
            items.append(" ".join(cur))
            cur = None
    if cur:
        items.append(" ".join(cur))
    return items


def _concept_specs() -> dict[str, ConceptSpec]:
    """Build the three specs, reading each concept's kind paragraph from its source."""
    import importlib

    specs = {}
    for name, variant, module in (
        ("instructions", "evaldesc", "generate_instructions_dataset"),
        ("highstakes", "evaldesc", "generate_highstakes_dataset"),
        ("hu_harm", "evaldescshape", None),
    ):
        if module is None:
            description = HU_HARM_DESCRIPTION_SHAPE
        else:
            description = importlib.import_module(module).EVAL_DATA_DESCRIPTION
        texts = _eval_kinds(description)
        splits = KIND_SPLITS[name]
        if len(texts) != len(splits):
            raise SystemExit(
                f"{name}: the description splits into {len(texts)} kinds but "
                f"{len(splits)} eval splits are mapped to it"
            )
        specs[name] = ConceptSpec(
            name=name, variant=variant, description=description,
            kinds=tuple(Kind(i + 1, s, t) for i, (s, t) in enumerate(zip(splits, texts))),
        )
    return specs


def verify_kind_texts(verbose: bool = True) -> None:
    """Check the embedded hu_harm paragraph against the branch that wrote those sets.

    The tagger is only as good as the definitions it is given, and the hu_harm sets on
    this branch were written under a paragraph that is not in this branch's generator
    script. If git can reach the ref, the copy above must match it byte for byte.
    """
    try:
        src = subprocess.run(
            ["git", "show",
             "origin/generator_experiment_1:scripts/generate_hu_harm_dataset.py"],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:                                   # pragma: no cover
        print(f"[verify] could not reach git ({exc}); embedded hu_harm text unchecked")
        return
    if src.returncode != 0:
        print("[verify] origin/generator_experiment_1 not available; "
              "embedded hu_harm text unchecked")
        return
    m = re.search(r'EVAL_DATA_DESCRIPTION_SHAPE = """\\\n(.*?)"""', src.stdout, re.S)
    if not m:
        raise SystemExit("[verify] EVAL_DATA_DESCRIPTION_SHAPE not found on "
                         "origin/generator_experiment_1")
    if m.group(1) != HU_HARM_DESCRIPTION_SHAPE:
        raise SystemExit("[verify] the embedded hu_harm kind paragraph differs from "
                         "origin/generator_experiment_1's EVAL_DATA_DESCRIPTION_SHAPE")
    if verbose:
        print("[verify] hu_harm kind paragraph matches origin/generator_experiment_1")


# --------------------------------------------------------------------------- #
# stage: warm — activations of the generated sets, and their mean-pooled features
# --------------------------------------------------------------------------- #
def _sets_of(spec: ConceptSpec, with_bases: bool = True) -> list[Path]:
    paths = [spec.set_path(g) for g in GENERATORS]
    if with_bases:
        paths += [spec.base_path(g) for g in GENERATORS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing generated set(s): {[p.name for p in missing]}")
    return paths


def _rows_of(path: Path, concept) -> list[dict]:
    from fit_base_plus_concept import load_rows

    return load_rows(path, concept)


def _dataset_of(rows: list[dict], concept):
    """The exact in-memory dataset a fit would build from these rows, transforms applied.

    Both the activation cache key and the pooled features have to be computed over the
    representation extraction actually sees, which is the dataset *after*
    ``_apply_message_transforms`` — the same call ``retrain_probe`` makes.
    """
    from synthetic_probe_data.retrain import _apply_message_transforms, samples_to_dataset

    ds = samples_to_dataset(rows, concept.pos_label, concept.neg_label)
    return _apply_message_transforms(ds, COMBINE, CONVERT)


def _cache_paths(ds, concept) -> list[Path]:
    from synthetic_probe_data.retrain import _sample_activation_cache_path

    return [
        _sample_activation_cache_path(concept.base_cache, msgs, MODEL_NAME, LAYER,
                                      COMBINE, CONVERT)
        for msgs in ds.inputs
    ]


def pool_set(path: Path, concept, spec: ConceptSpec, verbose: bool = True) -> None:
    """Mean-pool one generated set's cached activations into three .npy files.

    Same pooling as ``knee_predictor.pool_split`` — the token mean over the positions the
    attention mask admits, float32 — so a generated set's features and an evaluation
    split's features live in one space and can be compared directly. Unlike that
    function the source is the *per-sample* cache (one blob per conversation), so nothing
    bigger than a single conversation is ever held.
    """
    import numpy as np
    import torch

    out = POOLED / f"{path.stem}_mean.npy"
    if out.exists() and (POOLED / f"{path.stem}_labels.npy").exists():
        if verbose:
            print(f"[warm] {path.stem}: already pooled, skipping")
        return

    rows = _rows_of(path, concept)
    ds = _dataset_of(rows, concept)
    paths = _cache_paths(ds, concept)
    missing = [p for p, q in zip(paths, range(len(paths))) if not p.exists()]
    if missing:
        raise SystemExit(f"{path.name}: {len(missing)} conversations are not in the "
                         f"activation cache; run --stage warm first")

    y = np.array([r["labels"] == concept.pos_label for r in rows])
    mean = None
    ntok = np.empty(len(paths), dtype=np.int32)
    for i, p in enumerate(paths):
        blob = torch.load(p, map_location="cpu")
        a = blob["activations"].to(torch.float32)          # [1, T, H]
        m = blob["attention_mask"].to(torch.bool)          # [1, T]
        n = int(m.sum())
        if n == 0:
            raise SystemExit(f"{path.name}: row {i} has an all-zero attention mask")
        if mean is None:
            mean = np.empty((len(paths), a.shape[-1]), dtype=np.float32)
        mean[i] = ((a * m[:, :, None]).sum(1) / n).numpy()[0]
        ntok[i] = n
        del blob, a, m

    POOLED.mkdir(parents=True, exist_ok=True)
    np.save(out, mean)
    np.save(POOLED / f"{path.stem}_labels.npy", y)
    np.save(POOLED / f"{path.stem}_ntokens.npy", ntok)
    if verbose:
        print(f"[warm] {path.stem}: {len(paths)} rows x {mean.shape[1]} "
              f"({int(y.sum())} positive), tokens {ntok.min()}-{ntok.max()} "
              f"(mean {ntok.mean():.0f})")


def stage_warm(args) -> None:
    """Extract every generated set's activations (one model load per concept), then pool.

    The extraction is the only GPU work in the study and by far the longest pole
    (~2-3 s/sample for 2,600 samples a concept). It goes through
    ``retrain.warm_sample_activation_cache``, i.e. the same per-conversation cache every
    later fit reads, so a set warmed here costs nothing again — and the whole stage is a
    no-op on a second run.
    """
    from fit_base_plus_concept import CONCEPTS
    from synthetic_probe_data.retrain import warm_sample_activation_cache

    specs = _concept_specs()
    for name in args.concepts:
        spec, concept = specs[name], CONCEPTS[name]
        paths = _sets_of(spec, with_bases=not args.no_bases)
        rows: list[dict] = []
        for p in paths:
            rows.extend(_rows_of(p, concept))
        print(f"[warm] {name}: {len(rows)} rows from {len(paths)} sets", flush=True)
        computed = warm_sample_activation_cache(
            rows, base_probe_path=concept.base_probe,
            base_activation_cache_dir=concept.base_cache,
            combine_consecutive_messages=COMBINE, convert_tool_to_assistant=CONVERT,
            verbose=True,
        )
        print(f"[warm] {name}: {computed} newly extracted", flush=True)
        for p in paths:
            pool_set(p, concept, spec)


# --------------------------------------------------------------------------- #
# stage: tag — which kind is each generated sample?
# --------------------------------------------------------------------------- #
# One model, two passes, temperature 0, the kinds shown in two different orders. The
# second pass is not a second opinion from a second model — it is the SAME judgement
# asked again with the answer options permuted, which is what makes agreement evidence
# that the tag is a property of the conversation rather than of the list's order. The
# kinds keep their own numbers in both passes, so nothing has to be mapped back.
TAG_MODEL = "openai/gpt-5.6-luna"
TAG_MAX_TOKENS = 400
TAG_CONCURRENCY = 24
TAG_MSG_CHARS = 4000          # per-message truncation when rendering a conversation

_TAG_PROMPT = """\
You are auditing a generated dataset. Each sample was written to be one of the following
KINDS of conversation, and your job is to say which one it is.

The kinds, verbatim from the specification the data was written against:

{kinds}

The conversation to classify:

{conversation}

Which kind is this conversation? Judge the KIND — the situation, the shape of the exchange
and what is being asked — not whether the assistant's reply is good, correct, safe or
harmful, and not which label the sample carries. Answer with the number of the single
best-matching kind, or "none" if it matches none of them.

Reply with JSON and nothing else: {{"kind": <number or "none">}}
"""


def _render_conversation(messages: list[dict]) -> str:
    out = []
    for m in messages:
        content = str(m.get("content", ""))
        if len(content) > TAG_MSG_CHARS:
            content = content[:TAG_MSG_CHARS] + " …[truncated]"
        out.append(f"[{m.get('role', '?')}]\n{content}")
    return "\n\n".join(out)


def _render_kinds(kinds: tuple[Kind, ...], reverse: bool) -> str:
    order = list(reversed(kinds)) if reverse else list(kinds)
    return "\n\n".join(k.text for k in order)


def _parse_kind(text: str, n_kinds: int) -> str:
    """``"1".."n"`` or ``"none"``; ``""`` when the reply cannot be read as either."""
    if not text:
        return ""
    val = None
    m = re.search(r'"kind"\s*:\s*("?)([A-Za-z0-9_]+)\1', text)
    if m:
        val = m.group(2)
    else:
        m = re.search(r"\b(none|\d+)\b", text.strip().lower())
        val = m.group(1) if m else None
    if val is None:
        return ""
    s = str(val).strip().lower()
    if s in ("none", "null", "0"):
        return "none"
    if s.isdigit() and 1 <= int(s) <= n_kinds:
        return s
    return ""


def _tag_one(client, spec: ConceptSpec, messages: list[dict], reverse: bool) -> str:
    prompt = _TAG_PROMPT.format(kinds=_render_kinds(spec.kinds, reverse),
                                conversation=_render_conversation(messages))
    last = ""
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=TAG_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=TAG_MAX_TOKENS,
            )
            choices = getattr(resp, "choices", None)
            if not choices:
                from synthetic_probe_data.openrouter_client import extract_openrouter_error
                last = extract_openrouter_error(resp) or "no choices"
            else:
                got = _parse_kind(choices[0].message.content or "", len(spec.kinds))
                if got:
                    return got
                last = f"unparseable: {(choices[0].message.content or '')[:120]!r}"
        except Exception as exc:                                  # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:160]}"
        if attempt < 3:
            import time
            time.sleep(2 * attempt)
    print(f"  [warn] tag failed after 3 attempts: {last}", file=sys.stderr)
    return ""


TAG_FIELDS = ["row", "kind", "split", "pass1", "pass2", "agree"]


def _load_tags(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {int(r["row"]): r for r in csv.DictReader(fh) if r.get("row")}


def tag_set(client, path: Path, spec: ConceptSpec, concept, limit: int | None = None,
            verbose: bool = True) -> None:
    """Two-pass tag every row of one generated set into ``data/kind_tags/<stem>.csv``.

    Resumable at row granularity: rows already in the CSV are left alone, and the file is
    rewritten in row order at the end so a partial run is still a well-formed CSV.
    """
    from concurrent.futures import ThreadPoolExecutor

    TAGS.mkdir(parents=True, exist_ok=True)
    out = TAGS / f"{path.stem}.csv"
    have = _load_tags(out)
    rows = _rows_of(path, concept)
    todo = [i for i in range(len(rows)) if i not in have]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        if verbose:
            print(f"[tag] {path.stem}: all {len(rows)} rows tagged")
        return
    print(f"[tag] {path.stem}: {len(todo)} rows to tag ({len(have)} already done)",
          flush=True)

    def work(i: int) -> tuple[int, str, str]:
        msgs = rows[i]["inputs"]
        return (i,
                _tag_one(client, spec, msgs, reverse=False),
                _tag_one(client, spec, msgs, reverse=True))

    by_index = {k.index: k for k in spec.kinds}
    done = 0
    with ThreadPoolExecutor(max_workers=TAG_CONCURRENCY) as pool:
        for i, p1, p2 in pool.map(work, todo):
            agree = bool(p1) and p1 == p2
            kind = p1 if (agree and p1 != "none") else ""
            have[i] = {
                "row": i, "kind": kind,
                "split": by_index[int(kind)].split if kind else "",
                "pass1": p1, "pass2": p2, "agree": int(agree),
            }
            done += 1
            if verbose and done % 50 == 0:
                print(f"  [tag] {path.stem}: {done}/{len(todo)}", flush=True)

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TAG_FIELDS)
        w.writeheader()
        for i in sorted(have):
            w.writerow(have[i])
    if verbose:
        _report_tags(out, spec)


def _report_tags(out: Path, spec: ConceptSpec) -> None:
    import collections

    recs = list(_load_tags(out).values())
    n = len(recs)
    agree = sum(int(r["agree"]) for r in recs)
    none1 = sum(1 for r in recs if r["pass1"] == "none")
    counts = collections.Counter(r["kind"] for r in recs if r["kind"])
    by_index = {str(k.index): k for k in spec.kinds}
    per_kind = "  ".join(f"{i}:{counts.get(i, 0)}" for i in sorted(by_index))
    print(f"[tag] {out.stem}: n={n} agree={agree / n:.3f} none(pass1)={none1 / n:.3f} "
          f"kept={sum(counts.values())}  per kind {per_kind}", flush=True)


def stage_tag(args) -> None:
    from fit_base_plus_concept import CONCEPTS
    from synthetic_probe_data.openrouter_client import make_sync_client

    specs = _concept_specs()
    client = make_sync_client()
    for name in args.concepts:
        spec, concept = specs[name], CONCEPTS[name]
        for gen in args.generators:
            tag_set(client, spec.set_path(gen), spec, concept, limit=args.limit)


AUDIT_PER_KIND = 10
AUDIT_CHARS = 700


# A rule-based tagger, used as a CROSS-CHECK ONLY — never to tag the data. It reads the
# shape of a conversation, which is the part of a kind that is stated in the prompt and
# is not a judgement call: turn count, whether a source document is supplied, whether the
# system prompt carries a function list, and (for hu_harm, whose kinds are all two-turn
# and are separated by their measured turn lengths) the words in each turn. Where the
# rule is decisive its agreement with the LLM tagger is a check on the tagger; where it
# abstains it says nothing, which is most of the interesting cases.
def _rule_tag(concept: str, messages: list[dict]) -> int | None:
    def words(m):
        return len(str(m.get("content", "")).split())

    roles = [m.get("role") for m in messages]
    user = [m for m in messages if m.get("role") == "user"]
    asst = [m for m in messages if m.get("role") == "assistant"]
    text = " ".join(str(m.get("content", "")) for m in messages)
    # A supplied source is not always introduced with a "Document:" label — the llama70b
    # sets inline it ("According to the following passage, ... The passage states: '...'").
    # The looser pattern is what makes the cross-check a check on the tagger rather than
    # on one generator's formatting habit.
    has_doc = bool(re.search(
        r"\b(document|passage|context|source|text|excerpt|article|report|news)"
        r"\s*[A-Za-z0-9]{0,2}\s*:"
        r"|according to the (following|given|provided|passage|text)"
        r"|(the|this) (passage|text|document|source|excerpt) (states|says|reads)"
        r"|based on (the|this) (passage|text|document|excerpt|source)", text, re.I))

    if concept == "instructions":
        # Only the two context-drift kinds are multi-turn, and they are told apart by
        # whether each turn carries its own supplied document.
        if len(user) >= 2:
            return 3 if has_doc else 6
        return None                      # 1 / 2 / 4 / 5 are all one-turn: no rule
    if concept == "hu_harm":
        if not (len(messages) == 2 and user and asst):
            return None
        u, a = words(user[0]), words(asst[0])
        # p50 of the measured profiles: (80, 6), (8, 40), (11, 9), (50, 10).
        profiles = {1: (80, 6), 2: (8, 40), 3: (11, 9), 4: (50, 10)}
        import math
        return min(profiles,
                   key=lambda k: (math.log1p(u) - math.log1p(profiles[k][0])) ** 2
                   + (math.log1p(a) - math.log1p(profiles[k][1])) ** 2)
    if concept == "highstakes":
        sys_text = " ".join(str(m.get("content", "")) for m in messages
                            if m.get("role") == "system")
        if re.search(r"function|tool", sys_text, re.I) or "tool" in roles:
            return 4
        if len(messages) <= 3 and user and words(user[0]) > 120:
            return 2
        if len(messages) >= 5:
            return 1 if max(words(m) for m in messages) > 60 else 3
        return None
    return None


def rule_cross_check(spec: ConceptSpec, concept, generators) -> dict:
    """Agreement between the LLM tags and the rule, over the rows the rule decides."""
    import collections

    n_dec = n_agree = 0
    confusion = collections.Counter()
    for gen in generators:
        path = spec.set_path(gen)
        if not (TAGS / f"{path.stem}.csv").exists():
            continue
        rows = _rows_of(path, concept)
        have = _load_tags(TAGS / f"{path.stem}.csv")
        if len(have) != len(rows):
            continue
        for i, r in enumerate(rows):
            tag = have[i]["kind"]
            rule = _rule_tag(spec.name, r["inputs"])
            if not tag or rule is None:
                continue
            n_dec += 1
            n_agree += int(int(tag) == rule)
            confusion[(int(tag), rule)] += 1
    return {"decided": n_dec, "agree": n_agree,
            "rate": round(n_agree / n_dec, 4) if n_dec else float("nan"),
            "confusion": dict(confusion)}


def stage_audit(args) -> None:
    """Dump ten tagged samples per kind, seeded, for a human to read against the tags.

    The tagger decides which rows the kind-only and leave-one-kind-out arms are drawn
    from, so its errors would land in the study as a coverage effect it does not have.
    The dump is a plain file rather than a print so the exact rows read can be quoted in
    the analysis note; the draw is seeded on (concept, kind) and is the same every run.
    """
    import random

    from fit_base_plus_concept import CONCEPTS

    specs = _concept_specs()
    WORK.mkdir(parents=True, exist_ok=True)
    for name in args.concepts:
        spec, concept = specs[name], CONCEPTS[name]
        pool: dict[int, list[tuple]] = {k.index: [] for k in spec.kinds}
        for gen in args.generators:
            path = spec.set_path(gen)
            if not (TAGS / f"{path.stem}.csv").exists():
                continue
            rows = _rows_of(path, concept)
            for i, rec in sorted(_load_tags(TAGS / f"{path.stem}.csv").items()):
                if rec["kind"]:
                    pool[int(rec["kind"])].append((gen, i, rows[i], rec))
        out = [f"# Tagging audit — {name}\n"]
        for k in spec.kinds:
            rng = random.Random(f"{name}:{k.index}:{SEED}")
            picks = rng.sample(pool[k.index], min(AUDIT_PER_KIND, len(pool[k.index])))
            out.append(f"\n## kind {k.index} -> {k.split}  "
                       f"({len(pool[k.index])} tagged, showing {len(picks)})\n")
            out.append(f"_{k.text}_\n")
            for gen, i, row, rec in picks:
                msgs = " | ".join(
                    f"[{m['role']}] {str(m['content'])[:AUDIT_CHARS]}"
                    for m in row["inputs"])
                out.append(f"\n**{gen} row {i}** (label `{row['labels']}`)\n\n{msgs}\n")
        p = WORK / f"audit_{name}.md"
        p.write_text("".join(out), encoding="utf-8")
        print(f"[audit] wrote {p} ({sum(len(v) for v in pool.values())} tagged rows)")
        x = rule_cross_check(spec, concept, args.generators)
        print(f"[audit] {name}: rule-based cross-check decides {x['decided']} rows, "
              f"agrees on {x['rate']:.3f}")
        for (tag, rule), c in sorted(x["confusion"].items()):
            if tag != rule and c >= 10:
                print(f"         LLM kind {tag} vs rule kind {rule}: {c}")


# --------------------------------------------------------------------------- #
# stage: geometry — how many directions do a concept's kinds occupy?
# --------------------------------------------------------------------------- #
MIN_KIND_PER_CLASS = 20      # a kind direction needs this many samples of each class
N_GEOM_FOLDS = 5


def _auroc(y: "np.ndarray", score: "np.ndarray") -> float:
    """Rank AUROC, ties averaged — ``knee_predictor._auroc``, same body."""
    import numpy as np
    from scipy.stats import rankdata

    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _standardiser(X: "np.ndarray"):
    """Mean/scale fit on one set; zero-variance columns are left alone, not exploded."""
    import numpy as np

    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu, sd


def _direction(X: "np.ndarray", y: "np.ndarray") -> "np.ndarray | None":
    """Unit difference-of-means direction, or None if either class is empty."""
    import numpy as np

    if y.sum() == 0 or (~y).sum() == 0:
        return None
    d = X[y].mean(0) - X[~y].mean(0)
    n = float(np.linalg.norm(d))
    return d / n if n > 0 else None


def _load_set_features(stem: str):
    import numpy as np

    f = POOLED / f"{stem}_mean.npy"
    if not f.exists():
        raise SystemExit(f"{f} missing — run --stage warm first")
    return np.load(f), np.load(POOLED / f"{stem}_labels.npy")


def _load_eval_features(split_stem: str):
    """The evaluation split's mean-pooled features, from the knee study's pool."""
    import numpy as np

    import knee_predictor as kp

    f = kp.POOLED / f"{split_stem}_mean.npy"
    if not f.exists():
        raise SystemExit(
            f"{f} missing — the evaluation pool comes from "
            f"`python scripts/knee_predictor.py --stage pool`")
    return np.load(f), np.load(kp.POOLED / f"{split_stem}_labels.npy")


def _kind_tags(stem: str, n_rows: int) -> list[str]:
    """Per-row agreed kind index as a string, ``""`` where the passes disagreed."""
    path = TAGS / f"{stem}.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing — run --stage tag first")
    have = _load_tags(path)
    if len(have) != n_rows:
        raise SystemExit(f"{path.name}: {len(have)} tagged rows, the set has {n_rows}")
    return [have[i]["kind"] for i in range(n_rows)]


def geometry_for_set(spec: ConceptSpec, gen: str) -> tuple[list[dict], dict]:
    """Per-kind directions of one generated set, and what they do to each other.

    Everything is computed on the set's own standardised mean-pooled features, so a
    direction is a unit vector in a space whose axes have this set's scale. The
    evaluation splits are pushed through the *same* standardiser before being projected,
    which is what makes ``E[i, s]`` a statement about the direction rather than about a
    rescaling.
    """
    import numpy as np

    stem = spec.set_path(gen).stem
    X, y = _load_set_features(stem)
    tags = _kind_tags(stem, len(y))
    mu, sd = _standardiser(X)
    Z = (X - mu) / sd

    kinds = list(spec.kinds)
    idx = {k.index: np.array([t == str(k.index) for t in tags]) for k in kinds}
    d_all = _direction(Z, y)

    dirs: dict[int, np.ndarray] = {}
    npos, nneg = {}, {}
    for k in kinds:
        sel = idx[k.index]
        npos[k.index] = int((sel & y).sum())
        nneg[k.index] = int((sel & ~y).sum())
        if npos[k.index] >= MIN_KIND_PER_CLASS and nneg[k.index] >= MIN_KIND_PER_CLASS:
            dirs[k.index] = _direction(Z[sel], y[sel])

    rng = np.random.default_rng(SEED)

    def _held_out_auroc(i: int) -> float:
        """AUROC of kind i's own direction on kind i, direction fit on 4 of 5 folds.

        The diagonal of the transfer matrix would otherwise be fit and scored on the
        same rows, which is not comparable with the off-diagonal entries.
        """
        sel = np.where(idx[i])[0]
        order = rng.permutation(len(sel))
        folds = np.array_split(order, N_GEOM_FOLDS)
        aurocs = []
        for f in folds:
            te = sel[f]
            tr = sel[np.setdiff1d(order, f, assume_unique=False)]
            d = _direction(Z[tr], y[tr])
            if d is None or y[te].sum() == 0 or (~y[te]).sum() == 0:
                continue
            aurocs.append(_auroc(y[te], Z[te] @ d))
        return float(np.mean(aurocs)) if aurocs else float("nan")

    rows = []
    for ki in kinds:
        di = dirs.get(ki.index)
        for kj in kinds:
            dj = dirs.get(kj.index)
            sel_j = idx[kj.index]
            rows.append({
                "concept": spec.name, "gen": gen, "set": stem,
                "kind_i": ki.index, "kind_j": kj.index,
                "split_i": ki.split, "split_j": kj.split,
                "n_i_pos": npos[ki.index], "n_i_neg": nneg[ki.index],
                "cos_ij": "" if di is None or dj is None else round(float(di @ dj), 5),
                "T_ij": "" if di is None or sel_j.sum() == 0 else round(
                    _held_out_auroc(ki.index) if ki.index == kj.index
                    else _auroc(y[sel_j], Z[sel_j] @ di), 5),
            })

    # E[i, s]: the same directions against every evaluation split of the concept.
    evals = {}
    for s in {k.split for k in kinds}:
        Xe, ye = _load_eval_features(s)
        evals[s] = ((Xe - mu) / sd, ye)
    for ki in kinds:
        di = dirs.get(ki.index)
        for s, (Ze, ye) in evals.items():
            rows.append({
                "concept": spec.name, "gen": gen, "set": stem,
                "kind_i": ki.index, "kind_j": "eval",
                "split_i": ki.split, "split_j": s,
                "n_i_pos": npos[ki.index], "n_i_neg": nneg[ki.index],
                "cos_ij": "", "T_ij": "",
                "E_is": "" if di is None else round(_auroc(ye, Ze @ di), 5),
            })
    for s, (Ze, ye) in evals.items():
        rows.append({
            "concept": spec.name, "gen": gen, "set": stem,
            "kind_i": "all", "kind_j": "eval", "split_i": "all", "split_j": s,
            "n_i_pos": int(y.sum()), "n_i_neg": int((~y).sum()),
            "cos_ij": "", "T_ij": "",
            "E_is": "" if d_all is None else round(_auroc(ye, Ze @ d_all), 5),
        })

    # n_eff: the participation ratio of the Gram matrix of the unit kind directions.
    # Unit norms make the trace equal to the number of kinds, so n_eff is k when the
    # directions are orthogonal and 1 when they are all the same direction.
    present = [i for i in sorted(dirs)]
    D = np.stack([dirs[i] for i in present]) if present else np.zeros((0, Z.shape[1]))
    if len(present) >= 2:
        G = D @ D.T
        lam = np.linalg.eigvalsh(G)
        lam = np.clip(lam, 0.0, None)
        n_eff = float(lam.sum() ** 2 / (lam ** 2).sum())
        off = G[~np.eye(len(present), dtype=bool)]
        mean_off, max_off = float(off.mean()), float(off.max())
    else:
        n_eff, mean_off, max_off = float("nan"), float("nan"), float("nan")

    # Transfer summaries. n_eff is a cosine statistic and turns out to be blind to the
    # concept (3.4-4.0 everywhere); how well a kind's direction actually CLASSIFIES
    # another kind's samples is not. Both are recorded per set, and the per-split gaps
    # below (own minus other) are the split-level form, which is what a predictor of a
    # split's half-gain size has to be.
    t_dia = [r["T_ij"] for r in rows
             if r["kind_j"] != "eval" and r["T_ij"] != "" and r["kind_i"] == r["kind_j"]]
    t_off = [r["T_ij"] for r in rows
             if r["kind_j"] != "eval" and r["T_ij"] != "" and r["kind_i"] != r["kind_j"]]
    e_rows = [r for r in rows if r["kind_j"] == "eval" and r.get("E_is", "") != ""]
    e_own = [r["E_is"] for r in e_rows if r["split_i"] == r["split_j"]]
    e_oth = [r["E_is"] for r in e_rows
             if r["split_i"] not in ("all",) and r["split_i"] != r["split_j"]]
    e_all = [r["E_is"] for r in e_rows if r["split_i"] == "all"]

    def _mean(v):
        return round(float(np.mean(v)), 4) if v else ""

    summary = {
        "concept": spec.name, "gen": gen, "set": stem,
        "n_kinds": len(kinds), "n_kinds_used": len(present),
        "t_own": _mean(t_dia), "t_other": _mean(t_off),
        "e_own": _mean(e_own), "e_other": _mean(e_oth), "e_all": _mean(e_all),
        "n_eff": round(n_eff, 4) if n_eff == n_eff else "",
        "mean_off_cos": round(mean_off, 4) if mean_off == mean_off else "",
        "max_off_cos": round(max_off, 4) if max_off == max_off else "",
        "n_tagged": sum(1 for t in tags if t), "n_rows": len(y),
    }
    for k in kinds:
        di = dirs.get(k.index)
        summary[f"cos_all_k{k.index}"] = (
            "" if di is None or d_all is None else round(float(di @ d_all), 4))
        summary[f"n_k{k.index}"] = npos[k.index] + nneg[k.index]
        # Per-split gaps: how much the split's OWN kind direction buys over the other
        # kinds' directions, on that split's evaluation rows (e_gap) and on its own
        # samples (t_gap). A split whose kind is redundant with the rest has a gap near
        # zero, and is the case where a mixed sample should serve it as well as its own.
        own_e = [r["E_is"] for r in e_rows
                 if r["split_i"] == k.split and r["split_j"] == k.split]
        oth_e = [r["E_is"] for r in e_rows
                 if r["split_i"] not in ("all", k.split) and r["split_j"] == k.split]
        summary[f"e_gap_{k.split}"] = (
            round(float(np.mean(own_e) - np.mean(oth_e)), 4) if own_e and oth_e else "")
        own_t = [r["T_ij"] for r in rows if r["kind_j"] != "eval" and r["T_ij"] != ""
                 and r["kind_i"] == k.index and r["kind_j"] == k.index]
        oth_t = [r["T_ij"] for r in rows if r["kind_j"] != "eval" and r["T_ij"] != ""
                 and r["kind_i"] != k.index and r["kind_j"] == k.index]
        summary[f"t_gap_{k.split}"] = (
            round(float(np.mean(own_t) - np.mean(oth_t)), 4) if own_t and oth_t else "")
    return rows, summary


GEOM_FIELDS = ["concept", "gen", "set", "kind_i", "kind_j", "split_i", "split_j",
               "n_i_pos", "n_i_neg", "cos_ij", "T_ij", "E_is"]


def _keep_other_sets(path: Path, redone: set[str]) -> list[dict]:
    """Rows of a previous geometry CSV for sets this run is not recomputing.

    The stage is normally run one concept at a time (a concept's features land as its
    extraction finishes), so writing only what this run computed would silently drop the
    concepts computed earlier.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("set") not in redone]


def stage_geometry(args) -> None:
    specs = _concept_specs()
    rows, summaries = [], []
    for name in args.concepts:
        for gen in args.generators:
            r, s = geometry_for_set(specs[name], gen)
            rows.extend(r)
            summaries.append(s)
            print(f"[geometry] {s['set']}: n_eff={s['n_eff']} "
                  f"mean|cos|={s['mean_off_cos']} kinds used {s['n_kinds_used']}"
                  f"/{s['n_kinds']} tagged {s['n_tagged']}/{s['n_rows']}", flush=True)
    redone = {s["set"] for s in summaries}
    rows = _keep_other_sets(SCRIPTS / "dc_geometry.csv", redone) + rows
    summaries = _keep_other_sets(SCRIPTS / "dc_neff.csv", redone) + summaries
    _write_csv(SCRIPTS / "dc_geometry.csv", rows, GEOM_FIELDS)
    _write_csv(SCRIPTS / "dc_neff.csv", summaries,
               list(dict.fromkeys(k for s in summaries for k in s)))
    print(f"[geometry] wrote dc_geometry.csv ({len(rows)} rows) and dc_neff.csv "
          f"({len(summaries)} sets)")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# stage: arms — the three training sets every curve is drawn from
# --------------------------------------------------------------------------- #
# (a) kind-only stops at 80 because a kind holds only 100-150 tagged samples; (b) mixed
# and (c) leave-one-kind-out run the full range. The sizes below 10 are here to uncensor
# the high-stakes and harmful half-gain sizes, which the paper's curves left at "<= 10".
SIZES_KIND = (2, 4, 6, 10, 20, 30, 50, 80)
SIZES_FULL = (2, 4, 6, 10, 20, 30, 50, 80, 110, 170, 350, 590)
MIN_KIND_TAGGED = 60          # a kind with fewer tagged samples than this is skipped
DRAWS = 8


def _raw_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _arm_paths(spec: ConceptSpec, gen: str) -> dict[str, Path]:
    stem = f"dc_{spec.name}_{gen}"
    out = {"mixed": WORK / f"{stem}_mixed.jsonl"}
    for k in spec.kinds:
        out[f"kind:{k.split}"] = WORK / f"{stem}_{k.split}_kind.jsonl"
        out[f"loko:{k.split}"] = WORK / f"{stem}_{k.split}_loko.jsonl"
    return out


def build_arms(spec: ConceptSpec, gen: str, concept) -> list[dict]:
    """Write the mixed / kind-only / leave-one-kind-out JSONLs for one set.

    The rows are copied out of the source set **verbatim**, so the three arms are subsets
    of one file and nothing about a conversation (or its activation cache key) changes
    between arms. Rows the two tagging passes disagreed on are dropped from the kind-only
    and leave-one-kind-out arms — a row whose kind is uncertain is exactly the row that
    would blur the contrast those two arms exist to draw — and kept in the mixed arm,
    which is the whole set by definition.
    """
    path = spec.set_path(gen)
    lines = _raw_lines(path)
    rows = _rows_of(path, concept)
    tags = _kind_tags(path.stem, len(rows))
    agree = [t != "" or _load_tags(TAGS / f"{path.stem}.csv")[i]["agree"] == "1"
             for i, t in enumerate(tags)]
    pos = [r["labels"] == concept.pos_label for r in rows]

    WORK.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []

    def write(p: Path, keep: list[int], arm: str, split: str) -> None:
        p.write_text("\n".join(lines[i] for i in keep) + "\n", encoding="utf-8")
        npos = sum(1 for i in keep if pos[i])
        out.append({
            "concept": spec.name, "gen": gen, "arm": arm, "split": split,
            "file": p.name, "n": len(keep), "n_pos": npos, "n_neg": len(keep) - npos,
            "balanced": 2 * min(npos, len(keep) - npos),
        })

    write(WORK / f"dc_{spec.name}_{gen}_mixed.jsonl", list(range(len(rows))), "mixed", "")
    for k in spec.kinds:
        tgt = str(k.index)
        kind_idx = [i for i, t in enumerate(tags) if t == tgt]
        loko_idx = [i for i, t in enumerate(tags) if t != tgt and agree[i]]
        write(WORK / f"dc_{spec.name}_{gen}_{k.split}_kind.jsonl", kind_idx,
              "kind", k.split)
        write(WORK / f"dc_{spec.name}_{gen}_{k.split}_loko.jsonl", loko_idx,
              "loko", k.split)
    return out


def sizes_for(arm: str, balanced: int) -> list[int]:
    """The arm's size ladder, capped at what a class-balanced draw can supply.

    Skipped rather than drawn with replacement: a draw of n from fewer than n distinct
    rows is a different experiment, not a smaller one.
    """
    ladder = SIZES_KIND if arm == "kind" else SIZES_FULL
    return [n for n in ladder if n <= balanced]


ARM_FIELDS = ["concept", "gen", "arm", "split", "file", "n", "n_pos", "n_neg",
              "balanced", "sizes", "skipped"]


def stage_arms(args) -> None:
    from fit_base_plus_concept import CONCEPTS

    specs = _concept_specs()
    rows = []
    for name in args.concepts:
        for gen in args.generators:
            for r in build_arms(specs[name], gen, CONCEPTS[name]):
                sizes = sizes_for(r["arm"], r["balanced"])
                ladder = SIZES_KIND if r["arm"] == "kind" else SIZES_FULL
                if r["arm"] == "kind" and r["n"] < MIN_KIND_TAGGED:
                    r["skipped"] = f"only {r['n']} tagged (< {MIN_KIND_TAGGED})"
                    sizes = []
                else:
                    r["skipped"] = ("" if len(sizes) == len(ladder)
                                    else f"sizes above {r['balanced']} unavailable")
                r["sizes"] = " ".join(str(n) for n in sizes)
                rows.append(r)
                print(f"[arms] {r['file']:52s} n={r['n']:3d} "
                      f"({r['n_pos']}/{r['n_neg']}) sizes {r['sizes'] or '-'} "
                      f"{r['skipped']}", flush=True)
    _write_csv(SCRIPTS / "dc_arms.csv", rows, ARM_FIELDS)
    print(f"[arms] wrote dc_arms.csv ({len(rows)} arms)")


# --------------------------------------------------------------------------- #
# stage: fit — the size curves, through the unmodified harness
# --------------------------------------------------------------------------- #
def _fit_jobs(args) -> list[tuple]:
    """(concept, arm file, n) triples, in the order the brief asks for them."""
    if not (SCRIPTS / "dc_arms.csv").exists():
        raise SystemExit("dc_arms.csv missing — run --stage arms first")
    with (SCRIPTS / "dc_arms.csv").open(newline="", encoding="utf-8") as fh:
        arms = [r for r in csv.DictReader(fh)
                if r["concept"] in args.concepts and r["gen"] in args.generators]
    jobs = []
    for r in arms:
        for n in (int(x) for x in r["sizes"].split()):
            jobs.append((r["concept"], r["arm"], r["split"], r["file"], n))
    # Biggest first inside an arm file: a size that cannot be drawn fails immediately
    # rather than after hours of smaller fits.
    jobs.sort(key=lambda j: (j[0], j[3], -j[4]))
    return jobs


def stage_fit(args) -> None:
    """Run ``subsample_curve_concept.py`` once per (arm file, size).

    One subprocess per size because the optimiser regime is size-dependent: accumulation
    = ceil(n/16) at batch 16, i.e. exactly ONE optimiser step per epoch at every size.
    That is the regime ``run_pooled_sizecurve.sh`` pins and the reason the small sizes
    are measurable at all — at the inherited accumulation of 4 any set under ~49 rows
    takes zero steps and the fit hands back the untrained probe. The harness tags those
    rows ``base='none+ga<K>bs16'``, so they can never be pooled with a differently-fit
    curve, and it skips ``(samples, n, draw)`` triples already in the CSV, which is what
    makes this stage resumable at the level of a single fit.
    """
    import math
    import time

    jobs = _fit_jobs(args)
    print(f"[fit] {len(jobs)} (arm, size) cells, {DRAWS} draws each", flush=True)
    t0 = time.time()
    for i, (concept, arm, split, fname, n) in enumerate(jobs, 1):
        restrict = args.restrict_eval and arm in ("kind", "loko")
        # --out-tag keeps concurrent generators off one CSV. The harness appends a row per
        # fit and resumes off what the file already holds, so two processes sharing an
        # --out would interleave their writes and read each other's rows as "done".
        tag = f"__{args.out_tag}" if args.out_tag else ""
        out = SCRIPTS / (f"dc_curves_{concept}{tag}_{split}.csv" if restrict
                         else f"dc_curves_{concept}{tag}.csv")
        script = "dc_run_curve.py" if restrict else "subsample_curve_concept.py"
        cmd = [
            sys.executable, str(SCRIPTS / script),
            "--concept", concept, str(WORK / fname), "--no-base",
            "--grad-accum", str(math.ceil(n / 16)), "--batch-size", "16",
            "--sizes", str(n), "--draws", str(args.draws), "--out", str(out),
        ]
        if restrict:
            cmd += ["--eval-split", split]
        if concept == "highstakes" and args.highstakes_dev:
            cmd += ["--dev-data", args.highstakes_dev]
        print(f"[fit] [{i}/{len(jobs)}] {fname} n={n} "
              f"({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"[fit] FAILED {fname} n={n} (exit {r.returncode})", file=sys.stderr,
                  flush=True)
            if args.stop_on_error:
                raise SystemExit(r.returncode)


# --------------------------------------------------------------------------- #
# stage: analyse — fit the curves, take the ratios, link them to the paper's m
# --------------------------------------------------------------------------- #
N_RATIO_BOOT = 400            # paired bootstrap replicates for R = m_b / m_a
MIN_SIZES = 5                 # fit_curves_ref.fit_all's rule: a curve needs 5 sizes


def _ref():
    import fit_curves_ref

    return fit_curves_ref


def _curve_rows(concept: str) -> list[dict]:
    """Every fit of one concept, from the shared CSV and any per-split restricted ones.

    A kind-only or leave-one-kind-out arm run under ``--restrict-eval`` writes its own
    ``dc_curves_<concept>_<split>.csv`` (the harness will not append rows under a header
    it cannot fill), so the curves are reassembled here rather than in the file.
    """
    rows = []
    for path in sorted(SCRIPTS.glob(f"dc_curves_{concept}*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(r for r in csv.DictReader(fh) if r.get("samples"))
    return rows


def _parse_arm_file(fname: str, specs: dict[str, ConceptSpec]) -> tuple | None:
    """``dc_<concept>_<gen>_<split>_<arm>.jsonl`` -> (concept, gen, arm, split)."""
    m = re.match(r"dc_(\w+?)_(" + "|".join(GENERATORS) + r")_(.+)\.jsonl$", fname)
    if not m:
        return None
    concept, gen, rest = m.group(1), m.group(2), m.group(3)
    if concept not in specs:
        return None
    if rest == "mixed":
        return concept, gen, "mixed", ""
    for suffix, arm in (("_kind", "kind"), ("_loko", "loko")):
        if rest.endswith(suffix):
            return concept, gen, arm, rest[: -len(suffix)]
    return None


def _collect_curves(concepts: list[str]) -> dict[tuple, dict[int, list[float]]]:
    """``{(concept, gen, arm, target split): {n: [AUROC per draw]}}``.

    A mixed arm is scored on every split at once, so one mixed CSV row contributes a
    point to every split's mixed curve; a kind-only or leave-one-kind-out row contributes
    only to the split its file is named for. That asymmetry is the design: (b) is one fit
    per size read by every split, (a) and (c) are one fit per size per split.
    """
    import collections

    specs = _concept_specs()
    curves: dict[tuple, dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for concept in concepts:
        spec = specs[concept]
        for r in _curve_rows(concept):
            parsed = _parse_arm_file(r["samples"], specs)
            if parsed is None:
                continue
            _c, gen, arm, split = parsed
            targets = [k.split for k in spec.kinds] if arm == "mixed" else [split]
            for s in targets:
                col = next(k.eval_column for k in spec.kinds if k.split == s)
                val = r.get(col, "")
                if val in ("", None):
                    continue
                curves[(concept, gen, arm, s)][int(r["n"])].append(float(val))
    return {k: dict(v) for k, v in curves.items()}


FIT_FIELDS = ["model", "concept", "split", "recipe", "variant", "gen", "arm",
              "target_split", "kind_n", "sizes", "n_points", "L", "U", "U_lo", "U_hi",
              "m", "m_lo", "m_hi", "k", "n90", "n90_lo", "n90_hi", "gain_obs", "rmse",
              "U_bvar", "lm_bvar", "flat", "censored"]


def _arm_counts() -> dict[tuple, int]:
    path = SCRIPTS / "dc_arms.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["concept"], r["gen"], r["arm"], r["split"]): int(r["n"])
                for r in csv.DictReader(fh)}


def _expected_sizes() -> dict[tuple, list[int]]:
    """The size ladder each arm was supposed to be fit at, from dc_arms.csv."""
    path = SCRIPTS / "dc_arms.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["concept"], r["gen"], r["arm"], r["split"]):
                [int(x) for x in r["sizes"].split()]
                for r in csv.DictReader(fh)}


def fit_arm_curves(concepts: list[str], allow_partial: bool = False) -> list[dict]:
    """Fit every arm curve with the reference fitter, unmodified.

    A curve is fitted only once **every size of its ladder** has landed. The fit stage
    runs a cell at a time, biggest size first, so a half-run arm holds only its large
    sizes — a log-logistic fitted to that would put the half-gain size wherever the
    missing small end would have pinned it. ``--allow-partial`` lifts the rule for a
    look at work in progress; it is not how the committed numbers are produced.
    """
    ref = _ref()
    specs = _concept_specs()
    counts = _arm_counts()
    expected = _expected_sizes()
    rows = []
    for key, curve in sorted(_collect_curves(concepts).items()):
        concept, gen, arm, split = key
        if len(curve) < MIN_SIZES:
            continue
        want = expected.get((concept, gen, arm, split if arm != "mixed" else ""))
        if want and not allow_partial and set(want) - set(curve):
            continue
        f = ref.fit_curve(curve)
        sizes = sorted(curve)
        rows.append({
            "model": "gemma27b", "concept": concept, "split": split,
            "recipe": "detailed", "variant": specs[concept].variant, "gen": gen,
            "arm": arm, "target_split": split,
            "kind_n": counts.get((concept, gen, arm, split if arm != "mixed" else ""), ""),
            "sizes": " ".join(str(n) for n in sizes),
            "n_points": sum(len(v) for v in curve.values()),
            "flat": int(f["gain_obs"] < 0.02),
            # A half-gain size at or below the smallest size actually measured is
            # left-censored: the curve says "already half-way by the first point", not
            # "half-way at exactly this n".
            "censored": int(f["m"] <= sizes[0]),
            **{c: round(float(f[c]), 6) for c in
               ("L", "U", "U_lo", "U_hi", "m", "m_lo", "m_hi", "k", "n90", "n90_lo",
                "n90_hi", "gain_obs", "rmse", "U_bvar", "lm_bvar")},
        })
    return rows


def _paired_ratio_ci(curve_a: dict, curve_b: dict, rng) -> tuple[float, float]:
    """95% interval for m_b / m_a, resampling draws within each size of BOTH arms.

    Paired rather than two independent intervals: the question is about the ratio, and
    two marginal intervals would not say whether the ratio itself is above 1.
    """
    import numpy as np

    ref = _ref()
    out = []
    for _ in range(N_RATIO_BOOT):
        ms = []
        for curve in (curve_a, curve_b):
            c = {n: [v[i] for i in rng.integers(0, len(v), len(v))]
                 for n, v in curve.items()}
            ns, ys = ref._points(c)
            ms.append(ref.fit(ns, ys)["m"])
        if ms[0] > 0:
            out.append(ms[1] / ms[0])
    if not out:
        return float("nan"), float("nan")
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


RATIO_FIELDS = ["concept", "gen", "split", "usable", "m_a", "m_a_lo", "m_a_hi", "m_b",
                "m_b_lo", "m_b_hi", "m_c", "m_c_lo", "m_c_hi", "R", "R_lo", "R_hi", "G",
                "n_eff", "t_other", "e_gap", "t_gap", "a_s", "kind_n",
                "flat_a", "flat_b", "flat_c",
                "censored_a", "censored_b", "U_a", "U_b", "U_c", "L_b", "L_c"]


def _usable(r: dict) -> bool:
    """Is this row's ratio a measurement of anything?

    ``fit_curves_ref`` calls a curve **flat** when its fitted in-range gain is under 0.02
    — the probe never learned the split at any size — and the paper drops those before
    taking a median, because the half-gain size of a curve that never rose is wherever the
    grid's least-squares happened to land. The same has to hold for a *ratio* of two
    half-gain sizes, and more strongly: one flat arm is enough to make R meaningless, and
    it does not do so quietly. In the instructions interim the three rows with a flat
    kind-only arm read R = 0.02, 0.26 and 266 against a median of 5.3 for the eleven clean
    ones. Rows are kept in the CSV with ``usable = 0`` rather than dropped, so the count
    that was set aside is visible.
    """
    import numpy as np

    return (not int(r["flat_a"]) and not int(r["flat_b"])
            and np.isfinite(float(r["R"])) and float(r["R"]) > 0)


def build_ratios(fits: list[dict], concepts: list[str]) -> list[dict]:
    import numpy as np

    curves = _collect_curves(concepts)
    by_key = {(f["concept"], f["gen"], f["arm"], f["target_split"]): f for f in fits}
    geom = _load_geometry_summary()
    rng = np.random.default_rng(SEED)
    rows = []
    for (concept, gen, arm, split), f in sorted(by_key.items()):
        if arm != "kind":
            continue
        fb = by_key.get((concept, gen, "mixed", split))
        fc = by_key.get((concept, gen, "loko", split))
        if fb is None:
            continue
        R = fb["m"] / f["m"] if f["m"] > 0 else float("nan")
        lo, hi = _paired_ratio_ci(curves[(concept, gen, "kind", split)],
                                  curves[(concept, gen, "mixed", split)], rng)
        gain_b = fb["U"] - fb["L"]
        g = _concept_specs()[concept]
        kind_idx = next(k.index for k in g.kinds if k.split == split)
        key = (concept, gen)
        rows.append({
            "concept": concept, "gen": gen, "split": split,
            "m_a": f["m"], "m_a_lo": f["m_lo"], "m_a_hi": f["m_hi"],
            "m_b": fb["m"], "m_b_lo": fb["m_lo"], "m_b_hi": fb["m_hi"],
            "m_c": fc["m"] if fc else "", "m_c_lo": fc["m_lo"] if fc else "",
            "m_c_hi": fc["m_hi"] if fc else "",
            "R": round(R, 4), "R_lo": round(lo, 4), "R_hi": round(hi, 4),
            "G": round((fc["U"] - fc["L"]) / gain_b, 4) if fc and gain_b > 0 else "",
            "n_eff": geom.get(key, {}).get("n_eff", ""),
            "a_s": geom.get(key, {}).get(f"cos_all_k{kind_idx}", ""),
            "t_other": geom.get(key, {}).get("t_other", ""),
            "e_gap": geom.get(key, {}).get(f"e_gap_{split}", ""),
            "t_gap": geom.get(key, {}).get(f"t_gap_{split}", ""),
            "kind_n": f["kind_n"],
            "flat_a": f["flat"], "flat_b": fb["flat"], "flat_c": fc["flat"] if fc else "",
            "censored_a": f["censored"], "censored_b": fb["censored"],
            "U_a": round(f["U"], 4), "U_b": round(fb["U"], 4),
            "U_c": round(fc["U"], 4) if fc else "",
            "L_b": round(fb["L"], 4), "L_c": round(fc["L"], 4) if fc else "",
        })
        rows[-1]["usable"] = int(_usable(rows[-1]))
    return rows


def _load_geometry_summary() -> dict[tuple, dict]:
    path = SCRIPTS / "dc_neff.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["concept"], r["gen"]): r for r in csv.DictReader(fh)}


def _link_to_paper_m(ratios: list[dict]) -> list[dict]:
    """Do R_s or a_s predict the paper's per-split half-gain size?

    Reuses ``knee_predictor``'s statistics wholesale — the Spearman with tie-averaged
    ranks, the permutation p, the bootstrap over the curves behind each split's median,
    and the leave-one-split-out RMSE against the concept-only baseline — so the bar this
    study has to clear is exactly the bar that study set.
    """
    import collections

    import numpy as np

    import knee_predictor as kp

    targets = kp.load_targets()
    curves = kp.load_curves()
    per_split = collections.defaultdict(list)
    for c in curves:
        if not c["flat"]:
            per_split[(c["split"], c["recipe"])].append(c)

    # Per split: median R and median a_s over the generators that produced a curve.
    by_split = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ratios:
        knee = r["split"][5:] if r["split"].startswith("eval_") else r["split"]
        if int(r["usable"]):
            by_split[knee]["R"].append(float(r["R"]))
        for stat in GEOM_STATS:
            if r.get(stat, "") not in ("", None):
                by_split[knee][stat].append(float(r[stat]))
        by_split[knee]["concept"] = r["concept"]

    out = []
    splits = sorted(by_split)
    for name in ("R", "a_s", "e_gap", "t_gap", "n_eff", "t_other"):
        rows = [(s, float(np.median(by_split[s][name]))) for s in splits
                if by_split[s][name]]
        rows = [(s, v) for s, v in rows
                if targets.get(s, {}).get(kp.PRIMARY_TARGET) is not None]
        if len(rows) < 5:
            out.append({"predictor": name, "n_splits": len(rows)})
            continue
        x = np.array([v for _, v in rows])
        y = np.array([targets[s][kp.PRIMARY_TARGET] for s, _ in rows])
        cs = [by_split[s]["concept"] for s, _ in rows]
        rho = kp._spearman(x, y)
        rec = {
            "predictor": name, "n_splits": len(rows), "rho": round(rho, 4),
            "p_perm": round(kp._perm_p(x, y, rho, np.random.default_rng(SEED)), 5),
            "loo_rmse_pred": round(kp._loo_rmse(x, y, cs, True, False), 4),
            "loo_rmse_concept": round(kp._loo_rmse(x, y, cs, False, True), 4),
            "loo_rmse_grand": round(kp._loo_rmse(x, y, cs, False, False), 4),
            "loo_rmse_pred_concept": round(kp._loo_rmse(x, y, cs, True, True), 4),
        }
        rec["beats_concept"] = int(rec["loo_rmse_pred"] < rec["loo_rmse_concept"])
        lo, hi = kp._boot_ci(x, [s for s, _ in rows], "detailed", per_split,
                             np.random.default_rng(SEED + 1))
        rec["ci_lo"], rec["ci_hi"] = round(lo, 4), round(hi, 4)
        sel = np.array([c == "instructions" for c in cs])
        if sel.sum() >= 5:
            r_in = kp._spearman(x[sel], y[sel])
            rec["rho_instructions"] = round(r_in, 4)
            rec["p_instructions"] = round(
                kp._perm_p(x[sel], y[sel], r_in, np.random.default_rng(SEED + 2)), 5)
        out.append(rec)
    return out


def stage_analyse(args) -> None:
    import numpy as np

    fits = fit_arm_curves(args.concepts, allow_partial=args.allow_partial)
    _write_csv(SCRIPTS / "dc_fits.csv", fits, FIT_FIELDS)
    print(f"[analyse] {len(fits)} arm curves fitted -> dc_fits.csv")

    ratios = build_ratios(fits, args.concepts)
    _write_csv(SCRIPTS / "dc_ratios.csv", ratios, RATIO_FIELDS)
    print(f"[analyse] {len(ratios)} (concept, generator, split) ratios -> dc_ratios.csv")

    print("\n[analyse] R = m(mixed) / m(kind-only), per concept:")
    for concept in args.concepts:
        sel = [r for r in ratios if r["concept"] == concept and int(r["usable"])]
        dropped = sum(1 for r in ratios if r["concept"] == concept and not int(r["usable"]))
        rs = [float(r["R"]) for r in sel]
        if not rs:
            continue
        print(f"  {concept:13s} n={len(rs):3d}  median {np.median(rs):6.2f}  "
              f"range {min(rs):.2f}-{max(rs):.2f}  "
              f"(flat arms set aside: {dropped}; "
              f"censored at the smallest size: "
              f"{sum(int(r['censored_a']) for r in sel)} kind-only, "
              f"{sum(int(r['censored_b']) for r in sel)} mixed)")

    print("\n[analyse] leave-one-kind-out (arm c against arm b), per concept:")
    for concept in args.concepts:
        gs = [float(r["G"]) for r in ratios if r["concept"] == concept
              and r["G"] != "" and not int(r["flat_b"])]
        mc = [float(r["m_c"]) / float(r["m_b"]) for r in ratios
              if r["concept"] == concept and r["m_c"] not in ("", None)
              and not int(r["flat_b"]) and not int(r["flat_c"] or 0)
              and float(r["m_b"]) > 0]
        if gs:
            print(f"  {concept:13s} n={len(gs):3d}  median gain ratio G {np.median(gs):5.2f}"
                  f"  median m_c/m_b {np.median(mc):6.2f}")

    cov = _coverage_vs_difficulty(ratios)
    if cov:
        print("\n[analyse] coverage versus per-kind difficulty (log10 m, medians):")
        for k, v in cov.items():
            print(f"  {k:28s} {v}")

    neff = _neff_correlation(ratios)
    _write_csv(SCRIPTS / "dc_neff_corr.csv", neff,
               list(dict.fromkeys(k for r in neff for k in r)))
    print("\n[analyse] R against the geometry statistics:")
    for r in neff:
        label = f"{r['statistic']} ({r['subset']})"
        if "rho" in r:
            print(f"  {label:26s} n={r['n']:3d}  rho={r['rho']:+.3f} "
                  f"p={r['p_perm']:.4f}")
        else:
            print(f"  {label:26s} n={r['n']:3d}  not tested")

    stats = _link_to_paper_m(ratios)
    _write_csv(SCRIPTS / "dc_link_stats.csv", stats,
               list(dict.fromkeys(k for s in stats for k in s)))
    print("\n[analyse] link to the paper's per-split log10 m:")
    for r in stats:
        if "rho" not in r:
            print(f"  {r['predictor']}: only {r['n_splits']} splits, not tested")
            continue
        print(f"  {r['predictor']:5s} rho={r['rho']:+.3f} p={r['p_perm']:.4f} "
              f"CI [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] "
              f"LOO {r['loo_rmse_pred']:.3f} vs concept {r['loo_rmse_concept']:.3f} "
              f"(grand {r['loo_rmse_grand']:.3f})")

    rows = []
    import collections

    import knee_predictor as kp
    targets = kp.load_targets()
    # log_m_lo / log_m_hi are the spread of the individual detailed curves behind each
    # split's median — the same interval knee_predictor._write_scatter plots — so a point
    # here carries how well its target is pinned down, not just where it sits.
    pool = collections.defaultdict(list)
    for c in kp.load_curves():
        if not c["flat"] and c["recipe"] == "detailed":
            pool[c["split"]].append(c["lm"])
    for r in ratios:
        knee = r["split"][5:] if r["split"].startswith("eval_") else r["split"]
        lms = sorted(pool.get(knee, []))
        rows.append({
            "split": knee, "concept": r["concept"], "gen": r["gen"],
            "R": r["R"], "usable": r["usable"], "n_eff": r["n_eff"],
            "t_other": r.get("t_other", ""), "t_gap": r.get("t_gap", ""),
            "e_gap": r.get("e_gap", ""), "a_s": r["a_s"],
            "log_m": targets.get(knee, {}).get(kp.PRIMARY_TARGET, ""),
            "log_m_lo": f"{lms[0]:.4f}" if lms else "",
            "log_m_hi": f"{lms[-1]:.4f}" if lms else "",
            "n_curves": len(lms),
        })
    _write_csv(SCRIPTS / "dc_scatter.csv", rows,
               ["split", "concept", "gen", "R", "usable", "n_eff", "t_other", "t_gap",
                "e_gap", "a_s", "log_m", "log_m_lo", "log_m_hi", "n_curves"])
    _verdict(ratios, stats)


GEOM_STATS = ("n_eff", "t_other", "e_gap", "t_gap", "a_s")


def _neff_correlation(ratios: list[dict]) -> list[dict]:
    """Spearman of R against each geometry statistic, over the usable rows.

    n_eff is a property of a SET, so this correlation is carried by the between-set
    variation; a within-instructions version exists because that is the concept whose
    kinds the hypothesis says are separate, and it is the one place where the between-set
    spread of n_eff is not confounded with the concept.
    """
    import numpy as np

    import knee_predictor as kp

    out = []
    for stat in GEOM_STATS:
        for name, sel in (("all", lambda r: True),
                          ("instructions", lambda r: r["concept"] == "instructions")):
            pts = [(float(r[stat]), float(r["R"])) for r in ratios
                   if sel(r) and int(r["usable"]) and r.get(stat, "") not in ("", None)]
            row = {"statistic": stat, "subset": name, "n": len(pts)}
            if len(pts) >= 5:
                x = np.array([a for a, _ in pts])
                y = np.array([b for _, b in pts])
                if np.std(x) > 0:
                    row["rho"] = round(kp._spearman(x, y), 4)
                    row["p_perm"] = round(
                        kp._perm_p(x, y, row["rho"], np.random.default_rng(SEED + 5)), 5)
            out.append(row)
    return out


def _coverage_vs_difficulty(ratios: list[dict]) -> dict:
    """How much of instruction's larger half-gain size is coverage, and how much is the kind?

    The paper's concept effect could be either: *instruction* needs more samples because
    its kinds are separate directions and a mixed set spends most of itself elsewhere
    (coverage), or because one kind of *instruction* is simply harder to learn than one
    kind of the other two (per-kind difficulty). The kind-only arm separates them: it is
    the same concept with coverage removed. If instruction's m_a falls to the level of the
    other concepts', the gap was coverage; what is left of it is difficulty.
    """
    import numpy as np

    med = {}
    for concept in ("instructions", "hu_harm", "highstakes"):
        for arm, key in (("m_a", "kind"), ("m_b", "mixed")):
            vals = [float(r[arm]) for r in ratios if r["concept"] == concept
                    and int(r["usable"])]
            if vals:
                med[(concept, arm)] = float(np.median(np.log10(np.maximum(vals, 1.0))))
    others = [c for c in ("hu_harm", "highstakes") if (c, "m_b") in med]
    if ("instructions", "m_b") not in med or not others:
        return {}
    gap_mixed = med[("instructions", "m_b")] - float(
        np.mean([med[(c, "m_b")] for c in others]))
    if ("instructions", "m_a") not in med or not all((c, "m_a") in med for c in others):
        return {"gap_mixed": round(gap_mixed, 4)}
    gap_kind = med[("instructions", "m_a")] - float(
        np.mean([med[(c, "m_a")] for c in others]))
    return {
        "gap_mixed_log10": round(gap_mixed, 4),
        "gap_kind_log10": round(gap_kind, 4),
        # What closing the gap when coverage is removed says: 1.0 = the concept effect on
        # the mixed sets is entirely coverage, 0.0 = none of it is.
        "coverage_share": round((gap_mixed - gap_kind) / gap_mixed, 4) if gap_mixed else "",
        **{f"log_m_{arm}_{c}": round(med[(c, arm)], 4)
           for (c, arm) in med},
    }


def _verdict(ratios: list[dict], stats: list[dict]) -> None:
    """The brief's two conditions, applied without softening."""
    import numpy as np

    med = {}
    for concept in ("instructions", "hu_harm", "highstakes"):
        rs = [float(r["R"]) for r in ratios if r["concept"] == concept
              and int(r["usable"])]
        if rs:
            med[concept] = float(np.median(rs))
    cond_i = (med.get("instructions", 0) > 2
              and all(med.get(c, 9) < 1.5 for c in ("hu_harm", "highstakes")
                      if c in med))
    neff = [(float(r["n_eff"]), float(r["R"])) for r in ratios
            if r["n_eff"] not in ("", None) and int(r["usable"])]
    rho_neff = float("nan")
    if len(neff) >= 5:
        import knee_predictor as kp
        rho_neff = kp._spearman(np.array([a for a, _ in neff]),
                                np.array([b for _, b in neff]))
    beats = any(s.get("predictor") == "a_s" and s.get("beats_concept") == 1
                for s in stats)
    print("\n[analyse] VERDICT")
    print(f"  (i)  median R: " + ", ".join(f"{c} {v:.2f}" for c, v in med.items())
          + f"  -> {'holds' if cond_i else 'does not hold'}")
    print(f"  (ii) rho(R, n_eff) = {rho_neff:+.3f} over {len(neff)} set-splits; "
          f"a_s beats the concept-only baseline: {beats}")
    if not cond_i and not (rho_neff >= 0.6 or beats):
        print("  neither holds. The null is the result.")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
STAGES = ("tag", "audit", "warm", "geometry", "arms", "fit", "analyse", "all")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--concepts", nargs="+", default=["instructions", "hu_harm", "highstakes"],
                    choices=["instructions", "hu_harm", "highstakes"])
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS), choices=GENERATORS)
    ap.add_argument("--no-bases", action="store_true",
                    help="warm only the twelve 600-row detailed sets, not the 50-row bases")
    ap.add_argument("--limit", type=int, default=None,
                    help="tag at most this many untagged rows per set (a pilot)")
    ap.add_argument("--draws", type=int, default=DRAWS,
                    help=f"draws per size in --stage fit (default {DRAWS})")
    ap.add_argument("--highstakes-dev", default=None, metavar="DIR",
                    help="dev directory override for the highstakes fits only (the full "
                         "1908-row dev set is resident every epoch and is what makes that "
                         "concept ~20x the others)")
    ap.add_argument("--restrict-eval", action="store_true",
                    help="--stage fit: score the kind-only and leave-one-kind-out arms on "
                         "their own target split alone, through scripts/dc_run_curve.py. "
                         "Same fit, fewer columns; on highstakes the four eval blobs are "
                         "47 GB and reading all of them is ~95%% of a fit.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="--stage analyse: fit curves whose size ladder is not finished "
                         "yet (a look at work in progress, not how results are produced)")
    ap.add_argument("--out-tag", default="",
                    help="--stage fit: write to dc_curves_<concept>__<tag>[_<split>].csv "
                         "instead of the shared file, so several generators can be fit "
                         "concurrently. --stage analyse reads every dc_curves_<concept>*.csv.")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="--stage fit: abort on the first failing cell instead of going on")
    args = ap.parse_args(argv)

    if args.stage in ("tag", "all"):
        verify_kind_texts()
        stage_tag(args)
    if args.stage == "audit":
        stage_audit(args)
    if args.stage in ("warm", "all"):
        verify_kind_texts()
        stage_warm(args)
    if args.stage in ("geometry", "all"):
        stage_geometry(args)
    if args.stage in ("arms", "all"):
        stage_arms(args)
    if args.stage in ("fit", "all"):
        stage_fit(args)
    if args.stage in ("analyse", "all"):
        stage_analyse(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
