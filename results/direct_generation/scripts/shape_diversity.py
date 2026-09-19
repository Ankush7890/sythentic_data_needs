#!/usr/bin/env python
"""Two set-level filters for a `{inputs, labels}` training set, and the report that reads them.

Both answer the same question — *which rows is this set spending itself on twice?* — and
they disagree about what "twice" means, which is the point of having both.

**`filter_lexical_confounders`** is tuberlens'
`scripts/preprocessing_redteaming.filter_dataset`, generalized and instrumented. Same
mechanics (bag-of-words unigrams -> balanced logistic regression -> drop the most
confidently classified tail), with three changes that the original's shape made necessary
here:

1. the binary label comes from the two class labels actually present, not from a hardcoded
   comparison against the string `"high-stakes"` (which silently turns the function into a
   no-op passthrough on every other concept);
2. the percentile cut runs **inside each class** by default, so the kept set has exactly
   the class balance it started with. A global cut takes its 20% from wherever confidence
   is highest, and whichever class is more lexically distinctive loses more rows — a second,
   uncontrolled variable on top of the one being measured. `per_label=False` reproduces the
   original's global cut, and the report carries what that cut *would* have dropped per
   class either way;
3. it reports what it did.

Note what the confidence score is: `max(predict_proba)` is confidence in the class the
lexical model *picked*, so a row the model is confidently WRONG about is dropped too — the
row whose vocabulary points the wrong way, arguably the most valuable kind to keep. That is
inherited behaviour, kept deliberately so the two filters differ in what they measure and
not in how faithfully they were implemented.

**`filter_shape_redundancy`** is the new one, and it filters on CONVERSATION SHAPE with the
content thrown away: how many messages, in what role sequence, ending on whom, at what
lengths. It maximizes the MIX of shapes rather than cutting a tail — a percentile threshold
cannot express "keep one of each" — in two stages:

  A. **quota over exact role sequences.** Group by role string (`ua`, `uaua`, `sua`, ...),
     then hand out the budget one slot at a time, round-robin over the groups that still
     have rows. As-uniform-as-possible is the maximum-entropy allocation subject to what
     the set actually contains, so a shape holding 3% of the rows comes out at its fair
     share rather than at 3%.
  B. **max-min spread inside each group.** Rows sharing a role sequence still differ in
     length, and length is shape: a 45-character opening and a 3000-character clinical
     document are both `ua`. Within a group, greedy k-center on the standardized length
     vector — seed at the group medoid (deterministic; no RNG anywhere in this module),
     then repeatedly take the row whose minimum distance to the already-kept rows is
     largest.

The descriptor is deliberately CONTENT-FREE — no tokens, no n-grams, no topic — so the two
filters stay orthogonal and can be read against each other rather than measuring the same
thing twice.

**What the report is for.** Shape and content are correlated: "the user supplies a document,
then supplies a different one" *entails* four messages, so deduplicating on shape partly
deduplicates on scenario. And shape can carry real label information — if it does,
flattening the shape distribution deletes signal rather than redundancy. `ShapeReport`
therefore carries MI(role sequence; label) before and after, and `per_label=True` (the
default) selects inside each class so the marginal label distribution cannot move at all.

Both filters preserve class balance exactly under `per_label=True`, and both return
`floor(keep * n) // 2` rows per class, so the two filtered pools are the same size and a
draw of `n` rows from either is comparable to a draw of `n` from the unfiltered set.

    ${REPO_ROOT}/.venv_claude/bin/python scripts/shape_diversity.py \\
        --in data/highstakes_llama70b_600.jsonl --filter both --keep 0.8 \\
        --out-dir /tmp/pools --report-json /tmp/pools/report.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


# ---------------------------------------------------------------------------- rows


def load_rows(path: Path) -> list[dict]:
    """Read a `{inputs, labels}` JSONL. `inputs` may be a JSON string or a list.

    Also accepts the old red-team scaffold's `{id, inputs, label}` rows, whose `label` is
    `positive`/`negative` rather than a class name — those are normalized by the caller,
    which is the only place that knows the concept's two labels.
    """
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def messages_of(record: dict, text_key: str = "inputs",
                transform: bool = False) -> list[dict]:
    """The record's messages, optionally reshaped the way the extraction path will reshape
    them.

    `transform=True` applies `convert_tool_to_assistant` then `combine_consecutive_messages`
    — the pair every fit in this repo runs with (`fit_base_plus_concept.COMBINE/CONVERT`).
    It matters for the shape descriptor specifically: combining merges same-role
    neighbours, so `uaa` reaches the probe as `ua`. Describing the shape of the raw row
    would diversify a shape the probe never sees.
    """
    raw = record.get(text_key)
    msgs = json.loads(raw) if isinstance(raw, str) else raw
    msgs = [m for m in (msgs or []) if isinstance(m, dict)]
    if not transform:
        return msgs
    from synthetic_probe_data.token_budget import apply_message_transforms
    return apply_message_transforms(
        msgs, combine_consecutive_messages=True, convert_tool_to_assistant=True)


def flat_text(record: dict, text_key: str = "inputs") -> str:
    return " ".join(str(m.get("content", "")) for m in messages_of(record, text_key))


# ------------------------------------------------------------------- shape descriptor


ROLE_CODE = {"system": "s", "user": "u", "assistant": "a", "tool": "t"}


def role_signature(msgs: Sequence[dict]) -> str:
    """`sua`, `uaua`, ... — the role sequence, which subsumes turn count, whether a system
    turn is part of the row, and which role the row ends on: the whole `SHAPE (exact)` block
    of `scripts/split_specs.py` in one hashable string."""
    return "".join(ROLE_CODE.get(str(m.get("role", "")).strip().lower(), "?") for m in msgs)


#: The continuous half of the descriptor. Lengths are log-scaled because the real spread
#: across these splits is 45 -> 3000 characters and a linear scale would make every
#: difference below a thousand characters invisible next to one clinical document.
LENGTH_FEATURES = (
    "log_chars_total", "log_chars_first_user", "log_chars_last",
    "log_chars_mean_turn", "log_chars_max_turn", "assistant_user_ratio",
)


def length_vector(msgs: Sequence[dict]) -> np.ndarray:
    chars = [len(str(m.get("content", ""))) for m in msgs]
    first_user = next((len(str(m.get("content", ""))) for m in msgs
                       if str(m.get("role", "")).lower() == "user"), 0)
    user = sum(c for c, m in zip(chars, msgs) if str(m.get("role", "")).lower() == "user")
    asst = sum(c for c, m in zip(chars, msgs) if str(m.get("role", "")).lower() == "assistant")
    lg = lambda x: math.log10(1.0 + max(0.0, float(x)))
    return np.array([
        lg(sum(chars)), lg(first_user), lg(chars[-1] if chars else 0),
        lg(sum(chars) / len(chars) if chars else 0), lg(max(chars) if chars else 0),
        lg(asst) - lg(user),
    ], dtype=np.float64)


def _entropy_bits(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    p = np.array([c / total for c in counts if c > 0], dtype=np.float64)
    return float(-(p * np.log2(p)).sum())


def _normalized_entropy(counts: Sequence[int]) -> float:
    """Entropy over the *observed* signatures, divided by the entropy a uniform spread over
    the same number of signatures would have. 1.0 = every shape equally represented."""
    k = sum(1 for c in counts if c > 0)
    return 1.0 if k <= 1 else _entropy_bits(counts) / math.log2(k)


def _mutual_information_bits(sigs: Sequence[str], labels: Sequence[str]) -> float:
    """I(signature; label). The guard number: if shape carries real label information,
    flattening the shape distribution can delete signal rather than redundancy."""
    n = len(sigs)
    if n == 0:
        return 0.0
    joint = collections.Counter(zip(sigs, labels))
    ps = collections.Counter(sigs)
    pl = collections.Counter(labels)
    mi = 0.0
    for (s, l), c in joint.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / ((ps[s] / n) * (pl[l] / n)))
    return float(max(0.0, mi))


# ------------------------------------------------------------------------- reports


@dataclass
class ShapeReport:
    n_in: int
    n_out: int
    per_class_in: dict[str, int]
    per_class_out: dict[str, int]
    signatures_in: dict[str, int]
    signatures_out: dict[str, int]
    entropy_bits_in: float
    entropy_bits_out: float
    normalized_entropy_in: float
    normalized_entropy_out: float
    max_share_in: float
    max_share_out: float
    mi_shape_label_bits_in: float
    mi_shape_label_bits_out: float
    mean_nn_distance_in: float
    mean_nn_distance_out: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class LexicalReport:
    n_in: int
    n_out: int
    per_class_in: dict[str, int]
    per_class_out: dict[str, int]
    per_label_cut: bool
    n_features: int
    train_accuracy: float
    mean_confidence_in: float
    mean_confidence_out: float
    threshold: dict[str, float]
    global_cut_would_drop: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _class_split(records: Sequence[dict], label_key: str) -> tuple[str, str, dict[str, list[int]]]:
    by = collections.defaultdict(list)
    for i, r in enumerate(records):
        by[r[label_key]].append(i)
    labels = sorted(by)
    if len(labels) != 2:
        raise SystemExit(f"expected exactly two classes, saw {labels}")
    return labels[0], labels[1], by


def _budget_per_class(n_total: int, keep: int | float) -> int:
    """`floor(keep * n) // 2` per class — an even, class-balanced pool, so a balanced draw
    of any n <= pool size is possible from either filter's output."""
    target = int(keep) if keep > 1 else int(math.floor(keep * n_total))
    return max(1, target // 2)


# ------------------------------------------------------------------ lexical filter


def filter_lexical_confounders(
    records: Sequence[dict],
    *,
    keep: int | float = 0.8,
    per_label: bool = True,
    text_key: str = "inputs",
    label_key: str = "labels",
) -> tuple[list[dict], LexicalReport]:
    """Drop the rows a bag-of-words model classifies most confidently.

    `keep` is a fraction of the input (or an absolute row count); the result carries
    `floor(keep*n)//2` rows per class. `per_label=False` takes the cut globally, as
    tuberlens' `filter_dataset` does, and the class balance then falls where it falls.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    notes: list[str] = []
    lo, hi, by = _class_split(records, label_key)
    per_class_in = {k: len(v) for k, v in by.items()}
    per_class = _budget_per_class(len(records), keep)

    texts = [flat_text(r, text_key) for r in records]
    y = np.array([1 if r[label_key] == hi else 0 for r in records])

    vec = CountVectorizer(ngram_range=(1, 1), max_features=20000, min_df=3, max_df=0.9,
                          binary=False)
    try:
        X = vec.fit_transform(texts)
    except ValueError as exc:                      # empty vocabulary on a tiny set
        notes.append(f"vectorizer failed ({exc}); no lexical signal, kept the head of each class")
        X = None
    if X is None or X.shape[1] == 0:
        conf = np.zeros(len(records))
        n_features, acc = 0, float("nan")
        notes.append("empty vocabulary under min_df=3 — the filter degenerates to a no-op cut")
    else:
        clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        conf = proba.max(axis=1)
        n_features = X.shape[1]
        acc = float((proba.argmax(axis=1) == y).mean())

    # What a GLOBAL percentile cut would remove, per class — the diagnostic that says
    # whether `per_label` is doing real work on this set or none.
    n_drop_total = len(records) - 2 * per_class
    order = np.argsort(-conf, kind="stable")
    dropped_globally = set(order[:max(0, n_drop_total)].tolist())
    global_drop = collections.Counter(records[i][label_key] for i in dropped_globally)

    kept_idx: list[int] = []
    thresholds: dict[str, float] = {}
    if per_label:
        for label, idx in by.items():
            c = conf[idx]
            keep_local = np.argsort(c, kind="stable")[:per_class]      # least confident first
            chosen = [idx[j] for j in keep_local]
            kept_idx.extend(chosen)
            thresholds[label] = float(c[keep_local[-1]]) if len(keep_local) else float("nan")
            if len(idx) < per_class:
                notes.append(f"{label}: only {len(idx)} rows, wanted {per_class}")
    else:
        kept_idx = [i for i in order[::-1][: 2 * per_class]]            # least confident first
        thresholds["global"] = float(conf[kept_idx[-1]]) if kept_idx else float("nan")
        got = collections.Counter(records[i][label_key] for i in kept_idx)
        notes.append(f"global cut, class balance not preserved: {dict(got)}")

    kept_idx.sort()
    kept = [dict(records[i]) for i in kept_idx]
    return kept, LexicalReport(
        n_in=len(records), n_out=len(kept), per_class_in=per_class_in,
        per_class_out=dict(collections.Counter(r[label_key] for r in kept)),
        per_label_cut=per_label, n_features=n_features, train_accuracy=acc,
        mean_confidence_in=float(conf.mean()),
        mean_confidence_out=float(conf[kept_idx].mean()) if kept_idx else float("nan"),
        threshold=thresholds, global_cut_would_drop=dict(global_drop), notes=notes,
    )


# -------------------------------------------------------------------- shape filter


def _kcenter_order(V: np.ndarray) -> list[int]:
    """Greedy max-min ordering of the rows of `V`, seeded at the medoid.

    Deterministic: the seed is the row closest to the group mean, ties broken by index, and
    every later pick is the argmax of the minimum distance to what is already chosen.
    """
    n = len(V)
    if n <= 1:
        return list(range(n))
    centre = V.mean(axis=0)
    first = int(np.argmin(((V - centre) ** 2).sum(axis=1)))
    chosen = [first]
    dmin = np.sqrt(((V - V[first]) ** 2).sum(axis=1))
    for _ in range(n - 1):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.sqrt(((V - V[nxt]) ** 2).sum(axis=1)))
        dmin[nxt] = -1.0
    return chosen


def _round_robin_quota(groups: dict[str, list[int]], budget: int) -> dict[str, int]:
    """Hand out `budget` slots one at a time, cycling over the groups that still have rows.

    This is the maximum-entropy allocation subject to availability: every group gets
    `budget // k` and the remainder goes to the largest groups (which is what keeps the
    budget exactly met when a small group runs dry)."""
    quota = {g: 0 for g in groups}
    # Largest groups last in the cycle so the remainder lands where rows certainly remain.
    order = sorted(groups, key=lambda g: (len(groups[g]), g))
    remaining = budget
    while remaining > 0:
        progressed = False
        for g in order:
            if remaining == 0:
                break
            if quota[g] < len(groups[g]):
                quota[g] += 1
                remaining -= 1
                progressed = True
        if not progressed:                 # every group exhausted
            break
    return quota


def _nn_distance(V: np.ndarray) -> float:
    """Mean nearest-neighbour distance in the standardized length space — the direct
    read on "are the kept rows spread out or piled up"."""
    if len(V) < 2:
        return float("nan")
    D = np.sqrt(((V[:, None, :] - V[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    return float(D.min(axis=1).mean())


def filter_shape_redundancy(
    records: Sequence[dict],
    *,
    keep: int | float = 0.8,
    per_label: bool = True,
    transform: bool = True,
    text_key: str = "inputs",
    label_key: str = "labels",
) -> tuple[list[dict], ShapeReport]:
    """Keep the most shape-DIVERSE subset: role-sequence quota, then max-min on lengths.

    `keep` is a fraction of the input (or an absolute count); the result carries
    `floor(keep*n)//2` rows per class. With `per_label=True` the two stages run inside each
    class, so the marginal label distribution is untouched and only the conditional
    P(shape | label) moves.
    """
    notes: list[str] = []
    lo, hi, by = _class_split(records, label_key)
    per_class_in = {k: len(v) for k, v in by.items()}
    per_class = _budget_per_class(len(records), keep)

    msgs = [messages_of(r, text_key, transform=transform) for r in records]
    sigs = [role_signature(m) for m in msgs]
    bad = [i for i, m in enumerate(msgs) if not m or "?" in sigs[i]]
    if bad:
        notes.append(f"{len(bad)} row(s) carry an unknown or empty role sequence")

    V = np.vstack([length_vector(m) for m in msgs])
    sd = V.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (V - V.mean(axis=0)) / sd          # standardized ONCE over the whole set, so the
                                           # scales do not drift between groups

    strata = list(by.items()) if per_label else [("all", list(range(len(records))))]
    kept_idx: list[int] = []
    for label, idx in strata:
        budget = per_class if per_label else 2 * per_class
        groups: dict[str, list[int]] = collections.defaultdict(list)
        for i in idx:
            groups[sigs[i]].append(i)
        quota = _round_robin_quota(dict(groups), budget)
        for sig, members in groups.items():
            take = quota[sig]
            if take <= 0:
                continue
            order = _kcenter_order(Z[members])
            kept_idx.extend(members[j] for j in order[:take])
        if len(idx) < budget:
            notes.append(f"{label}: only {len(idx)} rows, wanted {budget}")

    kept_idx.sort()
    kept = [dict(records[i]) for i in kept_idx]
    sig_in = collections.Counter(sigs)
    sig_out = collections.Counter(sigs[i] for i in kept_idx)
    return kept, ShapeReport(
        n_in=len(records), n_out=len(kept), per_class_in=per_class_in,
        per_class_out=dict(collections.Counter(r[label_key] for r in kept)),
        signatures_in=dict(sorted(sig_in.items())), signatures_out=dict(sorted(sig_out.items())),
        entropy_bits_in=_entropy_bits(list(sig_in.values())),
        entropy_bits_out=_entropy_bits(list(sig_out.values())),
        normalized_entropy_in=_normalized_entropy(list(sig_in.values())),
        normalized_entropy_out=_normalized_entropy(list(sig_out.values())),
        max_share_in=max(sig_in.values()) / len(records),
        max_share_out=(max(sig_out.values()) / len(kept)) if kept else float("nan"),
        mi_shape_label_bits_in=_mutual_information_bits(sigs, [r[label_key] for r in records]),
        mi_shape_label_bits_out=_mutual_information_bits(
            [sigs[i] for i in kept_idx], [records[i][label_key] for i in kept_idx]),
        mean_nn_distance_in=_nn_distance(Z),
        mean_nn_distance_out=_nn_distance(Z[kept_idx]) if kept_idx else float("nan"),
        notes=notes,
    )


# ------------------------------------------------------------------------------ cli


def write_rows(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            inputs = r["inputs"]
            fh.write(json.dumps(
                {"inputs": inputs if isinstance(inputs, str)
                           else json.dumps(inputs, ensure_ascii=False),
                 "labels": r["labels"]}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--filter", choices=["lexical", "shape", "both"], default="both")
    ap.add_argument("--keep", type=float, default=0.8,
                    help="fraction of the input to keep (>1 is an absolute row count)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--report-json", type=Path, default=None)
    ap.add_argument("--no-transform", action="store_true",
                    help="shape filter only: describe the RAW messages instead of the "
                         "combine/convert-transformed ones the probe is fed")
    ap.add_argument("--global-cut", action="store_true",
                    help="lexical filter only: take the percentile cut globally, as "
                         "tuberlens' filter_dataset does, instead of inside each class")
    args = ap.parse_args()

    rows = load_rows(args.src)
    out_dir = args.out_dir or args.src.parent
    stem = args.src.stem
    reports: dict[str, Any] = {"source": str(args.src), "n": len(rows), "keep": args.keep}

    if args.filter in ("lexical", "both"):
        kept, rep = filter_lexical_confounders(rows, keep=args.keep,
                                               per_label=not args.global_cut)
        write_rows(kept, out_dir / f"{stem}_lex.jsonl")
        reports["lexical"] = rep.as_dict()
        print(f"lexical: {rep.n_in} -> {rep.n_out}  {rep.per_class_out}  "
              f"features={rep.n_features} train_acc={rep.train_accuracy:.3f}  "
              f"mean conf {rep.mean_confidence_in:.3f} -> {rep.mean_confidence_out:.3f}")
        for n in rep.notes:
            print(f"  note: {n}")

    if args.filter in ("shape", "both"):
        kept, rep = filter_shape_redundancy(rows, keep=args.keep,
                                            transform=not args.no_transform)
        write_rows(kept, out_dir / f"{stem}_shape.jsonl")
        reports["shape"] = rep.as_dict()
        print(f"shape:   {rep.n_in} -> {rep.n_out}  {rep.per_class_out}  "
              f"H_norm {rep.normalized_entropy_in:.3f} -> {rep.normalized_entropy_out:.3f}  "
              f"max share {rep.max_share_in:.3f} -> {rep.max_share_out:.3f}  "
              f"MI(shape;label) {rep.mi_shape_label_bits_in:.4f} -> "
              f"{rep.mi_shape_label_bits_out:.4f} bits")
        print(f"  signatures in : {rep.signatures_in}")
        print(f"  signatures out: {rep.signatures_out}")
        for n in rep.notes:
            print(f"  note: {n}")

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"report -> {args.report_json}")


if __name__ == "__main__":
    main()
