#!/usr/bin/env python
"""Can a cheap property of an evaluation split predict its learning-curve knee?

THE QUESTION
------------
Every synthetic-data learning curve in the paper is fit with a four-parameter
log-logistic ``A(n) = L + (U - L) / (1 + (n / m) ** -k)``. ``m`` — the half-gain size,
the "knee" — is how many generated samples it takes to get halfway from the floor to
the ceiling. The paper's variance decomposition says the *evaluation split* explains
42-45% of the variance of ``log m`` while the generator, probe model and prompt
together explain a few percent. So the knee is a property of the split. But that is
only useful if you can read it off the split *before* you generate anything.

This script tests whether you can. It computes cheap, purely descriptive properties of
each of the 14 evaluation splits from their cached Gemma-3-27B-IT layer-32 activations
and asks which — if any — correlate with the knee the paper measured.

NO MODEL IS EVER LOADED. Activations come from the published per-split blobs (see
``scripts/fetch_kaggle_eval_activations.py``); the only torch used here reads them.
No synthetic data is generated and no LLM is called.

WHAT IS PREDICTED (the target)
------------------------------
``scripts/knee_fits.csv`` holds the fitted curves. Restricted to ``model == gemma27b``
that is 140 rows. A curve whose fitted in-range gain is under 0.02 is **flat** — the
probe never learned the split at any size, so its ``m`` is meaningless and it is
dropped (the rule is ``fit_curves_ref.flat``, reused here rather than re-implemented).

    primary     median over the four ``detailed``-prompt curves of log10 m
    secondary   the same for ``general`` (10 splits), ``llm`` (6 splits),
                ``targeted`` (14 splits), and the n90 medians

``m <= 10`` means "at or below the smallest measured size" — left-censored. Those are
clamped to 10 (``fit_curves_ref.M_MIN``) and become ties, which is why every headline
number here is a rank statistic.

WHAT IS USED TO PREDICT IT (the predictors), per split
------------------------------------------------------
Everything is computed from two small matrices per split — the token-mean and the
last-token activation of every row — under 5-fold stratified CV with a fixed seed,
averaged over folds. Mean-pooled is primary; last-token is carried alongside.

    A  in-distribution few-shot curve: logistic regression on k balanced samples,
       k in {2 .. 256}, 32 draws each, held-out AUROC; the same log-logistic fitted
       to *that* curve gives m_ID / n90_ID / U_ID. The hypothesis under test: the
       synthetic knee tracks the in-distribution knee.
    B  one-direction share: how much of full logistic regression's AUROC gain a
       single difference-of-means direction recovers (r1), plus the same from two
       samples per class (AUROC_dom2).
    C  direction count: held-out AUROC of logistic regression on the top-d principal
       components; d95 / d90 are the smallest d reaching 95% / 90% of the
       all-dimension gain over 0.5.
    D  geometry: Fisher ratio along the difference-of-means direction, centroid
       cosine, participation ratio of the within-class covariance, within- vs
       between-class cosine, and (paired splits only) within-pair distance relative
       to between-pair spread.
    E  controls needing no activations: rows, mean tokens, mean turns, paired or not,
       concept identity.

Stages, each resumable, run in order::

    python scripts/knee_predictor.py --stage pool       # blobs  -> scripts/knee_pooled/*.npy
    python scripts/knee_predictor.py --stage features   # .npy   -> knee_predictors.csv, _curves.csv
    python scripts/knee_predictor.py --stage analyse    # + targets -> _stats.csv, _scatter.csv

``--stage all`` runs the three in sequence. ``pool`` is the only one that touches the
activation blobs (49 GB for high-stakes alone); it holds exactly one split at a time
and skips any split whose .npy files already exist.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
POOLED = SCRIPTS / "knee_pooled"

# One seed for the whole study, stated in the analysis. Folds, few-shot draws, PCA's
# randomized solver and every permutation/bootstrap are derived from it.
SEED = 20260917

# The blobs are all Gemma-3-27B-IT layer 32, extracted with these two transforms. They
# are part of what the cache key would have been, so they are asserted, not assumed.
MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 32
COMBINE_CONSECUTIVE_MESSAGES = True
CONVERT_TOOL_TO_ASSISTANT = True


# --------------------------------------------------------------------------- #
# the 14 splits
# --------------------------------------------------------------------------- #
class Split:
    """One evaluation split: where its rows, labels and activations live.

    ``knee`` is the name the split goes by in ``knee_fits.csv``, which drops the
    ``eval_`` prefix the human-harm files carry on disk. ``stem`` is the file name.
    """

    __slots__ = ("concept", "eval_dir", "stem", "knee", "pos", "neg", "cache_dir")

    def __init__(self, concept, eval_dir, stem, knee, pos, neg, cache_dir):
        self.concept, self.eval_dir, self.stem, self.knee = concept, eval_dir, stem, knee
        self.pos, self.neg, self.cache_dir = pos, neg, cache_dir

    @property
    def jsonl(self) -> Path:
        return REPO / "eval_sets" / self.eval_dir / f"{self.stem}.jsonl"

    @property
    def blob(self) -> Path:
        return REPO / self.cache_dir / f"{self.stem}-acts_full.pt"

    def pooled(self, what: str) -> Path:
        return POOLED / f"{self.stem}_{what}.npy"


_HS = ("high-stakes", "low-stakes")
_HU = ("harmful_to_human", "not_harmful_to_human")
_IN = ("assistant_follows_the_instruction", "assistant_does_not_follow_the_instruction")

# `oig_omission` is excluded from the study, so it is absent here rather than filtered
# later — there is no code path in which it can reappear.
SPLITS: list[Split] = [
    Split("highstakes", "highstakes", s, s, *_HS, "eval_activations_highstakes")
    for s in ("anthropic_hh_balanced", "mt_balanced", "mts_balanced", "toolace_balanced")
] + [
    Split("hu_harm", "hu_ha", f"eval_{s}", s, *_HU, "eval_activations_hu_harm")
    for s in ("ai_dilemmas", "ant_hh", "balanced_refusal", "daily_dilemmas")
] + [
    Split("instructions", "instructions", s, s, *_IN, "eval_activations_instructions")
    for s in ("anthropic_harmless_refusal", "bbq_substitution", "hc_context_drift",
              "hc_contradiction", "mm_substitution", "oig_context_drift")
]

BY_KNEE = {s.knee: s for s in SPLITS}


# --------------------------------------------------------------------------- #
# stage: pool
# --------------------------------------------------------------------------- #
def _rows(split: Split) -> list[dict]:
    """The split's JSONL records, in file order — the join key for everything here."""
    with split.jsonl.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _labels(split: Split, rows: list[dict]) -> np.ndarray:
    """Boolean positive-class mask, in file order, cross-checked against tuberlens.

    The raw ``labels`` column is the authority; ``LabelledDataset`` is loaded purely to
    confirm it maps those strings to the same 0/1 under this split's class labels. A
    disagreement means the label strings in the JSONL are not the ones the probes were
    trained against, which would silently invert a predictor.
    """
    from tuberlens.interfaces.dataset import LabelledDataset

    seen = collections.Counter(r["labels"] for r in rows)
    unexpected = set(seen) - {split.pos, split.neg}
    if unexpected:
        raise SystemExit(
            f"{split.stem}: unexpected label(s) {sorted(unexpected)} — this split's "
            f"classes are {split.pos!r} / {split.neg!r}."
        )
    y = np.array([r["labels"] == split.pos for r in rows])

    ds = LabelledDataset.load_from(
        split.jsonl,
        pos_class_label=split.pos,
        neg_class_label=split.neg,
        combine_consecutive_messages=COMBINE_CONSECUTIVE_MESSAGES,
        convert_tool_to_assistant=CONVERT_TOOL_TO_ASSISTANT,
    )
    if len(ds) != len(rows):
        raise SystemExit(f"{split.stem}: tuberlens read {len(ds)} rows, the file has {len(rows)}")
    theirs = np.array([lab.to_int() == 1 for lab in ds.labels])
    if not np.array_equal(y, theirs):
        raise SystemExit(f"{split.stem}: label disagreement between the JSONL and tuberlens")
    return y


def _validate_blob(split: Split, n_rows: int):
    """Header check, from ``eval_kfold_cv._load_split``: model, layer, row count.

    ``LLMModel.load_activations`` throws away the model name and layer the blob was
    saved with, and these caches load by path without validating their inputs, so the
    check has to happen against the raw dict before anything else touches it.
    """
    import torch

    if not split.blob.exists():
        raise SystemExit(
            f"missing eval activations for split {split.stem!r}: {split.blob}\n"
            f"Fill the cache first:\n"
            f"  python scripts/fetch_kaggle_eval_activations.py "
            f"--concept {split.concept} --cache-dir {split.cache_dir}"
        )
    data = torch.load(split.blob, map_location="cpu", mmap=True)
    got_model, got_layer = data.get("model_name"), data.get("layer")
    if got_model not in (None, MODEL_NAME) or (got_layer is not None and int(got_layer) != LAYER):
        raise SystemExit(
            f"{split.blob.name} was computed with {got_model} L{got_layer}, "
            f"but this study is {MODEL_NAME} L{LAYER}."
        )
    if int(data["activations"].shape[0]) != n_rows:
        raise SystemExit(
            f"{split.blob.name}: {data['activations'].shape[0]} rows but split "
            f"{split.stem!r} has {n_rows}. The blob does not describe this split."
        )
    return data


def pool_split(split: Split, chunk: int = 16, verbose: bool = True) -> None:
    """Reduce one split's activation blob to two small matrices, on disk.

    The blob is ``[rows, 1024, 5376]`` fp16 — 33 GB for ``anthropic_hh_balanced`` — so
    it is memory-mapped and consumed a few rows at a time; nothing larger than a chunk
    is ever materialised, and only one split is open at a time. What survives is the
    token-mean and the last real token of each row, float32, which is four orders of
    magnitude smaller and is all any predictor below needs.

    Padding is read off the attention mask rather than assumed: "last token" is the
    last position the mask admits, whichever side the padding landed on.
    """
    import torch

    if all(split.pooled(w).exists() for w in ("mean", "last", "labels", "ntokens")):
        if verbose:
            print(f"[pool] {split.stem}: already pooled, skipping")
        return

    rows = _rows(split)
    y = _labels(split, rows)
    data = _validate_blob(split, len(rows))
    acts, mask = data["activations"], data["attention_mask"]
    n, _, hidden = acts.shape

    mean = np.empty((n, hidden), dtype=np.float32)
    last = np.empty((n, hidden), dtype=np.float32)
    ntok = np.empty(n, dtype=np.int32)

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        a = acts[lo:hi].to(torch.float32)          # pages in exactly these rows
        m = mask[lo:hi].to(torch.bool)
        counts = m.sum(1)
        if int(counts.min()) == 0:
            bad = lo + int(torch.argmin(counts))
            raise SystemExit(f"{split.stem}: row {bad} has an all-zero attention mask")
        mean[lo:hi] = ((a * m[:, :, None]).sum(1) / counts[:, None]).numpy()
        # Last admitted position per row — no assumption about which side padding is on.
        idx = (m.shape[1] - 1) - torch.argmax(torch.flip(m, dims=[1]).to(torch.uint8), dim=1)
        last[lo:hi] = a[torch.arange(hi - lo), idx].numpy()
        ntok[lo:hi] = counts.numpy().astype(np.int32)
        del a, m

    del acts, mask, data

    POOLED.mkdir(parents=True, exist_ok=True)
    np.save(split.pooled("mean"), mean)
    np.save(split.pooled("last"), last)
    np.save(split.pooled("labels"), y)
    np.save(split.pooled("ntokens"), ntok)
    if verbose:
        print(f"[pool] {split.stem}: {n} rows x {hidden} "
              f"({int(y.sum())} positive), tokens {ntok.min()}-{ntok.max()} "
              f"(mean {ntok.mean():.0f})")


def stage_pool(args) -> None:
    for split in SPLITS:
        pool_split(split, chunk=args.chunk)
    write_pool_manifest()


def write_pool_manifest() -> None:
    """Shapes and content hashes of every pooled matrix.

    The .npy files are ~280 MB in total, over the threshold for committing them, so
    this manifest is what goes in the repo: enough to tell whether a rebuilt pool is
    the one the numbers came from.
    """
    entries = []
    for split in SPLITS:
        for what in ("mean", "last", "labels", "ntokens"):
            p = split.pooled(what)
            if not p.exists():
                continue
            arr = np.load(p, mmap_mode="r")
            entries.append({
                "split": split.knee, "file": p.name, "shape": list(arr.shape),
                "dtype": str(arr.dtype), "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            })
    manifest = {
        "model_name": MODEL_NAME, "layer": LAYER,
        "combine_consecutive_messages": COMBINE_CONSECUTIVE_MESSAGES,
        "convert_tool_to_assistant": CONVERT_TOOL_TO_ASSISTANT,
        "seed": SEED, "files": entries,
    }
    out = SCRIPTS / "knee_pooled_manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(e["bytes"] for e in entries) / 1e6
    print(f"[pool] manifest: {len(entries)} files, {total:.0f} MB -> {out.name}")


# --------------------------------------------------------------------------- #
# the targets: what knee_fits.csv says each split's knee is
# --------------------------------------------------------------------------- #
RECIPES = ("detailed", "general", "llm", "targeted")


def _fit_curves_ref():
    """``scripts/fit_curves_ref.py`` as a module — the reference fitter, imported.

    The flat rule, the censoring floor ``M_MIN`` and the log-logistic grid all live
    there. Re-implementing any of them would risk a curve counted here that the paper
    dropped, or a knee clamped differently, so the file is imported rather than copied.
    """
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import fit_curves_ref

    return fit_curves_ref


def load_curves() -> list[dict]:
    """The Gemma curve table: one dict per fitted curve, with ``flat`` and ``lm``.

    ``lm`` is ``log10(max(m, M_MIN))`` — the same clamp ``fit_curves_ref.run`` applies,
    which is what makes a knee at or below the smallest measured size (10) a tie rather
    than an extrapolation.
    """
    import csv

    ref = _fit_curves_ref()
    out = []
    with (SCRIPTS / "knee_fits.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["model"] != "gemma27b":
                continue
            if r["split"] not in BY_KNEE:          # oig_omission and anything else new
                continue
            d = {k: r[k] for k in ("model", "concept", "split", "recipe", "variant", "gen")}
            d.update({k: float(r[k]) for k in ("L", "U", "m", "m_lo", "m_hi", "k",
                                               "n90", "n90_lo", "n90_hi", "gain_obs",
                                               "rmse", "U_bvar", "lm_bvar")})
            d["flat"] = d["gain_obs"] < 0.02       # fit_curves_ref.run's `flat`
            d["lm"] = np.log10(max(d["m"], ref.M_MIN))
            d["ln90"] = np.log10(min(d["n90"], 1e6))
            out.append(d)
    return out


def load_targets(curves: list[dict] | None = None) -> dict[str, dict[str, float]]:
    """``{split: {target name: value}}`` — the median log10 m per recipe, and n90.

    Flat curves are excluded before the median is taken, so a split whose every curve
    under a recipe is flat has no target for that recipe (``general`` loses two splits
    this way) rather than a median of meaningless knees.
    """
    curves = curves if curves is not None else load_curves()
    out: dict[str, dict[str, float]] = {s.knee: {} for s in SPLITS}
    for recipe in RECIPES:
        for split in SPLITS:
            rs = [c for c in curves
                  if c["split"] == split.knee and c["recipe"] == recipe and not c["flat"]]
            if not rs:
                continue
            out[split.knee][f"log_m_{recipe}"] = float(np.median([c["lm"] for c in rs]))
            out[split.knee][f"log_n90_{recipe}"] = float(np.median([c["ln90"] for c in rs]))
            out[split.knee][f"n_curves_{recipe}"] = float(len(rs))
    return out


PRIMARY_TARGET = "log_m_detailed"


# --------------------------------------------------------------------------- #
# stage: features
# --------------------------------------------------------------------------- #
K_FEWSHOT = (2, 4, 8, 16, 32, 64, 128, 256)
N_DRAWS_FEWSHOT = 32
N_DRAWS_DOM2 = 200
D_PCA = (1, 2, 4, 8, 16, 32, 64, 128, 256)
C_VALUES = (1.0, 0.1)
N_FOLDS = 5
POOLINGS = ("mean", "last")


def _fold_views(X: np.ndarray, y: np.ndarray, seed: int) -> list[dict]:
    """5 stratified folds, each standardised on its own training folds.

    The scaler is fitted on the whole training side once and reused by every few-shot
    draw inside that fold. That is deliberate: estimating 5376 feature means from the
    2 samples of a k=2 draw would make the few-shot curve a measurement of how noisy
    standardisation is, not of how fast the classifier learns. The held-out fold never
    contributes to the scaler, so nothing leaks either way.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    views = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        scaler = StandardScaler().fit(X[tr])
        views.append({
            "tr": tr, "te": te,
            "Xtr": scaler.transform(X[tr]).astype(np.float32),
            "Xte": scaler.transform(X[te]).astype(np.float32),
            "ytr": y[tr], "yte": y[te],
        })
    return views


def _auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if y_true.all() or not y_true.any():
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _lr(C: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, max_iter=5000)


def _fewshot_curve(views: list[dict], C: float, rng) -> dict[int, list[float]]:
    """``{k: [held-out AUROC, one per (fold, draw)]}`` for the in-distribution curve.

    A ``k`` needing more samples per class than the smallest training fold holds is
    skipped outright rather than silently drawn with replacement — ``mts_balanced`` has
    43 rows per class, so it simply has no k=128 point.
    """
    curve: dict[int, list[float]] = {}
    for k in K_FEWSHOT:
        per_class = k // 2
        if any(min(int(v["ytr"].sum()), int((~v["ytr"]).sum())) < per_class for v in views):
            continue
        vals = []
        for v in views:
            pos = np.where(v["ytr"])[0]
            neg = np.where(~v["ytr"])[0]
            for _ in range(N_DRAWS_FEWSHOT):
                idx = np.concatenate([rng.choice(pos, per_class, replace=False),
                                      rng.choice(neg, per_class, replace=False)])
                model = _lr(C).fit(v["Xtr"][idx], v["ytr"][idx])
                vals.append(_auroc(v["yte"], model.decision_function(v["Xte"])))
        curve[k] = vals
    return curve


def _dom_direction(Xtr: np.ndarray, ytr: np.ndarray) -> np.ndarray:
    """Difference of class means — the one direction everything in B is measured against."""
    return Xtr[ytr].mean(0) - Xtr[~ytr].mean(0)


def _geometry(X: np.ndarray, y: np.ndarray, groups: list[tuple[list, list]] | None) -> dict:
    """Label-aware descriptive geometry of one split. No fit, so no cross-validation.

    All of these are properties of the split as a whole rather than of a model trained
    on part of it, which is the point: a practitioner can compute them before deciding
    to generate anything.
    """
    Xp, Xn = X[y], X[~y]
    mu_p, mu_n = Xp.mean(0), Xn.mean(0)

    w = mu_p - mu_n
    nw = np.linalg.norm(w)
    s = X @ (w / nw) if nw > 0 else np.zeros(len(X))
    sp, sn = s[y], s[~y]
    denom = sp.var() + sn.var()
    fisher = float((sp.mean() - sn.mean()) ** 2 / denom) if denom > 0 else float("nan")

    cos_centroid = float(mu_p @ mu_n / (np.linalg.norm(mu_p) * np.linalg.norm(mu_n)))

    # Participation ratio of the pooled WITHIN-class covariance, (tr S)^2 / tr(S^2).
    # S is 5376x5376 and never formed: with Xw the within-class-centred rows,
    # tr S = ||Xw||_F^2 / N and tr(S^2) = ||Xw Xw^T||_F^2 / N^2 (Gram is N x N).
    Xw = X.astype(np.float64).copy()
    Xw[y] -= mu_p
    Xw[~y] -= mu_n
    gram = Xw @ Xw.T
    tr_s = float(np.trace(gram)) / len(X)
    tr_s2 = float((gram * gram).sum()) / len(X) ** 2
    pr_within = float(tr_s ** 2 / tr_s2) if tr_s2 > 0 else float("nan")

    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Z @ Z.T
    same = y[:, None] == y[None, :]
    off = ~np.eye(len(X), dtype=bool)
    cos_within = float(S[same & off].mean())
    cos_between = float(S[~same].mean())

    out = {
        "fisher_dom": fisher,
        "cos_centroid": cos_centroid,
        "pr_within": pr_within,
        "cos_within": cos_within,
        "cos_between": cos_between,
        "cos_gap": cos_within - cos_between,
        "centroid_dist_rel": float(nw / np.sqrt(tr_s)) if tr_s > 0 else float("nan"),
    }

    # Class-paired splits: how big the label flip is next to how far apart the items
    # are. A split whose pairs differ only in the last message has a small numerator.
    if groups:
        within, mids = [], []
        for pos_idx, neg_idx in groups:
            cp, cn = X[pos_idx].mean(0), X[neg_idx].mean(0)
            within.append(np.linalg.norm(cp - cn))
            mids.append((cp + cn) / 2.0)
        M = np.stack(mids)
        centred = M - M.mean(0)
        between = float(np.sqrt((centred ** 2).sum(1).mean()) * np.sqrt(2.0))
        out["pair_within_rel"] = float(np.mean(within) / between) if between > 0 else float("nan")
    else:
        out["pair_within_rel"] = float("nan")
    return out


def _pair_groups(rows: list[dict], y: np.ndarray):
    """Prefix groups carrying both classes, or ``None`` if the split is not paired.

    Follows ``analyze_row_settling.load_split``: rows are grouped on every message but
    the last, so a pair is one context with both a following turn that has the property
    and one that does not.
    """
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        msgs = json.loads(r["inputs"])
        groups["\n".join(f"{m['role']}: {m['content']}" for m in msgs[:-1])].append(i)
    out = []
    for idx in groups.values():
        pos = [i for i in idx if y[i]]
        neg = [i for i in idx if not y[i]]
        if not pos or not neg:
            return None
        out.append((pos, neg))
    return out if len(out) > 1 else None


def features_for_split(split: Split, curve_rows: list[dict]) -> dict:
    """Every predictor for one split, both poolings. Appends to ``curve_rows`` in place."""
    from sklearn.decomposition import PCA

    ref = _fit_curves_ref()
    rows = _rows(split)
    y = np.load(split.pooled("labels"))
    ntok = np.load(split.pooled("ntokens"))
    groups = _pair_groups(rows, y)

    # --- E: controls, no activations involved
    turns = np.array([len(json.loads(r["inputs"])) for r in rows], dtype=float)
    out = {
        "split": split.knee, "concept": split.concept, "stem": split.stem,
        "n_rows": len(rows), "frac_positive": float(y.mean()),
        "mean_tokens": float(ntok.mean()), "median_tokens": float(np.median(ntok)),
        "frac_truncated": float((ntok >= 1024).mean()),
        "mean_turns": float(turns.mean()), "paired": int(groups is not None),
        "n_pairs": len(groups) if groups else 0,
    }

    for pooling in POOLINGS:
        X = np.load(split.pooled(pooling))
        views = _fold_views(X, y, SEED)
        # Per-split stream, but a REPRODUCIBLE one: Python's hash() of a str is salted
        # per process (PYTHONHASHSEED), so it would reseed differently on every run.
        tag = f"{split.knee}/{pooling}".encode()
        rng = np.random.default_rng(SEED + int.from_bytes(hashlib.sha256(tag).digest()[:4], "big"))
        suf = f"_{pooling}"

        # --- B: full logistic regression, one direction, and one direction from 4 rows
        auroc_full = float(np.mean([
            _auroc(v["yte"], _lr(1.0).fit(v["Xtr"], v["ytr"]).decision_function(v["Xte"]))
            for v in views]))
        auroc_dom = float(np.mean([
            _auroc(v["yte"], v["Xte"] @ _dom_direction(v["Xtr"], v["ytr"])) for v in views]))
        dom2 = []
        for v in views:
            pos, neg = np.where(v["ytr"])[0], np.where(~v["ytr"])[0]
            for _ in range(N_DRAWS_DOM2):
                p2 = rng.choice(pos, 2, replace=False)
                n2 = rng.choice(neg, 2, replace=False)
                w = v["Xtr"][p2].mean(0) - v["Xtr"][n2].mean(0)
                dom2.append(_auroc(v["yte"], v["Xte"] @ w))
        gain_full = auroc_full - 0.5
        out[f"auroc_full{suf}"] = auroc_full
        out[f"auroc_dom{suf}"] = auroc_dom
        out[f"auroc_dom2{suf}"] = float(np.mean(dom2))
        out[f"auroc_dom2_sd{suf}"] = float(np.std(dom2))
        # r1 is undefined when the full model itself barely beats chance: the ratio
        # would be noise over noise. 0.02 is four times the paper's single-fit jitter.
        out[f"r1{suf}"] = float((auroc_dom - 0.5) / gain_full) if gain_full > 0.02 else float("nan")

        # --- A: the in-distribution few-shot curve, fitted with the paper's own fitter
        for C in C_VALUES:
            curve = _fewshot_curve(views, C, rng)
            tag = f"{suf}_C{C:g}"
            for k, vals in curve.items():
                curve_rows.append({
                    "split": split.knee, "concept": split.concept, "curve": "A_fewshot",
                    "pooling": pooling, "C": C, "x": k,
                    "auroc_mean": float(np.mean(vals)), "auroc_sd": float(np.std(vals)),
                    "n_obs": len(vals),
                })
                out[f"auroc_ID_k{k}{tag}"] = float(np.mean(vals))
            if len(curve) >= 5:
                f = ref.fit_curve(curve, boot=False)
                out[f"m_ID{tag}"] = float(f["m"])
                # Censor at the smallest size THIS curve measured (k=2), not at the
                # synthetic curves' floor of 10 (ref.M_MIN): an in-distribution knee of
                # 3 is genuinely resolved here, and clamping it to 10 would tie together
                # eight of the fourteen splits on an artefact of the other experiment's
                # size grid.
                out[f"log_m_ID{tag}"] = float(np.log10(max(f["m"], min(K_FEWSHOT))))
                out[f"n90_ID{tag}"] = float(f["n90"])
                out[f"log_n90_ID{tag}"] = float(np.log10(min(f["n90"], 1e6)))
                out[f"U_ID{tag}"] = float(f["U"])
                out[f"k_ID{tag}"] = float(f["k"])
                out[f"gain_obs_ID{tag}"] = float(f["gain_obs"])
                out[f"flat_ID{tag}"] = int(f["gain_obs"] < 0.02)
                out[f"rmse_ID{tag}"] = float(f["rmse"])
            out[f"n_sizes_ID{tag}"] = len(curve)

        # --- C: how many principal directions the signal needs
        max_d = min(min(len(v["tr"]) for v in views) - 1, X.shape[1], max(D_PCA))
        ds = [d for d in D_PCA if d <= max_d]
        pcs = [PCA(n_components=max(ds), svd_solver="randomized",
                   random_state=SEED).fit(v["Xtr"]) for v in views]
        auroc_d = {}
        for d in ds:
            auroc_d[d] = float(np.mean([
                _auroc(v["yte"],
                       _lr(1.0).fit(p.transform(v["Xtr"])[:, :d], v["ytr"])
                       .decision_function(p.transform(v["Xte"])[:, :d]))
                for v, p in zip(views, pcs)]))
            curve_rows.append({
                "split": split.knee, "concept": split.concept, "curve": "C_pca",
                "pooling": pooling, "C": 1.0, "x": d,
                "auroc_mean": auroc_d[d], "auroc_sd": float("nan"), "n_obs": N_FOLDS,
            })
            out[f"auroc_pca_d{d}{suf}"] = auroc_d[d]
        # "all dimensions" is the full-feature model: logistic regression is invariant
        # to the rotation into the complete PC basis, so they are the same model.
        auroc_d[X.shape[1]] = auroc_full
        curve_rows.append({
            "split": split.knee, "concept": split.concept, "curve": "C_pca",
            "pooling": pooling, "C": 1.0, "x": X.shape[1],
            "auroc_mean": auroc_full, "auroc_sd": float("nan"), "n_obs": N_FOLDS,
        })
        for frac, name in ((0.95, "d95"), (0.90, "d90")):
            need = 0.5 + frac * gain_full
            hit = [d for d in sorted(auroc_d) if auroc_d[d] >= need]
            out[f"{name}{suf}"] = float(hit[0]) if hit else float("nan")
            out[f"log_{name}{suf}"] = float(np.log10(hit[0])) if hit else float("nan")

        # --- D: geometry
        for key, val in _geometry(X, y, groups).items():
            out[f"{key}{suf}"] = val
        del X, views, pcs
    return out


def stage_features(args) -> None:
    import csv

    missing = [s.stem for s in SPLITS if not s.pooled("mean").exists()]
    if missing:
        raise SystemExit(f"not pooled yet: {missing}. Run --stage pool first.")

    targets = load_targets()
    curve_rows: list[dict] = []
    rows = []
    for split in SPLITS:
        row = features_for_split(split, curve_rows)
        row.update(targets.get(split.knee, {}))
        rows.append(row)
        print(f"[features] {split.knee}: auroc_full_mean={row['auroc_full_mean']:.3f} "
              f"r1={row.get('r1_mean', float('nan')):.2f} "
              f"m_ID={row.get('m_ID_mean_C1', float('nan')):.0f} "
              f"target={row.get(PRIMARY_TARGET, float('nan')):.2f}")

    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (SCRIPTS / "knee_predictors.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with (SCRIPTS / "knee_predictor_curves.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curve_rows[0]))
        w.writeheader()
        w.writerows(curve_rows)
    print(f"[features] {len(rows)} splits -> knee_predictors.csv ({len(cols)} columns); "
          f"{len(curve_rows)} curve points -> knee_predictor_curves.csv")


# --------------------------------------------------------------------------- #
# stage: analyse
# --------------------------------------------------------------------------- #
N_PERM = 10_000
N_BOOT = 2_000
TARGETS = ([f"log_m_{r}" for r in RECIPES] + [f"log_n90_{r}" for r in RECIPES])

# Columns of knee_predictors.csv that are not predictors.
NON_PREDICTORS = {"split", "concept", "stem"} | set(TARGETS) | {
    f"n_curves_{r}" for r in RECIPES}


def _centred_ranks(a: np.ndarray) -> np.ndarray:
    """Mean-centred, unit-norm tie-averaged ranks: rho is then just a dot product."""
    from scipy.stats import rankdata

    r = rankdata(a)
    r = r - r.mean()
    n = np.sqrt(r @ r)
    return r / n if n > 0 else r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho, ties averaged — the target is censored at 10, so ties are the rule."""
    ra, rb = _centred_ranks(a), _centred_ranks(b)
    return float(ra @ rb) if ra.any() and rb.any() else float("nan")


def _perm_p(x: np.ndarray, y: np.ndarray, rho: float, rng) -> float:
    """Two-sided permutation p-value for rho: how often chance beats it in magnitude.

    With 14 points the asymptotic t-approximation is not to be trusted, and the tied,
    censored target breaks its assumptions anyway; shuffling the target is exact up to
    the number of permutations drawn. Permuting *ranks* rather than values is the same
    null and lets all N_PERM shuffles run as one matrix product.
    """
    if not np.isfinite(rho):
        return float("nan")
    rx, ry = _centred_ranks(x), _centred_ranks(y)
    perms = np.argsort(rng.random((N_PERM, len(y))), axis=1)
    null = ry[perms] @ rx
    hits = int((np.abs(null) >= abs(rho) - 1e-12).sum())
    return (hits + 1) / (N_PERM + 1)


def _boot_ci(x: np.ndarray, splits: list[str], recipe: str, per_split_curves: dict,
             rng) -> tuple[float, float]:
    """95% CI for rho, resampling the CURVES behind each split's median.

    Two sources of noise reach the target and both belong in the interval: which four
    generators happened to be run (resampling the curves with replacement) and how well
    each curve's own knee is pinned down (jittering log10 m by the bootstrap sd the
    fitter already stored in ``lm_bvar``). The re-clamp at log10(10) after jittering is
    what keeps a censored curve censored across replicates.
    """
    ref = _fit_curves_ref()
    floor = np.log10(ref.M_MIN)
    pools = [per_split_curves.get((s, recipe), []) for s in splits]
    if any(not p for p in pools):
        return float("nan"), float("nan")
    rx = _centred_ranks(x)
    # One (N_BOOT, n_splits) matrix of resampled medians, built column by column.
    med = np.empty((N_BOOT, len(pools)))
    for j, pool in enumerate(pools):
        lm = np.array([c["lm"] for c in pool])
        sd = np.sqrt(np.maximum([c["lm_bvar"] for c in pool], 0.0))
        pick = rng.integers(0, len(pool), (N_BOOT, len(pool)))
        drawn = lm[pick] + rng.normal(0.0, 1.0, pick.shape) * sd[pick]
        med[:, j] = np.median(np.maximum(drawn, floor), axis=1)
    rhos = [r for r in (float(rx @ _centred_ranks(m)) for m in med) if np.isfinite(r)]
    if not rhos:
        return float("nan"), float("nan")
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(lo), float(hi)


def _loo_rmse(pred: np.ndarray, y: np.ndarray, concept: list[str],
              use_predictor: bool, use_concept: bool) -> float:
    """Leave-one-split-out RMSE of a linear model, in log10 units.

    With neither term this is the grand-mean baseline; with concept alone it is the
    "predict a split from the other splits of its concept" baseline the brief names as
    the bar to clear. Every fit sees 13 splits and predicts the 14th.
    """
    n = len(y)
    levels = sorted(set(concept))
    errs = []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        cols = [np.ones((len(tr), 1))]
        test = [np.ones((1, 1))]
        if use_concept:
            # Drop the first level; a concept unseen in training contributes nothing,
            # which is the right behaviour for a leave-one-out on 3 concepts.
            cols.append(np.array([[1.0 if concept[j] == l else 0.0 for l in levels[1:]]
                                  for j in tr]))
            test.append(np.array([[1.0 if concept[i] == l else 0.0 for l in levels[1:]]]))
        if use_predictor:
            cols.append(pred[tr][:, None])
            test.append(np.array([[pred[i]]]))
        X = np.concatenate(cols, 1)
        coef, *_ = np.linalg.lstsq(X, y[tr], rcond=None)
        errs.append(float((np.concatenate(test, 1) @ coef)[0]) - y[i])
    return float(np.sqrt(np.mean(np.square(errs))))


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / tot if tot > 0 else float("nan")


def _design(rows: list[dict], factors: list[str], extra: np.ndarray | None = None):
    cols = [np.ones((len(rows), 1))]
    for f in factors:
        levels = sorted(set(r[f] for r in rows))
        cols.append(np.array([[1.0 if r[f] == l else 0.0 for l in levels[1:]] for r in rows]))
    if extra is not None:
        cols.append(extra[:, None])
    return np.concatenate(cols, 1)


def stage_analyse(args) -> None:
    import csv

    pred_path = SCRIPTS / "knee_predictors.csv"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path.name} missing. Run --stage features first.")
    with pred_path.open() as fh:
        prows = list(csv.DictReader(fh))
    order = [r["split"] for r in prows]
    concepts = [r["concept"] for r in prows]

    curves = load_curves()
    per_split_curves: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for c in curves:
        if not c["flat"]:
            per_split_curves[(c["split"], c["recipe"])].append(c)

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) if r.get(name, "") not in ("", None) else np.nan
                         for r in prows])

    predictors = [k for k in prows[0] if k not in NON_PREDICTORS]
    inst = np.array([c == "instructions" for c in concepts])

    stats = []
    for target in TARGETS:
        y_all = col(target)
        recipe = target.split("_")[-1]
        for name in predictors:
            x_all = col(name)
            ok = np.isfinite(x_all) & np.isfinite(y_all)
            n = int(ok.sum())
            row = {"predictor": name, "target": target, "n_splits": n}
            if n < 5 or np.nanstd(x_all[ok]) == 0:
                stats.append(row)
                continue
            x, y = x_all[ok], y_all[ok]
            rho = _spearman(x, y)
            rng = np.random.default_rng(SEED)
            row["rho"] = rho
            row["p_perm"] = _perm_p(x, y, rho, rng)
            if n == len(prows):   # CI only where every split has the target
                lo, hi = _boot_ci(x, order, recipe, per_split_curves,
                                  np.random.default_rng(SEED + 1))
                row["ci_lo"], row["ci_hi"] = lo, hi

            cs = [concepts[i] for i in np.where(ok)[0]]
            row["loo_rmse_grand"] = _loo_rmse(x, y, cs, False, False)
            row["loo_rmse_concept"] = _loo_rmse(x, y, cs, False, True)
            row["loo_rmse_pred"] = _loo_rmse(x, y, cs, True, False)
            row["loo_rmse_pred_concept"] = _loo_rmse(x, y, cs, True, True)
            row["beats_concept"] = int(row["loo_rmse_pred"] < row["loo_rmse_concept"])

            sel = ok & inst
            if sel.sum() >= 5 and np.nanstd(x_all[sel]) > 0:
                r_in = _spearman(x_all[sel], y_all[sel])
                row["rho_instructions"] = r_in
                row["p_instructions"] = _perm_p(x_all[sel], y_all[sel], r_in,
                                                np.random.default_rng(SEED + 2))
                row["sign_holds"] = int(np.sign(r_in) == np.sign(rho) and r_in != 0)
            stats.append(row)

    # --- curve level: does the predictor buy anything over generator + prompt?
    nonflat = [c for c in curves if not c["flat"]]
    for c in nonflat:
        c["prompt"] = c["recipe"]
    y_c = np.array([c["lm"] for c in nonflat])
    base_r2 = _r2(y_c, _design(nonflat, ["gen", "prompt"]))
    split_r2 = _r2(y_c, _design(nonflat, ["gen", "prompt", "split"]))
    by_split = {r["split"]: r for r in prows}
    for row in stats:
        if row["target"] != PRIMARY_TARGET or "rho" not in row:
            continue
        vals = np.array([float(by_split[c["split"]][row["predictor"]] or "nan")
                         for c in nonflat])
        if not np.isfinite(vals).all():
            continue
        row["curve_r2_base"] = base_r2
        row["curve_r2_with"] = _r2(y_c, _design(nonflat, ["gen", "prompt"], vals))
        row["curve_r2_gain"] = row["curve_r2_with"] - base_r2
        row["curve_r2_split_ceiling"] = split_r2 - base_r2

    cols = list(dict.fromkeys(k for r in stats for k in r))
    with (SCRIPTS / "knee_predictor_stats.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(stats)
    print(f"[analyse] {len(stats)} predictor x target rows -> knee_predictor_stats.csv")
    print(f"[analyse] curve level: {len(nonflat)} non-flat gemma curves; "
          f"R2(gen+prompt)={base_r2:.3f}, +split={split_r2:.3f} "
          f"(ceiling gain {split_r2 - base_r2:.3f})")

    # --- the headline: best predictor on the primary target, by |rho|
    primary = [r for r in stats if r["target"] == PRIMARY_TARGET
               and np.isfinite(r.get("rho", np.nan)) and r["n_splits"] == len(prows)]
    primary.sort(key=lambda r: -abs(r["rho"]))
    print("\n[analyse] top 12 predictors of " + PRIMARY_TARGET + " over 14 splits:")
    print(f"  {'predictor':34s} {'rho':>6s} {'p':>7s} {'95% CI':>16s} "
          f"{'LOO':>6s} {'concept':>7s} {'grand':>6s} {'rho_inst':>8s}")
    for r in primary[:12]:
        ci = (f"[{r.get('ci_lo', float('nan')):+.2f},{r.get('ci_hi', float('nan')):+.2f}]")
        print(f"  {r['predictor']:34s} {r['rho']:+.3f} {r['p_perm']:7.4f} {ci:>16s} "
              f"{r['loo_rmse_pred']:6.3f} {r['loo_rmse_concept']:7.3f} "
              f"{r['loo_rmse_grand']:6.3f} {r.get('rho_instructions', float('nan')):+8.3f}")

    _family_wise(prows, predictors, col(PRIMARY_TARGET), primary)

    if primary:
        best = primary[0]
        _write_scatter(best["predictor"], prows, per_split_curves)
        print(f"\n[analyse] scatter written for {best['predictor']}")
        _verdict(primary)


def _family_wise(prows: list[dict], predictors: list[str], y: np.ndarray,
                 primary: list[dict]) -> None:
    """Is the BEST of ~115 predictors better than the best of 115 noise predictors?

    Each predictor's own permutation p-value asks the wrong question once a hundred of
    them have been tried: at p < 0.05 roughly six would pass on noise alone. The honest
    test permutes the target and takes the largest |rho| across the whole family, which
    is the distribution the winner actually has to beat.
    """
    usable = [p for p in predictors
              if np.isfinite(np.array([float(r[p]) if r.get(p, "") != "" else np.nan
                                       for r in prows])).all()]
    R = np.stack([_centred_ranks(np.array([float(r[p]) for r in prows])) for p in usable])
    ry = _centred_ranks(y)
    obs = np.abs(R @ ry)
    rng = np.random.default_rng(SEED + 3)
    perms = np.argsort(rng.random((N_PERM, len(y))), axis=1)
    null_max = np.abs(R @ ry[perms].T).max(axis=0)          # (N_PERM,)
    p_fw = float(((null_max >= obs.max() - 1e-12).sum() + 1) / (N_PERM + 1))
    n_nominal = sum(1 for r in primary if r["p_perm"] < 0.05)
    print(f"\n[analyse] family-wise, {len(usable)} predictors x {N_PERM} permutations:")
    print(f"    best |rho| = {obs.max():.3f} ({usable[int(obs.argmax())]}), "
          f"family-wise p = {p_fw:.4f}")
    print(f"    predictors at nominal p < 0.05: {n_nominal} "
          f"(expected by chance alone: {0.05 * len(usable):.1f})")
    print(f"    median of the null's max |rho|: {np.median(null_max):.3f}")


def _write_scatter(predictor: str, prows: list[dict], per_split_curves: dict) -> None:
    import csv

    rows = []
    for r in prows:
        pool = per_split_curves.get((r["split"], "detailed"), [])
        lms = sorted(c["lm"] for c in pool)
        rows.append({
            "split": r["split"], "concept": r["concept"],
            "predictor": predictor, "predictor_value": r[predictor],
            "log_m": r.get(PRIMARY_TARGET, ""),
            "log_m_lo": f"{lms[0]:.4f}" if lms else "",
            "log_m_hi": f"{lms[-1]:.4f}" if lms else "",
            "n_curves": len(pool),
        })
    with (SCRIPTS / "knee_predictor_scatter.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _verdict(primary: list[dict]) -> None:
    """The brief's three conditions, applied without softening."""
    passed = [r for r in primary
              if abs(r["rho"]) >= 0.6 and r["p_perm"] < 0.05
              and r.get("beats_concept") == 1 and r.get("sign_holds") == 1]
    print("\n[analyse] rho >= 0.6, p < 0.05, beats concept-only LOO, sign holds within "
          f"instructions: {len(passed)} predictor(s)")
    for r in passed[:20]:
        print(f"    {r['predictor']:34s} rho={r['rho']:+.3f} p={r['p_perm']:.4f} "
              f"LOO {r['loo_rmse_pred']:.3f} < concept {r['loo_rmse_concept']:.3f} "
              f"rho_inst={r['rho_instructions']:+.3f}")
    if not passed:
        print("    none. The null is the result.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("pool", "features", "analyse", "all"))
    ap.add_argument("--chunk", type=int, default=16,
                    help="rows of an activation blob held at once during pooling")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="restrict to these splits (knee_fits.csv names); default all 14")
    args = ap.parse_args(argv)

    if args.splits:
        unknown = [s for s in args.splits if s not in BY_KNEE]
        if unknown:
            raise SystemExit(f"unknown split(s): {unknown}; known: {sorted(BY_KNEE)}")
        global SPLITS
        SPLITS = [BY_KNEE[s] for s in args.splits]

    if args.stage in ("pool", "all"):
        stage_pool(args)
    if args.stage in ("features", "all"):
        stage_features(args)
    if args.stage in ("analyse", "all"):
        stage_analyse(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
