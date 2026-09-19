#!/usr/bin/env python3
"""Build the base-activation blob for every attacker SUBSET by MERGING the four cached
per-attacker blobs — no model load, no forward passes.

WHY THIS IS EXACT. The four 50-row base cuts are pairwise disjoint slices of one dataset,
and a subset's base file is their concatenation in a fixed order. Activations are a
per-conversation function of a frozen model, so the activation of row i of the concatenated
file IS the activation of that row in its own part's blob. The only thing merging has to get
right is row order and padding width, and `_concatenate_consuming` — the same function the
fit itself uses to merge base with red-team — already does both.

The 200-row union is the control: it was extracted for real, so `--verify` merges the four
parts and checks the result against that blob element-by-element. If the merge reproduces an
independently-extracted blob, it reproduces the others too.

Usage:
  build_subset_base_activations.py --verify        # check the method against the 200-row blob
  build_subset_base_activations.py --sizes 2 3     # write the 6 pair + 4 triple blobs
"""
from __future__ import annotations
import argparse, importlib.util, itertools, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
_sp = importlib.util.spec_from_file_location("fcd", ROOT / "scripts/fit_combined_draws.py")
fcd = importlib.util.module_from_spec(_sp)
_argv = sys.argv[:]; sys.argv = [_argv[0]]
_sp.loader.exec_module(fcd); sys.argv = _argv

from tuberlens.interfaces.dataset import LabelledDataset   # noqa: E402
from tuberlens.model import LLMModel                     # noqa: E402
from agentic_redteam.retrain import (                    # noqa: E402
    _base_activation_cache_paths, _concatenate_consuming, stable_train_test_split,
)

CACHE = ROOT / "results_hs_gemma27b_devval/base_activations"
MODEL, LAYER, SEED = "google/gemma-3-27b-it", 32, 42
POS, NEG = "high-stakes", "low-stakes"
COMBINE = CONVERT = True          # the eval: transforms every high-stakes config pins


def _key(base_file: Path) -> Path:
    return _base_activation_cache_paths(CACHE, base_file, MODEL, LAYER, SEED,
                                        0.0, None, COMBINE, CONVERT, 1.0)[0]


def _part_dataset(code: str):
    """One attacker's 50-row base, activated from its own cached blob."""
    f = ROOT / fcd.BASE_PARTS[code]
    ds = LabelledDataset.load_from(f, pos_class_label=POS, neg_class_label=NEG,
                                   combine_consecutive_messages=COMBINE,
                                   convert_tool_to_assistant=CONVERT)
    train, _ = stable_train_test_split(ds, test_size=0.0, split_field=None, seed=SEED)
    blob = _key(f)
    if not blob.exists():
        raise SystemExit(f"missing cached blob for {code}: {blob}")
    acts = LLMModel.load_activations(blob)
    if len(acts.activations) != len(train):
        raise SystemExit(f"{code}: blob has {len(acts.activations)} rows, dataset has {len(train)}")
    return train.assign(activations=acts.activations, attention_mask=acts.attention_mask,
                        input_ids=acts.input_ids)


class _Acts:
    """The three tensors, however the dataset happens to carry them.

    `_concatenate_consuming` returns a LabelledDataset whose activation tensors live in
    `other_fields`, not as attributes, so reach for them there rather than through
    attribute access (which pydantic rejects)."""
    def __init__(self, ds):
        f = ds.other_fields
        self.activations, self.attention_mask, self.input_ids = (
            f["activations"], f["attention_mask"], f["input_ids"])


def merged_activation(codes: str) -> "_Acts":
    """Parts merged in PART_ORDER — the order subset_base_path concatenates the FILE in.
    Any other order silently pairs each row with another row's activations."""
    ordered = [c for c in fcd.PART_ORDER if c in codes]
    return _Acts(_concatenate_consuming([_part_dataset(c) for c in ordered]))


def _trim(t: torch.Tensor, width: int) -> torch.Tensor:
    """Blobs are padded to their own call's max length, so two correct blobs of the same
    rows can differ in width. Compare on the common width — the tail is pad, and
    Activation.__post_init__ has already zeroed it under the mask."""
    return t[:, :width] if t.dim() >= 2 else t


def verify() -> int:
    ref_file = ROOT / fcd.BASE
    ref_blob = _key(ref_file)
    if not ref_blob.exists():
        raise SystemExit(f"no reference blob to verify against: {ref_blob}")
    ref = LLMModel.load_activations(ref_blob)
    got = merged_activation("".join(fcd.PART_ORDER))
    w = min(ref.activations.shape[1], got.activations.shape[1])
    print(f"reference {tuple(ref.activations.shape)}  merged {tuple(got.activations.shape)}"
          f"  -> comparing on width {w}")
    bad = 0
    for name, a, b in [("activations", ref.activations, got.activations),
                       ("attention_mask", ref.attention_mask, got.attention_mask),
                       ("input_ids", ref.input_ids, got.input_ids)]:
        a, b = _trim(a, w), _trim(b, w)
        same = torch.equal(a, b)
        if not same:
            d = (a.float() - b.float()).abs()
            print(f"  {name}: MISMATCH  max|diff|={d.max():.3e}  n_differing={int((d>0).sum())}")
            bad += 1
        else:
            print(f"  {name}: identical")
    # the pad tail beyond the common width must be all-zero on whichever is wider
    for name, t in [("reference", ref.activations), ("merged", got.activations)]:
        if t.shape[1] > w and t[:, w:].abs().sum() != 0:
            print(f"  {name}: NON-ZERO data beyond width {w}"); bad += 1
    print("VERIFY OK — merging reproduces an independently extracted blob" if not bad
          else f"VERIFY FAILED ({bad} mismatches)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[])
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify and verify():
        raise SystemExit(1)
    for k in sorted(set(args.sizes)):
        for codes in itertools.combinations(fcd.PART_ORDER, k):
            codes = "".join(codes)
            base_file = fcd.subset_base_path(codes)
            dst = _key(base_file)
            if dst.exists():
                print(f"{codes}: blob already present  {dst.name}")
                continue
            m = merged_activation(codes)
            torch.save({"activations": m.activations, "attention_mask": m.attention_mask,
                        "input_ids": m.input_ids, "layer": LAYER, "model_name": MODEL}, dst)
            print(f"{codes}: wrote {dst.name}  {tuple(m.activations.shape)}  "
                  f"({dst.stat().st_size / 1e6:.0f} MB)  from {base_file.name}")


if __name__ == "__main__":
    main()
