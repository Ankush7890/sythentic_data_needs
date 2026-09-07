#!/usr/bin/env python3
"""COMBINATION study for HUMAN-HARM: pool k attackers' red-team successes, refit, resample.

The high-stakes study (origin/experiment_hs_last, scripts/fit_combined_draws.py) applied to
this concept, mirroring its three configurations exactly:

  memo  = rolling + cross-iteration memos only          (no eval.data_description)
  desc  = + eval.data_description, judge-side only      (the OLDER text)
  att   = + attacker.show_eval_data_description         (the REWRITTEN text)

ONE DELIBERATE DEPARTURE: THE BASE IS MATCHED TO THE SUBSET.
The high-stakes build_combos() held the base at all 200 rows for every subset size, so that
base size could not confound the k-curve. Here a k-attacker subset trains on exactly those
k attackers' 50-row bases (k*50 rows) — what a practitioner who ran k attackers would
actually have had. The cost is that base size now moves with k, so the k-curve mixes "more
red-team data" with "more base data"; the k=1 column is the per-arm subsample grid, which
is already on disk and was built the same way, so the two are directly comparable.
Each subset therefore needs its OWN base activation blob (the base cache is keyed on the
base file's hash) — 11 blobs for k=2,3,4, about 1400 rows of extraction, computed once.

WHAT IS HELD FIXED across the three groups: the member bases (the four 50-row sets are
pairwise disjoint), the probe template (architecture + metadata only — retrain_probe does
not warm-start weights), the dev set, the eval splits, the judge-confidence gate, the fit
seed, the message transforms and the filter + contrastive recipe.

WHAT NECESSARILY VARIES: each group passes its own eval.data_description, because that text
is folded into the contrastive cache key — memo's pairs were minted without it, desc's under
the older text, att's under the rewritten one. That is the arms' own recipe, and omitting it
would regenerate every pair under a prompt no arm ever ran.

A CAVEAT THE HIGH-STAKES VERSION DOES NOT HAVE. For the memo and desc groups, gpt-oss and
deepseek come from experiment25 / experiment26, which ran judge.eval_scope_check ON (it
defaults True and their configs do not set it) and, for desc, the older description. The
llama-70b and nemotron members of those groups ran with the knob OFF. So memo and desc are
internally mixed on that knob; the att group is clean (all four arms on human_harm_last).
Recorded per row as `scope_mixed` so it cannot silently shape the k-curve.

Draws are seeded on (combo, fraction, draw) so a re-run extends the grid rather than
recomputing it; every finished fit is skipped on resume.
"""
from __future__ import annotations

import argparse, itertools, json, random, shutil, subprocess, sys, time, traceback
from pathlib import Path

REPO = Path("/workspace/probe_auto_improvement")
sys.path.insert(0, str(REPO / "src"))
from agentic_redteam.retrain import retrain_probe                    # noqa: E402
from agentic_redteam.evaluation import evaluate_probe                # noqa: E402
from agentic_redteam.config import PreprocessingConfig, load_config  # noqa: E402
from agentic_redteam.circuit_breaker import OpenRouterOutageError    # noqa: E402

OUT = REPO / "analysis/refit_studies/hu_harm_combined"
WORK = REPO / ".hu_harm_combined_work"          # bases, pooled JSONLs, caches (gitignored)
DEV = REPO / "dev_samples/hu_ha"
EVAL = REPO / "eval_sets/hu_ha"
CACHE = REPO / "results_hu_harm_gemma27b_batch_ablation/base_activations"
ECACHE = REPO / "results_hu_harm_gemma27b_batch_ablation/eval_activations"
SPLITS = ["eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal", "eval_daily_dilemmas"]
PREP = PreprocessingConfig(provider="openrouter", model="openai/gpt-5.1", max_concurrent=50,
                           max_tokens=2048, filter_percentile=0.8, assistant_centric=True)

ATTACKERS = ["g", "d", "l", "n"]
ATTACKER_NAMES = {"g": "gpt-oss", "d": "deepseek", "l": "llama70b", "n": "nemotron"}
BASES = {"g": "data/hu_harm_gptoss_50.jsonl",       "d": "data/hu_harm_deepseekv4pro_50.jsonl",
         "l": "data/hu_harm_llama70b_50.jsonl",     "n": "data/hu_harm_nemotron_50.jsonl"}

# arm spec: (results dir, jsonl stem, probe dir, branch or None for local)
E25 = "origin/experiment25_gptoss_base_cloud"
E26 = "origin/experiment26_deepseek_cloud"
ARMS = {
 # memo
 "memo_g": ("results_hu_harm_gemma27b_gptoss120b_gptossbase_itermemo150", "gptossbase_itermemo150",
            "probes/hu_harm_gemma27b_gptoss120b_gptossbase_itermemo150", E25),
 "memo_d": ("results_hu_harm_gemma27b_deepseekv4pro_dsbase_itermemo150", "dsbase_itermemo150",
            "probes/hu_harm_gemma27b_deepseekv4pro_dsbase_itermemo150", E26),
 "memo_l": ("results_hu_harm_gemma27b_llama70b_l70base_itermemo150", "l70base_itermemo150",
            "probes/hu_harm_gemma27b_llama70b_l70base_itermemo150", None),
 "memo_n": ("results_hu_harm_gemma27b_nemotron_nmbase_itermemo150", "nmbase_itermemo150",
            "probes/hu_harm_gemma27b_nemotron_nmbase_itermemo150", None),
 # desc (OLDER text)
 "desc_g": ("results_hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc", "gptossbase_evaldesc",
            "probes/hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc", E25),
 "desc_d": ("results_hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc", "dsbase_evaldesc",
            "probes/hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc", E26),
 "desc_l": ("results_hu_harm_gemma27b_llama70b_l70base_evaldesc", "l70base_evaldesc",
            "probes/hu_harm_gemma27b_llama70b_l70base_evaldesc", None),
 "desc_n": ("results_hu_harm_gemma27b_nemotron_nmbase_evaldesc", "nmbase_evaldesc",
            "probes/hu_harm_gemma27b_nemotron_nmbase_evaldesc", None),
 # att (REWRITTEN text, shown to the attacker) — all four local
 "att_g": ("results_hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc_attacker",
           "gptossbase_evaldesc_attacker", "probes/hu_harm_gemma27b_gptoss120b_gptossbase_evaldesc_attacker", None),
 "att_d": ("results_hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker",
           "dsbase_evaldesc_attacker", "probes/hu_harm_gemma27b_deepseekv4pro_dsbase_evaldesc_attacker", None),
 "att_l": ("results_hu_harm_gemma27b_llama70b_l70base_evaldesc_attacker",
           "l70base_evaldesc_attacker", "probes/hu_harm_gemma27b_llama70b_l70base_evaldesc_attacker", None),
 "att_n": ("results_hu_harm_gemma27b_nemotron_nmbase_evaldesc_attacker",
           "nmbase_evaldesc_attacker", "probes/hu_harm_gemma27b_nemotron_nmbase_evaldesc_attacker", None),
}
# The description each GROUP's arms actually ran under; drives the contrastive cache key.
GROUP_DESC_CONFIG = {
    "memo": None,   # no description at all
    "desc": "configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc.md",            # the OLDER text
    "att":  "configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_attacker.md",  # the REWRITTEN text
}
# Groups whose gpt-oss / deepseek members ran eval_scope_check ON while l/n ran it off.
SCOPE_MIXED = {"memo": True, "desc": True, "att": False}
TEMPLATE_PROBE = REPO / "probes/hu_harm_gemma27b_llama70b_l70base_itermemo150/probe_iter0.pkl"


def materialize(key: str) -> Path:
    """Local dir with fp.jsonl / fn.jsonl / contrastive_cache.jsonl for one arm.

    Arms living on experiment25 / experiment26 are extracted from those refs with `git show`
    rather than checked out, so this branch's tree is never disturbed.
    """
    res, stem, probe, branch = ARMS[key]
    d = WORK / "arms" / key
    if (d / "fn.jsonl").exists() and (d / "contrastive_cache.jsonl").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    for et in ("fp", "fn"):
        dst = d / f"{et}.jsonl"
        if dst.exists():
            continue
        src = f"{res}/{stem}_probing_{et}.jsonl"
        if branch is None:
            shutil.copy(REPO / src, dst)
        else:
            dst.write_bytes(subprocess.run(["git", "show", f"{branch}:{src}"], cwd=REPO,
                                           check=True, capture_output=True).stdout)
    cc = d / "contrastive_cache.jsonl"
    if not cc.exists():
        src = f"{probe}/contrastive_cache.jsonl"
        if branch is None:
            shutil.copy(REPO / src, cc)
        else:
            cc.write_bytes(subprocess.run(["git", "show", f"{branch}:{src}"], cwd=REPO,
                                          check=True, capture_output=True).stdout)
    return d


def successes(key: str) -> list[str]:
    d = materialize(key)
    rows = []
    for et in ("fp", "fn"):
        for line in (d / f"{et}.jsonl").open():
            if json.loads(line).get("success"):
                rows.append(line if line.endswith("\n") else line + "\n")
    return rows


def subset_base(codes: str) -> Path:
    """Union of the member attackers' 50-row bases, in fixed code order.

    Written once per subset and content-stable, so its activation blob is minted once and
    every group and draw at that subset reuses it.
    """
    p = WORK / "bases" / f"hu_harm_base_{codes}.jsonl"
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as out:
        for c in codes:                      # codes are kept in ATTACKERS order by build_combos
            for line in (REPO / BASES[c]).open():
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
    return p


def build_combos(sizes) -> dict:
    combos = {}
    for g in ("memo", "desc", "att"):
        for k in sizes:
            for codes in itertools.combinations(ATTACKERS, k):
                cs = "".join(codes)
                combos[f"{g}_{cs}"] = dict(group=g, codes=cs, k=k,
                                           arms=[f"{g}_{c}" for c in codes],
                                           label=f"{'+'.join(ATTACKER_NAMES[c] for c in codes)} · {g}")
    return combos


def run(name: str, spec: dict, frac: float, draw: int, desc_cache: dict):
    tag = f"{name}_f{int(frac * 100)}_d{draw}"
    res_path = OUT / f"{tag}.json"
    if res_path.exists():
        return json.load(res_path.open())

    pool = []
    per_arm = {}
    for a in spec["arms"]:
        rows = successes(a)
        per_arm[a] = len(rows)
        pool.extend(rows)
    k = max(1, round(len(pool) * frac))
    # One pool over both error types AND all member attackers, drawn once: a draw must not
    # be able to rebalance fp against fn, nor one attacker against another — those ratios
    # are properties of the configuration under test.
    keep = random.Random(f'{name}|{frac}|{draw}').sample(pool, k)

    jl = WORK / "draws" / f"{tag}.jsonl"
    jl.parent.mkdir(parents=True, exist_ok=True)
    jl.write_text("".join(keep))

    cc = WORK / "draws" / f"{tag}_contrastive.jsonl"
    if not cc.exists():
        with cc.open("w") as out:           # concatenate members' caches; _load_cache is last-row-wins
            for a in spec["arms"]:
                src = WORK / "arms" / a / "contrastive_cache.jsonl"
                if src.exists():
                    for line in src.open():
                        if line.strip():
                            out.write(line if line.endswith("\n") else line + "\n")

    base = subset_base(spec["codes"])
    probe_out = WORK / "draws" / f"{tag}.pkl"
    t0 = time.time()
    print(f"\n===== {tag}: {k}/{len(pool)} successes, base {spec['k']}x50 =====", flush=True)
    retrain_probe(jsonl_path=[jl], base_probe_path=TEMPLATE_PROBE,
                  base_training_data_path=base, new_probe_path=probe_out,
                  preprocessing=PREP, contrastive_cache_path=cc, min_judge_confidence=7,
                  dev_data_path=DEV, seed=42, ensemble_size=None,
                  base_activation_cache_dir=CACHE,
                  combine_consecutive_messages=True, convert_tool_to_assistant=True,
                  eval_data_description=desc_cache[spec["group"]], verbose=True)
    df = evaluate_probe(str(probe_out), str(EVAL), str(ECACHE), splits=None, max_samples=None,
                        seed=42, combine_consecutive_messages=True, convert_tool_to_assistant=True)
    p = df.set_index("dataset")["auroc"]
    sp = p[SPLITS]
    res = dict(combo=name, group=spec["group"], codes=spec["codes"], k=spec["k"],
               label=spec["label"], frac=frac, draw=draw, n=k, n_pool=len(pool),
               per_arm=per_arm, base_rows=spec["k"] * 50, scope_mixed=SCOPE_MIXED[spec["group"]],
               mean=round(float(sp.mean()), 4),
               splits={kk: round(float(v), 4) for kk, v in sp.items()},
               minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, res_path.open("w"), indent=1)
    print("RESULT " + json.dumps({kk: res[kk] for kk in ("combo", "frac", "draw", "n", "mean")}),
          flush=True)
    probe_out.unlink(missing_ok=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4], choices=[1, 2, 3, 4],
                    help="subset sizes (default 2 3 4; k=1 is the per-arm subsample grid)")
    ap.add_argument("--combos", nargs="+", default=None)
    ap.add_argument("--draws", type=int, default=None,
                    help="draws per subset (default: 8 at k=4, 4 at k=2,3 — the hs cadence)")
    ap.add_argument("--fraction", type=float, default=0.9)
    ap.add_argument("--groups", nargs="+", default=["memo", "desc", "att"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    desc_cache = {g: (load_config(REPO / c).eval.data_description or "") if c else ""
                  for g, c in GROUP_DESC_CONFIG.items()}

    combos = build_combos(sorted(set(args.sizes)))
    names = args.combos or [n for n in combos if combos[n]["group"] in args.groups]
    # Ascending k: an interrupted sweep leaves whole cells rather than fragments of all.
    names.sort(key=lambda n: (combos[n]["k"], combos[n]["group"], combos[n]["codes"]))

    for name in names:
        spec = combos[name]
        ndraws = args.draws if args.draws is not None else (8 if spec["k"] == 4 else 4)
        for draw in range(ndraws):
            try:
                run(name, spec, args.fraction, draw, desc_cache)
            except OpenRouterOutageError as exc:
                print(f"\n[ABORT] OpenRouter unusable: {exc}", flush=True)
                print(f"[ABORT] stopped at {name} d{draw}; finished fits are in {OUT}.", flush=True)
                sys.exit(3)
            except Exception:
                traceback.print_exc()
                print(f"[FAILED] {name} d{draw}", flush=True)


if __name__ == "__main__":
    main()
