"""90% resampling for the four human-harm `_evaldesc_new` arms (arms 9-12).

Same design and same code path as run_subsample_attacker.py — for every arm, 8 independent
random subsets holding 90% of that arm's successes, refit and re-evaluated; everything else
is the arm's own recipe (its 50-row base, its probe_iter0, its own contrastive cache,
judge-confidence 7, the shared dev set, fit seed 42, ensemble size inherited), so the only
thing varying within an arm is which successes are kept. The draw seed is the same string,
`{arm}|{frac}|{draw}`, so a draw here is built exactly as its memo / evaldesc / +att
siblings' draws were and the four new arms drop straight into the existing grid.

These are the four arms the earlier passes could not cover because they had not been run
yet: `eval.data_description` delivered JUDGE-SIDE ONLY (attacker.show_eval_data_description
false) on all four attackers.

`eval_data_description` is passed per arm because that text is folded into the contrastive
cache key: omit it and every lookup misses, the records come back unpaired, and the pairs
are regenerated under a different prompt than the arm itself used. These four carry the
REWRITTEN description (d793fe5d), which is what their own retrains ran under.

Results land in analysis/refit_studies/subsample/{arm}_f90_d{n}.json, the same directory
and naming the other 22 arms use, so summarize_subsample.py picks them up unchanged.
"""
import json, sys, time, random, shutil, pathlib, traceback

REPO = pathlib.Path("/workspace/probe_auto_improvement")
sys.path.insert(0, str(REPO / "src"))
from agentic_redteam.retrain import retrain_probe                # noqa: E402
from agentic_redteam.evaluation import evaluate_probe            # noqa: E402
from agentic_redteam.config import PreprocessingConfig, load_config  # noqa: E402
from agentic_redteam.circuit_breaker import OpenRouterOutageError    # noqa: E402

OUT = REPO / "analysis/refit_studies/subsample"
WORK = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("/")
                    else "/tmp/claude-1000/-workspace-probe-auto-improvement/"
                         "8d901f36-9756-413b-8a24-6c44cbc475af/scratchpad/ednew")
WORK.mkdir(parents=True, exist_ok=True)

DEV = REPO / "dev_samples/hu_ha"
EVAL = REPO / "eval_sets/hu_ha"
CACHE = REPO / "results_hu_harm_gemma27b_batch_ablation/base_activations"
ECACHE = REPO / "results_hu_harm_gemma27b_batch_ablation/eval_activations"
SPLITS = ["eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal", "eval_daily_dilemmas"]

# Identical to the `preprocessing:` block in all four arms' configs (verified byte-for-byte
# across the twelve human-harm configs), and to what the other 22 arms' draws used.
PREP = PreprocessingConfig(provider="openrouter", model="openai/gpt-5.1", max_concurrent=50,
                           max_tokens=2048, filter_percentile=0.8, assistant_centric=True)

ARMS = [
    dict(key="hh_l70_ednew", stem="llama70b_l70base",      jl="l70base_evaldesc_new",
         base="data/hu_harm_llama70b_50.jsonl",
         config="configs/llama70b_hu_harm_gemma27b_l70base_evaldesc_new.md"),
    dict(key="hh_go_ednew", stem="gptoss120b_gptossbase",  jl="gptossbase_evaldesc_new",
         base="data/hu_harm_gptoss_50.jsonl",
         config="configs/gptoss120b_hu_harm_gemma27b_gptossbase_evaldesc_new.md"),
    dict(key="hh_nm_ednew", stem="nemotron_nmbase",        jl="nmbase_evaldesc_new",
         base="data/hu_harm_nemotron_50.jsonl",
         config="configs/nemotron_hu_harm_gemma27b_nmbase_evaldesc_new.md"),
    dict(key="hh_ds_ednew", stem="deepseekv4pro_dsbase",   jl="dsbase_evaldesc_new",
         base="data/hu_harm_deepseekv4pro_50.jsonl",
         config="configs/deepseekv4pro_hu_harm_gemma27b_dsbase_evaldesc_new.md"),
]


def paths(arm):
    res = REPO / f"results_hu_harm_gemma27b_{arm['stem']}_evaldesc_new"
    prb = REPO / f"probes/hu_harm_gemma27b_{arm['stem']}_evaldesc_new"
    return res, prb


def successes(arm):
    """Every successful attempt of the arm, both error types, as raw JSONL lines."""
    res, _ = paths(arm)
    rows = []
    for et in ("fp", "fn"):
        f = res / f"{arm['jl']}_probing_{et}.jsonl"
        for line in f.open():
            if json.loads(line).get("success"):
                rows.append(line if line.endswith("\n") else line + "\n")
    return rows


def run(arm, frac, draw):
    tag = f'{arm["key"]}_f{int(frac * 100)}_d{draw}'
    res_path = OUT / f"{tag}.json"
    if res_path.exists():
        return json.load(res_path.open())
    _, prb = paths(arm)
    rows = successes(arm)
    k = max(1, round(len(rows) * frac))
    rnd = random.Random(f'{arm["key"]}|{frac}|{draw}')   # same seed string as every sibling
    keep = rnd.sample(rows, k)
    jl = WORK / f"{tag}.jsonl"
    jl.write_text("".join(keep))
    cc = WORK / f"{tag}_contrastive.jsonl"
    if not cc.exists() and (prb / "contrastive_cache.jsonl").exists():
        shutil.copy(prb / "contrastive_cache.jsonl", cc)  # copy: never mutate the arm's cache
    probe_out = WORK / f"{tag}.pkl"
    t0 = time.time()
    print(f"\n===== {tag}: {k}/{len(rows)} successes =====", flush=True)
    retrain_probe(jsonl_path=[jl], base_probe_path=prb / "probe_iter0.pkl",
                  base_training_data_path=REPO / arm["base"], new_probe_path=probe_out,
                  preprocessing=PREP, contrastive_cache_path=cc, min_judge_confidence=7,
                  dev_data_path=DEV, seed=42, ensemble_size=None,
                  base_activation_cache_dir=CACHE,
                  combine_consecutive_messages=True, convert_tool_to_assistant=True,
                  eval_data_description=arm["desc"], verbose=True)
    df = evaluate_probe(str(probe_out), str(EVAL), str(ECACHE), splits=None, max_samples=None,
                        seed=42, combine_consecutive_messages=True, convert_tool_to_assistant=True)
    p = df.set_index("dataset")["auroc"]
    sp = p[SPLITS]
    res = dict(arm=arm["key"], concept="hu_harm", frac=frac, draw=draw, n=k, n_all=len(rows),
               mean=round(float(sp.mean()), 4),
               splits={kk: round(float(v), 4) for kk, v in sp.items()},
               minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, res_path.open("w"), indent=1)
    print("RESULT " + json.dumps(res), flush=True)
    probe_out.unlink(missing_ok=True)     # 32 ensemble pickles is a lot of disk for no reuse
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    only = [a for a in sys.argv[1:] if not a.startswith("/")] or None
    for arm in ARMS:
        arm["desc"] = load_config(REPO / arm["config"]).eval.data_description or ""
    for arm in ARMS:
        if only and arm["key"] not in only:
            continue
        for draw in range(8):
            try:
                run(arm, 0.9, draw)
            except OpenRouterOutageError as exc:
                # Never swallow this one (see CLAUDE.md): the contrastive step needs
                # OpenRouter, so once the breaker trips every remaining fit is a no-op
                # failure. Stop and leave the finished draws on disk to resume from.
                print(f"\n[ABORT] OpenRouter unusable: {exc}", flush=True)
                print(f"[ABORT] stopped at {arm['key']} d{draw}; finished draws are in "
                      f"{OUT} and are skipped on re-run.", flush=True)
                sys.exit(3)
            except Exception:
                traceback.print_exc()
                print(f"[FAILED] {arm['key']} d{draw}", flush=True)
