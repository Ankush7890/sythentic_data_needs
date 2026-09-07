"""8-draw 90%/80% resampling for the two human-harm +att arms E25 and E26.

Same design and same code path as analysis/refit_studies/scripts/run_subsample_attacker.py
on origin/human_harm_last — the draw seed string, the judge-confidence gate, the
preprocessing config, the fit seed and the inherited ensemble size are all copied from it,
so these draws are built exactly as their +desc/general siblings' were. The one thing that
had to be supplied per arm is `eval_data_description`: it is folded into the contrastive
cache key, so passing the arm's own REWRITTEN text is what makes the cached pairs hit
instead of being regenerated under a different prompt.
"""
import json, sys, time, random, shutil, pathlib, traceback
sys.path.insert(0, "/workspace/probe_auto_improvement/src")
from agentic_redteam.retrain import retrain_probe
from agentic_redteam.evaluation import evaluate_probe
from agentic_redteam.config import PreprocessingConfig
from agentic_redteam.circuit_breaker import OpenRouterOutageError

R = pathlib.Path("/workspace/probe_auto_improvement")
S = pathlib.Path("/tmp/claude-1000/-workspace-probe-auto-improvement/9dd30d0b-7bf2-4222-a38c-b921265a2f45/scratchpad")
D = S / "attexp"; OUT = D / "out"; OUT.mkdir(exist_ok=True)
CACHE = R / ".gen1_worktree/cache_gen_gemma27b_hu_harm/base_activations"
ECACHE = R / ".gen1_worktree/cache_gen_gemma27b_hu_harm/eval_activations"
DEV = R / "dev_samples/hu_ha"; EVAL = R / "eval_sets/hu_ha"
SPLITS = ["eval_ai_dilemmas", "eval_ant_hh", "eval_balanced_refusal", "eval_daily_dilemmas"]
PREP = PreprocessingConfig(provider="openrouter", model="openai/gpt-5.1", max_concurrent=50,
                           max_tokens=2048, filter_percentile=0.8, assistant_centric=True)
DESCS = json.load(open(D / "descs.json"))
ARMS = [{"key": "e25_edatt", "base": R / "data/hu_harm_gptoss_50.jsonl"},
        {"key": "e26_edatt", "base": R / "data/hu_harm_deepseekv4pro_50.jsonl"}]

def successes(d):
    rows = []
    for et in ("fp", "fn"):
        for line in open(d / f"{et}.jsonl"):
            if json.loads(line).get("success"):
                rows.append(line if line.endswith("\n") else line + "\n")
    return rows

def run(arm, frac, draw):
    tag = f'{arm["key"]}_f{int(frac*100)}_d{draw}'
    res_path = OUT / f"{tag}.json"
    if res_path.exists():
        return json.load(open(res_path))
    d = D / arm["key"]
    rows = successes(d)
    k = max(1, round(len(rows) * frac))
    rnd = random.Random(f'{arm["key"]}|{frac}|{draw}')     # same seed string as the siblings
    keep = rnd.sample(rows, k)
    jl = OUT / f"{tag}.jsonl"; jl.write_text("".join(keep))
    cc = OUT / f"{tag}_contrastive.jsonl"
    if not cc.exists() and (d / "contrastive_cache.jsonl").exists():
        shutil.copy(d / "contrastive_cache.jsonl", cc)     # copy: never mutate the arm's cache
    probe_out = OUT / f"{tag}.pkl"
    t0 = time.time()
    print(f"\n===== {tag}: {k}/{len(rows)} successes =====", flush=True)
    retrain_probe(jsonl_path=[jl], base_probe_path=d / "probe_iter0.pkl",
                  base_training_data_path=arm["base"], new_probe_path=probe_out,
                  preprocessing=PREP, contrastive_cache_path=cc, min_judge_confidence=7,
                  dev_data_path=DEV, seed=42, ensemble_size=None,
                  base_activation_cache_dir=CACHE,
                  combine_consecutive_messages=True, convert_tool_to_assistant=True,
                  eval_data_description=DESCS[arm["key"]], verbose=True)
    df = evaluate_probe(str(probe_out), str(EVAL), str(ECACHE), splits=None, max_samples=None,
                        seed=42, combine_consecutive_messages=True, convert_tool_to_assistant=True)
    p = df.set_index("dataset")["auroc"]; sp = p[SPLITS]
    res = dict(arm=arm["key"], concept="hu_harm", frac=frac, draw=draw, n=k, n_all=len(rows),
               mean=round(float(sp.mean()), 4),
               splits={kk: round(float(v), 4) for kk, v in sp.items()},
               minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, open(res_path, "w"), indent=1)
    print("RESULT " + json.dumps(res), flush=True)
    probe_out.unlink(missing_ok=True)
    return res

if __name__ == "__main__":
    for frac in (0.9, 0.8):
        for arm in ARMS:
            for draw in range(8):
                try:
                    run(arm, frac, draw)
                except OpenRouterOutageError as exc:
                    print(f"\n[ABORT] OpenRouter unusable: {exc}", flush=True); sys.exit(3)
                except Exception:
                    traceback.print_exc(); print(f"[FAILED] {arm['key']} {frac} d{draw}", flush=True)
    print("ATT_REFITS_DONE", flush=True)
