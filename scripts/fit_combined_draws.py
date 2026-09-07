#!/usr/bin/env python3
"""COMBINATION analysis for the INSTRUCTION-FOLLOWING concept: pool the red-team successes
of all four attackers within one configuration, refit, and resample.

This is the high-stakes combination study (experiment_hs_last's
scripts/fit_combined_draws.py) re-run on instruction-following, against the twelve arms of
run_gemma27b_instructions_selfbase_arms{,2}.sh + run_gemma27b_instructions_evaldesc_attacker{,_rest}.sh.
The design, the controls and the resume semantics are carried over unchanged; only the
concept, the arms and the eval/dev sets differ.

THE QUESTION. A per-arm reading answers "does showing the eval description to *this*
attacker help?", one rotation at a time. This script asks the pooled version: if a
practitioner ran all four attackers under one configuration and trained on everything they
found, would the configuration itself show up? Three configurations, one probe each per
draw:

  combo_memo  = the four `_itermemo150` arms       (rolling + cross-iteration memos only)
  combo_desc  = the four `_evaldesc` arms          (+ eval.data_description, reaching the
                                                    attacker only through the judge's memos)
  combo_att   = the four `_evaldesc_attacker` arms (+ attacker.show_eval_data_description:
                                                    the same paragraph verbatim in the
                                                    attacker's own system prompt)

WHAT IS HELD FIXED. The base training data is the UNION of the four arms' 50-row bases —
they are pairwise disjoint (verified: 0 overlap on `inputs`, 100 follows / 100
does-not-follow), so the union is exactly the base data those four attackers collectively
had, and it is the same 200 rows for all three combos. The base probe (architecture +
metadata template only; retrain_probe does not warm-start weights) is the gpt-oss memo arm's
probe_iter0 for all three. The dev set (dev_samples/instructions, 404 rows — the SIX-split
set these arms trained against, not the seven-split one), the eval splits, the
judge-confidence gate, the seed, the message transforms and the filter + contrastive recipe
are the arms' own.

Note the arms fit TEN-member ensembles and this grid fits SINGLE probes (ensemble_size=1),
exactly as the high-stakes grid did: a draw grid is about the spread across draws, and 8x
the fit cost buys none of it. So these AUROCs are not comparable to the arms' own
comparison CSVs — only to each other and to the base-only reference fit the same way.

WHAT NECESSARILY VARIES. Each combo keeps its own group's config, because
`eval.data_description` reaches `generate_contrastive_dataset`'s prompt and is folded into
the contrastive cache key — so combo_memo's pairs are minted without it and combo_desc /
combo_att's with it. That is the arms' own recipe. combo_desc and combo_att have
byte-identical retrain recipes (verified: the `preprocessing:` block is identical across all
twelve configs and the `data_description` text across all eight that carry it), so those two
differ in their DATA alone — which is the pair the question turns on.

The two error types are drawn together as one pool, draws are seeded on (combo, draw,
--seed) so a re-run extends the grid instead of recomputing it, and each --fraction and
--base-mode gets its own CSV and probe dir.
"""
from __future__ import annotations
import argparse, csv, itertools, json, random, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_redteam.config import load_config              # noqa: E402
from agentic_redteam.retrain import retrain_probe, train_initial_probe   # noqa: E402
from agentic_redteam.evaluation import evaluate_probe       # noqa: E402

OUT_PROBES = ROOT / "probes/ins_combined_draws"
OUT_RES = ROOT / "results_ins_combined_draws"
EVAL_DIR = ROOT / "eval_sets/instructions"
BASE = "data/instructions_combined_200.jsonl"        # union of the four disjoint 50-row bases
BASE_PARTS = {"g": "data/instructions_gptoss_50.jsonl",
              "d": "data/instructions_deepseekv4pro_50.jsonl",
              "l": "data/instructions_llama70b_50.jsonl",
              "n": "data/instructions_nemotron_50.jsonl"}

# THE ORDER PARTS ARE CONCATENATED IN IS LOAD-BEARING and must never change: the base
# activation blob is a row-ordered tensor built by merging the per-attacker blobs
# (scripts/build_subset_base_activations.py), so the merge order has to equal the order the
# base FILE was written in, or every row gets the wrong activation. It is sorted by the part
# FILENAME — which is the order data/instructions_combined_200.jsonl was built in, so the
# already-extracted 200-row blob stays valid. Not ATTACKERS order, which is a display order.
PART_ORDER = sorted(BASE_PARTS, key=lambda c: BASE_PARTS[c])          # d, g, l, n

# The four attackers, and the twelve arms. Subsets are named by these one-letter codes in
# this fixed order (memo_gd, desc_gln, ...), so a subset's name is stable regardless of how
# it was enumerated.
ATTACKERS = ["g", "d", "l", "n"]
ATTACKER_NAMES = {"g": "gpt-oss", "d": "deepseek", "l": "llama70b", "n": "nemotron"}
# code -> (config-file attacker prefix, run key stem, base-data tag)
_ATT = {"g": ("gptoss120b", "gptoss", "gobase"),
        "d": ("deepseekv4pro", "deepseekv4pro", "dsbase"),
        "l": ("llama70b", "llama70b", "l70base"),
        "n": ("nemotron", "nemotron", "nmbase")}
_VARIANT = {"memo": "itermemo150", "desc": "evaldesc", "att": "evaldesc_attacker"}


def _arm(code: str, group: str) -> dict:
    """One of the twelve runs. Every path this study reads is derived from one key —
    `<attacker>_<basetag>_<variant>` — because the runs were laid out that way: the results
    dir, the probe dir, the JSONL stem and the config name are all that key plus a fixed
    prefix or suffix. Deriving them beats a hand-written table that can disagree with disk."""
    cfg_prefix, stem_attacker, basetag = _ATT[code]
    variant = _VARIANT[group]
    key = f"{stem_attacker}_{basetag}_{variant}"
    return dict(
        label=f"{ATTACKER_NAMES[code]} · {group}",
        config=ROOT / f"configs/{cfg_prefix}_instructions_gemma27b_{basetag}_{variant}.md",
        res=ROOT / f"results_instructions_gemma27b_{key}",
        stem=f"{key}_probing",
        probes=ROOT / f"probes/instructions_gemma27b_{key}",
        base=BASE_PARTS[code],
    )


ARMS = {f"{g}_{c}": _arm(c, g) for g in _VARIANT for c in ATTACKERS}

GROUPS = {
    "memo": dict(label="memo", arms={c: f"memo_{c}" for c in ATTACKERS}),
    "desc": dict(label="+eval-desc", arms={c: f"desc_{c}" for c in ATTACKERS}),
    "att": dict(label="+eval-desc→attacker", arms={c: f"att_{c}" for c in ATTACKERS}),
}
for _g in GROUPS:
    # The group's config for the RETRAIN recipe. Within a group the four arms' configs
    # differ only in their base data and output paths — the preprocessing block, the
    # transforms, the confidence gate and the data_description are identical (asserted
    # below) — so any one of them is the group's recipe. gpt-oss is picked for all three.
    GROUPS[_g]["config"] = ARMS[f"{_g}_g"]["config"]

TEMPLATE_PROBE = ARMS["memo_g"]["probes"] / "probe_iter0.pkl"


def subset_base_path(codes: str) -> Path:
    """The base file for a subset under --base-mode subset: the union of just THOSE
    attackers' 50-row cuts, 50*k rows.

    Written on demand and content-deterministic — the parts are concatenated in PART_ORDER,
    so the same subset always produces a byte-identical file and therefore the same
    base-activation cache key (which hashes the file's bytes). The four cuts are pairwise
    disjoint, so a k-subset's base is exactly 50*k rows, class-balanced 25/25 per part."""
    codes = "".join(c for c in PART_ORDER if c in codes)
    if len(codes) == len(PART_ORDER):
        return ROOT / BASE                      # the 200-row union already on disk
    dst = ROOT / f"data/instructions_base_{codes}.jsonl"    # codes already in PART_ORDER
    if not dst.exists():
        dst.write_text("".join((ROOT / BASE_PARTS[c]).read_text() for c in codes))
    return dst


def build_combos(sizes) -> dict:
    """Every subset of `sizes` attackers, in every configuration.

    THE BASE IS HELD AT ALL 200 ROWS FOR EVERY SUBSET SIZE under --base-mode fixed —
    deliberately, and it is the one thing here that is not the "what would a practitioner
    have had" choice. Someone who ran two attackers would have had those two 50-row bases,
    not four; but then base size would move with subset size and the k-curve would confound
    the two. Fixing the base leaves the red-team pool as the ONLY thing that varies with k,
    which is the variable under study. It costs nothing in fidelity — the four base cuts are
    disjoint samples of one source distribution, not attacker-specific data — and it means
    every subset reuses the single cached 200-row activation blob instead of extracting its
    own. --base-mode subset is the companion design that scales the base with k instead.

    k=4 keeps the names combo_memo / combo_desc / combo_att, so those rows stay addressable
    and are skipped on resume rather than recomputed."""
    combos = {}
    for g, gs in GROUPS.items():
        for k in sizes:
            for codes in itertools.combinations(ATTACKERS, k):
                name = f"combo_{g}" if k == len(ATTACKERS) else f"{g}_{''.join(codes)}"
                who = "ALL 4" if k == len(ATTACKERS) else "+".join(ATTACKER_NAMES[c] for c in codes)
                combos[name] = dict(label=f"{who} · {gs['label']}",
                                    arms=[gs["arms"][c] for c in codes],
                                    config=gs["config"], group=g, k=k, codes="".join(codes))
    return combos


COMBOS = build_combos([4])          # rebuilt from --sizes in main()


def seed_contrastive_cache(arms: list[str], dst: Path) -> int:
    """Concatenate the member arms' contrastive caches into this configuration's cache.

    Keys are sha256(source messages + target label + guidance fingerprint) and _load_cache
    is last-row-wins, so concatenating cannot corrupt an entry — a duplicated key just
    resolves to the later copy. Only pairs that no member arm ever minted (a record the
    pooled filter keeps but every per-arm filter dropped) are regenerated."""
    if dst.exists():
        return sum(1 for _ in dst.open())
    n = 0
    with dst.open("w") as out:
        for a in arms:
            src = ARMS[a]["probes"] / "contrastive_cache.jsonl"
            if src.exists():
                for line in src.open():
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def draw_subset(arms: list[str], frac: float, rng: random.Random, out_prefix: Path):
    """Write `frac` of the POOLED successes of all member arms to fresh fp/fn JSONLs.

    One pool over both error types AND all member attackers, drawn once: a draw must not be
    able to rebalance fp against fn, nor one attacker against another — those ratios are
    properties of the configuration, which is the thing under test."""
    pool, per_arm = [], {}
    for a in arms:
        spec = ARMS[a]
        n = 0
        for et in ("fp", "fn"):
            path = Path(spec["res"]) / f"{spec['stem']}_{et}.jsonl"
            with path.open() as fh:
                for line in fh:
                    if line.strip() and json.loads(line).get("success"):
                        pool.append((et, line))
                        n += 1
        per_arm[a] = n
    k = max(1, round(len(pool) * frac))
    keep = rng.sample(pool, k)
    paths = []
    for et in ("fp", "fn"):
        dst = out_prefix.with_name(out_prefix.name + f"_{et}.jsonl")
        dst.write_text("".join(l for e, l in keep if e == et))
        paths.append(dst)
    return paths, len(pool), k, per_arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", nargs="+", default=None,
                    help="subset names (e.g. memo_gd desc_gln); default: every subset of --sizes")
    ap.add_argument("--sizes", nargs="+", type=int, default=[4], choices=[1, 2, 3, 4],
                    help="attacker-subset sizes to enumerate (default 4 = the whole rotation)")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-dir", default="dev_samples/instructions")
    ap.add_argument("--base-mode", choices=["fixed", "subset"], default="fixed",
                    help="fixed: every subset trains on all 200 base rows, so ONLY the "
                         "red-team pool varies with k (the controlled comparison). "
                         "subset: a k-subset trains on just those k attackers' 50-row cuts "
                         "(50*k rows), which is what a practitioner running those k "
                         "attackers would actually have had.")
    ap.add_argument("--base-only", action="store_true",
                    help="also fit the base ALONE (no red-team data) as the reference line "
                         "every combo is read against; written as combo 'base'/'base_<codes>' draw 0")
    args = ap.parse_args()

    global COMBOS
    COMBOS_ALL = build_combos([1, 2, 3, 4])
    COMBOS = build_combos(sorted(set(args.sizes)))
    if args.combos:
        COMBOS = COMBOS_ALL
        unknown = [c for c in args.combos if c not in COMBOS]
        if unknown:
            raise SystemExit(f"unknown combos {unknown}; known: {' '.join(COMBOS)}")
        wanted = list(args.combos)
    else:
        wanted = list(COMBOS)

    OUT_PROBES.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    dev_dir = ROOT / args.dev_dir
    if not dev_dir.is_dir():
        raise SystemExit(f"missing dev dir {dev_dir}")
    if not (ROOT / BASE).exists():
        raise SystemExit(f"missing combined base {BASE}")
    if not TEMPLATE_PROBE.exists():
        raise SystemExit(f"missing template probe {TEMPLATE_PROBE}")
    n_dev = sum(1 for p in dev_dir.glob("*.jsonl") for l in p.read_text().splitlines() if l.strip())
    # Same rule as the high-stakes grid: the resume key is (combo, draw) and carries neither
    # the fraction nor the base mode, so each of those needs its own CSV and probe dir.
    suffix = "" if abs(args.fraction - 0.9) < 1e-9 else f"_f{round(args.fraction * 100)}"
    if args.base_mode == "subset":
        suffix += "_subsetbase"
    probe_dir = Path(str(OUT_PROBES) + suffix)
    probe_dir.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_RES / f"combined_draws{suffix}.csv"
    print(f"combination analysis (instructions) | base {BASE} (200 rows) | "
          f"validation {args.dev_dir} ({n_dev} rows)\n  {args.draws} draws x "
          f"{args.fraction:.0%} of each configuration's POOLED successes | base-mode "
          f"{args.base_mode} | seed {args.seed}\n  -> {csv_path.name} , {probe_dir.name}/\n")

    done = set()
    if csv_path.exists():
        with csv_path.open() as fh:
            for row in csv.reader(fh):
                if len(row) > 2 and row[0] not in ("combo", ""):
                    done.add((row[0], row[1]))

    def score(probe_out: Path, cfg, tag: str, combo: str, d: int, n_keep: int, n_pool: int):
        df = evaluate_probe(probe_out, EVAL_DIR, cfg.output.activations_cache_dir,
                            max_samples=None, seed=args.seed,
                            combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                            convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant)
        print(df.to_string(index=False))
        df.insert(0, "combo", combo); df.insert(1, "draw", d)
        df.insert(2, "n_kept", n_keep); df.insert(3, "n_pool", n_pool)
        df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        m = float(df.loc[df["dataset"] == "mean", "auroc"].iloc[0])
        print(f"    {tag}: mean {m:.5f}  -> {csv_path.name}")

    def fit_base_only(base_path: Path, combo_name: str):
        """The no-red-team reference for one base. Every red-team cell is read against the
        probe that same base trains ALONE, so a subset-base cell needs its own reference —
        the 200-row one is not the right yardstick for a 100-row cell."""
        if (combo_name, "0") in done:
            print(f"--- {combo_name}: already scored, skipping")
            return
        cfg = load_config(COMBOS_ALL["combo_memo"]["config"])
        n_base = sum(1 for l in base_path.read_text().splitlines() if l.strip())
        probe_out = probe_dir / f"{combo_name}.pkl"
        print(f"\n--- {combo_name}: {n_base} base rows, no red-team data ({base_path.name})")
        train_initial_probe(
            base_training_data_path=base_path,
            model_name=cfg.probe.model, layer=cfg.probe.layer,
            new_probe_path=probe_out,
            pos_class_label=cfg.probe.pos_class_label,
            neg_class_label=cfg.probe.neg_class_label,
            probe_description=cfg.probe.description,
            probe_spec=cfg.probe.architecture,
            test_size=0.2, split_field=None, dev_data_path=dev_dir,
            seed=args.seed, ensemble_size=1,
            base_activation_cache_dir=cfg.output.base_activation_cache_dir,
            combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
            convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant,
            verbose=True,
        )
        score(probe_out, cfg, combo_name, combo_name, 0, 0, 0)

    if args.base_only:
        if args.base_mode == "subset":
            seen = set()
            for spec in COMBOS.values():
                codes = "".join(c for c in PART_ORDER if c in spec["codes"])
                if codes in seen:
                    continue
                seen.add(codes)
                fit_base_only(subset_base_path(codes), f"base_{codes}")
        else:
            fit_base_only(ROOT / BASE, "base")

    for combo in wanted:
        spec = COMBOS[combo]
        cfg = load_config(spec["config"])
        # ONE CACHE PER CONFIGURATION, shared by all of its subsets. The keys are content
        # hashes, so a subset can only ever read entries minted from its own conversations;
        # sharing just means the pair a triple mints is already there when a pair needs it.
        cache = OUT_RES / f"combo_{spec['group']}_contrastive_cache.jsonl"
        n_seed = seed_contrastive_cache([GROUPS[spec["group"]]["arms"][c] for c in ATTACKERS], cache)
        print(f"\n### {combo} ({spec['label']}): arms {'+'.join(spec['arms'])}, "
              f"contrastive cache {n_seed} rows")
        for d in range(args.draws):
            if (combo, str(d)) in done:
                print(f"--- {combo} draw {d}: already scored, skipping")
                continue
            rng = random.Random(f"{combo}:{d}:{args.seed}")
            tag = f"{combo}_d{d}{suffix}"
            paths, n_pool, n_keep, per_arm = draw_subset(
                spec["arms"], args.fraction, rng, OUT_RES / f"{tag}_probing")
            base_path = (subset_base_path(spec["codes"]) if args.base_mode == "subset"
                         else ROOT / BASE)
            probe_out = probe_dir / f"{tag}.pkl"
            n_base = sum(1 for l in base_path.read_text().splitlines() if l.strip())
            print(f"\n--- {combo} draw {d}  ({spec['label']}): {n_keep} of {n_pool} "
                  f"pooled successes  {per_arm}  | base {base_path.name} ({n_base} rows)")
            retrain_probe(
                jsonl_path=paths,
                base_probe_path=TEMPLATE_PROBE,
                base_training_data_path=base_path,
                new_probe_path=probe_out,
                layer=None, probe_spec=None,
                preprocessing=cfg.preprocessing,
                contrastive_cache_path=cache,
                min_judge_confidence=cfg.judge.confidence_threshold,
                test_size=0.2, split_field=None,     # ignored: dev_data_path forces 0.0
                dev_data_path=dev_dir,
                seed=args.seed, ensemble_size=1,
                base_activation_cache_dir=cfg.output.base_activation_cache_dir,
                combine_consecutive_messages=cfg.eval.combine_consecutive_messages,
                convert_tool_to_assistant=cfg.eval.convert_tool_to_assistant,
                eval_data_description=cfg.eval.data_description,
                verbose=True,
            )
            score(probe_out, cfg, tag, combo, d, n_keep, n_pool)

    vals: dict[str, list[float]] = {}
    if csv_path.exists():
        with csv_path.open() as fh:
            for row in csv.reader(fh):
                if len(row) > 5 and row[0] not in ("combo", "") and row[4] == "mean":
                    vals.setdefault(row[0], []).append(float(row[5]))
    if vals:
        print(f"\n{'configuration':38}{'draws':>7}{'mean':>10}{'sd':>9}{'min':>10}{'max':>10}{'range':>9}")
        allc = build_combos([1, 2, 3, 4])
        refs = ["base"] + [f"base_{''.join(c)}" for k in (2, 3)
                           for c in itertools.combinations(PART_ORDER, k)]
        for c in refs + list(allc):
            v = vals.get(c, [])
            if not v:
                continue
            sd = st.stdev(v) if len(v) > 1 else float("nan")
            if c == "base":
                lbl = "200-row base only"
            elif c.startswith("base_"):
                codes = c.removeprefix("base_")
                lbl = f"{50 * len(codes)}-row base only ({'+'.join(ATTACKER_NAMES[x] for x in codes)})"
            else:
                lbl = allc[c]["label"]
            print(f"{lbl:38}{len(v):7d}{st.mean(v):10.5f}{sd:9.5f}"
                  f"{min(v):10.5f}{max(v):10.5f}{max(v)-min(v):9.5f}")


if __name__ == "__main__":
    main()
