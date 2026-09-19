# Five prompts each for `hu_ha / eval_ant_hh` and `hu_ha / eval_ai_dilemmas`

The toolace / anthropic_hh protocol applied to the two `hu_ha` splits: measure the split,
write five deliberately different generator prompts from the measurements, generate 600
rows per prompt with deepseek-v4-pro, fit with **no base data**, take a size curve.

560 fits. Each arm validates on the split's own dev file (`dev_samples/hu_ha_ant_hh`, 44
rows; `dev_samples/hu_ha_ai_dilemmas`, 46 rows) and is scored on that split alone.
Prompts: `ANT_HH_GENERATOR_PROMPTS.md`, `AI_DILEMMAS_GENERATOR_PROMPTS.md`. Results:
`scripts/hu_harm_{ant_hh,ai_dilemmas}_prompts5{,_sizecurve}.csv`.

**Reference:** `probe_iter0` — 50 base rows, no generated data — scores **0.737** on
eval_ant_hh and **0.862** on eval_ai_dilemmas.

n = 40, 20, 10 run with `--grad-accum 1` (`base='none+ga1'`); see the note in
`analysis/prompts5_size_curves.md`.

## eval_ant_hh — the split that would not move

`scripts/fit_base_plus_concept.py` records this as the one eval split neither earlier
generator could shift: 0.737 base → **0.721** gptoss → **0.729** deepseek, both *below*
the base probe while the concept's other three splits improved. The stated diagnosis was
that those sets' one-shot pair made the harmful class casual and the safe class careful —
a surface cue `eval_ant_hh` does not supply, since it is the one split in the concept that
is not class-paired.

| set | 590 | 300 | 150 | 75 | 40* | 20* | 10* | 590→10 |
|---|---|---|---|---|---|---|---|---|
| nearmiss | **0.920** | 0.913 | 0.917 | 0.901 | 0.877 | 0.846 | 0.802 | 0.118 |
| replica | 0.905 | 0.906 | 0.893 | 0.871 | 0.864 | 0.834 | **0.837** | 0.068 |
| longtail | 0.875 | 0.863 | 0.869 | 0.860 | 0.843 | 0.830 | 0.804 | 0.072 |
| victim | 0.871 | 0.879 | 0.875 | 0.858 | 0.863 | 0.830 | 0.822 | 0.049 |
| intent | 0.868 | 0.867 | 0.872 | 0.877 | 0.876 | **0.856** | 0.830 | **0.038** |
| **mean** | 0.888 | 0.885 | 0.885 | 0.873 | 0.865 | 0.839 | 0.819 | |
| **sd across sets** | 0.021 | 0.020 | 0.018 | 0.016 | 0.012 | 0.010 | 0.014 | |
| mean draw sd | 0.009 | 0.014 | 0.018 | 0.021 | 0.028 | 0.040 | 0.054 | |

**All five prompts beat the base probe by 0.13–0.18, and the best by 0.204.** Ten rows of
any of them reach 0.80–0.84 — above what either earlier generator reached with 600.

**What was done differently.** Every prompt was written from a measurement of what the label
actually means here, which is *not* whether the assistant helped:

- refusal language appears in 33% of harmful rows and 31% of safe ones, so refusal carries
  no signal;
- a row is `harmful_to_human` by any of three routes — the assistant supplies harm, **the
  user announces intent to harm a person even when the assistant refuses**, or the user is
  themselves the person being harmed;
- animals, fictional characters, untargeted profanity and literal-benign readings of
  sinister-sounding questions are all `not_harmful_to_human`.

Before fitting, each generated set was checked against the cue that sank the earlier
attempts. Refusal language, harmful vs safe: replica 33/38, intent 27/30, longtail 24/25,
victim 19/17, nearmiss 9/5. Assistant-turn lengths matched within 10 characters on every
set. So the gain is not the old cue reappearing in a new form.

`nearmiss` — the prompt built **entirely** from the boundary cases above — leads at every
size ≥ 20.

**On the flatness.** A curve this flat (`intent` moves 0.038 across a 59× cut) is, by the
argument in `analysis/prompts5_size_curves.md`, the signature of a simple learned cue; that
is how `redteam` was diagnosed on anthropic_hh. That reading is probably wrong here, for
two reasons: the flatness is uniform across five prompts built on deliberately different
axes, so a shared shortcut would have to survive all five constructions; and all five were
verified balanced on the one cue this split is known to leak. The likelier reading is that
"is a human's wellbeing implicated" is **already represented** in gemma-3-27b's layer 32,
and these sets only have to *locate* the direction rather than teach it. That is consistent
with 10 rows outperforming a 50-row base probe by 0.07–0.10, and with the boundary-case
prompt winning: boundary cases are what locate a direction, not what build one.

This is a hypothesis the present data does not settle. What would settle it: whether
probes fit on different prompts here converge on the same direction (cosine similarity
between the fitted classifiers), which none of these 1120 fits measures.

## eval_ai_dilemmas

68 AI-ethics dilemmas, each appearing **twice** — the user turn byte-identical, the label
carried entirely by a ~38-character `I will ...` decision clause.

| set | 590 | 300 | 150 | 75 | 40* | 20* | 10* | 590→10 |
|---|---|---|---|---|---|---|---|---|
| replica | **1.000** | **1.000** | 0.998 | **1.000** | 0.999 | 0.891 | **0.680** | 0.320 |
| grid | **1.000** | **1.000** | **0.998** | 0.995 | 0.997 | **0.914** | 0.585 | 0.415 |
| domains | 0.996 | 0.995 | 0.996 | 0.997 | 0.995 | 0.858 | 0.564 | 0.432 |
| verbose | 0.969 | 0.982 | 0.965 | 0.906 | 0.894 | 0.804 | 0.648 | 0.321 |
| nearmiss | 0.939 | 0.957 | 0.965 | 0.968 | 0.965 | 0.929 | 0.650 | **0.289** |
| **mean** | 0.981 | 0.987 | 0.984 | 0.973 | 0.970 | 0.879 | 0.626 | |
| **sd across sets** | 0.024 | 0.016 | 0.016 | 0.036 | 0.040 | 0.045 | 0.044 | |
| mean draw sd | 0.008 | 0.006 | 0.012 | 0.017 | 0.021 | **0.149** | **0.164** | |

**Saturated from 40 rows up**: every set except `verbose` and `nearmiss` sits at 0.995–1.000
across the whole range 40–590, then falls off a cliff at 20 and collapses at 10 (0.56–0.68,
with a draw sd of 0.16 — that column is not a measurement).

### The negation cue does not matter here — a clean negative result

The split's safe decisions carry negation ("I will not …", "refuse", "decline") in 31% of
rows against 3% of harmful ones. Prompt 2 (`grid`) was written to destroy that, with five
rows in each cell of (harmful, safe) × (affirmative, negated) — harmful decisions phrased as
omissions ("I will not disclose the flaw to the review board") and safe ones as actions
("I will report the vulnerability to the oversight board").

Measured on the generated sets, negation in the decision clause, harmful vs safe:

| set | harmful | safe | eval @590 |
|---|---|---|---|
| replica | 1% | 61% | 1.000 |
| verbose | 29% | 90% | 0.969 |
| domains | 30% | 15% | 0.996 |
| nearmiss | 0% | 11% | 0.939 |
| grid | *decoupled by construction* | | 1.000 |

`replica` reproduces the cue and **doubles** it (61% vs the split's 31%). `domains`
accidentally *inverted* it (30% harmful vs 15% safe). `grid` removed it deliberately. All
three score 0.996–1.000, indistinguishable.

**So on this split the lexical cue carries none of the performance.** This is the opposite
of what the same experiment found on `oig_context_drift`, where inverting that split's cue
cost 0.097 AUROC — and it is worth recording as a negative result, because the cue-inflation
worry does not generalise. `eval_ai_dilemmas` is paired: the user turn is identical across
the two rows of a pair, so nothing about the scenario can separate them and the probe is
forced onto the decision's *meaning*. Pairing is the property that makes a split
cue-resistant, and this is direct evidence for it.

### The two weaker sets are weak for reasons worth naming

- **`verbose` (0.969)** replaces the one-clause decision with 3–6 sentences of reasoning.
  It also **failed an instruction**: the prompt asked for the two classes to be matched on
  length and the result is 523 characters for harmful decisions against 316 for safe ones.
  So this arm carries an *extra* cue the others do not — and still scores worst of the
  three form-faithful sets. Being off-distribution on the answer format costs more than a
  free length cue is worth.
- **`nearmiss` (0.939)** deliberately makes the decision's surface vocabulary
  anti-correlated with the label — alarming-sounding safe decisions ("I will shut down my
  own deployment pending review"), benign-sounding harmful ones ("I will maintain the
  current reporting arrangement"). It is the lowest at n≥150 and the **highest at n=20**
  (0.929). Adversarial surface features cost a little where data is plentiful and help
  where it is scarce.

## Reading the two splits together

Both are in the same concept, both use the same 2-message shape, and both saturate — but
they saturate at different distances from the base probe (0.737 → 0.92 against 0.862 →
1.00), and only one of them is cue-resistant. The difference is pairing:
`eval_ai_dilemmas` is perfectly paired and its cue is inert; `eval_ant_hh` is not paired at
all, and it is the split where a generator that encoded the wrong cue went *backwards*.

## Caveats

- **Both evals are tiny** — 134 and 136 rows. Draw sd at n=590 is 0.008–0.009, so
  differences under ~0.02 between sets are not differences. Do not rank `victim` against
  `longtail` (0.004 apart) or `replica` against `grid` (0.000 apart) on these numbers.
- **The n=10 and n=20 columns on ai_dilemmas have draw sds of 0.16 and 0.15** — larger than
  every between-set gap they contain. They say "it breaks down here", nothing finer.
- **Target split only.** Nothing here says how these sets travel to
  `eval_balanced_refusal` or `eval_daily_dilemmas`, which is where the earlier generators'
  gains actually sat. That was the deliberate scope of this study and is its main gap.
- **`aidil5_grid` was generated on a second OpenRouter key** after the first hit its
  spending limit mid-run; same prompt, same parameters, 300/300 balanced, 36 calls.
