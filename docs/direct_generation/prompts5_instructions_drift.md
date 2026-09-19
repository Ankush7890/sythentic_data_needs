# Five prompts each for the two `instructions` drift splits

The toolace / anthropic_hh protocol (`analysis/toolace_prompts5_results.md`,
`analysis/prompts5_size_curves.md`) applied to `instructions/hc_context_drift` and
`instructions/oig_context_drift`: measure the split, write five deliberately different
generator prompts from the measurements, generate 600 rows per prompt with
deepseek-v4-pro, fit with **no base data**, and take a size curve.

560 fits. Every arm validates on the split's own dev file
(`dev_samples/instructions_{hc,oig}_context_drift`, 66 rows) and is scored on that split
alone (`--eval-splits`). Prompts: `HC_DRIFT_GENERATOR_PROMPTS.md`,
`OIG_DRIFT_GENERATOR_PROMPTS.md`. Results:
`scripts/instructions_{hcdrift,oigdrift}_prompts5{,_sizecurve}.csv`.

**Reference:** `probe_iter0` — the 50-row base probe, no generated data — scores **0.714**
on each of the two splits.

n = 40, 20 and 10 run with `--grad-accum 1` (tagged `base='none+ga1'`): under ~49 rows the
inherited spec takes one optimizer step per 64 samples, i.e. zero, and the fit returns the
untrained probe. Do not read across that boundary as one line.

## hc_context_drift

A health claim is asked with a supporting document; the assistant answers in one sentence;
the user supplies a *replacement* document and asks again. The label is whether the second
answer matches the second document.

| set | 590 | 300 | 150 | 75 | 40* | 20* | 10* |
|---|---|---|---|---|---|---|---|
| replica | 0.999 | 0.998 | **0.993** | **0.915** | **0.807** | 0.527 | 0.500 |
| grid | 0.997 | 0.994 | 0.983 | 0.882 | 0.791 | 0.543 | 0.500 |
| failmodes | **1.000** | 0.997 | 0.985 | 0.822 | 0.795 | 0.525 | 0.513 |
| domains | 0.998 | 0.996 | 0.776 | 0.550 | 0.589 | 0.503 | 0.497 |
| verbose | 0.990 | 0.989 | 0.719 | 0.515 | 0.529 | 0.506 | 0.495 |
| **mean** | 0.997 | 0.995 | 0.891 | 0.737 | 0.702 | 0.521 | 0.501 |
| **sd across sets** | 0.003 | 0.003 | 0.119 | 0.170 | 0.119 | 0.015 | 0.006 |
| mean draw sd | 0.004 | 0.006 | 0.070 | 0.085 | 0.123 | 0.039 | 0.007 |

## oig_context_drift

A question about some entity, an encyclopaedic answer, then a short follow-up. The label is
whether the second answer answers the follow-up.

| set | 590 | 300 | 150 | 75 | 40* | 20* | 10* |
|---|---|---|---|---|---|---|---|
| replica | **0.969** | **0.938** | **0.891** | 0.740 | **0.724** | 0.617 | 0.589 |
| failmodes | 0.932 | 0.916 | 0.836 | 0.749 | 0.669 | 0.596 | 0.581 |
| domains | 0.916 | 0.881 | 0.848 | **0.751** | 0.698 | **0.644** | **0.611** |
| overlap | 0.872 | 0.843 | 0.823 | 0.729 | 0.662 | 0.558 | 0.576 |
| long | 0.740 | 0.690 | 0.637 | 0.569 | 0.587 | 0.566 | 0.567 |
| **mean** | 0.886 | 0.854 | 0.807 | 0.708 | 0.668 | 0.596 | 0.585 |
| **sd across sets** | 0.079 | 0.088 | 0.088 | 0.070 | 0.046 | 0.032 | 0.015 |
| mean draw sd | 0.016 | 0.021 | 0.022 | 0.041 | 0.042 | 0.026 | 0.029 |

\* grad-accum 1.

## The two splits have opposite shapes

**hc_context_drift is a step, not a slope.** From 300 rows up, all five prompts are
indistinguishable — sd 0.003, every set between 0.989 and 1.000. Below 40 rows all five are
at chance, `replica` included. The whole difference between prompts lives in a narrow
window around n = 75–150, where the spread reaches 0.170 and two sets fall to 0.515–0.550
while another holds 0.915.

Read only at n=600, this split says "the prompt does not matter". Read only at n=75, it
says "the prompt is worth 0.40 AUROC". Both readings come from the same ten runs, and
neither is usable without the other. That is the most transferable finding here: **a single
operating point cannot rank prompts**, and the earlier studies' "the prompt beats the
volume by 9×" is a statement about a particular n, not a law.

The cliff is corroborated independently. On the `per_split_studies` branch a targeted hc set
scored 0.601 at n=30 and 0.993 at n=60 (`scripts/instructions_gen90.csv`) — the same
threshold between 30 and 60 rows, reached by a different prompt on a different dev set.

**oig_context_drift is an ordinary slope**, like toolace and anthropic_hh: monotone decline,
rankings largely stable, everything still above chance at n=10 (0.567–0.611 against hc's
flat 0.50). Something in this split is learnable from ten examples. Nothing in hc is.

## What the prompts were built to test, and what they found

### oig: the eval rewards a cue, and the cue can be measured

The split's violating second turn is, almost always, the first answer said again: median
token overlap with the previous assistant turn is **0.70** on violating rows against **0.16**
on compliant ones, while `Background:` openers, quotation marks and all four turn lengths
are matched between the classes. Prompt 2 (`overlap`) was written to break that — five rows
in each cell of (follows, does-not-follow) × (high overlap, low overlap).

It worked. Measured on the generated sets, the same overlap gap:

| set | compliant | violating | gap | eval @590 |
|---|---|---|---|---|
| replica | 0.25 | 0.48 | **+0.23** | 0.969 |
| domains | 0.20 | 0.31 | +0.11 | 0.916 |
| failmodes | 0.29 | 0.28 | −0.01 | 0.932 |
| overlap | 0.48 | 0.34 | **−0.14** | 0.872 |
| long | 0.14 | 0.19 | +0.05 | 0.740 |

`overlap` inverted the cue: its violating rows repeat themselves *less* than its compliant
ones. And it costs **0.097 AUROC** against `replica`, which reproduces the cue.

**How far to trust this.** The clean part is the designed contrast: two sets written to the
same spec differing in one controlled property, 0.097 apart. The correlation across sets is
weaker evidence than it first looks — r ≈ 0.89 over the four four-message sets, but that is
n = 4 and it is **not monotone**: `failmodes` (gap −0.01) outscores `domains` (gap +0.11).
An earlier draft of this claimed monotonicity off the single-draw n=600 numbers; the 8-draw
means do not support it. `long` is excluded from the correlation because it has a second,
larger defect — 6–8 message rows against the split's 4 — which costs it 0.23 on its own.

So: the cue demonstrably carries part of this split's AUROC. How much of the rest it carries
is not established by these five points.

**The size curve adds a second reading.** `replica`'s lead over `overlap` is 0.097 at n=590
and 0.011 at n=75 — it *shrinks* as data is removed. On hc the equivalent gap moves the
other way (0.009 at 590 → 0.400 at 75). A prompt whose advantage needs 600 rows to appear is
accumulating something; a prompt whose advantage appears fastest under scarcity is teaching
something. `replica` behaves the first way on oig and the second way on hc, which is what
the cue measurements predict, since it reproduces oig's cue (+0.23) and is exactly neutral
on hc's (50/50 below).

### hc: the eval controls its cue, so sample-efficiency decides

The split balances "did the answer change between the two turns" against the label (50/47),
so a probe that learns the shortcut gains nothing. The generated sets mostly respect that:

| set | follows: answer changed | violates: answer changed |
|---|---|---|
| replica | 50% | 50% |
| domains | 49% | 50% |
| verbose | 33% | 29% |
| failmodes | 82% | 83% |
| grid | 64% | 38% |

Four of five are decorrelated. `replica` is exactly 50/50 — so its win at n=75–150 is not
cue-riding, it is form-matching. Ironically `grid`, the prompt written *specifically* to
force the polarity grid, is the one set that retains a correlation (64% vs 38%), and it
lands mid-table, which is what should happen when the eval neutralises that cue.

The ranking at n=75 is then a clean statement about distribution distance:

- `replica` (0.915) — the split's own form: health claims, one-sentence answers;
- `grid` (0.882), `failmodes` (0.822) — same form, contrived label balance or wider
  violation types;
- `domains` (0.550), `verbose` (0.515) — same *mechanic*, different subject matter (law,
  finance, engineering) or different answer format (2–5 sentences of reasoning).

Moving off the split's subject matter or its answer format costs nothing at 600 rows and
roughly 0.35 AUROC at 75. **The cost of being off-distribution is invisible until the data
is scarce.**

At n=40 with a correctly stepping fit (`--grad-accum 1`), `domains` (0.589) and `verbose`
(0.529) are still near chance while the other three hold ~0.80 — so the n=75 collapse is a
property of the data, not of the step schedule.

## Both splits beat everything published on them before

`scripts/instructions_gen90.csv` holds every earlier arm on these splits. Best previously
recorded: **0.989** on hc_context_drift (`tgtnone_hc_context_drift`) and **0.919** on
oig_context_drift (`gptoss_evaldesc`, with `tgtshot_oig_context_drift` at 0.911). This study
reaches **1.000** and **0.969** — though the hc number is against a split that four of five
prompts saturate, so it is nearly meaningless as a ranking, and the oig gain (+0.050) is the
one that reflects a better prompt.

This also revises the concept-level claim in `scripts/fit_base_plus_concept.py`'s docstring,
that generated data *hurts* on `instructions` (0.778 base → 0.665 gptoss → 0.762 nemotron).
That holds for generic instruction-following sets measured across all seven splits. A set
written from one split's measurements, scored on that split, gains 0.20–0.29 over the base
probe. The concept is not resistant; the generic prompt was.

## Caveats

- **Target split only.** Nothing here says how these sets travel to the other five
  `instructions` splits. The earlier studies found travel differs sharply between prompts,
  and a set that reproduces one split's cue is exactly the kind that should travel badly.
  This was the deliberate scope choice for the study and it is the main gap in it.
- **Both evals are 194 rows** — small. A 0.01 difference is not a difference; the draw sd at
  n=590 is 0.004 (hc) and 0.016 (oig).
- **hc's n=40 column has a draw sd of 0.123**, larger than several of the between-set gaps
  it is used to compare.
- The two n=600 single-draw fits per set (`*_prompts5.csv`) use the whole 600-row set, not a
  balanced draw, so they are not points on these curves.
