# Does the number of directions the generated data must cover set the half-gain size?

*(draft — the fits are still running; every number below that is not marked FINAL is
provisional)*

## Method, in one paragraph

Every synthetic-data learning curve in the paper is fit with a log-logistic
`A(n) = L + (U − L) / (1 + (n/m)^−k)`, and `m` — the half-gain size — is the quantity
this study tries to explain. The detailed prompt writes each batch of a generated set for
one of the concept's **kinds**, one kind per evaluation split, in round-robin. If the
kinds share one direction in activation space, every generated sample serves every split;
if they are separate directions, `n` mixed samples give one split only about `n/n_kinds`
useful ones. So each of the twelve detailed sets (three concepts × four generators) is
tagged sample-by-sample with its kind by two LLM passes, and three training sets are cut
from it per evaluation split `s`: **(a) kind-only**, the samples tagged kind(s); **(b)
mixed**, the whole set; **(c) leave-one-kind-out**, the set minus kind(s). Each is run
through `scripts/subsample_curve_concept.py` unedited — class-balanced draws at 2…590
samples, eight draws a size, `--no-base`, accumulation `ceil(n/16)` at batch 16 so every
size takes exactly one optimiser step per epoch — and every curve is fit by
`scripts/fit_curves_ref.py`, imported, so the half-gain sizes stay on the scale of
`scripts/knee_fits.csv`. The headline is `R_s = m(mixed) / m(kind-only)`.

## Kind → evaluation split: verified

The brief's mapping table was checked against samples drawn from each evaluation split
(seeded on the split name; `eval_sets/<concept>/<split>.jsonl`) before anything relied on
it. All fourteen rows hold.

| concept | kind | split | what the split's rows actually look like |
|---|---|---|---|
| instructions | 1 unjustified refusal | `anthropic_harmless_refusal` | harmless question; reply answers it or declines ("I can't provide guidance on that topic") |
| instructions | 2 biased substitution | `bbq_substitution` | two named people, question about one; reply answers about the right one, the other, or declines to infer |
| instructions | 3 context drift, retrieved-document | `hc_context_drift` | two turns, each with its own supplied source, same question; second reply answers from the new source or repeats the stale answer |
| instructions | 4 contradicting the source | `hc_contradiction` | one supplied passage; reply follows it or states its opposite |
| instructions | 5 exaggeration / embellishment | `mm_substitution` | short quoted text + question; reply quotes plainly or asserts more than the text supports |
| instructions | 6 context drift, dialogue | `oig_context_drift` | multi-turn chat, no documents; later reply answers the new question or repeats the earlier answer |
| highstakes | 1 open-ended chatbot conversation | `anthropic_hh_balanced` | `You are an LLM as part of a general purpose chatbot`, 5–9 turns |
| highstakes | 2 clinical document | `mt_balanced` | medical-assistant system prompt, ONE long user turn holding an operative note |
| highstakes | 3 doctor-patient dialogue | `mts_balanced` | medical assistant talking to a patient, many short turns |
| highstakes | 4 tool-calling agent | `toolace_balanced` | function list in the system prompt; reply emits a call or asks for the missing argument |
| hu_harm | 1 AI in an institutional role | `eval_ai_dilemmas` | long "You are an AI system…" scenario, one-line decision |
| hu_harm | 2 everyday request answered straight | `eval_ant_hh` | short ask, full answer, nothing declined |
| hu_harm | 3 request that should not be carried out | `eval_balanced_refusal` | short harmful ask; one-line refusal or the procedure |
| hu_harm | 4 interpersonal dilemma | `eval_daily_dilemmas` | first-person account, one line of advice |

## Stage 0: the tagging, and its audit

Two passes of `openai/gpt-5.6-luna` at temperature 0 per sample, the concept's numbered
kinds quoted verbatim, the second pass with the kinds listed in the reverse order and
their own numbers kept (so nothing has to be mapped back). A sample is kept only when the
two passes agree. `data/kind_tags/<set stem>.csv` holds `row,kind,split,pass1,pass2,agree`.

| set | pass agreement | `none` (pass 1) | kept | per-kind counts |
|---|---|---|---|---|
| instructions llama70b | 0.902 | 0.063 | 523 | 72 / 102 / 38 / 160 / 59 / 92 |
| instructions gptoss | 0.910 | 0.053 | 539 | 75 / 96 / 102 / 127 / 53 / 86 |
| instructions nemotron | 0.912 | 0.025 | 543 | 74 / 97 / 97 / 141 / 43 / 91 |
| instructions deepseekv4pro | 0.942 | 0.022 | 565 | 90 / 103 / 102 / 125 / 53 / 92 |
| hu_harm llama70b | 0.947 | 0.008 | 567 | 147 / 107 / 152 / 161 |
| hu_harm gptoss | 0.968 | 0.000 | 581 | 154 / 128 / 150 / 149 |
| hu_harm nemotron | 0.977 | 0.003 | 586 | 160 / 125 / 149 / 152 |
| hu_harm deepseekv4pro | 0.995 | 0.000 | 597 | 160 / 135 / 153 / 149 |
| highstakes llama70b | 0.910 | 0.062 | 522 | 211 / 120 / 93 / 98 |
| highstakes gptoss | 0.980 | 0.015 | 587 | 169 / 157 / 133 / 128 |
| highstakes nemotron | 0.993 | 0.000 | 596 | 143 / 160 / 144 / 149 |
| highstakes deepseekv4pro | 0.985 | 0.000 | 591 | 155 / 150 / 139 / 147 |

Round-robin predicts ~100 per kind for *instruction* and ~150 for the other two. The
four-kind concepts land there; *instruction* does not, and the miss is systematic — kind
4 (contradicting the source) runs 25–60% over and kind 5 (exaggeration) 45–55% under, in
every generator.

**Kind 5 has almost no positive class at all** (1, 8, 2 and 7 positives in the four sets,
against 41–58 negatives), and that is a property of the data, not of the tagger: kinds 3,
4 and 5 differ only in what the *negative* reply does — drift, contradict, embellish —
while their positive is one and the same conversation, a passage question answered
plainly. A tagger asked "which kind is this" cannot split a shared positive class, and
neither could a human. The consequence is recorded rather than patched: the kind-only arm
for `mm_substitution` is **skipped for all four generators** (43–53 tagged rows, under
the 60-row floor, and never more than 8 of one class), and the geometry's per-kind
direction for kind 5 is undefined for want of 20 positives.

**Audit.** Ten tagged samples per kind per concept were read by hand (180 conversations,
drawn seeded on `(concept, kind)`, `.dc_work/audit_<concept>.md`): **180/180 agree with
the tagger**, with one borderline — an instructions row opening "I have two documents.
First, read this one" but supplying only one, tagged kind 4, which is what it is.

**Rule-based cross-check** (structure only, never used to tag): turn count and whether
each turn carries its own supplied source for *instruction*; the two turns' word counts
against the four measured profiles for *harmful*; function list / one long user turn /
many short turns for *high-stakes*. Where the rule is decisive it agrees with the LLM
tagger on **0.991** of 693 instructions rows, **0.946** of 2,331 hu_harm rows and
**0.941** of 1,242 highstakes rows.

## The geometry does not separate the concepts — this is the study's first result

The hypothesis needs the *high-stakes* and *harmful* kinds to share a direction and the
*instruction* kinds to be separate. They are all separate, to the same degree:

| concept | kinds used | n_eff (four generators) | mean off-diagonal \|cos\| |
|---|---|---|---|
| instructions | 4–5 of 6 | 3.46, 3.62, 3.91, 4.03 | 0.14–0.29 |
| hu_harm | 4 of 4 | 3.71, 3.80, 3.80, 3.82 | 0.11–0.14 |
| highstakes | 4 of 4 | 3.43, 3.52, 3.64, 3.64 | 0.17–0.19 |

`n_eff` sits within 0.6 of the number of kinds in every one of the twelve sets. Prediction
2 — that `R` tracks the effective number of directions — cannot hold in the form it was
posed, because the quantity it is supposed to track is the same for all three concepts.
Whatever makes *instruction* expensive, a cosine statistic over mean-pooled per-kind
difference-of-means directions does not see it.

## Stage 2: direction geometry — *instruction* (FINAL for this concept)

Per set, on its own standardised mean-pooled features: a unit difference-of-means
direction per kind with at least 20 samples of each class, the Gram matrix of those unit
directions, and `n_eff = (Σλ)² / Σλ²` — the number of kinds if they were orthogonal, 1 if
they were all the same direction.

| set | kinds used | n_eff | mean off-diagonal \|cos\| | max |
|---|---|---|---|---|
| instructions llama70b | 4 of 6 | 3.46 | 0.202 | 0.323 |
| instructions gptoss | 5 of 6 | 3.62 | 0.292 | 0.499 |
| instructions nemotron | 5 of 6 | 3.91 | 0.152 | 0.644 |
| instructions deepseekv4pro | 5 of 6 | 4.03 | 0.139 | 0.450 |

So the *instruction* kinds are close to orthogonal: `n_eff` is 3.5–4.0 against a ceiling
of 4–5, and no pair of kind directions has a cosine above 0.65. The prediction for this
concept (`n_eff` above 2) holds with room to spare. Kind 5 has no direction anywhere, for
the reason in stage 0 — it has no positive class to subtract.

Transfer, averaged over the four sets: a kind's own direction, fit on four of five folds,
scores **0.980** AUROC on the held-out fold of its own kind and **0.814** on the other
kinds' samples. Against the *evaluation* splits the same directions score **0.705** on
their own split and **0.601** on the others, while the whole-set direction scores
**0.722** — i.e. in this linear, mean-pooled geometry the mixed set's single direction is
already the best one for every split. Hold that beside the fits below: the two are not in
conflict, because the fits are about how many samples it takes to get half-way, not about
where the curve ends up.

## Stage 3: the ratio and the leave-one-kind-out control — *instruction* (FINAL)

3,720 fits: four generators × (one mixed arm, five kind-only arms, six leave-one-kind-out
arms) × their size ladders × eight draws. `mm_substitution` has no kind-only arm in any
generator (stage 0). `hc_context_drift` has none under llama70b (38 tagged rows).

**R = m(mixed) / m(kind-only), per (generator, split).** Four of the nineteen rows have a
flat arm and are set aside (`usable = 0` in `scripts/dc_ratios.csv`); none of the fifteen
that remain is censored at the smallest measured size (2).

| generator | refusal | bbq | hc_drift | hc_contra | oig_drift |
|---|---|---|---|---|---|
| deepseekv4pro | 5.3 | *flat* | 6.3 | 5.3 | 30.5 |
| gptoss | 15.7 | *flat* | 2.3 | 3.0 | 1.1 |
| llama70b | 2.5 | 1.8 | — | 3.5 | 33.2 |
| nemotron | 7.4 | 15.7 | *flat* | *flat* | 266.3 |

**Median R = 5.30** over the fifteen usable rows, range 1.09–266. **Every one of the
fifteen is above 1**: on this concept, a sample of the split's own kind is worth several
mixed samples, everywhere, under every generator. The prediction for *instruction* (R
"well above 1, up to about 6") holds, and the median is at the top of that range.

**Leave-one-kind-out.** Median gain ratio `G = (U_c − L_c) / (U_b − L_b) = 0.67` and
median `m_c / m_b = 0.78` over the eighteen rows whose mixed arm is not flat. Removing
the split's own kind from the training set costs about a third of the total gain, and the
extreme cases are extreme: `anthropic_harmless_refusal` keeps 11% of its gain under
deepseekv4pro and goes **flat** under gptoss — remove the refusal samples and the probe
never learns the refusal split at any size.

**R against n_eff** (Spearman over the fifteen usable set-splits): ρ = +0.379, p = 0.16.
Within one concept `n_eff` barely varies (3.46–4.03), so this is not yet a test of
prediction 2; it becomes one when the four-kind concepts are in.

*The link to the paper's per-split m needs all three concepts (six of fourteen splits
cannot rank against a fourteen-split baseline) and is left until then.*

## Stage 3 — *harmful* (FINAL)

2,816 fits. Every hu_harm kind has both classes, so all four splits get a kind-only arm
under all four generators.

| split | deepseekv4pro | gptoss | llama70b | nemotron |
|---|---|---|---|---|
| eval_ai_dilemmas | 0.8 | 2.5 | 1.8 | 0.8 |
| eval_ant_hh | *flat* | 2.3 | *flat* | 30.5 |
| eval_balanced_refusal | 3.0 | 2.3 | 1.9 | 1.6 |
| eval_daily_dilemmas | 1.0 | 1.6 | 2.5 | 3.2 |

**Median R = 2.12** over the fourteen usable rows (two set aside for a flat arm), range
0.85–30.5. Against *instruction*'s 5.30. The ordering the hypothesis predicts is there and
it is not small — but 2.12 is **above the brief's 1.5 bar**, and four of the fourteen rows
are at or below 1, so this is not the "R ≈ 1, samples of any kind serve any split" the
hypothesis asks for.

**Leave-one-kind-out: median G = 0.90, median m_c/m_b = 1.00.** This is the cleanest
contrast in the study. Take away the split's own kind and *harmful* keeps 90% of its gain
and does not slow down at all; *instruction* keeps 67% and, on `anthropic_harmless_refusal`
under gptoss, learns nothing at any size. Prediction 3 holds as stated.

## Coverage versus per-kind difficulty (instruction against harmful)

The paper's concept effect could be coverage (a mixed set spends most of itself on kinds
that do not serve the split) or per-kind difficulty (one kind of *instruction* is simply
harder than one kind of *harmful*). The kind-only arm is the same concept with coverage
removed, so the two are separable:

| | median log10 m, mixed | median log10 m, kind-only |
|---|---|---|
| instructions | 2.07 | 1.06 |
| hu_harm | 1.26 | 0.86 |
| **gap** | **0.81** (6.5x) | **0.20** (1.6x) |

**76% of the instruction-vs-harmful gap in half-gain size is coverage**, and 24% survives
as per-kind difficulty. This is the study's main positive finding, and note that it stands
on its own feet: it does not depend on the direction geometry, which (above) sees no
difference between the concepts at all.

## Caveats

*(pending)*
