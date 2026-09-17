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

## Stage 2: direction geometry

*(pending)*

## Stage 3: the ratio, the leave-one-kind-out control, and the link to the paper's m

*(pending)*

## Caveats

*(pending)*
