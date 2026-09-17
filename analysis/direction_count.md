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

*(agreement rates, per-kind counts and the `none` rate: filled in when the stage
finishes)*

## Stage 2: direction geometry

*(pending)*

## Stage 3: the ratio, the leave-one-kind-out control, and the link to the paper's m

*(pending)*

## Caveats

*(pending)*
