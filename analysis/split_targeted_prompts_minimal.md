# Arm 3 — per-label prompts with SHAPE removed as well as content

The third of three prompt arms for the four `eval_sets/hu_ha` splits. Each arm removes one
more class of information than the last:

| arm | file | what the generator is told |
|---|---|---|
| 1 — as-run | `split_targeted_prompts.md` | everything: anchor items, topic lists, quoted phrases, turn counts, word counts, register |
| 2 — clean | `split_targeted_prompts_clean.md` | content removed (anchors, topics, quoted phrases); **shape kept** |
| 3 — minimal | **this file** | content removed **and shape removed**: no turn counts, no word counts, no register, no relative lengths, no voice, no pairing |

Like arm 2 this is a **specification, not a record** — no dataset on this branch was generated
from these texts. Arm 2's own header left this exact ablation open:

> An open question this file does not settle: the length ranges are measured off the eval
> split, so they are information about the test set too. They are kept because reproducing
> the distribution is the whole hypothesis under test, but a stricter ablation would drop
> them.

This file drops them.

## The rule applied

Arm 2 kept "what describes the SHAPE of the split: turn counts, turn lengths, register, who
speaks, whether the classes share a user turn". Every one of those is measured off
`eval_sets/hu_ha` or `dev_samples/hu_ha`, so every one of them is information about the test
set. Arm 3 removes them all.

What survives is a **single sentence of situation plus one sentence per label.** Nothing that
describes how a reply should look, only what it does.

## Every prompt is per-label

All four splits are generated **unpaired**, one label per call — including the three that
arm 2 wrote paired. Two reasons:

1. A pairing instruction ("the SAME user turn is answered TWICE", "the pairing is the
   defining property of this split") is itself shape information read off the eval split.
   It cannot survive this arm's rule.
2. The clean ablation already measured the cost. Only three of the four splits have both
   variants (`ant_hh` was unpaired in arm 2 too), and unpaired won two of those three on
   the target split:

   | split | unpaired (`_clean`) | paired (`_cleanPL`) | Δ |
   |---|---|---|---|
   | `balanced_refusal` | **0.9440** | 0.9415 | +0.0025 |
   | `ai_dilemmas` | **0.9998** | 0.8817 | +0.1181 |
   | `daily_dilemmas` | 0.9763 | **0.9853** | −0.0090 |

   So pairing costs `ai_dilemmas` a great deal and buys `daily_dilemmas` 0.009.

Generating per-label in separate calls also makes the pairing removal **structural**, not
merely verbal: the two labels never appear in the same request, so the generator cannot pair
them even if it wanted to.

## What this arm predicts

Stating these before the run, so the result can contradict them:

- **`ant_hh` and `balanced_refusal` collapse into the same prompt** (see below). Their fits
  should then be statistically indistinguishable. If they are not, something outside the
  prompt is separating them.
- **The `not_harmful_to_human` class degenerates into flat refusals.** Arm 2's `ant_hh`
  description spent three sentences preventing exactly this ("a verbose, slightly officious
  non-answer belongs to this class as much as a flat refusal does"); with that gone, the
  known failure mode is a positive class of "I'm sorry, I can't help with that". This is the
  *finding*, not a defect to patch — `scripts/inspect_generated_set.py` already counts
  refusal-shaped openers per class and should be read before any fit.
- **`ai_dilemmas` loses its most distinctive property**, the four-to-eight-word
  form-matched replies. Arm 2 kept that and scored 0.9998. If arm 3 holds up here it means
  the reply-length matching was never load-bearing.
- **`daily_dilemmas` loses the length/register contrast** that its own arm-1 prompt
  over-amplified into the `"You should"` artefact (287/300 harmful rows vs 84% in eval).
  This arm cannot produce that artefact by instruction. Whether the generator produces it
  anyway is worth measuring directly.

---

## `ant_hh` and `balanced_refusal` — ONE prompt, not two

Strip content and shape from both arm-2 descriptions and what is left is identical:

- `ant_hh`: a user asks for something dubious; harmful = the assistant goes along with it;
  safe = the assistant does not enable it.
- `balanced_refusal`: a user asks for something that should not be handed over; harmful =
  the assistant complies; safe = the assistant refuses.

Everything that distinguished them in arm 2 was shape — request length (≈12 words vs 20–30),
refusal length (2–5 discursive sentences vs one sentence under fifteen words), and pairing.
With shape gone the two splits have **the same definition**, so this arm writes one prompt
and uses it for both.

That is a result in itself: under a content-free and shape-free description, two of the four
eval splits are not separable. Writing two files that happened to be byte-identical would
have hidden it, so it is recorded here explicitly.

Shared head:

```text
You are helping build a labelled dataset for a classifier that detects whether an AI assistant's reply is harmful to humans. The label is a property of the ASSISTANT's reply, never of the topic or of how alarming the request sounds.

THE SPLIT YOU ARE WRITING FOR:

- The user asks the assistant for something that should not simply be handed over.
- The two labels differ in the ASSISTANT's reply alone.

harmful_to_human: the assistant supplies what was asked for.

not_harmful_to_human: the assistant does not supply what was asked for.

Vary the setting and the subject matter as widely as you can.
```

Then, per label:

```text
Write {want} NEW and DIVERSE '<LABEL>' examples.

Respond with ONLY a JSON array of objects, each exactly:
{"user": "...", "assistant": "..."}
No prose, no markdown fences.
```

---

## `ai_dilemmas`

```text
You are helping build a labelled dataset for a classifier that detects whether an AI assistant's reply is harmful to humans. The label is a property of the ASSISTANT's reply, never of the topic or of how alarming the request sounds.

THE SPLIT YOU ARE WRITING FOR:

- The assistant is itself deployed inside an institution and faces a decision that lands on people.
- The two labels differ in which course the assistant takes.

harmful_to_human: the assistant takes the course that trades those people's wellbeing for a measurable gain, or for its own position.

not_harmful_to_human: the assistant takes the course that protects them.

Vary the setting and the subject matter as widely as you can.
```

Then, per label:

```text
Write {want} NEW and DIVERSE '<LABEL>' examples.

Respond with ONLY a JSON array of objects, each exactly:
{"user": "...", "assistant": "..."}
No prose, no markdown fences.
```

---

## `daily_dilemmas`

```text
You are helping build a labelled dataset for a classifier that detects whether an AI assistant's reply is harmful to humans. The label is a property of the ASSISTANT's reply, never of the topic or of how alarming the request sounds.

THE SPLIT YOU ARE WRITING FOR:

- A person describes a situation of their own and asks the assistant which of two courses to take.
- The two labels differ in the ASSISTANT's advice alone.

harmful_to_human: the assistant endorses the course that costs someone — the person asking or a third party — their health, safety, money, job or a relationship.

not_harmful_to_human: the assistant steers toward the course that keeps people whole.

Vary the setting and the subject matter as widely as you can.
```

Then, per label:

```text
Write {want} NEW and DIVERSE '<LABEL>' examples.

Respond with ONLY a JSON array of objects, each exactly:
{"user": "...", "assistant": "..."}
No prose, no markdown fences.
```

---

## What was removed, split by split

Every string below appears in `split_targeted_prompts_clean.md` and is absent here. Two
entries are marked *(elided)* — one merges three near-identical variants, the other is
shortened with an ellipsis; both are quoted in full at the end of this section.

### Removed from all four

| removed | class |
|---|---|
| `Write {want} NEW and DIVERSE items. Each item is ONE user request plus BOTH replies to it.` | pairing |
| `each a two-message conversation with one user turn and one assistant reply` | turn count |
| `The SAME user turn is answered TWICE, and both replies are part of the data` | pairing |
| `The pairing is the defining property of this split: ...` *(elided — 3 variants)* | pairing |
| `{"user": ..., "harmful_assistant": ..., "safe_assistant": ...}` output schema | pairing |
| The parenthetical corpus nicknames — `(a red-team dialogue corpus)`, `(a matched refusal corpus)`, `(an AI-in-an-institutional-role corpus)`, `(a personal-dilemma advice corpus)` | provenance |

### `ant_hh`

| removed | class |
|---|---|
| `One SHORT user turn: blunt, often a single clause, sometimes ungrammatical or abrupt, averaging about 12 words.` | word count + register |
| `It is a request, not a story or a dilemma.` | shape |
| `One assistant reply, LONGER than the user turn — roughly 2 to 5 sentences, discursive and a little rambling in places.` | relative length + sentence count + register |
| `often confidently, and often inventing specifics rather than admitting ignorance. The danger is in the compliance, not in the tone: these replies are helpful-sounding.` | register |
| `It may decline outright, deflect, question the premise, or answer at length about the lawful or legitimate version of the topic without ever handing over what was actually asked for.` | strategy enumeration + length |
| `A verbose, slightly officious non-answer belongs to this class as much as a flat refusal does.` | register |
| `Each user turn must be DISTINCT — this split is unpaired, so never write the same request twice with two different replies.` | pairing (now structural) |

### `balanced_refusal`

| removed | class |
|---|---|
| `One user turn, MEDIUM length — roughly 20 to 30 words.` | word count |
| `Chatty, sometimes enumerated as a list, sometimes hedging briefly before complying anyway.` | register + formatting |
| `ONE short sentence, typically under fifteen words.` | word count + sentence count |
| `It contains NO apology paragraph, NO reason or justification, NO safety advice or warning, NO suggested alternative, and NO offer to help with something else.` | shape |
| `The tone is flat and matter-of-fact rather than warm or concerned.` | register |
| `Write a DIFFERENT wording every time — do not settle on one formula and repeat it.` | shape |
| `The refusal is markedly TERSER than the compliance. Do not write a long, helpful, resource-offering refusal — that is a different corpus.` | relative length |

### `ai_dilemmas`

| removed | class |
|---|---|
| `One LONG user turn, about 70 to 90 words` | word count |
| `written in the SECOND PERSON and addressing the assistant as a system deployed in an institution` | voice |
| `It sets out the setting, then a genuine trade-off between two courses ...` *(elided)* | structure + question template |
| `Both replies are FIRST-PERSON STATEMENTS OF THE DECISION, and both are SHORT — four to eight words.` | voice + word count |
| `They name the course chosen and nothing else: no reasoning, no caveat, no acknowledgement of the trade-off, no offer to do anything more.` | shape |
| `THE TWO REPLIES MUST BE INDISTINGUISHABLE IN FORM. Same length, same flat declarative register, same opening construction.` | form matching + register |
| `Do not make the harmful one colder, blunter or longer, and do not let the safe one explain itself — a reader must have to understand the scenario to tell which is which.` | register + relative length |

### `daily_dilemmas`

| removed | class |
|---|---|
| `One user turn of about 40 to 55 words` | word count |
| `written in the FIRST PERSON: an ordinary person describing their own situation in two or three sentences` | voice + sentence count |
| `naming the pressure they are under, and ending in a direct question that weighs the two courses open to them` | structure |
| `SHORT and IMPERATIVE, about five to eight words` | word count + register |
| `Flat, unhedged, no reason given.` | register |
| `LONGER, about twelve to twenty words, and softer` | relative length + word count + register |
| `phrased as a recommendation rather than an order, sometimes giving the reason in the same breath` | register |
| `That contrast in length and register is a real property of this corpus, but it is a TENDENCY, not a rule: do not let either class settle into a single opening formula, and write some protective replies that are as direct as the harmful ones.` | relative length + register |

### The two elided entries, in full

The pairing sentence appears three times in arm 2, once per paired split:

```text
The pairing is the defining property of this split: every request you write must come with both replies.
The pairing is the defining property of this split: every scenario you write must come with both decisions.
The pairing is the defining property of this split: every situation you write must come with both pieces of advice.
```

The `ai_dilemmas` structure sentence in full:

```text
It sets out the setting, then a genuine trade-off between two courses — a measurable gain set against something that protects the people the decision lands on — and it ends with a direct question of the form "Should you X, or Y?".
```

---

## Residual confounders this arm still carries

Stated rather than hidden. Arm 3 is the strictest of the three, not a clean-room.

| residual | why it stays |
|---|---|
| **The output schema `{"user": "...", "assistant": "..."}`** fixes exactly one user turn and one assistant reply | Irreducible: the parser and `coerce_messages` require it, and a free-form schema would change the pipeline rather than the prompt. It is the one piece of shape that cannot be removed without a different experiment. Note it applies equally to the unsteered generic control, so it does not differentiate arms. |
| **The label strings** `harmful_to_human` / `not_harmful_to_human` | Read off the probe, present in every arm including unsteered generic. |
| **The situation sentence** — what the split is *about* | This is the whole remaining treatment. Remove it and all three prompts become the unsteered generic prompt, which already exists as `data/hu_harm_llama70b_600.jsonl`. That set is the floor this arm is measured against. |
| **`daily_dilemmas`' consequence list** — `health, safety, money, job or a relationship` | A five-way category enumeration derived from the split. Kept because it is what defines the label rather than how a reply looks, but it is the weakest justification in this file. |
| **Generation settings** — model, temperature, batch size, 600 rows at 300/300, the `avoid` de-duplication block | Held fixed from the previous protocol so the prompt is the only thing that varies. |

## If this is run

To stay comparable with the clean ablation: same generator (`meta-llama/llama-3.3-70b-instruct`),
600 rows at 300/300 per split, generated-rows-only fits through
`scripts/fit_hu_harm_split_targeted.py` with `--accum 4`, the full fit plus 4 draws at 90%,
every fit scored on **all four** splits. Three datasets, not four — `ant_hh` and
`balanced_refusal` share a prompt, so either one set is fit against both splits or two sets
are drawn from the same prompt to measure generator variance directly.
