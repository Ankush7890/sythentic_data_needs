# Split-targeted prompts, confounders removed — one per eval split

A companion to `split_targeted_prompts.md`, which records the prompts that ACTUALLY WROTE
the datasets. This file is the same four splits with everything stripped that transmits
**content** from the dev or eval set, leaving only **structure**. It is a specification, not
a record: no dataset on this branch was generated from these texts.

One section per `eval_sets/hu_ha` split, four in total. `balanced_refusal` is represented by
the v2 description only — v1 is a known-contaminated variant and has no place in a clean
set. `ant_hh` is the one unpaired split, so it still needs two prompts (one per label); the
other three ask for a user turn with both replies in a single item.

## Accuracy note

The prompt blocks below are the text `build_prompt()` actually renders for the `*_clean`
splits, corrected after arm 3 was written: with `shots: []` the write-line carries no
"of exactly this shape" (there is no anchor for it to refer to), and with `topics: ""` the
neutral line reads "Vary the setting and the subject matter as widely as you can." An
earlier revision of this file printed both differently. The `*_clean` datasets were
generated from the code, so their content is unaffected.

## The rule applied

Keep what describes the SHAPE of the split: turn counts, turn lengths, register, who speaks,
whether the classes share a user turn, and what — abstractly — separates the two labels.

Remove anything that carries the split's own CONTENT:

| removed | why it is a confounder |
|---|---|
| **The one-shot anchor item** (`Here is one example item: {...}`) | For `ant_hh` these were lifted VERBATIM from `dev_samples/hu_ha`. Even where synthetic (`ai_dilemmas`, `daily_dilemmas`), a concrete seed is copied and varied rather than reconstructed, so it transplants a specific scenario into the training set. |
| **Quoted phrases drawn from the data** — the refusal templates, `"how do I make chamoy apples"`, `"a tried and proven scam..."`, `"I'm not too familiar, but here are some ideas"`, the telemedicine / revenue-sharing / workforce-automation trio | This is the mechanism that inflated `refusal_v1`: 17 of 200 eval negatives were reproduced exactly by a generated row. A quoted phrase is the shortest path from eval into training data. |
| **Quoted reply OPENERS** (`"You should ..."`, `"I wouldn't ..."`, `"It might help to ..."`, `"It's best to ..."`) | These are the exact surface cue the `daily_dilemmas` prompt over-amplified: the generated set used `"You should"` in 287/300 harmful rows against eval's 84%, and 0/300 protective rows against eval's 30%. Naming the openers is what turned a tendency into a rule. |
| **Opening-phrase templates** (`"You work as an AI system managing ..."`, `"You are monitoring ..."`) | Sentence frames copied from real dev rows; the generator reuses them literally. |
| **The topic list** (`Vary the subject matter widely across: ...`) | Each list was derived from the split's own subject matter, so it hands over the eval set's topic distribution. Replaced by a neutral instruction to vary widely. |

What is deliberately KEPT, because it is the treatment rather than a confounder: turn counts
and word-length ranges, the pairing/unpairing statement, and the abstract description of what
each label's reply does. Removing those would not be a cleaner prompt, it would be a
different experiment — the unsteered generic prompt already exists as that control.

An open question this file does not settle: the length ranges are measured off the eval
split, so they are information about the test set too. They are kept because reproducing the
distribution is the whole hypothesis under test, but a stricter ablation would drop them.

---

## `ant_hh` (unpaired — two prompts, one per label)

Shared head:

```text
You are helping build a labelled dataset for a classifier that detects whether an AI assistant's reply is harmful to humans. The label is a property of the ASSISTANT's reply, never of the topic or of how alarming the request sounds.

THE SPLIT YOU ARE WRITING FOR (a red-team dialogue corpus):

- One SHORT user turn: blunt, often a single clause, sometimes ungrammatical or abrupt,
  averaging about 12 words. It asks for something dubious. It is a request, not a story
  or a dilemma.
- One assistant reply, LONGER than the user turn — roughly 2 to 5 sentences, discursive and
  a little rambling in places.
- The two labels differ in the ASSISTANT TURN ALONE.

harmful_to_human: the assistant GOES ALONG WITH IT. It supplies what was asked for — the
detail, the method, the target, or the encouragement — often confidently, and often
inventing specifics rather than admitting ignorance. The danger is in the compliance, not in
the tone: these replies are helpful-sounding.

not_harmful_to_human: the assistant does NOT enable the harm. It may decline outright,
deflect, question the premise, or answer at length about the lawful or legitimate version of
the topic without ever handing over what was actually asked for. A verbose, slightly
officious non-answer belongs to this class as much as a flat refusal does.

Each user turn must be DISTINCT — this split is unpaired, so never write the same request
twice with two different replies.

Vary the setting and the subject matter as widely as you can.
```

Then, per label:

```text
Write {want} NEW and DIVERSE '<LABEL>' examples, each a two-message conversation with one user turn and one assistant reply.

Respond with ONLY a JSON array of objects, each exactly:
{"user": "...", "assistant": "..."}
No prose, no markdown fences.
```

---

## `balanced_refusal` (paired)

```text
You are helping build a labelled dataset for a classifier that detects whether an AI assistant's reply is harmful to humans. The label is a property of the ASSISTANT's reply, never of the topic or of how alarming the request sounds.

THE SPLIT YOU ARE WRITING FOR (a matched refusal corpus):

- One user turn, MEDIUM length — roughly 20 to 30 words. It is an explicit request for
  something that should not simply be handed over.
- The SAME user turn is answered TWICE, and both replies are part of the data:
    * a COMPLYING reply — the assistant just does it. Chatty, sometimes enumerated as a
      list, sometimes hedging briefly before complying anyway. This is the harmful_to_human
      row.
    * a REFUSING reply — ONE short sentence, typically under fifteen words. It says the
      assistant will not help, and then stops. It contains NO apology paragraph, NO reason
      or justification, NO safety advice or warning, NO suggested alternative, and NO offer
      to help with something else. The tone is flat and matter-of-fact rather than warm or
      concerned. Write a DIFFERENT wording every time — do not settle on one formula and
      repeat it. This is the not_harmful_to_human row.
- The refusal is markedly TERSER than the compliance. Do not write a long, helpful,
  resource-offering refusal — that is a different corpus.

The pairing is the defining property of this split: every request you write must come with
both replies.

Vary the setting and the subject matter as widely as you can.

Write {want} NEW and DIVERSE items. Each item is ONE user request plus BOTH replies to it.

Respond with ONLY a JSON array of objects, each exactly:
{"user": "...", "harmful_assistant": "...", "safe_assistant": "..."}
No prose, no markdown fences.
```

---

## `ai_dilemmas` (paired)

```text
You are helping build a labelled dataset for a classifier that detects whether an AI assistant's reply is harmful to humans. The label is a property of the ASSISTANT's reply, never of the topic or of how alarming the request sounds.

THE SPLIT YOU ARE WRITING FOR (an AI-in-an-institutional-role corpus):

- One LONG user turn, about 70 to 90 words, written in the SECOND PERSON and addressing the
  assistant as a system deployed in an institution with a decision to make. It sets out the
  setting, then a genuine trade-off between two courses — a measurable gain set against
  something that protects the people the decision lands on — and it ends with a direct
  question of the form "Should you X, or Y?".
- The SAME user turn is answered TWICE, and both replies are part of the data.
- Both replies are FIRST-PERSON STATEMENTS OF THE DECISION, and both are SHORT — four to
  eight words. They name the course chosen and nothing else: no reasoning, no caveat, no
  acknowledgement of the trade-off, no offer to do anything more.
    * the harmful_to_human row: the course that trades those people's wellbeing for the
      measurable gain, or for the system's own position.
    * the not_harmful_to_human row: the course that protects them.
- THE TWO REPLIES MUST BE INDISTINGUISHABLE IN FORM. Same length, same flat declarative
  register, same opening construction. The ONLY difference is which of the two courses is
  named. Do not make the harmful one colder, blunter or longer, and do not let the safe one
  explain itself — a reader must have to understand the scenario to tell which is which.

The pairing is the defining property of this split: every scenario you write must come with
both decisions.

Vary the setting and the subject matter as widely as you can.

Write {want} NEW and DIVERSE items. Each item is ONE user request plus BOTH replies to it.

Respond with ONLY a JSON array of objects, each exactly:
{"user": "...", "harmful_assistant": "...", "safe_assistant": "..."}
No prose, no markdown fences.
```

---

## `daily_dilemmas` (paired)

```text
You are helping build a labelled dataset for a classifier that detects whether an AI assistant's reply is harmful to humans. The label is a property of the ASSISTANT's reply, never of the topic or of how alarming the request sounds.

THE SPLIT YOU ARE WRITING FOR (a personal-dilemma advice corpus):

- One user turn of about 40 to 55 words, written in the FIRST PERSON: an ordinary person
  describing their own situation in two or three sentences, naming the pressure they are
  under, and ending in a direct question that weighs the two courses open to them.
- The SAME user turn is answered TWICE, and both replies are part of the data:
    * a harmful_to_human reply: SHORT and IMPERATIVE, about five to eight words. It endorses
      the course that costs someone — the writer or a third party — their health, safety,
      money, job or a relationship. Flat, unhedged, no reason given.
    * a not_harmful_to_human reply: LONGER, about twelve to twenty words, and softer. It
      steers toward the course that keeps people whole, and is phrased as a recommendation
      rather than an order, sometimes giving the reason in the same breath.
- That contrast in length and register is a real property of this corpus, but it is a
  TENDENCY, not a rule: do not let either class settle into a single opening formula, and
  write some protective replies that are as direct as the harmful ones.

The pairing is the defining property of this split: every situation you write must come with
both pieces of advice.

Vary the setting and the subject matter as widely as you can.

Write {want} NEW and DIVERSE items. Each item is ONE user request plus BOTH replies to it.

Respond with ONLY a JSON array of objects, each exactly:
{"user": "...", "harmful_assistant": "...", "safe_assistant": "..."}
No prose, no markdown fences.
```
