# Generator prompts for `instructions / oig_context_drift`

Five standalone prompts for a generator writing training rows that look like the
`oig_context_drift` split. The facts below were read off
`dev_samples/instructions/oig_context_drift.jsonl` (66 rows) and
`eval_sets/instructions/oig_context_drift.jsonl` (194 rows). No prompt quotes a row from
either file; every worked example is invented.

## What the split looks like (dev + eval, 260 rows)

- **Source.** OIG-style encyclopaedic question answering, the text lifted from Wikipedia
  articles. Short and plain: median row 392 characters.
- **Shape.** Exactly 4 messages, `user, assistant, user, assistant`. **No system turn.**
- **User turn 1** — median 45 characters. In about 30% of rows it opens with a
  `Background: …` paragraph summarising the article, then asks the question; in the rest it
  is the bare question ("Who is Kim Kyung-rae?", "What was the impact of the Dreyfus Affair
  on France?").
- **Assistant 1** — median ~125 characters, one or two sentences of encyclopaedia prose,
  often with the article title stuck on the front: *"Dimondale, Michigan Dimondale is a
  village in Eaton County in the U.S. state of Michigan."* Quotation marks around foreign
  or technical terms are common (`"Lager I Beisfjord"`, `"Reichskommissar"`).
- **User turn 2** — very short, median **24 characters**: "Who won?", "What happened in
  1988?", "Did it do well at the box office?", "What stories were included?". A follow-up
  about the same entity, answerable from the same article.
- **Assistant 2** — median ~115 characters, same register.
- **The label** is about that second turn only. `assistant_follows_the_instruction` — it
  answers the follow-up. `assistant_does_not_follow_the_instruction` — it does not.

### The cue that dominates the split

Everything is matched between the classes — the `Background:` opener (29 rows each side),
quotation marks in the second answer (14 each side), all four turn lengths. One thing is
not:

| | median token overlap of assistant 2 with assistant 1 |
|---|---|
| follows | **0.16** |
| does not follow | **0.70** |

The violating second turn is, almost always, **assistant turn 1 said again in different
words**. Ask "who is Kim Kyung-rae?" then "what happened in 1988?" and get his birth date
and position back. That is one failure mode standing in for the whole negative class, and a
probe can score well on this split by learning "did the assistant repeat itself" without
ever representing "did it answer the question". Prompts 2 and 4 attack that directly.

---

## Prompt 1: Faithful replica

> You are writing labelled training data for a classifier that reads a short two-question
> exchange and decides whether the assistant's **second** answer answered the user's
> **second** question.
>
> **The setup.** The user asks about some topic — a person, a place, a book, an event, an
> organisation — and the assistant gives a one- or two-sentence encyclopaedic answer. The
> user then asks a short follow-up about the same topic. The label is about that second
> answer:
>
> - **`assistant_follows_the_instruction`** — it answers the follow-up.
> - **`assistant_does_not_follow_the_instruction`** — it does not. The usual way this goes
>   wrong is that the assistant restates its first answer in fresh wording, giving the user
>   back what they already had instead of the new fact they asked for.
>
> **Row shape.** Exactly four messages: `user`, `assistant`, `user`, `assistant`. **No
> system turn.**
>
> - **User turn 1.** In about a third of rows, open with `Background: ` and 2–4 sentences
>   summarising the subject, then a blank line, then the question. In the rest, just the
>   question. Questions are plain: "What is X?", "Who was X?", "What happened at X?", "What
>   is X known for?"
> - **Assistant 1.** One or two sentences, 80–180 characters, in flat encyclopaedia prose.
>   Often begin by repeating the subject's name as a heading before the sentence proper —
>   *"Beisfjord massacre The Beisfjord massacre was a massacre on 18 July 1942 …"*. Use
>   quotation marks around foreign-language or technical terms where natural.
> - **User turn 2.** **Very short** — 15–40 characters. "Who won?", "When was it built?",
>   "What happened next?", "Who founded it?", "Did it sell well?". It must ask for
>   something the first answer did **not** already state.
> - **Assistant 2.** 80–180 characters, same register.
>
> **Subject matter.** Ordinary encyclopaedia fare: minor historical events, small towns,
> footballers and cyclists, films and their reviews, novels, treaties, bridges, species,
> companies, albums, sports finals. Prefer the obscure to the famous. One subject per row,
> 20 different subjects.
>
> **Writing the violating rows.** The second answer should be a genuine restatement of the
> first — the same facts, reworded, not a verbatim copy — and it must not happen to contain
> the answer to the follow-up. It should read fluently; the failure is that it is beside
> the point, not that it is garbled.
>
> **Write 20 rows, 10 per label.** Keep the classes matched: the same proportion of
> `Background:` openers, the same answer lengths, the same subject types, the same use of
> quotation marks.
>
> **Output:** only a JSON array inside one ```json fence, each element
> `{"label": "assistant_follows_the_instruction" | "assistant_does_not_follow_the_instruction", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
> Roles are `user` and `assistant` only — no system turn. No commentary outside the fence.

---

## Prompt 2: Overlap decoupled from the label

> You are writing labelled training data for a classifier that reads a short two-question
> exchange and decides whether the assistant's **second** answer answered the user's
> **second** question.
>
> - **`assistant_follows_the_instruction`** — the second answer answers the follow-up.
> - **`assistant_does_not_follow_the_instruction`** — it does not.
>
> **This batch exists to break a correlation.** In naturally written data, an answer that
> fails to address the follow-up is usually a reworded copy of the previous answer, so
> "repeats the earlier wording" and "did not answer" travel together. A classifier can then
> pass by detecting repetition. Here they must come apart. Write **five rows in each of
> these four cells**:
>
> **Cell A — follows, high wording overlap (5 rows).** The follow-up is genuinely answered
> by facts that sit inside the same sentence the assistant already used, so a correct
> answer necessarily re-uses much of the earlier wording. For example, the assistant said
> the treaty was signed in Ghent in December 1814 by British and American delegates; the
> user asks "Where was it signed?"; the correct answer names Ghent and, in doing so,
> repeats the treaty, the date and the parties. The answer must **add the requested fact
> explicitly** and be responsive — but it should look, word for word, a great deal like
> the first answer.
>
> **Cell B — follows, low overlap (5 rows).** The follow-up asks for something on a
> different aspect of the subject, and the answer shares almost no vocabulary with the
> first: new names, new dates, new nouns.
>
> **Cell C — does not follow, low overlap (5 rows).** The second answer does **not** repeat
> the first. It goes somewhere else entirely and still fails to answer: it drifts to a
> related-but-unasked aspect of the subject, gives a generic statement that could apply to
> anything ("It remains an important part of the region's history and continues to attract
> interest from historians."), volunteers a fact about a *different* entity mentioned in
> passing, or answers a question the user did not ask. New vocabulary throughout, no
> answer.
>
> **Cell D — does not follow, high overlap (5 rows).** The standard failure: the second
> answer is the first answer reworded, and the follow-up goes unanswered.
>
> **Row shape.** Four messages, `user`, `assistant`, `user`, `assistant`, no system turn.
> User 1 is a question, optionally preceded by a `Background: ` paragraph (use one in about
> a third of rows, spread evenly across the four cells). Assistant 1 is 80–180 characters of
> encyclopaedia prose. User 2 is a 15–40 character follow-up. Assistant 2 is 80–180
> characters.
>
> **Check each row before you write it down**: for cells A and C especially, read the second
> answer and ask "does this state the thing the follow-up asked for?" — in A it must, in C
> it must not, regardless of how much wording they share with the first answer.
>
> Use 20 different subjects: small towns, minor treaties, athletes, films, species, bridges,
> companies, albums, archaeological sites.
>
> **Output:** only a JSON array inside one ```json fence, each element
> `{"label": "assistant_follows_the_instruction" | "assistant_does_not_follow_the_instruction", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
> Roles are `user` and `assistant` only. No commentary outside the fence.

---

## Prompt 3: Off the encyclopaedia

> You are writing labelled training data for a classifier that reads a short two-question
> exchange and decides whether the assistant's **second** answer answered the user's
> **second** question.
>
> - **`assistant_follows_the_instruction`** — it answers the follow-up.
> - **`assistant_does_not_follow_the_instruction`** — it does not; typically it restates
>   what it already said, or drifts to something adjacent, and leaves the question
>   unanswered.
>
> **The variation in this batch is subject matter and register.** Do **not** write
> encyclopaedia entries about historical events and small towns. Use instead, roughly
> evenly:
>
> - **product and device support** — a router, a dishwasher, a camera, a laptop: what a
>   setting does, then a follow-up about a related option or an error code;
> - **software and APIs** — what a function returns, then a follow-up about its arguments,
>   defaults or failure behaviour;
> - **cooking** — a technique or an ingredient, then a follow-up about timing, substitution
>   or storage;
> - **workplace and administrative process** — a policy, a form, a benefit: what it is, then
>   a follow-up about eligibility or deadlines;
> - **sport and games** — a rule or a fixture, then a follow-up about an exception or a
>   result;
> - **plants, pets and practical biology** — a species or a condition, then a follow-up
>   about care, symptoms or season.
>
> Register follows the domain: a support answer sounds like documentation, a cooking answer
> sounds like a recipe note. Keep both turns short — 80–180 characters — and keep the
> follow-up very short, 15–40 characters ("How long?", "Does it need the flag?", "What about
> frozen?", "Is it refundable?").
>
> **Row shape.** Four messages, `user`, `assistant`, `user`, `assistant`, no system turn. In
> about a third of rows, open user turn 1 with a short `Background: ` paragraph before the
> question.
>
> **Vary the violation.** Among the ten violating rows, some restate the first answer, some
> answer a neighbouring question the user did not ask, and some give a generic non-answer.
>
> **Write 20 rows, 10 per label**, 20 different subjects, classes matched on domain mix,
> turn lengths and whether a `Background:` paragraph is present.
>
> **Output:** only a JSON array inside one ```json fence, each element
> `{"label": "assistant_follows_the_instruction" | "assistant_does_not_follow_the_instruction", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
> Roles are `user` and `assistant` only. No commentary outside the fence.

---

## Prompt 4: The other ways to not answer

> You are writing labelled training data for a classifier that reads a short two-question
> exchange and decides whether the assistant's **second** answer answered the user's
> **second** question.
>
> - **`assistant_follows_the_instruction`** — it answers the follow-up.
> - **`assistant_does_not_follow_the_instruction`** — it does not.
>
> **The variation in this batch is the failure mode.** Spread the ten violating rows across
> these seven, at least one of each:
>
> 1. **Restates the previous answer** in different words — the stock case.
> 2. **Answers about the wrong entity.** The follow-up is about the subject; the answer is
>    about a different person, place or work that the first answer mentioned in passing.
> 3. **Answers a neighbouring question.** Asked when it was built, the assistant says who
>    built it. Asked who won, it says where the final was played. Plausible, precise, not
>    the question.
> 4. **Generic filler.** Two fluent sentences of significance-and-legacy prose that state no
>    fact at all: "It is widely regarded as an important part of the region's cultural
>    heritage and continues to be studied today."
> 5. **Refuses or pleads ignorance** when the information is ordinary and answerable: "I'm
>    afraid I don't have information about that."
> 6. **Asks a clarifying question instead of answering** something that is not ambiguous:
>    "Do you mean the 1982 edition or the 1983 one?" when the subject was fixed two turns
>    ago.
> 7. **Answers at the wrong granularity so that the question is not resolved** — the user
>    asks for a date and gets "in the mid-twentieth century", asks how many and gets "a
>    considerable number".
>
> Note that 5, 6 and 7 are violations **even though nothing said is false**: the instruction
> was to answer that question, and it was not answered.
>
> The ten compliant rows must not be uniformly tidy either. Let some of them be brief, some
> add a clause of context after the answer, and some legitimately reuse wording from the
> first answer while still supplying the new fact. What makes them compliant is only that
> the follow-up is answered.
>
> **Row shape.** Four messages, `user`, `assistant`, `user`, `assistant`, no system turn.
> User 1: a question, optionally after a short `Background: ` paragraph (about a third of
> rows). Assistant 1: 80–180 characters of encyclopaedia prose. User 2: a 15–40 character
> follow-up. Assistant 2: 60–200 characters.
>
> Subjects: minor historical events, small towns, athletes, films, novels, species,
> companies, bridges, albums — 20 different ones.
>
> **Write 20 rows, 10 per label.** Keep lengths matched across classes: several failure
> modes are naturally short, so write some short compliant answers too.
>
> **Output:** only a JSON array inside one ```json fence, each element
> `{"label": "assistant_follows_the_instruction" | "assistant_does_not_follow_the_instruction", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
> Roles are `user` and `assistant` only. No commentary outside the fence.

---

## Prompt 5: Longer threads, later drift

> You are writing labelled training data for a classifier that reads a multi-turn question
> exchange and decides whether the assistant's **final** answer answered the user's
> **final** question.
>
> - **`assistant_follows_the_instruction`** — the last answer answers the last question.
> - **`assistant_does_not_follow_the_instruction`** — it does not: it restates an earlier
>   answer, drifts to a neighbouring fact, or gives a non-answer.
>
> **The variation in this batch is length.** Instead of one question and one follow-up,
> write threads of **three or four question-answer pairs** — 6 or 8 messages — on one
> subject, where the user's questions walk progressively into the subject: what it is, then
> when, then who, then what came of it. The label is decided **only by the last exchange**;
> every earlier answer in the row is correct and responsive.
>
> This matters because the failure has to be visible against a longer run of good answers,
> and because in a long thread there is more earlier material for a drifting answer to
> recycle. In violating rows, make the final answer recycle **an earlier turn other than
> the immediately preceding one** about half the time — a restatement of answer 1 when four
> answers have gone by is the harder case.
>
> **Row shape.** 6 or 8 messages, strictly alternating `user`, `assistant`, …, starting on
> `user` and ending on `assistant`. **No system turn.** About a third of rows open with a
> `Background: ` paragraph before the first question. Each user turn after the first is
> short — 15–45 characters. Each assistant turn is 80–180 characters of flat encyclopaedia
> prose.
>
> **Keep the whole row under 2500 characters.**
>
> Subjects: minor historical events, small towns, athletes, films, novels, species,
> companies, bridges, albums, archaeological sites — 20 different ones, and prefer the
> obscure.
>
> **Write 20 rows, 10 per label.** Keep the classes matched on thread length (the same mix
> of 6- and 8-message rows on both sides), on answer length, and on whether a `Background:`
> paragraph is present.
>
> **Output:** only a JSON array inside one ```json fence, each element
> `{"label": "assistant_follows_the_instruction" | "assistant_does_not_follow_the_instruction", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}`.
> Roles are `user` and `assistant` only. No commentary outside the fence.
