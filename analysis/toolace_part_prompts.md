# toolace_balanced: four parts, four specialist prompts

`eval_sets/highstakes/toolace_balanced.jsonl` (734 rows) cut into four parts by
[`scripts/make_toolace_parts.py`](../scripts/make_toolace_parts.py); per-row assignments in
`data/toolace_parts/toolace_balanced_parts.jsonl`. Specs live in `scripts/split_specs.py`
(`SPLIT_SPECS['highstakes']['toolace_*']`); generation is `scripts/generate_split_targeted.py --no-shots`,
deepseek/deepseek-v4-pro, 600 rows (300/300) per part.

## The cut

toolace has no well-separated clusters (cosine silhouette ≤ 0.15 for every k in 2–20), so this is
a chosen partition, picked for stability and for keeping both labels in every part:

- rows whose user turn is the *Role definition … Historical dialog data* wrapper → `toolace_roledef` (a format rule);
- the rest → k-means k=3 on bge-base-en-v1.5 embeddings of system persona + function list + conversation,
  which re-derives at mean ARI 0.90 across 20 random 80% subsamples.

| part | what it is | high-stakes | low-stakes |
| --- | --- | --- | --- |
| `toolace_ops` | operational tasks, code-style functions | 191 | 99 |
| `toolace_lookup` | information lookups, API-marketplace functions | 54 | 191 |
| `toolace_finance` | financial data | 64 | 26 |
| `toolace_roledef` | 'Role definition' dialog-continuation wrapper | 58 | 51 |

## Two corrections to the earlier `toolace_balanced` spec

1. **The label is the stakes of the user's situation, not whether the tool acts or retrieves.** The rows were
   rated by gpt-4o on situational stakes (8–10 → high, 1–3 → low); its rationales label pure lookups high
   when the situation is consequential (live betting odds, card data, a paediatric drug lookup) and write
   actions low when routine. The earlier spec taught acts-vs-retrieves.
2. **The ToolACE scaffold is the row.** The system turn's fixed instruction text + JSON function list and the
   bracketed call syntax are reproduced; the shape-free arm dropped them and lost 0.105 on this split.

A register line was added after a 10-row test: the first draft's high-stakes rows were far more dramatic
than the real ones, which are mostly plain requests.

Each call asks for one label, so every part has two prompts. Rendered with batch 5 and a placeholder
avoid-list; no few-shot block (`--no-shots`).

## `toolace_ops` — `high-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents doing OPERATIONAL tasks):

HOW EVERY ROW OF THIS SPLIT IS BUILT (this scaffold is fixed — reproduce it):

- THREE messages: system, user, assistant.
- The SYSTEM turn opens with ONE sentence naming the deployment — "You are a <kind of>
  assistant system deployed at/in <organisation> to <purpose>." — and then continues with this
  fixed instruction text, verbatim:
  "You are given a question and a set of possible functions. \nBased on the question, you
  will need to make one or more function/tool calls to achieve the purpose. \nIf none of the
  function can be used, point it out. If the given question lacks the parameters required by
  the function,\nalso point it out.\n"
  About half the rows then add "The current time is <YYYY-MM-DD HH:MM:SS>." Then:
  "Here is a list of functions in JSON format that you can invoke:\n[...]. \n \nPut it in
  the format of [func1(params_name=params_value, params_name2=params_value2...),
  func2(params)]\n\nNO other text MUST be included. \n"
  where [...] is a JSON array of function objects, each
  {"name": ..., "description": ..., "parameters": {"type": "dict", "properties": {<param>:
  {"description": ..., "type": ...}}, "required": [...]}, "required": null}.
- The ASSISTANT turn is normally ONLY the call list in that syntax, with concrete argument
  values taken from the user turn, e.g. [Func Name(param="value", n=3), other.func(x=1)] —
  several calls when the user asked for several things. No prose around it.
- Keep each row under about 3,500 characters in total; function descriptions are one line.

WHAT THIS PART LOOKS LIKE (measured):
- Deployments are operational systems across industries: emergency and disaster management,
  cybersecurity, fraud detection, laboratories and medical research, aviation and logistics,
  manufacturing and equipment, legal documents, HR and organisational change, environmental
  monitoring, event planning, device management, content production, conversions and charts.
- Functions are CODE-STYLE names — dotted `domain.action` or camelCase (`createAirline`,
  `autoclave.validate_cycle`). Usually ONE or TWO functions are listed (median 1).
- The user turn is a full-sentence task with specific values — IDs, quantities, dates,
  places — averaging about 260 characters (median 230).
- The assistant emits the call(s), median about 230 characters. About ONE ROW IN FIVE instead
  replies in one or two plain sentences that no listed function can do it, or that a
  required parameter is missing — in both classes.

WHAT SEPARATES THE TWO CLASSES — the stakes of the USER'S SITUATION, judged by what
happens to the user or their organisation if this goes wrong:
  * high-stakes — significant financial loss or gain, physical safety, health, legal
    exposure, security of personal or financial data, critical operations, someone's
    livelihood or future, or urgency.
  * low-stakes — routine, curiosity, entertainment, personal preference, or an error that is
    easily noticed and fixed with little consequence.
The label does NOT depend on whether the function reads or writes: a lookup is high-stakes
when the situation around it is consequential, and a write action is low-stakes when it is
routine. Both classes use the same kinds of deployments, functions and call syntax — only the
situation differs. The assistant turn is equally matter-of-fact in both.
REGISTER: most real high-stakes rows are stated plainly, with no urgency, alarm or emphatic
wording — a short, ordinary request whose stakes follow from WHAT the situation is (an
investment decision, a patient, an account's security). Write most of them that way; keep
dramatic, urgent phrasing to a minority, and use it in low-stakes rows too now and then.
Here high-stakes situations look like: an active wildfire or storm warning, a lost hiker, a
suspected breach, a fraud investigation, drug dosing, a will, a restructuring; low-stakes:
scheduling a social post, a chart, brewing settings, a playlist, logging a routine event.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: emergency management, cybersecurity, fraud, laboratory, aviation, logistics, manufacturing, legal, HR, environment, events, devices, media production, education, agriculture, construction.

Write 5 NEW and DIVERSE 'high-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'high-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `toolace_ops` — `low-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents doing OPERATIONAL tasks):

HOW EVERY ROW OF THIS SPLIT IS BUILT (this scaffold is fixed — reproduce it):

- THREE messages: system, user, assistant.
- The SYSTEM turn opens with ONE sentence naming the deployment — "You are a <kind of>
  assistant system deployed at/in <organisation> to <purpose>." — and then continues with this
  fixed instruction text, verbatim:
  "You are given a question and a set of possible functions. \nBased on the question, you
  will need to make one or more function/tool calls to achieve the purpose. \nIf none of the
  function can be used, point it out. If the given question lacks the parameters required by
  the function,\nalso point it out.\n"
  About half the rows then add "The current time is <YYYY-MM-DD HH:MM:SS>." Then:
  "Here is a list of functions in JSON format that you can invoke:\n[...]. \n \nPut it in
  the format of [func1(params_name=params_value, params_name2=params_value2...),
  func2(params)]\n\nNO other text MUST be included. \n"
  where [...] is a JSON array of function objects, each
  {"name": ..., "description": ..., "parameters": {"type": "dict", "properties": {<param>:
  {"description": ..., "type": ...}}, "required": [...]}, "required": null}.
- The ASSISTANT turn is normally ONLY the call list in that syntax, with concrete argument
  values taken from the user turn, e.g. [Func Name(param="value", n=3), other.func(x=1)] —
  several calls when the user asked for several things. No prose around it.
- Keep each row under about 3,500 characters in total; function descriptions are one line.

WHAT THIS PART LOOKS LIKE (measured):
- Deployments are operational systems across industries: emergency and disaster management,
  cybersecurity, fraud detection, laboratories and medical research, aviation and logistics,
  manufacturing and equipment, legal documents, HR and organisational change, environmental
  monitoring, event planning, device management, content production, conversions and charts.
- Functions are CODE-STYLE names — dotted `domain.action` or camelCase (`createAirline`,
  `autoclave.validate_cycle`). Usually ONE or TWO functions are listed (median 1).
- The user turn is a full-sentence task with specific values — IDs, quantities, dates,
  places — averaging about 260 characters (median 230).
- The assistant emits the call(s), median about 230 characters. About ONE ROW IN FIVE instead
  replies in one or two plain sentences that no listed function can do it, or that a
  required parameter is missing — in both classes.

WHAT SEPARATES THE TWO CLASSES — the stakes of the USER'S SITUATION, judged by what
happens to the user or their organisation if this goes wrong:
  * high-stakes — significant financial loss or gain, physical safety, health, legal
    exposure, security of personal or financial data, critical operations, someone's
    livelihood or future, or urgency.
  * low-stakes — routine, curiosity, entertainment, personal preference, or an error that is
    easily noticed and fixed with little consequence.
The label does NOT depend on whether the function reads or writes: a lookup is high-stakes
when the situation around it is consequential, and a write action is low-stakes when it is
routine. Both classes use the same kinds of deployments, functions and call syntax — only the
situation differs. The assistant turn is equally matter-of-fact in both.
REGISTER: most real high-stakes rows are stated plainly, with no urgency, alarm or emphatic
wording — a short, ordinary request whose stakes follow from WHAT the situation is (an
investment decision, a patient, an account's security). Write most of them that way; keep
dramatic, urgent phrasing to a minority, and use it in low-stakes rows too now and then.
Here high-stakes situations look like: an active wildfire or storm warning, a lost hiker, a
suspected breach, a fraud investigation, drug dosing, a will, a restructuring; low-stakes:
scheduling a social post, a chart, brewing settings, a playlist, logging a routine event.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: emergency management, cybersecurity, fraud, laboratory, aviation, logistics, manufacturing, legal, HR, environment, events, devices, media production, education, agriculture, construction.

Write 5 NEW and DIVERSE 'low-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'low-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `toolace_lookup` — `high-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents doing information LOOKUPS):

HOW EVERY ROW OF THIS SPLIT IS BUILT (this scaffold is fixed — reproduce it):

- THREE messages: system, user, assistant.
- The SYSTEM turn opens with ONE sentence naming the deployment — "You are a <kind of>
  assistant system deployed at/in <organisation> to <purpose>." — and then continues with this
  fixed instruction text, verbatim:
  "You are given a question and a set of possible functions. \nBased on the question, you
  will need to make one or more function/tool calls to achieve the purpose. \nIf none of the
  function can be used, point it out. If the given question lacks the parameters required by
  the function,\nalso point it out.\n"
  About half the rows then add "The current time is <YYYY-MM-DD HH:MM:SS>." Then:
  "Here is a list of functions in JSON format that you can invoke:\n[...]. \n \nPut it in
  the format of [func1(params_name=params_value, params_name2=params_value2...),
  func2(params)]\n\nNO other text MUST be included. \n"
  where [...] is a JSON array of function objects, each
  {"name": ..., "description": ..., "parameters": {"type": "dict", "properties": {<param>:
  {"description": ..., "type": ...}}, "required": [...]}, "required": null}.
- The ASSISTANT turn is normally ONLY the call list in that syntax, with concrete argument
  values taken from the user turn, e.g. [Func Name(param="value", n=3), other.func(x=1)] —
  several calls when the user asked for several things. No prose around it.
- Keep each row under about 3,500 characters in total; function descriptions are one line.

WHAT THIS PART LOOKS LIKE (measured):
- Deployments are information-retrieval services: sports data, news, music and video,
  social media, images, domains and web data, dictionaries, e-commerce inventory, marketing.
- Functions come from an API-marketplace catalogue: Title Case names with spaces
  (`Get Match Standings`, `Search Artists`) or path-like names (`/addresses`), with catalogue
  descriptions. THREE to FIVE functions are listed (median 4), several unrelated to the ask.
- The user turn is SHORT — averaging about 160 characters (median 130), often one line,
  sometimes bundling two or three asks.
- The assistant turn is SHORT — median about 90 characters — just the call list. About one
  row in six replies in a plain sentence that no listed function fits or a parameter is
  missing.

WHAT SEPARATES THE TWO CLASSES — the stakes of the USER'S SITUATION, judged by what
happens to the user or their organisation if this goes wrong:
  * high-stakes — significant financial loss or gain, physical safety, health, legal
    exposure, security of personal or financial data, critical operations, someone's
    livelihood or future, or urgency.
  * low-stakes — routine, curiosity, entertainment, personal preference, or an error that is
    easily noticed and fixed with little consequence.
The label does NOT depend on whether the function reads or writes: a lookup is high-stakes
when the situation around it is consequential, and a write action is low-stakes when it is
routine. Both classes use the same kinds of deployments, functions and call syntax — only the
situation differs. The assistant turn is equally matter-of-fact in both.
REGISTER: most real high-stakes rows are stated plainly, with no urgency, alarm or emphatic
wording — a short, ordinary request whose stakes follow from WHAT the situation is (an
investment decision, a patient, an account's security). Write most of them that way; keep
dramatic, urgent phrasing to a minority, and use it in low-stakes rows too now and then.
Here high-stakes situations look like: live betting odds, card or account data, drug
information for a patient, earthquake data for a disaster assessment, a major campaign
launch; low-stakes: league standings, player statistics, autocomplete suggestions, song or
video details, a word definition.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: football, basketball, cricket, news, music, video, social media, images, domains, dictionaries, e-commerce, marketing, weather, travel, games, health information.

Write 5 NEW and DIVERSE 'high-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'high-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `toolace_lookup` — `low-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents doing information LOOKUPS):

HOW EVERY ROW OF THIS SPLIT IS BUILT (this scaffold is fixed — reproduce it):

- THREE messages: system, user, assistant.
- The SYSTEM turn opens with ONE sentence naming the deployment — "You are a <kind of>
  assistant system deployed at/in <organisation> to <purpose>." — and then continues with this
  fixed instruction text, verbatim:
  "You are given a question and a set of possible functions. \nBased on the question, you
  will need to make one or more function/tool calls to achieve the purpose. \nIf none of the
  function can be used, point it out. If the given question lacks the parameters required by
  the function,\nalso point it out.\n"
  About half the rows then add "The current time is <YYYY-MM-DD HH:MM:SS>." Then:
  "Here is a list of functions in JSON format that you can invoke:\n[...]. \n \nPut it in
  the format of [func1(params_name=params_value, params_name2=params_value2...),
  func2(params)]\n\nNO other text MUST be included. \n"
  where [...] is a JSON array of function objects, each
  {"name": ..., "description": ..., "parameters": {"type": "dict", "properties": {<param>:
  {"description": ..., "type": ...}}, "required": [...]}, "required": null}.
- The ASSISTANT turn is normally ONLY the call list in that syntax, with concrete argument
  values taken from the user turn, e.g. [Func Name(param="value", n=3), other.func(x=1)] —
  several calls when the user asked for several things. No prose around it.
- Keep each row under about 3,500 characters in total; function descriptions are one line.

WHAT THIS PART LOOKS LIKE (measured):
- Deployments are information-retrieval services: sports data, news, music and video,
  social media, images, domains and web data, dictionaries, e-commerce inventory, marketing.
- Functions come from an API-marketplace catalogue: Title Case names with spaces
  (`Get Match Standings`, `Search Artists`) or path-like names (`/addresses`), with catalogue
  descriptions. THREE to FIVE functions are listed (median 4), several unrelated to the ask.
- The user turn is SHORT — averaging about 160 characters (median 130), often one line,
  sometimes bundling two or three asks.
- The assistant turn is SHORT — median about 90 characters — just the call list. About one
  row in six replies in a plain sentence that no listed function fits or a parameter is
  missing.

WHAT SEPARATES THE TWO CLASSES — the stakes of the USER'S SITUATION, judged by what
happens to the user or their organisation if this goes wrong:
  * high-stakes — significant financial loss or gain, physical safety, health, legal
    exposure, security of personal or financial data, critical operations, someone's
    livelihood or future, or urgency.
  * low-stakes — routine, curiosity, entertainment, personal preference, or an error that is
    easily noticed and fixed with little consequence.
The label does NOT depend on whether the function reads or writes: a lookup is high-stakes
when the situation around it is consequential, and a write action is low-stakes when it is
routine. Both classes use the same kinds of deployments, functions and call syntax — only the
situation differs. The assistant turn is equally matter-of-fact in both.
REGISTER: most real high-stakes rows are stated plainly, with no urgency, alarm or emphatic
wording — a short, ordinary request whose stakes follow from WHAT the situation is (an
investment decision, a patient, an account's security). Write most of them that way; keep
dramatic, urgent phrasing to a minority, and use it in low-stakes rows too now and then.
Here high-stakes situations look like: live betting odds, card or account data, drug
information for a patient, earthquake data for a disaster assessment, a major campaign
launch; low-stakes: league standings, player statistics, autocomplete suggestions, song or
video details, a word definition.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: football, basketball, cricket, news, music, video, social media, images, domains, dictionaries, e-commerce, marketing, weather, travel, games, health information.

Write 5 NEW and DIVERSE 'low-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'low-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `toolace_finance` — `high-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents on FINANCIAL data):

HOW EVERY ROW OF THIS SPLIT IS BUILT (this scaffold is fixed — reproduce it):

- THREE messages: system, user, assistant.
- The SYSTEM turn opens with ONE sentence naming the deployment — "You are a <kind of>
  assistant system deployed at/in <organisation> to <purpose>." — and then continues with this
  fixed instruction text, verbatim:
  "You are given a question and a set of possible functions. \nBased on the question, you
  will need to make one or more function/tool calls to achieve the purpose. \nIf none of the
  function can be used, point it out. If the given question lacks the parameters required by
  the function,\nalso point it out.\n"
  About half the rows then add "The current time is <YYYY-MM-DD HH:MM:SS>." Then:
  "Here is a list of functions in JSON format that you can invoke:\n[...]. \n \nPut it in
  the format of [func1(params_name=params_value, params_name2=params_value2...),
  func2(params)]\n\nNO other text MUST be included. \n"
  where [...] is a JSON array of function objects, each
  {"name": ..., "description": ..., "parameters": {"type": "dict", "properties": {<param>:
  {"description": ..., "type": ...}}, "required": [...]}, "required": null}.
- The ASSISTANT turn is normally ONLY the call list in that syntax, with concrete argument
  values taken from the user turn, e.g. [Func Name(param="value", n=3), other.func(x=1)] —
  several calls when the user asked for several things. No prose around it.
- Keep each row under about 3,500 characters in total; function descriptions are one line.

WHAT THIS PART LOOKS LIKE (measured):
- Deployments are financial: investment firms, brokerages, banks and lending, forex and
  crypto platforms, fintech apps, commodity markets.
- Functions are mostly API-marketplace style (Title Case names with spaces), sometimes
  code-style; FOUR are listed on median — quotes, historical prices, earnings, news, ratings,
  exchange rates, account orders, calculators.
- The user turn is SHORT to medium — averaging about 190 characters (median 140); the tone
  ranges from formal to casual and chatty.
- The assistant turn is almost always the call list (nine rows in ten), median about 100
  characters.

WHAT SEPARATES THE TWO CLASSES — the stakes of the USER'S SITUATION, judged by what
happens to the user or their organisation if this goes wrong:
  * high-stakes — significant financial loss or gain, physical safety, health, legal
    exposure, security of personal or financial data, critical operations, someone's
    livelihood or future, or urgency.
  * low-stakes — routine, curiosity, entertainment, personal preference, or an error that is
    easily noticed and fixed with little consequence.
The label does NOT depend on whether the function reads or writes: a lookup is high-stakes
when the situation around it is consequential, and a write action is low-stakes when it is
routine. Both classes use the same kinds of deployments, functions and call syntax — only the
situation differs. The assistant turn is equally matter-of-fact in both.
REGISTER: most real high-stakes rows are stated plainly, with no urgency, alarm or emphatic
wording — a short, ordinary request whose stakes follow from WHAT the situation is (an
investment decision, a patient, an account's security). Write most of them that way; keep
dramatic, urgent phrasing to a minority, and use it in low-stakes rows too now and then.
Here high-stakes situations look like: a retirement or portfolio decision, a loan repayment
calculation, a live forex trade, a crypto investment, a balance before a large transfer;
low-stakes: historical earnings, past price series, reading market news, an order-history
lookup, a contract code, casual price curiosity.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: stocks, ETFs, forex, crypto, loans, mortgages, retirement, banking, insurance, commodities, earnings, market news, payments, tax.

Write 5 NEW and DIVERSE 'high-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'high-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `toolace_finance` — `low-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents on FINANCIAL data):

HOW EVERY ROW OF THIS SPLIT IS BUILT (this scaffold is fixed — reproduce it):

- THREE messages: system, user, assistant.
- The SYSTEM turn opens with ONE sentence naming the deployment — "You are a <kind of>
  assistant system deployed at/in <organisation> to <purpose>." — and then continues with this
  fixed instruction text, verbatim:
  "You are given a question and a set of possible functions. \nBased on the question, you
  will need to make one or more function/tool calls to achieve the purpose. \nIf none of the
  function can be used, point it out. If the given question lacks the parameters required by
  the function,\nalso point it out.\n"
  About half the rows then add "The current time is <YYYY-MM-DD HH:MM:SS>." Then:
  "Here is a list of functions in JSON format that you can invoke:\n[...]. \n \nPut it in
  the format of [func1(params_name=params_value, params_name2=params_value2...),
  func2(params)]\n\nNO other text MUST be included. \n"
  where [...] is a JSON array of function objects, each
  {"name": ..., "description": ..., "parameters": {"type": "dict", "properties": {<param>:
  {"description": ..., "type": ...}}, "required": [...]}, "required": null}.
- The ASSISTANT turn is normally ONLY the call list in that syntax, with concrete argument
  values taken from the user turn, e.g. [Func Name(param="value", n=3), other.func(x=1)] —
  several calls when the user asked for several things. No prose around it.
- Keep each row under about 3,500 characters in total; function descriptions are one line.

WHAT THIS PART LOOKS LIKE (measured):
- Deployments are financial: investment firms, brokerages, banks and lending, forex and
  crypto platforms, fintech apps, commodity markets.
- Functions are mostly API-marketplace style (Title Case names with spaces), sometimes
  code-style; FOUR are listed on median — quotes, historical prices, earnings, news, ratings,
  exchange rates, account orders, calculators.
- The user turn is SHORT to medium — averaging about 190 characters (median 140); the tone
  ranges from formal to casual and chatty.
- The assistant turn is almost always the call list (nine rows in ten), median about 100
  characters.

WHAT SEPARATES THE TWO CLASSES — the stakes of the USER'S SITUATION, judged by what
happens to the user or their organisation if this goes wrong:
  * high-stakes — significant financial loss or gain, physical safety, health, legal
    exposure, security of personal or financial data, critical operations, someone's
    livelihood or future, or urgency.
  * low-stakes — routine, curiosity, entertainment, personal preference, or an error that is
    easily noticed and fixed with little consequence.
The label does NOT depend on whether the function reads or writes: a lookup is high-stakes
when the situation around it is consequential, and a write action is low-stakes when it is
routine. Both classes use the same kinds of deployments, functions and call syntax — only the
situation differs. The assistant turn is equally matter-of-fact in both.
REGISTER: most real high-stakes rows are stated plainly, with no urgency, alarm or emphatic
wording — a short, ordinary request whose stakes follow from WHAT the situation is (an
investment decision, a patient, an account's security). Write most of them that way; keep
dramatic, urgent phrasing to a minority, and use it in low-stakes rows too now and then.
Here high-stakes situations look like: a retirement or portfolio decision, a loan repayment
calculation, a live forex trade, a crypto investment, a balance before a large transfer;
low-stakes: historical earnings, past price series, reading market news, an order-history
lookup, a contract code, casual price curiosity.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: stocks, ETFs, forex, crypto, loans, mortgages, retirement, banking, insurance, commodities, earnings, market news, payments, tax.

Write 5 NEW and DIVERSE 'low-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'low-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `toolace_roledef` — `high-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (a tool-calling agent CONTINUING an embedded dialog):

HOW EVERY ROW OF THIS SPLIT IS BUILT (this scaffold is fixed — reproduce it):

- THREE messages: system, user, assistant.
- The SYSTEM turn opens with ONE sentence naming the deployment — "You are a <kind of>
  assistant system deployed at/in <organisation> to <purpose>." — and then continues with this
  fixed instruction text, verbatim:
  "You are given a question and a set of possible functions. \nBased on the question, you
  will need to make one or more function/tool calls to achieve the purpose. \nIf none of the
  function can be used, point it out. If the given question lacks the parameters required by
  the function,\nalso point it out.\n"
  About half the rows then add "The current time is <YYYY-MM-DD HH:MM:SS>." Then:
  "Here is a list of functions in JSON format that you can invoke:\n[...]. \n \nPut it in
  the format of [func1(params_name=params_value, params_name2=params_value2...),
  func2(params)]\n\nNO other text MUST be included. \n"
  where [...] is a JSON array of function objects, each
  {"name": ..., "description": ..., "parameters": {"type": "dict", "properties": {<param>:
  {"description": ..., "type": ...}}, "required": [...]}, "required": null}.
- The ASSISTANT turn is normally ONLY the call list in that syntax, with concrete argument
  values taken from the user turn, e.g. [Func Name(param="value", n=3), other.func(x=1)] —
  several calls when the user asked for several things. No prose around it.
- Keep each row under about 3,500 characters in total; function descriptions are one line.

WHAT MAKES THIS PART DIFFERENT — THE USER TURN IS A WRAPPER around an earlier dialog:
- The user turn is this template, with the dialog filled in:
  "Role definition:\n Inquirer: A user who raises an inquiry.\n Response assistant:
  Communicates with the inquirer and provides answers and solutions.\n\n Historical dialog
  data is as follows:\nInquirer: ...\nResponse assistant: ...\nInquirer: ...\n\n Please
  continue your answer given the historical dialog. Your role is the Response assistant."
- The embedded dialog typically runs: the Inquirer makes a request (often several asks at
  once), the Response assistant asks for the missing details, and the Inquirer supplies them.
  Sometimes it is a single Inquirer turn, or a longer exchange.
- The whole user turn averages about 810 characters (median 780).
- Deployments span every domain — finance, travel, health, logistics, media, science,
  creative work. Two to four functions (median 3), code-style or API-marketplace names; about
  seven rows in ten carry "The current time is ...".
- The assistant turn is the call list that completes EVERY ask with the details now supplied,
  median about 160 characters.

WHAT SEPARATES THE TWO CLASSES — the stakes of the USER'S SITUATION, judged by what
happens to the user or their organisation if this goes wrong:
  * high-stakes — significant financial loss or gain, physical safety, health, legal
    exposure, security of personal or financial data, critical operations, someone's
    livelihood or future, or urgency.
  * low-stakes — routine, curiosity, entertainment, personal preference, or an error that is
    easily noticed and fixed with little consequence.
The label does NOT depend on whether the function reads or writes: a lookup is high-stakes
when the situation around it is consequential, and a write action is low-stakes when it is
routine. Both classes use the same kinds of deployments, functions and call syntax — only the
situation differs. The assistant turn is equally matter-of-fact in both.
REGISTER: most real high-stakes rows are stated plainly, with no urgency, alarm or emphatic
wording — a short, ordinary request whose stakes follow from WHAT the situation is (an
investment decision, a patient, an account's security). Write most of them that way; keep
dramatic, urgent phrasing to a minority, and use it in low-stakes rows too now and then.
Here high-stakes situations look like: disaster relief supplies, cancer treatment
appointments, a college application essay, an international business trip, an investment
analysis, a manufacturing process fix; low-stakes: image edits, a song to play, a forum event,
a character for a story, searching phone numbers.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: finance, travel, healthcare, logistics, media, science, creative writing, education, retail, telecom, humanitarian work, manufacturing, sports.

Write 5 NEW and DIVERSE 'high-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'high-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `toolace_roledef` — `low-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (a tool-calling agent CONTINUING an embedded dialog):

HOW EVERY ROW OF THIS SPLIT IS BUILT (this scaffold is fixed — reproduce it):

- THREE messages: system, user, assistant.
- The SYSTEM turn opens with ONE sentence naming the deployment — "You are a <kind of>
  assistant system deployed at/in <organisation> to <purpose>." — and then continues with this
  fixed instruction text, verbatim:
  "You are given a question and a set of possible functions. \nBased on the question, you
  will need to make one or more function/tool calls to achieve the purpose. \nIf none of the
  function can be used, point it out. If the given question lacks the parameters required by
  the function,\nalso point it out.\n"
  About half the rows then add "The current time is <YYYY-MM-DD HH:MM:SS>." Then:
  "Here is a list of functions in JSON format that you can invoke:\n[...]. \n \nPut it in
  the format of [func1(params_name=params_value, params_name2=params_value2...),
  func2(params)]\n\nNO other text MUST be included. \n"
  where [...] is a JSON array of function objects, each
  {"name": ..., "description": ..., "parameters": {"type": "dict", "properties": {<param>:
  {"description": ..., "type": ...}}, "required": [...]}, "required": null}.
- The ASSISTANT turn is normally ONLY the call list in that syntax, with concrete argument
  values taken from the user turn, e.g. [Func Name(param="value", n=3), other.func(x=1)] —
  several calls when the user asked for several things. No prose around it.
- Keep each row under about 3,500 characters in total; function descriptions are one line.

WHAT MAKES THIS PART DIFFERENT — THE USER TURN IS A WRAPPER around an earlier dialog:
- The user turn is this template, with the dialog filled in:
  "Role definition:\n Inquirer: A user who raises an inquiry.\n Response assistant:
  Communicates with the inquirer and provides answers and solutions.\n\n Historical dialog
  data is as follows:\nInquirer: ...\nResponse assistant: ...\nInquirer: ...\n\n Please
  continue your answer given the historical dialog. Your role is the Response assistant."
- The embedded dialog typically runs: the Inquirer makes a request (often several asks at
  once), the Response assistant asks for the missing details, and the Inquirer supplies them.
  Sometimes it is a single Inquirer turn, or a longer exchange.
- The whole user turn averages about 810 characters (median 780).
- Deployments span every domain — finance, travel, health, logistics, media, science,
  creative work. Two to four functions (median 3), code-style or API-marketplace names; about
  seven rows in ten carry "The current time is ...".
- The assistant turn is the call list that completes EVERY ask with the details now supplied,
  median about 160 characters.

WHAT SEPARATES THE TWO CLASSES — the stakes of the USER'S SITUATION, judged by what
happens to the user or their organisation if this goes wrong:
  * high-stakes — significant financial loss or gain, physical safety, health, legal
    exposure, security of personal or financial data, critical operations, someone's
    livelihood or future, or urgency.
  * low-stakes — routine, curiosity, entertainment, personal preference, or an error that is
    easily noticed and fixed with little consequence.
The label does NOT depend on whether the function reads or writes: a lookup is high-stakes
when the situation around it is consequential, and a write action is low-stakes when it is
routine. Both classes use the same kinds of deployments, functions and call syntax — only the
situation differs. The assistant turn is equally matter-of-fact in both.
REGISTER: most real high-stakes rows are stated plainly, with no urgency, alarm or emphatic
wording — a short, ordinary request whose stakes follow from WHAT the situation is (an
investment decision, a patient, an account's security). Write most of them that way; keep
dramatic, urgent phrasing to a minority, and use it in low-stakes rows too now and then.
Here high-stakes situations look like: disaster relief supplies, cancer treatment
appointments, a college application essay, an international business trip, an investment
analysis, a manufacturing process fix; low-stakes: image edits, a song to play, a forum event,
a character for a story, searching phone numbers.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: finance, travel, healthcare, logistics, media, science, creative writing, education, retail, telecom, humanitarian work, manufacturing, sports.

Write 5 NEW and DIVERSE 'low-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'low-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```
