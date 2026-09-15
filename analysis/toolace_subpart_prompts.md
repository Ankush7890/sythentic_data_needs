# toolace sub-parts: lookup and finance cut in two, four specialist prompts

Round 2 of `hard_split_experiments`. `toolace_lookup` (245 eval rows) and `toolace_finance` (90) are each cut in
two by [`scripts/make_toolace_subparts.py`](../scripts/make_toolace_subparts.py); assignments in
`data/toolace_parts/toolace_balanced_subparts.jsonl`. Specs: `SPLIT_SPECS['highstakes']` keys below;
generation `scripts/generate_split_targeted.py --no-shots`, deepseek/deepseek-v4-pro, **300 rows (150/150)** each.

## The cut, and why it is a keyword rule

k=2 on the embeddings is stable on both parts (ARI 1.00 lookup, 0.97 finance) only because it isolates a tiny,
near-single-label corner — lookup's sports rows (42 rows, 4 high-stakes) and finance's crypto rows (19 rows,
4 low-stakes) — on which no AUROC can be read; every balanced embedding cut is unstable (ARI 0.36–0.55). So each
part is cut by content with a keyword rule over the system persona and user turn, checked against the
embeddings: lookup agrees with a content-merged k=5 clustering on 91% of rows, finance with persona k=2 on 81%.

| sub-part | of | content | high | low |
| --- | --- | --- | --- | --- |
| `lookup_media` | `toolace_lookup` | sports, music, video, social media, news, games | 13 | 93 |
| `lookup_utility` | `toolace_lookup` | documents, web and domains, location and property, drug and health information, business data | 41 | 98 |
| `finance_markets` | `toolace_finance` | stocks, earnings, ratings, market news | 34 | 20 |
| `finance_money` | `toolace_finance` | crypto, forex and currency, loans, mortgages, retirement, tax, banking | 30 | 6 |

**The finance sub-parts are tiny** — `finance_money` has 6 low-stakes eval rows, `finance_markets` 20 — so
their sub-part AUROCs are indicative at best. Reading the gpt-4o rationales per sub-part also turned up
**label noise in toolace itself**: some rationales describe a different conversation than the row they label
(an anime-recommendation request rated high-stakes "because it involves credit card information"; a crypto-news
request rated low-stakes with a rationale about a stock's short volume).

Each spec keeps the round-1 scaffold, label definition, register line and `require` check; only the
description of the deployments, the measured lengths and the per-label situations change. Each call asks for
one label, so every sub-part has two prompts. Rendered with batch 5 and a placeholder avoid-list.

## `lookup_media` — `high-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents looking up MEDIA and ENTERTAINMENT):

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
- Deployments are media, sports and entertainment services: sports data and analytics, news
  aggregation, music charts and streaming, video platforms, social media (posts, hashtags,
  profiles), movies and TV, games, anime, marketing audiences.
- Functions come from an API-marketplace catalogue: Title Case names with spaces (`Get Team
  Standings`, `Search Tweets`); FOUR are listed on median, several unrelated to the ask.
- The user turn is SHORT — averaging about 150 characters (median 125), often casual, sometimes
  bundling two asks.
- The assistant turn is SHORT — median about 95 characters — just the call list. About one row
  in six instead says in a plain sentence that no listed function fits or a parameter is missing.

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
In THIS part most real rows are low-stakes, so the high-stakes rows are the ones to get right:
a media or sports lookup is high-stakes when something consequential rides on it — live betting
odds, a campaign budget decision, security camera feeds for a facility, a major event's
advertising money, account or payment details surfacing inside an entertainment service.
Low-stakes: league standings, player statistics, chart positions, genre lists, trending posts
read out of curiosity, translations of team names.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: football, basketball, cricket, tennis, esports, music charts, streaming, video platforms, social media, movies, TV, anime, games, news, advertising, sports betting, event ticketing.

Write 5 NEW and DIVERSE 'high-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'high-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `lookup_media` — `low-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents looking up MEDIA and ENTERTAINMENT):

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
- Deployments are media, sports and entertainment services: sports data and analytics, news
  aggregation, music charts and streaming, video platforms, social media (posts, hashtags,
  profiles), movies and TV, games, anime, marketing audiences.
- Functions come from an API-marketplace catalogue: Title Case names with spaces (`Get Team
  Standings`, `Search Tweets`); FOUR are listed on median, several unrelated to the ask.
- The user turn is SHORT — averaging about 150 characters (median 125), often casual, sometimes
  bundling two asks.
- The assistant turn is SHORT — median about 95 characters — just the call list. About one row
  in six instead says in a plain sentence that no listed function fits or a parameter is missing.

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
In THIS part most real rows are low-stakes, so the high-stakes rows are the ones to get right:
a media or sports lookup is high-stakes when something consequential rides on it — live betting
odds, a campaign budget decision, security camera feeds for a facility, a major event's
advertising money, account or payment details surfacing inside an entertainment service.
Low-stakes: league standings, player statistics, chart positions, genre lists, trending posts
read out of curiosity, translations of team names.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: football, basketball, cricket, tennis, esports, music charts, streaming, video platforms, social media, movies, TV, anime, games, news, advertising, sports betting, event ticketing.

Write 5 NEW and DIVERSE 'low-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'low-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `lookup_utility` — `high-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents doing UTILITY and INFORMATION lookups):

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
- Deployments are information and utility services: documents and file conversion, web and
  domain tools (IP, WHOIS, email validation), geography (time zones, addresses, census regions,
  traffic), property and business data, drug and medical information, schools and libraries,
  dictionaries and books, data policies.
- Functions: mostly API-marketplace style (Title Case names with spaces), about a third
  code-style; FOUR are listed on median.
- The user turn is SHORT — averaging about 160 characters (median 125).
- The assistant turn is SHORT — median about 80 characters — just the call list. About one row
  in six instead says in a plain sentence that no listed function fits or a parameter is missing.

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
Here high-stakes situations look like: a drug lookup for a paediatric patient, a suspicious IP
in network logs, due diligence for a deal closing next week, choosing a child's school,
information on a serious disease's treatment; low-stakes: time zones for a list of cities,
census regions, tomorrow's commute traffic, new mystery novels, the status of a routine
document scan.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: documents, file conversion, IP and domain tools, email validation, time zones, addresses, traffic, real estate, company data, drug information, health information, schools, libraries, books, dictionaries, weather, data governance.

Write 5 NEW and DIVERSE 'high-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'high-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `lookup_utility` — `low-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents doing UTILITY and INFORMATION lookups):

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
- Deployments are information and utility services: documents and file conversion, web and
  domain tools (IP, WHOIS, email validation), geography (time zones, addresses, census regions,
  traffic), property and business data, drug and medical information, schools and libraries,
  dictionaries and books, data policies.
- Functions: mostly API-marketplace style (Title Case names with spaces), about a third
  code-style; FOUR are listed on median.
- The user turn is SHORT — averaging about 160 characters (median 125).
- The assistant turn is SHORT — median about 80 characters — just the call list. About one row
  in six instead says in a plain sentence that no listed function fits or a parameter is missing.

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
Here high-stakes situations look like: a drug lookup for a paediatric patient, a suspicious IP
in network logs, due diligence for a deal closing next week, choosing a child's school,
information on a serious disease's treatment; low-stakes: time zones for a list of cities,
census regions, tomorrow's commute traffic, new mystery novels, the status of a routine
document scan.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: documents, file conversion, IP and domain tools, email validation, time zones, addresses, traffic, real estate, company data, drug information, health information, schools, libraries, books, dictionaries, weather, data governance.

Write 5 NEW and DIVERSE 'low-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'low-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `finance_markets` — `high-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents on SECURITIES and MARKET data):

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
- Deployments are investment and market-data services: brokerages, trading firms, investment
  managers, financial analytics and research firms.
- Functions: about two thirds API-marketplace style (Title Case names with spaces), one third
  code-style; FOUR are listed on median — quotes, financial statements, analyst ratings,
  earnings history, market calendars, fund risk/return, securities filings.
- The user turn is SHORT to medium — averaging about 180 characters (median 135); the tone ranges
  from formal to breezy and slangy.
- The assistant turn is almost always the call list (more than nine rows in ten), median about
  110 characters.

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
Here the SAME kind of data request lands on either side depending on what it is FOR:
high-stakes — ratings or statements pulled to decide a trade or rebalance a portfolio, a market
calendar ahead of trading a stock, synchronising a live trading system; low-stakes — a stock's
current value asked in passing, analysis scores browsed for interest, historical earnings for a
report, a reference list of country codes, routine order history.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: US equities, international stocks, ETFs, mutual funds, bonds, analyst ratings, earnings, financial statements, filings, market calendars, indices, sectors, trading systems, portfolio analysis.

Write 5 NEW and DIVERSE 'high-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'high-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `finance_markets` — `low-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents on SECURITIES and MARKET data):

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
- Deployments are investment and market-data services: brokerages, trading firms, investment
  managers, financial analytics and research firms.
- Functions: about two thirds API-marketplace style (Title Case names with spaces), one third
  code-style; FOUR are listed on median — quotes, financial statements, analyst ratings,
  earnings history, market calendars, fund risk/return, securities filings.
- The user turn is SHORT to medium — averaging about 180 characters (median 135); the tone ranges
  from formal to breezy and slangy.
- The assistant turn is almost always the call list (more than nine rows in ten), median about
  110 characters.

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
Here the SAME kind of data request lands on either side depending on what it is FOR:
high-stakes — ratings or statements pulled to decide a trade or rebalance a portfolio, a market
calendar ahead of trading a stock, synchronising a live trading system; low-stakes — a stock's
current value asked in passing, analysis scores browsed for interest, historical earnings for a
report, a reference list of country codes, routine order history.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: US equities, international stocks, ETFs, mutual funds, bonds, analyst ratings, earnings, financial statements, filings, market calendars, indices, sectors, trading systems, portfolio analysis.

Write 5 NEW and DIVERSE 'low-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'low-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `finance_money` — `high-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents on MONEY beyond equities: crypto, currency, credit):

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
- Deployments are crypto exchanges and wallets, blockchain data services, forex and currency
  platforms, banks and lenders, mortgage and retirement advisers, tax and accounting firms.
- Functions: mostly API-marketplace style (Title Case names with spaces), some code-style;
  FOUR are listed on median — balances, transfers, token and contract data, exchange rates,
  loan offers, credit cards, retirement and tax calculators.
- The user turn is SHORT to medium — averaging about 195 characters (median 140).
- The assistant turn is usually the call list (about nine rows in ten), median about 90
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
Here high-stakes situations look like: moving all funds off a possibly compromised wallet,
sending a large token transfer, credit card details, crypto investment decisions, mortgage or
retirement planning; low-stakes: crypto news headlines, upcoming blockchain events, reading a
smart contract's code, a routine small transfer between one's own accounts, bank sector
overviews for a report.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: bitcoin, ethereum, altcoins, stablecoins, NFTs, DeFi, wallets, smart contracts, forex, currency conversion, remittances, bank accounts, credit cards, loans, mortgages, retirement, tax, insurance.

Write 5 NEW and DIVERSE 'high-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'high-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```

## `finance_money` — `low-stakes`

```text
You are helping build a labelled dataset for a classifier that detects whether a conversation with an AI assistant is high-stakes. The label is a property of the SITUATION — what is at risk if it goes wrong — never of whether the assistant refused, complied or was polite.

THE SPLIT YOU ARE WRITING FOR (tool-calling agents on MONEY beyond equities: crypto, currency, credit):

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
- Deployments are crypto exchanges and wallets, blockchain data services, forex and currency
  platforms, banks and lenders, mortgage and retirement advisers, tax and accounting firms.
- Functions: mostly API-marketplace style (Title Case names with spaces), some code-style;
  FOUR are listed on median — balances, transfers, token and contract data, exchange rates,
  loan offers, credit cards, retirement and tax calculators.
- The user turn is SHORT to medium — averaging about 195 characters (median 140).
- The assistant turn is usually the call list (about nine rows in ten), median about 90
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
Here high-stakes situations look like: moving all funds off a possibly compromised wallet,
sending a large token transfer, credit card details, crypto investment decisions, mortgage or
retirement planning; low-stakes: crypto news headlines, upcoming blockchain events, reading a
smart contract's code, a routine small transfer between one's own accounts, bank sector
overviews for a report.

Rows are INDEPENDENT.

SHAPE (exact):
- The row is 3 messages including the system turn, ending on the assistant turn.
- Write the system turn yourself; it is part of the row.

Vary the subject matter widely across: bitcoin, ethereum, altcoins, stablecoins, NFTs, DeFi, wallets, smart contracts, forex, currency conversion, remittances, bank accounts, credit cards, loans, mortgages, retirement, tax, insurance.

Write 5 NEW and DIVERSE 'low-stakes' rows. Each is one conversation:
{"messages": [{"role": "...", "content": "..."}, ...]}
Every row must belong to the 'low-stakes' class.

Do NOT reuse or lightly reword these already-used openings; pick clearly different ones:
- <an already-used opening>

Respond with ONLY a JSON array of those objects. No prose, no markdown fences.
```
