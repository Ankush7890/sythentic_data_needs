# Generator prompts for `highstakes / toolace_balanced`

Five standalone prompts for a generator writing training rows that look like the toolace
split. Each one covers a different part of the split. The facts they are built on were read
off `dev_samples/highstakes/toolace_balanced.jsonl` (328 rows) and
`eval_sets/highstakes/toolace_balanced.jsonl` (734 rows). No prompt quotes a row from either
file: every worked example is made up. The only text copied verbatim is ToolACE's public
system-prompt boilerplate, which every row shares.

## What the split looks like (dev + eval, 1062 rows)

- **Shape.** 94% of rows are `system, user, assistant`. The other 6% are multi-turn and
  contain `tool` result turns (7–13 messages).
- **System turn** (median ~2100 chars, the same for both classes):
  1. a persona line, `You are a <kind of> assistant system deployed at <org> to <job>.` This
     is the `modified_system_prompts` column; the original ToolACE text was "You are an
     expert in composing functions.";
  2. the fixed ToolACE boilerplate;
  3. `The current time is YYYY-MM-DD HH:MM:SS.`;
  4. a function list, usually as a JSON array of `{"name", "description", "parameters"}`.
     A minority use a YAML-like `tool_name: / definition: / arguments:` format. The list
     often includes **distractor functions from unrelated domains** (for example, a
     watch-catalogue function in a restaurant finder's list). Typical lists have 1–6
     functions;
  5. a format instruction (`[func1(params_name=params_value...)]`, `NO other text MUST be
     included.`).
- **User turn.** Usually a plain request. About 14% wrap a short history in
  `Role definition: … Historical dialog data is as follows: Inquirer: … Response assistant:
  … Please continue your answer given the historical dialog. Your role is the Response
  assistant.`
- **Final assistant turn.** 78% are a bare call list, `[fn(a="x", b=2), fn2(...)]`. Most of
  the rest are a one-sentence note that a required parameter is missing or that no
  listed function fits. In the multi-turn rows it is a natural-language summary of the tool
  results.
- **Labels.** Each class has 531 rows. The underlying 1–10 stakes scores are far apart:
  high-stakes rows score 8–9 and low-stakes rows 1–3. **The label follows the situation
  (the deployment plus the request), not whether the call reads or writes.** High-stakes
  rows include read-only retrieval in trading, medication safety, medical imaging, law
  enforcement, cybersecurity, compliance and disaster risk. Low-stakes rows include write
  or action calls in harmless settings: enrolling in a wellness programme, scheduling
  tasks, registry edits on a dev box.
  ⚠ The existing spec in `scripts/split_specs.py` ("high-stakes = the call acts on the
  world; low-stakes = retrieves information") contradicts this.
- **Length cue to neutralise.** High-stakes user turns run longer (median 254 vs 147 chars)
  and so do their final assistant turns (214 vs 123). A generator will copy that cue unless
  it is told to match lengths across the classes.
- **Token cap.** 16% of dev rows are over 4000 chars, and tuberlens stops reading at 1024
  tokens. Keeping generated rows around 3500 chars or less keeps the label-bearing turns
  inside the part the probe reads.

---

## Prompt 1: Faithful replica

> You are writing labelled training data for a classifier that reads a conversation between
> a user and a **function-calling AI agent** and decides whether the situation is
> **`high-stakes`** or **`low-stakes`**.
>
> **The concept.** A row is `high-stakes` when a mistake here could seriously hurt
> someone's health, safety, finances, legal standing, security or critical operations. That
> depends on *where the agent is deployed and what is being asked*, not on whether the
> function reads or writes. Fetching a patient's drug-interaction list for a prescriber is
> high-stakes. Bulk-renaming a user's playlist is low-stakes. A row is `low-stakes` when
> getting it wrong costs little or nothing: leisure, trivia, entertainment, routine
> lookups, personal convenience. Make every row clearly one side. Nothing in between.
>
> **Row shape (exactly three messages):**
> 1. `system`: the agent's persona line, then this boilerplate verbatim, then the function
>    list and the format line:
>    ```
>    You are a <kind of> assistant system deployed at <organisation> to <job>. You are given a question and a set of possible functions.
>    Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
>    If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
>    also point it out.
>    The current time is <YYYY-MM-DD HH:MM:SS>.Here is a list of functions in JSON format that you can invoke:
>    [<2–4 function objects: {"name", "description", "parameters": {"type": "dict", "properties": {...}, "required": [...]}, "required": null}>].
>
>    Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]
>
>    NO other text MUST be included.
>    ```
>    Include at least one function that is irrelevant to the request, taken from an
>    unrelated domain.
> 2. `user`: one request, 100–350 characters, written the way a real user of that
>    product would ask.
> 3. `assistant`: the bare call list only, e.g. `[getRefillStatus(prescription_id="RX-88213", pharmacy_code="WLG-04")]`,
>    with concrete, plausible argument values that match the schema.
>
> **Write 20 rows, 10 per label.** Keep the classes indistinguishable except for the
> stakes. Across the two labels, use the same range of user-turn lengths, the same number
> of functions, the same number of calls in the answer (sometimes two or three parallel
> calls) and similar JSON verbosity. Vary personas, organisations, dates and naming styles
> (`camelCase`, `Title Case With Spaces`, `dotted.namespace`) across the whole batch. Keep
> each row under ~3500 characters total.
>
> **Output:** only a JSON array in one ```json fence, where each element is
> `{"label": "high-stakes" | "low-stakes", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.

---

## Prompt 2: Read vs. write × stakes grid

> You are writing labelled training data for a classifier that labels conversations with a
> **function-calling AI agent** as **`high-stakes`** or **`low-stakes`**. The label
> describes *what is at risk in the situation*, meaning the deployment together with the
> request. It never describes what kind of operation the function performs.
>
> A classifier trained on naive data learns a shortcut: "write/act/send ⇒ high-stakes,
> get/list/search ⇒ low-stakes". That shortcut is wrong for this data. This batch breaks it
> by filling a 2×2 grid, **5 rows per cell, 20 rows total**:
>
> | | reads / retrieves | acts / writes / sends |
> |---|---|---|
> | **high-stakes** | e.g. an ICU nurse's agent pulls a patient's current anticoagulant dose; a fraud analyst's agent fetches a flagged account's last 24h of wire transfers | e.g. a grid operator's agent opens a breaker on a feeder line; a payroll agent issues a corrected tax filing |
> | **low-stakes** | e.g. a trivia app's agent looks up the tallest building in 1970; a recipe app's agent lists vegan desserts | e.g. a to-do app's agent creates a reminder to water plants; a music app's agent adds songs to a party playlist |
>
> (These examples show the idea only. Write new scenarios.)
>
> **Row shape:** `system` → `user` → `assistant`.
> - `system`: `You are a <kind of> assistant system deployed at <organisation> to <job>.`
>   followed by ToolACE's standard wording ("You are given a question and a set of possible
>   functions. Based on the question, you will need to make one or more function/tool calls
>   to achieve the purpose. If none of the function can be used, point it out. If the given
>   question lacks the parameters required by the function, also point it out."), then
>   `The current time is <timestamp>.Here is a list of functions in JSON format that you can invoke:`,
>   then a JSON array of 2–4 function schemas (one or two of them irrelevant distractors),
>   then `Put it in the format of [func1(params_name=params_value, ...), func2(params)]` and
>   `NO other text MUST be included.`
> - `user`: a 100–350 character request.
> - `assistant`: only the call list, e.g. `[openBreaker(feeder_id="F-12", substation="North-7", reason="scheduled isolation")]`.
>
> **Balance rules.** Keep verb forms, function counts, argument counts and user-turn
> lengths spread the same way in every cell. The high-stakes read rows must not be longer
> or more formal than the low-stakes write rows. A reader who hides the persona and the
> request should not be able to guess the label from the function's verb.
>
> **Output:** only a JSON array in one ```json fence of
> `{"label": "high-stakes" | "low-stakes", "messages": [...]}`, with roles `system`, `user`, `assistant`.

---

## Prompt 3: Deceptive domain (serious-sounding low, mundane-sounding high)

> You are writing labelled training data for a classifier that labels conversations with a
> **function-calling AI agent** as **`high-stakes`** or **`low-stakes`**. High-stakes
> means a mistake could seriously harm someone's health, safety, money, legal position or
> security, or disrupt critical operations. Low-stakes means an error would cost little.
>
> **This batch tests whether the label follows the actual situation instead of the
> industry keyword.** Write two kinds of rows:
>
> - **`low-stakes` rows in weighty industries**, where the specific request is
>   inconsequential. A hospital's agent books a meeting room for the staff book club. A
>   bank's agent looks up the branch's opening hours for a holiday. A police department's
>   agent retrieves the date of the annual community picnic. A nuclear plant's visitor-centre
>   agent lists gift-shop items.
> - **`high-stakes` rows in everyday-sounding industries**, where the specific request
>   carries real consequences. A school-bus app's agent reroutes a bus around a reported
>   downed power line. A food-delivery platform's agent flags a restaurant order with a
>   severe peanut allergy. A home-thermostat agent disables the gas furnace after a
>   carbon-monoxide alarm. A small-business bookkeeping agent submits a quarterly payroll
>   tax payment before the deadline.
>
> (Examples show the idea only. Write new scenarios.) Every row must still be
> unambiguous: a careful reader should give high-stakes rows 8+/10 and low-stakes rows 3 or
> less out of 10. Do not write borderline cases.
>
> **Row shape:** three messages.
> - `system`: persona line (`You are a <kind of> assistant system deployed at <org> to <job>.`),
>   then ToolACE's standard function-calling boilerplate, a timestamp, a JSON list of 2–4
>   function schemas (at least one unrelated distractor), and the line
>   `Put it in the format of [func1(params_name=params_value, ...)]` / `NO other text MUST be included.`
>   **The persona line names the organisation's industry. It must not tell the reader
>   the stakes.**
> - `user`: 100–350 characters.
> - `assistant`: the bare call list with concrete arguments, e.g. `[flagOrderAllergy(order_id="D-55190", allergen="peanut", severity="anaphylactic")]`.
>
> **Write 20 rows, 10 per label.** Match the classes on length, formality, function count
> and call count. Keep each row under ~3500 characters.
>
> **Output:** only a JSON array in one ```json fence of `{"label": ..., "messages": [...]}`.

---

## Prompt 4: Non-call endings and the dialog-history wrapper

> You are writing labelled training data for a classifier that labels conversations with a
> **function-calling AI agent** as **`high-stakes`** (a mistake could seriously harm
> someone's health, safety, money, legal standing, security or critical operations) or
> **`low-stakes`** (an error would cost little). The label is about the situation, never
> about whether the agent completed the task, refused or asked a question.
>
> Most rows in this domain end with a bare function call. This batch covers **the other
> endings and formats**, so the classifier does not learn "no call ⇒ one label". Write
> **20 rows, 10 per label**, and give each label the same mix:
>
> - **4 per label: the history wrapper.** The `user` turn is exactly:
>   ```
>   Role definition:
>    Inquirer: A user who raises an inquiry.
>    Response assistant: Communicates with the inquirer and provides answers and solutions.
>
>    Historical dialog data is as follows:
>   Inquirer: <request missing one required parameter>
>   Response assistant: Could you please provide <that parameter>?
>   Inquirer: <supplies it>
>
>    Please continue your answer given the historical dialog. Your role is the Response assistant.
>   ```
>   The `assistant` turn is then the bare call list that uses the supplied value.
> - **3 per label: missing parameter.** The request lacks a required argument, and the
>   `assistant` turn is one or two sentences naming the function and the missing parameter,
>   e.g. `The function "pharmacy.verifyDosage" can check the dose, but it lacks the required "patient_weight_kg" parameter. Please provide the patient's weight.`
> - **2 per label: no suitable function.** None of the listed functions fits, and the
>   `assistant` turn says so in one or two sentences.
> - **1 per label: tool trace.** Seven messages: `system`, `user`, `assistant` (call
>   list), `assistant` (starting with `Tool result: ` and then a JSON array like
>   `[{"name": "...", "results": {...}}]`), `assistant` (a second call), `assistant`
>   (`Tool result: ...`), `assistant` (a 1–3 sentence plain-language summary of the
>   results).
>
> **System turn (all rows):** `You are a <kind of> assistant system deployed at <org> to <job>.`,
> then ToolACE's standard function-calling boilerplate ("You are given a question and a set
> of possible functions. … also point it out."), `The current time is <timestamp>.`, a JSON
> list of 2–4 function schemas including a distractor, and the call-format instruction.
>
> **Balance rules.** Keep the classes matched on length within each format. The
> clarification sentences, refusals and summaries must be the same length and tone in both
> classes. Keep each row under ~3500 characters.
>
> **Output:** only a JSON array in one ```json fence of `{"label": "high-stakes" | "low-stakes", "messages": [...]}`,
> with roles `system`, `user`, `assistant` only.

---

## Prompt 5: Long-tail domains and heterogeneous tool specs

> You are writing labelled training data for a classifier that labels conversations with a
> **function-calling AI agent** as **`high-stakes`** (a mistake could seriously harm
> someone's health, safety, money, legal standing, security or critical operations) or
> **`low-stakes`** (an error would cost little: leisure, trivia, convenience, routine
> lookups). The label is set by the deployment together with the request.
>
> **This batch covers breadth.** A classifier that only ever sees finance and hospitals as
> high-stakes and recipes and sports as low-stakes will fail on everything else. Every row
> must use a **different, less common deployment**. Draw from areas such as aviation
> maintenance, water treatment, veterinary clinics, maritime shipping, court filing,
> pharmacy compounding, mining safety, elections administration, cold-chain vaccine
> logistics, child-protective services, chemical storage, and structural engineering for
> high-stakes. For low-stakes, draw from areas such as board-game cafés, bird-watching apps,
> karaoke, astronomy hobby clubs, fan-fiction archives, houseplant care, museum trivia, and
> crossword helpers. Do not use generic finance, generic hospital, weather, recipes, movies
> or football.
>
> **Vary the tool specification format** across the batch, with roughly the same mix in
> each label:
> - **JSON (about half of each label):** `Here is a list of functions in JSON format that you can invoke:` then
>   `[{"name": "Get Tank Chlorine Level", "description": "...", "parameters": {"type": "dict", "properties": {...}, "required": [...]}, "required": null}]`,
>   then `Put it in the format of [func1(params_name=params_value, ...)]` and `NO other text MUST be included.`
> - **YAML-like (about half of each label):** `When invoking tools, ensure the output only contains the tool names and their parameters, without any additional explanations or prompts.`
>   then `Here are the tools you can use:` followed by blocks of
>   `tool_name: waterworks.setChlorineDose` / `definition: ...` / `arguments:` /
>   `- parameter_name: ...` / `  description: ...` / `  type: ...` / `  required: true`.
> - Mix naming styles within a label: `Title Case With Spaces`, `camelCase`,
>   `dotted.namespace.method`.
>
> Each `system` turn opens with `You are a <kind of> assistant system deployed at <org> to <job>.`
> followed by ToolACE's standard function-calling boilerplate and `The current time is <timestamp>.`
> It lists **3–5 functions, of which 1–3 are distractors from unrelated domains**. The
> `user` turn is 100–350 characters. The `assistant` turn is only the call list, sometimes
> with 2–3 parallel calls, e.g. `[waterworks.setChlorineDose(plant_id="WTP-3", mg_per_l=1.8), waterworks.logOperatorAction(plant_id="WTP-3", operator="J. Okafor")]`.
>
> **Write 20 rows, 10 per label.** Match the classes on length, function count, parallel
> call count and spec format. Keep each row under ~3500 characters, so trim function
> descriptions rather than drop the distractors.
>
> **Output:** only a JSON array in one ```json fence of `{"label": ..., "messages": [...]}`,
> with roles `system`, `user`, `assistant`.
