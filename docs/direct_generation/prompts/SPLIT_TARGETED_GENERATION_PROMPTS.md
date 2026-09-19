# Split-targeted generation prompts — instructions and high-stakes

Where to read the exact text that was sent to a generator when a training set was aimed at
ONE eval split. Three families of prompt did that on this branch — A and B are not variants
of each other, C is the shape-free ablation of B — and each has its own verbatim dump:

| Family | Script | Prompt dump | Data files |
| --- | --- | --- | --- |
| **A. `--kind` pin** | `scripts/generate_{concept}_dataset.py --eval-description --kind N` | [`PER_SPLIT_KIND_PROMPTS.txt`](PER_SPLIT_KIND_PROMPTS.txt) | `data/{concept}_{tag}_{split}_600.jsonl` |
| **B. measured spec** | `scripts/generate_split_targeted.py --concept … --split …` | [`SPLIT_TARGETED_PROMPTS.txt`](SPLIT_TARGETED_PROMPTS.txt) | `data/{concept}_{tag}_{tgtshot,tgtnone}_{split}_600.jsonl` |
| **C. shape-free** | `scripts/generate_split_targeted.py --minimal …` | [`SPLIT_TARGETED_PROMPTS_MINIMAL.txt`](SPLIT_TARGETED_PROMPTS_MINIMAL.txt) | `data/{concept}_{tag}_tgtmin_{split}_600.jsonl` |

All three dumps are **rendered from the generator code itself**, not transcribed, so they
cannot drift from what was sent. Re-render any of them at any time:

```bash
.venv_claude/bin/python scripts/dump_persplit_prompts.py --out PER_SPLIT_KIND_PROMPTS.txt
.venv_claude/bin/python scripts/generate_split_targeted.py --dump-prompts SPLIT_TARGETED_PROMPTS.txt
.venv_claude/bin/python scripts/generate_split_targeted.py --minimal --dump-prompts SPLIT_TARGETED_PROMPTS_MINIMAL.txt
```

In every dump the batch size is shown as 5 and the avoid-block as a single placeholder; a
real call carries up to 20 sampled openings. Nothing else differs from what was sent. Every
prompt is one **user** message — no family uses a system prompt.

---

## Family A — the `--kind` pin

Driver: `run_persplit_gen.sh`. Each call is

```bash
.venv_claude/bin/python scripts/generate_<concept>_dataset.py \
    --model <generator> --n-per-label 300 --batch-size 5 --concurrency 8 \
    --call-budget-factor 8 --max-tokens 8192 --eval-description --kind <N> \
    --out data/<concept>_<tag>_<split>_600.jsonl
```

`--eval-description` shows the concept's `EVAL_DATA_DESCRIPTION` — the numbered kinds, one
per eval split, carried **verbatim** from the `eval.data_description` block of the red-team
configs — in full. `--kind N` pins **every** call to kind N instead of round-robining over
them, so all 600 rows sit inside the one kind that corresponds to a single eval split. That
pin is the `For THIS batch write every example in ONE kind only:` block near the end of each
prompt; everything above it is identical to the mixed `evaldesc` arm.

`--call-budget-factor 8` rather than 6: 300 rows per label inside one kind is far more dedup
pressure than a mixed set, so more calls come back as near-repeats.

### Kind → split mapping (20 prompts = 10 splits × 2 labels)

**instructions** — each kind names its split in parentheses, so the mapping is read off the
description text itself:

| `--kind` | Split | Kind |
| --- | --- | --- |
| 1 | `anthropic_harmless_refusal` | unjustified refusal |
| 2 | `bbq_substitution` | biased substitution |
| 3 | `hc_context_drift` | context drift, retrieved-document flavour |
| 4 | `hc_contradiction` | contradicting the source |
| 5 | `mm_substitution` | exaggeration / embellishment |
| 6 | `oig_context_drift` | context drift, dialogue flavour |

`oig_omission` — the seventh eval split — has **no kind**: the red-team branch removed it
before that text was written, and keeping the text verbatim is the point. It has no prompt
and stays the untouched control.

**highstakes** — the kinds do not name splits; the correspondence is positional and was
checked against the split contents:

| `--kind` | Split | Kind |
| --- | --- | --- |
| 1 | `anthropic_hh_balanced` | open-ended chatbot conversation |
| 2 | `mt_balanced` | one long clinical document |
| 3 | `mts_balanced` | many-turn doctor-patient dialogue |
| 4 | `toolace_balanced` | tool-calling agent, function list in a system prompt |

### What this family does not say

The description names a split's **subject matter** and nothing about its **shape** — no turn
count, no pairing, no measured lengths. Measured against the split files the misses were
large: every instruction split is a PAIRED corpus (the same user turn answered once each
way) and every one of these arms was generated unpaired. That gap is what family B closes.

---

## Family B — the measured per-split spec

Driver: `run_tgt_gen.sh`, specs in `scripts/split_specs.py`. Each call is

```bash
.venv_claude/bin/python scripts/generate_split_targeted.py \
    --concept <concept> --split <split> --model <generator> --n 600 [--no-shots] \
    --out data/<concept>_<tag>_<tgtshot|tgtnone>_<split>_600.jsonl
```

Each split gets, instead of one line of a shared description:

1. a description written from **measurements** of the split file (row counts, pairing, role
   sequence, mean/median character lengths) rather than from recollection;
2. a few-shot anchor drawn from `dev_samples/`, **never** `eval_sets/` — generated rows
   become training data and the probe is scored on eval, so anchoring on eval would copy test
   content into the training set;
3. a generation **mode** matching the split: PAIRED (one opening, two final assistant turns,
   two rows) for all six instruction splits, UNPAIRED for all four high-stakes splits;
4. an **enforced shape** — turn count, whether a system turn is part of the row, whether the
   row ends on the assistant or the user. A generated row that misses it is DROPPED, not
   repaired, and the log reports how many.

Register is **described, never quoted**: on `human_harm_last` a description that quoted real
refusal strings had 17/200 eval negatives come back reproduced verbatim, and that cell fell
0.9832 → 0.9403 once the quotes went — a third of the apparent margin was memorisation.

**Two variants per split**, differing by the anchor block **alone**:

- `tgtshot` — with the dev-derived few-shot anchor;
- `tgtnone` (`--no-shots`) — without it, so the measured description is the only thing
  describing the split.

The pair measures how much of a targeted set's quality is the DESCRIPTION and how much is
seeing two real rows. Prompt count in the dump: 6 instruction splits × 1 (paired mode emits
both labels in one call) + 4 high-stakes splits × 2 labels = 14, × 2 variants = **28**.

---

---

## Family C — the shape-free ablation of family B

Driver: `run_tgt_minimal_gen.sh`, descriptions in `scripts/split_specs.py` under
`MINIMAL_DESCS`, full write-up in
[`analysis/split_targeted_prompts_minimal.md`](analysis/split_targeted_prompts_minimal.md).
**Not yet run** — the prompts are written and rendered, no `tgtmin` set has been generated.

Family B's descriptions were written from **measurements of the split file** — which is both
their selling point and their confound: a turn count, a role sequence, a mean character
length, a pinned system string, a pairing statement and a topic list are all read off the eval
split. Family C keeps the situation and the label boundary and removes every one of them, so a
`tgtmin` set differs from a `tgtnone` set by exactly that.

```bash
.venv_claude/bin/python scripts/generate_split_targeted.py --minimal \
    --concept <concept> --split <split> --model <generator> --n 600 \
    --out data/<concept>_<tag>_tgtmin_<split>_600.jsonl
```

What survives per split is **one sentence of situation plus one sentence per label: what a
reply DOES, never what it looks like.** Pairing goes with the rest — a pairing instruction is
itself shape — so all ten splits generate one label per call and the two labels never share an
opening, and shape is no longer enforced on the way in either.

| Arm | Description | Anchor | Topics | Shape | Mode |
| --- | --- | --- | --- | --- | --- |
| `tgtshot` | measured | dev few-shot | yes | stated + enforced | split's own |
| `tgtnone` | measured | — | yes | stated + enforced | split's own |
| `tgtmin` | situation + labels only | — | — | — | unpaired |

The motivation is a measured result, not a hunch: the identical ablation on `hu_harm`
(`generator_experiment_1`, `5c59c568`) **inverted** which split a targeted set was good at, so
the shape cues rather than the situation were carrying those results.
`scripts/check_minimal_descs.py` guards the arm — 0 banned shape terms in the ten minimal
descriptions, against 205 in the full ones.

---

## Which prompt wrote which file

| File pattern | Family | Prompt in the dump |
| --- | --- | --- |
| `data/instructions_{tag}_{split}_600.jsonl` | A | `instructions / {split} [--kind N]`, both labels |
| `data/highstakes_{tag}_{split}_600.jsonl` | A | `highstakes / {split} [--kind N]`, both labels |
| `data/instructions_{tag}_tgtshot_{split}_600.jsonl` | B | `instructions / {split} [WITH few-shot anchor] [paired]` |
| `data/instructions_{tag}_tgtnone_{split}_600.jsonl` | B | `instructions / {split} [NO few-shot anchor] [paired]` |
| `data/highstakes_{tag}_tgtshot_{split}_600.jsonl` | B | `highstakes / {split} [WITH few-shot anchor]`, both labels |
| `data/highstakes_{tag}_tgtnone_{split}_600.jsonl` | B | `highstakes / {split} [NO few-shot anchor]`, both labels |
| `data/{concept}_{tag}_tgtmin_{split}_600.jsonl` | C | `{concept} / {split} [MINIMAL: no shape, no anchor, no topics]`, both labels |

`{tag}` is the generator: `deepseekv4pro` (`deepseek/deepseek-v4-pro`) throughout, plus a
partial `llama70b` sweep of family A. Sets not listed here are the *un*-targeted baselines —
`{concept}_{tag}_600.jsonl` (no eval description at all) and `{concept}_{tag}_evaldesc_600.jsonl`
(the full description, kinds round-robined, no pin) — and their prompts are the same
`build_prompt` with `focus_kind` empty.
