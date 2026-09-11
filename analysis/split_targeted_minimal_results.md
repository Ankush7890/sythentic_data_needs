# The shape-free arm: results

Ten shape-free (`tgtmin`) sets, 6000 rows, 90 fits. The arm and its prompts are specified in
[`split_targeted_prompts_minimal.md`](split_targeted_prompts_minimal.md); the numbers below
come from `scripts/compare_tgtmin_arms.py` over `scripts/instructions_gen90.csv` and
`scripts/highstakes_gen90_dev500.csv`, at n=540 × 8 draws, deepseek-v4-pro throughout.

## The question

A split-targeted description was written from MEASUREMENTS of the eval split — turn counts,
role sequences, mean character lengths, the split's own system prompt, its pairing, its topic
list. All of that is test-set information reaching the training set through the prompt. Does
a targeted set win because of the SITUATION it describes, or because it was told the answer's
SHAPE?

## The answer: the shape was nearly all redundant

`tgtmin` against the `--kind` arm, on each split's own eval split:

| Split | `tgtmin` | `kind-pin` | Δ |
| --- | --- | --- | --- |
| `oig_context_drift` | 0.9252 ±0.0125 | 0.8150 ±0.0532 | **+0.110** |
| `hc_context_drift` | 0.9898 ±0.0024 | 0.9422 ±0.0268 | **+0.048** |
| `anthropic_hh_balanced` | 0.9634 ±0.0053 | 0.9207 ±0.0088 | **+0.043** |
| `mm_substitution` | 0.9456 ±0.0279 | 0.9150 ±0.0275 | **+0.031** |
| `mts_balanced` | 0.9761 ±0.0082 | 0.9661 ±0.0077 | +0.010 |
| `anthropic_harmless_refusal` | 0.9871 ±0.0048 | 0.9796 ±0.0080 | +0.008 |
| `mt_balanced` | 0.9861 ±0.0021 | 0.9835 ±0.0035 | +0.003 |
| `hc_contradiction` | 0.9032 ±0.0245 | 0.9098 ±0.0112 | −0.007 |
| `bbq_substitution` | 0.8226 ±0.0592 | 0.8340 ±0.0304 | −0.011 |
| `toolace_balanced` | 0.7065 ±0.0232 | 0.8111 ±0.0117 | **−0.105** |
| **mean of ten** | **0.9206** | **0.9077** | **+0.013** |

Four wins, four washes, two losses at |Δ| > 0.01. **Removing every measured confounder made
the targeted sets slightly BETTER on average**, and on the two biggest wins it also cut
draw-to-draw variance by a factor of four (`oig_context_drift` ±0.0125 against ±0.0532).

Against the family-B arms, which exist only for the first two instruction splits:

| Split | `tgtmin` | `tgtnone` (measured, no anchor) | `tgtshot` (measured + dev anchor) |
| --- | --- | --- | --- |
| `anthropic_harmless_refusal` | 0.9871 ±0.0048 | 0.9856 ±0.0069 | 0.9897 ±0.0048 |
| `bbq_substitution` | 0.8226 ±0.0592 | **0.9367 ±0.0133** | 0.9079 ±0.0230 |

On `anthropic_harmless_refusal` the three arms are inside ±0.005 of each other — the measured
description, the dev few-shot anchor and the shape bought **nothing** over one sentence of
situation and one sentence per label.

`bbq_substitution` is the opposite, and the control is what shows it. `tgtnone` — the measured
description with the anchor dropped — is the best arm on that split at 0.9367, above `tgtshot`
(so the anchor HURT by 0.029) and 0.114 above `tgtmin`. The whole deficit is the SHAPE, none
of it the anchor. So `bbq_substitution` is a second genuinely shape-dependent split alongside
`toolace_balanced`, not the marginal −0.011 the `--kind` comparison alone suggested: measured
against the arm it actually ablates, the loss is ten times larger.

## Why: shape that FOLLOWS from the situation is not information

The generated sets were profiled against the real splits. The six instruction splits
reproduced their shape EXACTLY without being told it:

| split | generated turns | gen system | gen ends | real turns | real system | real ends |
| --- | --- | --- | --- | --- | --- | --- |
| `anthropic_harmless_refusal` | 2:600 | 0/600 | assistant | 2:200 | 0/200 | assistant |
| `bbq_substitution` | 2:600 | 0/600 | assistant | 2:200 | 0/200 | assistant |
| `hc_context_drift` | 4:600 | 0/600 | assistant | 4:194 | 0/194 | assistant |
| `hc_contradiction` | 2:596, 3:4 | 4/600 | assistant | 2:200 | 0/200 | assistant |
| `mm_substitution` | 2:595, 3:5 | 5/600 | assistant | 2:200 | 0/200 | assistant |
| `oig_context_drift` | 4:600 | 0/600 | assistant | 4:194 | 0/194 | assistant |
| `anthropic_hh_balanced` | 2:585, 4:15 | 0/600 | assistant | 3:829, 5:854, 7:766 | 2984/2984 | assistant |
| `mt_balanced` | 2:600 | 0/600 | assistant | 2:604 | 604/604 | **user** |
| `mts_balanced` | 2:165, 4:314, 6:104, 8:17 | 0/600 | assistant | 2:1, 3:7, 4:2, 5:16 | 86/86 | **user** |
| `toolace_balanced` | 2:593, 4:7 | 0/600 | assistant | 3:695, 7:18, 9:5, 11:8 | 734/734 | assistant |

"The user supplies a document and asks about it, then supplies a different document and asks
about that one" ENTAILS four messages. The generator recovered `{4: 600}` against the split's
`{4: 194}` with the turn count never mentioned. Six for six on instructions.

Not so on high-stakes: **every one of the four missed the system turn**, because a deployment's
system prompt is a fact about the harness, not about the user's problem — nothing in the
situation implies it. `mt_balanced` additionally ends on the assistant where all 604 real rows
end on the user, and its clinical documents are ~7× too short.

**And it did not matter.** `mt_balanced` got the system turn, the terminal role and the length
all wrong and still scored 0.9861 against `kind-pin`'s 0.9835. What carries the label there is
the clinical content and what is at risk in it, not the document's wrapper.

## The exception, and what it marks

`toolace_balanced` lost 0.105 on-target — and its OFF-target mean rose to 0.9059, the highest
of any high-stakes arm (`kind-pin` 0.8474). It stopped being a tool-calling set and became a
general high-stakes set. That is the same signature hu_harm arm 3 found: a stripped prompt
buying off-target transfer by giving up its own split.

The boundary this marks is not "shape matters sometimes". It is that for `toolace_balanced`
the stripped material was not shape at all. A function list in a system prompt and an emitted
call ARE what the row is; removing them removed the split. The arm's own rule — situation in,
form out — misclassified the split's defining content as form. Every other split's form was
genuinely incidental.

## Size curve: half the data is free, and there is no plateau

Six sizes x 8 class-balanced draws on each of the six shape-free instruction sets, 288 fits
(`scripts/instructions_tgtmin_size_curve.csv`). Every fit is `own 50-row base + n drawn rows`,
so n=10 trains on 60 and clears tuberlens' ~49-row optimizer-step threshold. On-target AUROC,
mean ±sd:

| Split | n=300 | n=120 | n=60 | n=30 | n=15 | n=10 | n=540 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `anthropic_harmless_refusal` | 0.9787±0.0064 | 0.9552±0.0117 | 0.9025±0.0185 | 0.8014±0.0611 | 0.7140±0.0329 | 0.7332±0.0495 | 0.9871±0.0048 |
| `bbq_substitution` | 0.8468±0.0446 | 0.8456±0.0563 | 0.7938±0.0509 | 0.7719±0.0820 | 0.6278±0.0727 | 0.5932±0.0783 | 0.8226±0.0592 |
| `hc_context_drift` | 0.9934±0.0048 | 0.9744±0.0259 | 0.9303±0.1334 | 0.5404±0.0574 | 0.5104±0.0119 | 0.5049±0.0063 | 0.9898±0.0024 |
| `hc_contradiction` | 0.9172±0.0157 | 0.8867±0.0365 | 0.8509±0.0195 | 0.8191±0.1021 | 0.6740±0.0811 | 0.5716±0.0886 | 0.9032±0.0245 |
| `mm_substitution` | 0.9514±0.0121 | 0.8896±0.0482 | 0.8558±0.0411 | 0.7903±0.0339 | 0.7863±0.0505 | 0.7144±0.0343 | 0.9456±0.0279 |
| `oig_context_drift` | 0.8564±0.0468 | 0.6452±0.0359 | 0.6029±0.0279 | 0.5898±0.0200 | 0.5696±0.0132 | 0.5695±0.0149 | 0.9252±0.0125 |
| **mean of six** | **0.9240** | **0.8661** | **0.8227** | **0.7188** | **0.6470** | **0.6144** | 0.9289 |

**NO 30-ROW PLATEAU** — the direct opposite of the hu_harm curve, where 30 rows matched 600.
Here the mean falls monotonically and has already lost 0.21 by n=30.

**Half of every set is free.** n=300 (0.9240) matches n=540 (0.9289) within noise on five of
six splits. Whatever the back 300 rows add, the probe does not use it.

**`hc_context_drift` falls off a CLIFF, not a slope**: 0.9303 at n=60 to 0.5404 — chance — at
n=30. Its n=60 sd is ±0.1334, ten times its n=300 sd, so the edge is bimodal: some 60-row
draws carry the drift pattern and some do not. A mean alone would have hidden that; the sd is
what makes it readable.

**`oig_context_drift` is the hungriest split**, down to 0.6452 by n=120 where every other
split is still at or above 0.84 — it needs several hundred rows to teach what its
document-flavoured twin learns from sixty.

## What this does NOT settle

- **Family B is missing for eight of ten splits.** The earlier run died at 4/20 sets, so eight
  of the ten comparisons above are against `--kind`, not against the measured-description arm
  the ablation is nominally of. `tgtmin` beating `--kind` is a weaker claim than `tgtmin`
  matching `tgtshot`, which is only shown on `anthropic_harmless_refusal`.
- **One generator, one draw per split.** hu_harm arm 3 needed a second draw to read its own
  result; generator variance here is unmeasured, and the two losses are the cells where that
  matters most.
- **The size curve is `tgtmin` only.** Whether the arm's small-n behaviour differs from the
  measured-description or `--kind` arms is unmeasured; the cliff on `hc_context_drift` in
  particular may or may not be a property of shape-free data specifically.
- **`anthropic_hh_balanced` rewards no targeting at all** — unsteered generic scores 0.9699,
  above every targeted arm. It is a poor test of shape in either direction.
