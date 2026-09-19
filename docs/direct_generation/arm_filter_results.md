# Two set-level filters, three concepts, 43 arms

Does removing *redundant* rows from a red-team or generated training set make the probe
better at equal training-set size? Two filters — one that drops the most lexically
predictable rows, one that maximizes the mix of conversation shapes — measured on every
red-team and generated arm of three concepts, **1,080 fits**.

**Answer: no. Pooled over 43 arms neither filter differs from not filtering**, and no
property of a set that was measured predicts which arms will gain or lose.

| | arms | Δ lexical | Δ shape-mix |
| --- | --- | --- | --- |
| high-stakes | 16 | −0.0022 ±0.0027 (t=−0.80) | −0.0040 ±0.0044 (t=−0.90) |
| **instructions** | 16 | +0.0034 ±0.0053 (t=+0.64) | **+0.0119 ±0.0045 (t=+2.63)** |
| hu_harm | 11 | −0.0020 ±0.0041 (t=−0.49) | −0.0035 ±0.0048 (t=−0.72) |
| **pooled** | **43** | **−0.0001 ±0.0024 (t=−0.03)** | **+0.0021 ±0.0028 (t=+0.73)** |

The one exception is instructions' shape-mix column, and it does not survive a
leave-one-attacker-out check: drop llama70b's four arms (mean +0.033) and the other twelve
give **+0.005, t ≈ 1.6**. Treat it as the one place worth a follow-up, not as a result.

Filters: [`scripts/shape_diversity.py`](../scripts/shape_diversity.py). Fits:
`scripts/{highstakes,instructions,hu_harm}_armfilter.csv`, rendered by
[`scripts/report_armfilter.py`](../scripts/report_armfilter.py) with per-set diagnostics in
`scripts/<concept>_armfilter_sets.json` (built by
[`scripts/build_armfilter_sets.py`](../scripts/build_armfilter_sets.py)).

## The arms

Each concept contributes the red-team arms of its `*_last` branch plus the generator
scaffold's one-shot 600-row set, per attacker. Every arm is fitted on **its own attacker's**
50-row base, as its red-team run was, so an arm is single-source. The three red-team arms
differ by one config knob each, which is what their names mean:

| arm | what it is |
| --- | --- |
| `general` | red-team loop, **no** `eval.data_description` anywhere (`itermemo150`) |
| `+desc` | adds `eval.data_description` on the judge / eval-scope side |
| `+attacker` | adds `show_eval_data_description: true` — the description also enters the **attacker's** prompt |
| `gen` | the generator scaffold's one-shot 600-row set, `data/<concept>_<tag>_600.jsonl` |

| concept | branch | arms | note |
| --- | --- | --- | --- |
| high-stakes | `origin/experiment_hs_last` | 16 (4 attackers × 4) | complete |
| instructions | `origin/experiment_instruction_last` | 16 (4 × 4) | complete |
| hu_harm | `origin/human_harm_last` | 11 | see below |

**hu_harm is 11, not 16, and the reason is the branch, not the protocol.** `+desc` there is
`evaldesc_new` — its config is documented as "ARM 5's config with
`show_eval_data_description` flipped to FALSE", i.e. the exact one-knob control for
`evaldesc_attacker`, the same pairing the other concepts have; the older `evaldesc` is not
that control and exists for only two attackers. `itermemo150` exists for only two attackers.
And llama70b's three hu_harm red-team arms hold 16 / 50 / 78 rows: at n = 0.6·N those are
9–46 added rows and a 0.8 keep removes 3–15 of them, so they are excluded as unmeasurable
rather than reported as noise.

## The two filters

**Lexical confounders** (`filter_lexical_confounders`) is tuberlens'
`scripts/preprocessing_redteaming.filter_dataset`, generalized: bag-of-words unigrams
(`min_df=3`, `max_df=0.9`, ≤20k features) → balanced logistic regression → drop the rows
classified with the highest confidence. Three changes were needed to use it here: the
binary label comes from the two class labels actually present (the original compares against
the literal string `"high-stakes"`, which makes it a silent no-op on any other concept); the
percentile cut runs **inside each class**, so class balance cannot drift and become a second
uncontrolled variable; and it reports what it did. `max(predict_proba)` is confidence in the
*predicted* class, so a row the lexical model is confidently **wrong** about is dropped too —
inherited behaviour, kept deliberately.

**Shape mix** (`filter_shape_redundancy`) is new, and maximizes the **variety of
conversation shapes** rather than cutting a tail — a percentile threshold cannot express
"keep one of each". Content is thrown away entirely; the descriptor is the role sequence
(`ua`, `uaua`, `sua`, … — which subsumes turn count, whether a system turn is part of the
row, and which role the row ends on: the whole `SHAPE (exact)` block of
`scripts/split_specs.py` in one string) plus six log-scaled length features. Two stages:

1. **quota over role sequences** — hand the budget out one slot at a time, round-robin over
   the groups that still have rows. As-uniform-as-possible is the maximum-entropy allocation
   subject to what the set contains, so a shape holding 3% of the rows comes out at its fair
   share instead of at 3%.
2. **max-min spread inside each group** — rows sharing a role sequence still differ in
   length, and length is shape (45 characters and 3000 characters are both `ua`). Greedy
   k-center on the standardized length vector, seeded at the group medoid. No RNG anywhere.

The descriptor is computed on the **combine/convert-transformed** messages — the ones the
extraction path actually tokenizes — since `combine_consecutive_messages` merges same-role
neighbours and `uaa` reaches the probe as `ua`. (Measured: the transform is a no-op on all
four of these sets, but it is now checked rather than assumed.)

Both filters run `per_label=True` and return `floor(keep·N)/2` rows per class, so the two
filtered pools are the same size, exactly class-balanced, and a balanced draw from either is
comparable to one from the unfiltered set.

## Results

Every cell: 8 class-balanced draws per pool, `Δ` against that arm's unfiltered column with
the standard error of the difference. `H_norm` is the normalized entropy of the set's role
sequences, `top %` the share of its most common one — both measured before filtering.

### high-stakes — 16 arms

| attacker | arm | N | n | H_norm | top % | original | Δ lex | Δ shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama70b | general | 262 | 156 | 0.792 | 47.3 | 0.9110 ±0.0143 | +0.0031 ±0.0056 | +0.0063 ±0.0067 |
| llama70b | +desc | 40 | 24 | 0.878 | 37.5 | 0.9110 ±0.0236 | −0.0054 ±0.0119 | −0.0099 ±0.0098 |
| llama70b | +attacker | 62 | 36 | 0.703 | 61.3 | 0.8849 ±0.0197 | +0.0166 ±0.0104 | **+0.0273 ±0.0087** |
| llama70b | gen | 600 | 360 | 1.000 | 100.0 | 0.8844 ±0.0222 | −0.0060 ±0.0096 | **−0.0292 ±0.0153** |
| deepseek | general | 394 | 236 | 0.585 | 65.0 | 0.8657 ±0.0089 | −0.0147 ±0.0101 | **−0.0446 ±0.0160** |
| deepseek | +desc | 390 | 234 | 0.531 | 67.9 | 0.8845 ±0.0233 | +0.0062 ±0.0090 | +0.0071 ±0.0094 |
| deepseek | +attacker | 440 | 264 | 0.721 | 31.1 | 0.9022 ±0.0076 | +0.0015 ±0.0040 | −0.0076 ±0.0055 |
| deepseek | gen | 600 | 360 | 1.000 | 100.0 | 0.8965 ±0.0151 | −0.0021 ±0.0067 | −0.0114 ±0.0062 |
| gpt-oss | general | 442 | 264 | 0.349 | 85.1 | 0.8996 ±0.0109 | +0.0016 ±0.0043 | −0.0071 ±0.0046 |
| gpt-oss | +desc | 312 | 186 | 0.253 | 89.7 | 0.8927 ±0.0176 | −0.0030 ±0.0100 | **+0.0166 ±0.0065** |
| gpt-oss | +attacker | 310 | 186 | 0.468 | 73.9 | 0.9220 ±0.0076 | **−0.0149 ±0.0040** | −0.0030 ±0.0049 |
| gpt-oss | gen | 600 | 360 | 1.000 | 100.0 | 0.8705 ±0.0146 | +0.0118 ±0.0075 | +0.0088 ±0.0065 |
| nemotron | general | 558 | 334 | 0.720 | 28.1 | 0.8102 ±0.0136 | −0.0250 ±0.0257 | −0.0213 ±0.0180 |
| nemotron | +desc | 380 | 228 | 0.820 | 20.5 | 0.8604 ±0.0305 | −0.0131 ±0.0127 | +0.0065 ±0.0136 |
| nemotron | +attacker | 368 | 220 | 0.682 | 35.3 | 0.8560 ±0.0211 | −0.0002 ±0.0094 | −0.0043 ±0.0086 |
| nemotron | gen | 600 | 360 | 1.000 | 100.0 | 0.8777 ±0.0118 | +0.0088 ±0.0046 | +0.0027 ±0.0051 |

### instructions — 16 arms

| attacker | arm | N | n | H_norm | top % | original | Δ lex | Δ shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama70b | general | 310 | 186 | 0.275 | 89.0 | 0.7933 ±0.0365 | +0.0047 ±0.0165 | +0.0193 ±0.0158 |
| llama70b | +desc | 188 | 112 | 0.578 | 72.3 | 0.7808 ±0.0482 | +0.0002 ±0.0234 | **+0.0541 ±0.0252** |
| llama70b | +attacker | 228 | 136 | 0.668 | 53.5 | 0.7856 ±0.0406 | **+0.0435 ±0.0163** | **+0.0472 ±0.0159** |
| llama70b | gen | 600 | 360 | 1.000 | 100.0 | 0.7367 ±0.0309 | +0.0225 ±0.0150 | +0.0108 ±0.0175 |
| deepseek | general | 616 | 368 | 0.316 | 85.2 | 0.7198 ±0.0341 | +0.0252 ±0.0174 | +0.0193 ±0.0178 |
| deepseek | +desc | 456 | 272 | 0.328 | 81.6 | 0.7609 ±0.0332 | −0.0226 ±0.0133 | −0.0029 ±0.0162 |
| deepseek | +attacker | 516 | 308 | 0.521 | 72.9 | 0.8142 ±0.0315 | +0.0054 ±0.0124 | +0.0066 ±0.0122 |
| deepseek | gen | 600 | 360 | 1.000 | 100.0 | 0.7476 ±0.0475 | −0.0147 ±0.0213 | −0.0068 ±0.0222 |
| gpt-oss | general | 410 | 246 | 0.243 | 93.2 | 0.7422 ±0.0400 | −0.0026 ±0.0202 | +0.0136 ±0.0176 |
| gpt-oss | +desc | 410 | 246 | 0.256 | 91.2 | 0.7780 ±0.0336 | −0.0220 ±0.0205 | +0.0001 ±0.0156 |
| gpt-oss | +attacker | 376 | 224 | 0.346 | 83.8 | 0.7643 ±0.0431 | +0.0095 ±0.0185 | +0.0208 ±0.0174 |
| gpt-oss | gen | 600 | 360 | 1.000 | 100.0 | 0.6508 ±0.0366 | −0.0305 ±0.0247 | +0.0047 ±0.0258 |
| nemotron | general | 482 | 288 | 0.267 | 87.8 | 0.6992 ±0.0311 | +0.0092 ±0.0185 | −0.0011 ±0.0185 |
| nemotron | +desc | 394 | 236 | 0.384 | 78.4 | 0.8296 ±0.0211 | −0.0092 ±0.0089 | −0.0158 ±0.0126 |
| nemotron | +attacker | 326 | 194 | 0.375 | 76.7 | 0.7717 ±0.0316 | **+0.0364 ±0.0129** | +0.0121 ±0.0158 |
| nemotron | gen | 600 | 360 | 1.000 | 100.0 | 0.6782 ±0.0392 | −0.0015 ±0.0158 | +0.0090 ±0.0180 |

### hu_harm — 11 arms

| attacker | arm | N | n | H_norm | top % | original | Δ lex | Δ shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama70b | gen | 600 | 360 | 1.000 | 100.0 | 0.8764 ±0.0335 | +0.0147 ±0.0130 | −0.0054 ±0.0143 |
| deepseek | +desc | 382 | 228 | 0.328 | 94.0 | 0.8825 ±0.0202 | +0.0110 ±0.0088 | +0.0055 ±0.0094 |
| deepseek | +attacker | 336 | 200 | 0.053 | 99.4 | 0.8470 ±0.0402 | −0.0254 ±0.0247 | **−0.0416 ±0.0198** |
| deepseek | gen | 600 | 360 | 1.000 | 100.0 | 0.8933 ±0.0062 | −0.0033 ±0.0032 | +0.0030 ±0.0031 |
| gpt-oss | +desc | 302 | 180 | 0.376 | 92.7 | 0.8495 ±0.0177 | +0.0047 ±0.0079 | −0.0044 ±0.0114 |
| gpt-oss | +attacker | 290 | 174 | 0.490 | 89.3 | 0.8056 ±0.0461 | +0.0101 ±0.0221 | +0.0211 ±0.0229 |
| gpt-oss | gen | 600 | 360 | 1.000 | 100.0 | 0.8860 ±0.0075 | −0.0061 ±0.0069 | −0.0004 ±0.0035 |
| nemotron | general | 436 | 260 | 0.438 | 73.2 | 0.8859 ±0.0219 | +0.0014 ±0.0082 | −0.0066 ±0.0092 |
| nemotron | +desc | 364 | 218 | 0.321 | 82.7 | 0.8899 ±0.0068 | +0.0049 ±0.0052 | **+0.0097 ±0.0035** |
| nemotron | +attacker | 400 | 240 | 0.439 | 81.8 | 0.8736 ±0.0304 | **−0.0262 ±0.0130** | −0.0164 ±0.0140 |
| nemotron | gen | 600 | 360 | 1.000 | 100.0 | 0.8913 ±0.0060 | −0.0078 ±0.0033 | −0.0025 ±0.0041 |

## Nothing predicts which arms gain

Correlation of Δ shape-mix with every set property measured, pooled and per concept:

| candidate predictor | pooled (43) | high-stakes | instructions | hu_harm |
| --- | --- | --- | --- | --- |
| shape entropy `H_norm` | −0.044 | −0.140 | +0.012 | +0.331 |
| dominant-shape share | −0.019 | +0.025 | −0.538 | −0.130 |
| distinct shapes in the set | −0.132 | −0.060 | −0.165 | +0.073 |
| rows in the set | −0.298 | −0.375 | −0.603 | +0.039 |
| unfiltered AUROC of the arm | −0.303 | +0.241 | +0.105 | −0.046 |
| MI(shape; label) | −0.040 | −0.158 | −0.299 | +0.307 |
| bag-of-words train accuracy | −0.170 | +0.158 | +0.273 | −0.045 |
| **draw sd of the arm** | **+0.379** | +0.283 | +0.577 | −0.264 |

Every one either sits near zero pooled or changes sign between concepts. **Four hypotheses
formed during the run and were each refuted by later arms:**

1. **Δ is monotone in shape entropy** (r = −0.995 on the first four high-stakes arms, with a
   crossover at `H_norm ≈ 0.84`). Pooled r = −0.044. Its sharpest prediction — `gpt-oss`
   high-stakes `general` at `H_norm` 0.349 should gain +0.092 — came back −0.007.
2. **Arm type carries.** `+attacker` gives +0.027 / −0.008 / −0.003 / −0.004 on high-stakes
   and +0.047 / +0.007 / +0.021 / +0.012 on instructions: the effect tracks concept and
   attacker, not arm.
3. **Single-shape sets are harmed** (the six 600/600 `ua` `gen` sets, where the role-sequence
   quota provably cannot act and only the length stage runs). They give −0.029 / −0.011 /
   +0.009 / +0.003 / +0.011 / −0.005 / +0.003 / −0.000 / −0.003. The *mechanism* is real —
   greedy k-center is outlier-seeking, so on a homogeneous set it buys length extremes — but
   it does not fix the sign.
4. **Filtering helps weak sets.** Pooled r = −0.303 looks supportive, but it is +0.241 on
   high-stakes; the pooled value is driven by instructions being both the weakest concept and
   the only positive one, i.e. it is the concept effect wearing a disguise.

The one correlate that is neither near zero nor sign-flipping between two of three concepts
is the arm's own **draw-to-draw sd** (+0.379 pooled, +0.577 on instructions). That is the
least comforting possible predictor: it says the filters look best exactly where the
measurement is noisiest.

**Sibling arms settle it.** Four pairs share an attacker, a base, a concept and near-identical
shape statistics, and split in opposite directions: high-stakes deepseek `general` (−0.045)
vs `+desc` (+0.007) at `H_norm` 0.585/0.531; high-stakes gpt-oss `general` (−0.007) vs
`+desc` (+0.017); instructions deepseek `general` (+0.019 lex +0.025) vs `+desc` (−0.003 lex
−0.023); hu_harm nemotron `+desc` (+0.010) vs `+attacker` (−0.016). Whatever moves these
numbers is a property of *which particular rows* a filter removes, not of any summary
statistic of the set.

### The one hypothesis still standing, weakly

Near-degenerate shape distributions — one role sequence holding ≥99% of the rows, where the
quota stage cannot act at all:

| | arms | Δ shape-mix |
| --- | --- | --- |
| ≥99% one shape | 13 | −0.0045 ±0.0043 (t=−1.05) |
| everything else | 30 | +0.0049 ±0.0036 (t=+1.39) |

The gap is ~0.009 with each side about one standard error from zero — suggestive, not
established, and it is the fourth incarnation of a hypothesis whose first three versions
died. It contains the study's two worst cells (high-stakes deepseek `general` at 65% one
shape is *not* in this group, but hu_harm deepseek `+attacker` at 99.4% and llama70b
high-stakes `gen` at 100% are). If the shape filter is used again, gating stage B when one
shape dominates — or replacing max-min with a quantile-stratified length pick — is the
change to make.

## Variance is not systematically improved either

| filter | median sd ratio (filtered / original) | range | inflated |
| --- | --- | --- | --- |
| lexical | 0.79 | 0.34 – 5.28 | 18/43 arms |
| shape-mix | 0.86 | 0.33 – 4.99 | 15/43 arms |

A mild median tightening with violent tails in both directions, landing on different arms for
the two filters. The extremes: lexical on high-stakes `nemotron/general` (±0.0136 → ±0.0715,
draws running 0.645–0.857 where the unfiltered pool never leaves 0.79–0.83) and shape-mix on
high-stakes `deepseek/general` (±0.0089 → ±0.0443). An earlier reading of the first four arms
— "the lexical filter buys stability" — does not hold at 43.

## What to do with this

**Don't filter.** At `keep=0.8`, on 43 arms across three concepts, neither criterion beats
leaving the set alone, and both can cost up to 0.045 AUROC on an individual set with no way
to tell in advance which. The bag-of-words confound is real and available in every set
(train accuracy 0.81–1.000, the generated sets at 0.96–0.98 mean confidence) — but removing
it changes nothing, which is the useful negative: **lexical separability is not what limits
these probes.**

The single follow-up worth the compute is instructions' shape-mix column: +0.0119 ±0.0045
over 16 arms, carried by one attacker. A keep-curve (0.5 / 0.65 / 0.9) on instructions'
non-llama70b arms would settle whether there is anything there, and the pools already exist,
so it is fits only.

## What this does not settle

- **One keep fraction.** Everything is `keep=0.8`. A filter that removes 20% may be removing
  too little to matter and too much to be free.
- **Draws are unpaired.** The harness seeds each draw on the pool's filename, so an arm's
  three columns do not share subsets. Pairing them would cut the standard error on every Δ
  and is the cheapest available improvement.
- **The red-team sets were filtered whole**, attacker submissions together with the
  LLM-written opposite-class partners generated from them. The partners are frequently
  minimal word swaps of their originals ($100 → $100,000) — the most confounded thing in
  these sets — and filtering only the originals would measure something different.
- **One implementation each.** The shape filter's stage B is outlier-seeking by construction;
  the lexical filter inherits `filter_dataset`'s quirk of scoring confidence in the
  *predicted* class, so a confidently-wrong row is dropped along with the confidently-right
  ones. Different choices are different filters.
- **43 arms × 8 draws rejects a large effect, not a small one.** A true +0.005 from either
  filter would be invisible here.
- **hu_harm's llama70b red-team arms (16/50/78 rows) were never measured**, so that concept
  is 11 arms and three of its four attackers contribute no `general` arm.

## Protocol for the AUROC numbers

Per arm: three pools (unfiltered, lexical, shape-mix), each drawn from **8 times** at the
**same n** = 0.6·N, class-balanced n/2 per class, through
`scripts/subsample_curve_concept.py`. Pools are 0.8·N, so every pool can serve every draw
and no pool is exhausted (which would collapse its sd to zero). The three columns of an arm
therefore differ in *which rows were available to draw from* and in nothing else.

    dev            the concept's dev_samples/ dir (highstakes: the 500-row cut)
    eval           the concept's eval_sets/ dir, FULL splits, no subsampling
    probe          gemma-3-27b L32, linear_then_softmax, single, seed 42
    transforms     combine_consecutive_messages = convert_tool_to_assistant = True
    base           each attacker's OWN 50-row set, always in full
    n              0.6 x N per arm, 24-360 rows

Draws are seeded on `(pool filename, n, draw)`, so the three pools of an arm get
**independent** draws rather than paired ones; Δ carries the standard error of the
difference, `sqrt(sd_a²/8 + sd_b²/8)`.

**Read llama70b's two small high-stakes arms with care.** The probe steps the optimizer every 4th batch
of `ceil(rows/16)`, so llama70b's `+desc` (50 + 24 = 74 rows → 5 batches) and `+attacker`
(86 rows → 6 batches) take **one optimizer step per epoch**: they train (200 epochs,
patience 50), but with an effective batch covering the whole training set. Every other arm
in the sweep is 186–360 drawn rows and gets 3–6 steps per epoch. All sixteen run at the
repo-default `gradient_accumulation_steps=4`, so no accum-boundary caveat applies, but
llama70b's two small arms are coarser fits than the rest — which matters, because the single
largest effect in the experiment is one of them.
