# Which eval rows stop improving, and which never do

The size curves in *Pooled, Steered, Targeted* are averages over an eval split. An average
that flattens can hide three different things: rows still climbing at the last size, rows
that peaked and are sliding back, and rows that were never learned at all and never will
be. This reads the same curves one row at a time.

Three arms, the ones the question was about:

| arm | training set | target split |
|---|---|---|
| shape-free | `tgtmin` | `hc_context_drift` |
| shape-free | `tgtmin` | `oig_context_drift` |
| + shape info | `tgtnone` | `mm_substitution` |

**384 fits** — 3 arms x {60, 120, 300, 540} rows x **32 draws**. Thirty-two and not eight
because a per-sample statistic has a far wider across-draw spread than a 50-row part AUROC;
because the settling threshold has to be read in units of that spread; and because half the
draws are then available to check the other half's conclusions (see *Does any of this
replicate?* below, which is where the original 16-draw version of this note got caught
out). Everything else follows the campaign protocol exactly: gemma-3-27b-it L32,
`linear_then_softmax`, single probe, no base data, default accumulation,
`dev_samples/instructions` for early stopping, full eval splits from the Kaggle activations.

`n=30` is absent on purpose. It ran at accumulation 1, a different optimizer regime, and
`run_instrparts_fits.sh` documents why it must never sit on the same curve.

## The unit: two statistics, because they disagree

A single sample has no AUROC, so something has to stand in for one. Two things do, and the
gap between them turns out to be one of the main results.

**Placement value `u_i`.** For a positive row, the fraction of the split's negative rows it
outranks (ties counting half); for a negative row, the fraction of positives above it. Its
mean over either class *is* the split's AUROC, exactly — so the per-row curves average back
into the published curve with nothing left over. The script asserts this rather than
assuming it, against `tuberlens.calculate_metrics`, on every fit.

**Pair solve rate `c_i`.** All three splits are fully class-paired — `hc_drift` and
`oig_drift` are 97 prefixes x 2 rows, `mm_sub` is 84 x 2 plus 8 x 4 — so the two rows of a
pair share every message but the last. Comparing them holds topic, length and register
exactly constant. `c_i` is the fraction of draws in which the pair comes out in the right
order.

Both are computed on **ranks within each fit**, never on raw probabilities: these are 384
separately-fit probes whose output scales are not calibrated to one another, and only a
rank statistic is comparable across `n`.

### The threshold

"Improves negligibly" has to be read against the row's own noise, not a fixed 0.01. Each
row gets `delta_i = max(2 * SE_i, 0.03)`, where `SE_i` is its pooled across-draw standard
error, and settles at the smallest `n` from which it never again departs from its `n=540`
value by more than `delta_i`. Median `delta` at 32 draws is 0.042 / 0.045 / 0.031 on the
three arms — that is the real resolution of these settling calls, and no claim below is
finer than it.

### Reproduction

Draws 0–7 use the same row subsets as `scripts/instructions_parts_size_curve.csv` (the draw
key is `<stem>:<n>:<draw>` and nothing about it changed). All twelve arm x size cells
reproduce the published AUROC **exactly, to 0.0000, worst single draw 0.0000**. So these
per-row scores are not a re-derivation of the published curves; they are the scores those
curves were computed from.

## Does any of this replicate?

The settling labels are, by construction, **in-sample**: the same draws supply the mean
curve, the tolerance *and* the selection, so a row whose draws happened to cluster gets
labelled "settled" partly by luck — and the flatness then reported is the very quantity it
was selected on. `--validate` tests this properly: classify each row on the first half of
the draws, then measure how much that population moves on the second half, which had no say
in choosing it. The column to read is the **120→540 span**, the only one every population
shares.

Classify on draws 0–15, measure on the 192 fits of draws 16–31:

| arm | selected population | rows | moves on fresh draws |
|---|---|---|---|
| `hc_drift` | settles by 120 | 84 | −0.016 ±0.004 |
| | settles by 300 | 109 | +0.004 ±0.005 |
| `oig_drift` | settles by 120 | 6 | +0.048 ±0.016 |
| | settles by 300 | 91 | +0.189 ±0.012 |
| | still moving at 540 | 97 | +0.330 ±0.016 |
| `mm_sub` | settles by 60 | 30 | +0.005 ±0.002 |
| | settles by 120 | 8 | +0.022 ±0.004 |
| | settles by 300 | 72 | +0.097 ±0.008 |
| | still moving at 540 | 90 | +0.207 ±0.014 |

**`oig_drift` and `mm_sub` validate cleanly** — monotone, well separated, out of sample. A
row called settled moves 4–20x less than one called unsettled, on fits that did not exist
when the label was assigned.

**`hc_drift` does not, and the reason matters.** Its two populations are both flat on fresh
draws (−0.016 and +0.004), and their half-to-half label agreement is 45% against 75% and
72% for the other two arms. The cause is visible in the split curve itself:

| arm | sd across draws at n=120 | at n=540 |
|---|---|---|
| `hc_drift` | **0.110** | 0.024 |
| `oig_drift` | 0.038 | 0.011 |
| `mm_sub` | 0.039 | 0.034 |

`hc_drift`'s AUROC at n=120 swings by 0.11 depending on which 120 rows you draw — draws
0–15 average 0.884 there, draws 16–31 average 0.943. So whether a row "still has room at
120" is largely a fact about the draw set, not the row. The fine 120-vs-300 boundary is not
measurable on this arm at this draw count, and the share of rows on either side of it is
unstable: 43% settled by 120 at 16 draws, 64% at 32.

What survives for `hc_drift` is the coarse statement, which the fresh draws support
strongly: **it is done by 120.** On the B half nothing moves after 120 at all.

## What settles when

At 32 draws, percentages of the split's rows, by `u_i`:

| arm | settles by 120 | by 300 | still moving at 540 | declining |
|---|---|---|---|---|
| `hc_drift` shape-free | 64% | 35% | 1 row | 5 rows |
| `oig_drift` shape-free | 3% | 35% | **63%** | 7 rows |
| `mm_sub` + shape info | 21% (17% free at 60) | 32% | **46%** | 3 rows |

Read the `hc_drift` row as "settled by 120" in full, per the section above.

**`oig_drift` has not converged.** 63% of the split is still climbing at the last size, and
those rows carry the bulk of the 60→540 gain. The published curve reaching 0.948 is not a
plateau; the arm was stopped mid-climb.

**`mm_sub` splits in two.** 17% of rows are *free* — 0.977 at n=60, 0.987 at 540, so
training barely touches them and the raw activation geometry already separates them. At the
other end, 46% are still moving at 540. There is very little in between.

## The two statistics disagree, and that is a finding

On `hc_drift`, the pair solve rate reaches **0.996 at n=120 and 1.000 by 300**, and it never
leaves. Yet in placement units a third of rows are still classed as moving until 300.

Both numbers are correct, and together they say something the split curve cannot:

> Past n=120, `hc_drift` is no longer learning to tell a compliant ending from a
> non-compliant one. It is learning to rank prefixes against each other. What is left is
> calibration, not discrimination.

That distinction matters for what you would do next. If the remaining headroom were
discrimination, more training data of the same kind would be the answer. It is not, so it
would not be.

The other two arms are not in that position. `oig_drift` still gets 6 pairs wrong at 540 and
`mm_sub` 11 — and in both, a small pocket gets **worse** with more rows:

| | pairs declining | solve rate, 60 → 540 |
|---|---|---|
| `oig_drift` | 3 of 97 | 0.615 → 0.729 (dipping to 0.552 at 120) |
| `mm_sub` | 2 of 92 | 0.375 → **0.047** |

A solve rate below 0.5 means the probe orders the pair backwards, consistently, in most
draws. Those `mm_sub` pairs are not unlearned; they are confidently wrong, and training is
making them more so.

## Why those rows are hard — and why the first test for it failed

The obvious next question is what the non-settling rows have in common. The first test
looked for a topic: the settling classes were projected onto the bge-base-en-v1.5 prefix
embeddings used to cut the published parts, and scored against a label-permutation null. All
three arms came back at **z ≈ 0** — no separation. The class-by-part crosstabs say the same
thing in plainer terms; on `hc_drift` the four k-means parts hold 33/35/29/28 early settlers
and 15/13/17/19 late ones, about as close to uniform as 194 rows get.

So settling is **not topical**, and targeting data collection at a subject area would not
address it. Two of the three published per-part panels are, in this light, measuring
something orthogonal to what limits them.

The prefix embedding was also the wrong instrument, and it is worth being precise about
why: **the prefix is shared by both rows of a pair.** Whatever makes one pair harder than
another cannot live there, by construction. It has to be in the two endings. Looking there
answers immediately:

**`mm_sub` — the probe confounds compliance with length, and never fully stops.**
Correlating pair solve rate against `len(non-compliant ending) − len(compliant ending)`:

| n | spearman | p |
|---|---|---|
| 60 | −0.276 | 0.0076 |
| 120 | −0.322 | 0.0017 |
| 300 | −0.330 | 0.0013 |
| 540 | −0.231 | 0.027 |

Negative means the longer the non-compliant ending, the more likely the probe prefers it.
*(The 16-draw version of this note reported the effect as washing out by 540 — at 32 draws
it is still significant there. It weakens; it does not go away.)* In pair 24 the compliant
ending is *"They initially thought the scandal was backed by real evidence"* and the
non-compliant one a 40-word florid restatement — *"possessed corroborating evidence, thereby
inciting its relentless intensification"* — and the probe prefers the florid one at every
size.

**`hc_drift` — the same confound, opposite sign, gone by 120.** Spearman +0.578 (p = 6e-10)
at n=60 and nothing significant after, which is exactly consistent with its pair curve
saturating at 120: the confound *was* the n=60 behaviour.

**`oig_drift` — refusal-shaped distractors.** Length explains nothing here (no size
significant). But the two pairs whose non-compliant ending is a refusal — *"I apologize, but
I don't have enough information"* — sit at a solve rate of **0.062 at n=540, against 0.968
for the other 95 pairs**. Nearly every draw, backwards. Only the *final* message counts for
this test; `oig_drift` carries refusals in the prefix too, and those are shared by the pair
and explain nothing.

Two pairs is two pairs, and it is the effect being near-total rather than the sample size
that makes it worth recording. It is also the one finding here pointing at a concrete fix:
shape-free training gives the probe no reason to distinguish "refused because the
instruction said to" from "refused", and these splits contain both.

## Five readings

1. **`hc_drift` shape-free is done at 120 rows.** The pair statistic is at 0.996 there and
   1.000 by 300; whatever the AUROC gains after that, it is not discrimination.
2. **`oig_drift` shape-free was stopped mid-climb.** 63% of rows are still moving at 540; its
   published 0.948 is not a ceiling.
3. **`mm_sub` carries a length confound at every size measured.** Significant from 60 through
   540; its 46% unsettled population is a surface cue being slowly unlearned.
4. **Settling is not topical in any of the three arms.** All three permutation tests are at
   z ≈ 0 and the class-by-part crosstabs are flat. What separates hard pairs from easy ones
   lives in the endings, which a prefix-based partition cannot see.
5. **A small pocket in each split gets actively worse with more data** — 2 `mm_sub` pairs
   falling to 0.047, and `oig_drift`'s refusal pairs at 0.062. Split-level means hide these
   completely; they are the reason to do this per-row at all.

## Caveats

- **Per-row labels are noisy; population statements are not.** Half-to-half label agreement
  is 45% / 75% / 72%. The group curves and the out-of-sample table are what should be
  quoted; individual row labels should not be leaned on.
- **"Settled" means "moves much less", not "stops".** The validated populations still drift
  by up to +0.05 over 120→540 on fresh draws. The claim that survives is relative.
- **The `hc_drift` 120-vs-300 boundary is not measurable** at this draw count — see above.
- **`u` and `c` are different questions, not two measurements of one.** `u` is what the
  published curve decomposes into; `c` is what instruction-following means once prefix
  difficulty is removed. Where they disagree, both are right.
- **The declining and refusal pockets are 2–11 pairs each.** Large, consistent effects;
  small populations.
- The k-means parts used for the crosstabs have weak stability (best ARI 0.67), so read
  those as "settling does not line up with the partition", not as a claim about the
  partition itself.

## Files

| file | what |
|---|---|
| `run_instrparts_rowscores.sh` | the fits (`DRAWS=32`) |
| `scripts/fit_instructions_parts.py` | `--row-scores DIR` persists `predict_proba` per split per fit |
| `data/instructions_row_scores/<stem>/gad_n<n>_d<d>.npz` | the per-row probabilities, in file order |
| `scripts/instructions_rowscores_size_curve.csv` | the 384 fits' split and part AUROCs |
| `scripts/analyze_row_settling.py` | placement values, pair statistics, taxonomy, validation, diagnostics |
| `scripts/instructions_row_settling.csv` | per-row: class, settling point, delta, `u` and `c` at each size |
| `analysis/instructions_row_settling_report.txt` | the full printed report |

Reproduce with:

```bash
DRAWS=32 ./run_instrparts_rowscores.sh
.venv_claude/bin/python scripts/analyze_row_settling.py --repro --validate
```
