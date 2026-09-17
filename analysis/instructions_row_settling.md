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

**192 fits** — 3 arms x {60, 120, 300, 540} rows x **16 draws**. Sixteen and not eight
because a per-sample statistic has a far wider across-draw spread than a 50-row part AUROC,
and the settling threshold has to be read in units of that spread. Everything else follows
the campaign protocol exactly: gemma-3-27b-it L32, `linear_then_softmax`, single probe, no
base data, default accumulation, `dev_samples/instructions` for early stopping, full eval
splits from the Kaggle activations.

`n=30` is absent on purpose. It ran at accumulation 1, a different optimizer regime, and
`run_instrparts_fits.sh` documents why it must never sit on the same curve.

## The unit: two statistics, because they disagree

A single sample has no AUROC, so something has to stand in for one. Two things do, and the
gap between them turns out to be the main result.

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

Both are computed on **ranks within each fit**, never on raw probabilities: these are 192
separately-fit probes whose output scales are not calibrated to one another, and only a
rank statistic is comparable across `n`.

### The threshold

"Improves negligibly" has to be read against the row's own noise, not a fixed 0.01. Each
row gets `delta_i = max(2 * SE_i, 0.03)`, where `SE_i` is its pooled across-draw standard
error, and settles at the smallest `n` from which it never again departs from its `n=540`
value by more than `delta_i`. Median `delta` came out at 0.058 / 0.064 / 0.038 on the three
arms — that is the real resolution of these settling calls, and no claim below is finer
than it.

### Reproduction

Draws 0–7 use the same row subsets as `scripts/instructions_parts_size_curve.csv` (the draw
key is `<stem>:<n>:<draw>` and nothing about it changed). All twelve arm x size cells
reproduce the published AUROC **exactly, to 0.0000, worst single draw 0.0000**. So these
per-row scores are not a re-derivation of the published curves; they are the scores those
curves were computed from.

## What settles when

Percentages are of the split's rows, by `u_i`.

| arm | settles by 120 | settles by 300 | still moving at 540 | declining |
|---|---|---|---|---|
| `hc_drift` shape-free | 43% | 56% | 1 row | 2 rows |
| `oig_drift` shape-free | 3% | 47% | **50%** | 6 rows |
| `mm_sub` + shape info | 19% (15% free at 60) | 36% | **45%** | 2 rows |

Three quite different regimes:

**`hc_drift` is finished.** Nothing is still moving at 540 beyond noise. 43% of rows are
done at 120 and the rest at 300. The two declining rows are the same conversation twice
(*"Is a tepid sponge bath a good way to reduce fever in children?"*), which ends at a
placement value of 0.36 — below chance, and worse than it was at 60.

**`oig_drift` has not converged.** Half the split is still climbing at the last size, and
those rows carry **58% of the entire 60→540 gain**. The published curve reaching 0.948 is
not a plateau; the arm was stopped mid-climb.

**`mm_sub` splits in two.** 15% of rows are *free* — they sit at 0.984 from n=60 and end at
0.994, so training barely touches them; the raw activation geometry already separates them.
At the other end, 44% are still moving at 540 and carry **70% of the gain**. There is very
little in between.

## The two statistics disagree, and that is the finding

On `hc_drift`, the pair solve rate reaches **1.000 by n=300 and no pair is wrong at 540** —
90% of pairs are done at 120. Yet 56% of *rows* only settle at 300 in placement units.

Both numbers are correct, and together they say something the split curve cannot:

> Past n=120, `hc_drift` is no longer learning to tell a compliant ending from a
> non-compliant one. It is learning to rank prefixes against each other. The AUROC gain
> from 120 to 540 is calibration, not discrimination.

That distinction matters for what you'd do next. If the remaining headroom were
discrimination, more training data of the same kind would be the answer. It isn't, so it
wouldn't be.

The other two arms are not in that position. `oig_drift` still has 7 pairs it gets wrong at
540, and `mm_sub` has 8 — and in both, some of those pairs get **worse** with more rows:

| | pairs declining | solve rate, 60 → 540 |
|---|---|---|
| `oig_drift` | 6 of 97 | 0.917 → **0.219** |
| `mm_sub` | 3 of 92 | 0.521 → **0.271** |

A solve rate below 0.5 means the probe orders the pair backwards, consistently, in most
draws. These are not unlearned rows; they are confidently wrong ones, and the training is
making them more so.

## Why those rows are hard — and why the first test for it failed

The obvious next question is what the non-settling rows have in common. The first test
looked for a topic: the settling classes were projected back onto the bge-base-en-v1.5
prefix embeddings used to cut the published parts, and scored against a label-permutation
null. All three arms came back at **z = +0.16, −0.70, −0.12** — no separation whatsoever.
The class-by-part crosstabs say the same thing in plainer terms; on `hc_drift` the four
k-means parts hold 23/20/21/20 early settlers and 25/30/25/28 late ones, which is as close
to uniform as 194 rows get.

So settling is **not topical**, and no amount of targeting data collection at a subject
area would address it. Two of the three published per-part panels are, in this light,
measuring something orthogonal to what limits them.

But the prefix embedding was also the wrong instrument, and it is worth being precise about
why: **the prefix is shared by both rows of a pair.** Whatever makes one pair harder than
another cannot be in the prefix, by construction. It has to be in the two endings. Looking
there gives an answer immediately:

**`mm_sub` — the probe confounds compliance with length.** Correlating pair solve rate
against `len(non-compliant ending) − len(compliant ending)`:

| n | spearman | p |
|---|---|---|
| 60 | −0.251 | 0.016 |
| 120 | −0.269 | 0.010 |
| 300 | −0.359 | 0.00044 |
| 540 | −0.097 | 0.36 |

Negative means: the longer the non-compliant ending, the more likely the probe prefers it.
The confound is real and significant through 300 rows, and **washes out by 540**. That is
what `mm_sub`'s 44% unsettled population is doing — those rows are not learning the concept
late, they are having a surface confound trained out of them. The backwards pairs are the
extreme case; in pair 24 the compliant ending is *"They initially thought the scandal was
backed by real evidence"* and the non-compliant one is a 40-word florid restatement
(*"possessed corroborating evidence, thereby inciting its relentless intensification"*),
and the probe prefers the florid one in 7 of 8 draws at every size.

**`hc_drift` — the same confound, in the opposite direction, gone by 120.** Spearman +0.576
(p = 7e-10) at n=60, and nothing significant after. Which is exactly consistent with its
pair curve saturating at 120: the confound *was* the n=60 behaviour.

**`oig_drift` — refusal-shaped distractors.** Length explains nothing here (no size
significant). But the two pairs whose non-compliant ending is a refusal — *"I apologize, but
I don't have enough information"* — sit at a solve rate of **0.000 at n=540, against 0.964
for the other 95 pairs**. Every draw, backwards. Only the *final* message counts for this
test; `oig_drift` carries refusals in the prefix too, and those are shared by the pair and
explain nothing.

Two pairs is two pairs, and the effect being total rather than partial is what makes it
worth writing down, not the sample size. It is also the one finding here that points at a
concrete fix: shape-free training gives the probe no reason to distinguish "refused because
the instruction said to" from "refused", and these splits contain both.

## Five readings

1. **`hc_drift` shape-free stops learning the concept at 120 rows.** Everything after that
   is cross-prefix calibration. The pair statistic is at 1.000 and stays there.
2. **`oig_drift` shape-free was stopped mid-climb.** Half its rows are still moving at 540
   and carry 58% of the gain; its published 0.948 is not a ceiling.
3. **`mm_sub`'s late population is a confound being unlearned, not a concept being
   learned.** The length correlation is significant through n=300 and gone at 540.
4. **Settling is not topical in any of the three arms.** All three permutation tests are at
   z ≈ 0, and the class-by-part crosstabs are flat. What separates hard pairs from easy ones
   lives in the endings, which the prefix-based partition cannot see.
5. **A small pocket in each split gets actively worse with more data** — 6 pairs in
   `oig_drift` falling to 0.219, 3 in `mm_sub` to 0.271. Split-level means hide these
   completely; they are the reason to do this per-row at all.

## Caveats

- **16 draws is the resolution limit.** Median `delta` is 0.038–0.064, so a row is called
  settled when its curve is flat *to within about five AUROC points*. Group-level statements
  (the class curves, the crosstabs, the correlations) are much tighter than the per-row
  labels they are built from; prefer them.
- **`u` and `c` are different questions, not two measurements of one.** `u` is what the
  published curve decomposes into; `c` is what instruction-following actually means once
  prefix difficulty is removed. Where they disagree, both are right.
- **The declining and refusal pockets are 2–8 pairs each.** The effects within them are
  large and consistent across draws, but the populations are small.
- The k-means parts used for the crosstabs have weak stability (best ARI 0.67), so read the
  crosstabs as "settling does not line up with the partition", not as a statement about the
  partition itself.

## Files

| file | what |
|---|---|
| `run_instrparts_rowscores.sh` | the 192 fits |
| `scripts/fit_instructions_parts.py` | `--row-scores DIR` persists `predict_proba` per split per fit |
| `data/instructions_row_scores/<stem>/gad_n<n>_d<d>.npz` | the per-row probabilities, in file order |
| `scripts/instructions_rowscores_size_curve.csv` | the 192 fits' split and part AUROCs |
| `scripts/analyze_row_settling.py` | placement values, pair statistics, taxonomy, diagnostics |
| `scripts/instructions_row_settling.csv` | per-row: class, settling point, delta, `u` and `c` at each size |

Reproduce with:

```bash
./run_instrparts_rowscores.sh
.venv_claude/bin/python scripts/analyze_row_settling.py --repro
```
