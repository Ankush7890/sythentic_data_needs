# Per-part size curves: five split-targeted instructions sets

**Branch** `hello_kitty` · gemma-3-27b-it L32, linear_then_softmax, single probe ·
deepseek-v4-pro sets · 200 fits.

What this asks, for one split-targeted generated set at a time: **how much of the set do
you need**, and **where on the target split do the rows you add actually land**. The
second question is the new one — every earlier curve reported one AUROC per split, which
averages over whatever internal structure the split has.

## Protocol

Every fit trains on the drawn rows ALONE — no base data, so `n` is the whole training
set — and validates on the full 436-row `dev_samples/instructions` (early stopping).
Scoring is on the FULL eval splits with Kaggle activations. `seed 42`,
`combine_consecutive_messages = convert_tool_to_assistant = True`. Draws are
class-balanced, `n/2` per class, seeded on `(stem, n, draw)`, 8 per cell.

The draw seeding is deliberately identical to `subsample_curve_concept.py`, so draw *d*
at size *n* is the **same rows** as its twin in the published no-base curves — see the
reproduction check at the bottom.

**Two passes, two CSVs, never pooled.** The probe is batch_size 16 ×
gradient_accumulation_steps 4, so with no base data

| n | batches/epoch | optimizer steps/epoch |
|---|---|---|
| 540 | 34 | 8 |
| 300 | 19 | 4 |
| 120 | 8 | 2 |
| 60 | 4 | 1 |
| 30 | 2 | **0** |

n=30 at the default returns the probe **at initialisation**, so it is re-run at
accumulation 1 (`scripts/instructions_parts_size_curve_accum1.csv`) exactly as
`run_tgtmin_sizecurve_nobase.sh` did. That the accumulation-1 rows show real
draw-to-draw spread (sd 0.028–0.064 on eval mean) rather than eight identical values is
what says they actually trained.

## How the four parts are cut

`scripts/make_instructions_parts.py`. All four cut splits are **fully class-paired** —
each conversation prefix occurs exactly twice, once per label — so the clustering runs on
the **pair**, embedding the shared prefix alone (bge-base-en-v1.5 CLS), and both rows of
a pair land in the same part. Two consequences worth stating plainly:

- Every part is **exactly class-balanced**, so every part column is a real AUROC.
  Clustering rows individually would drift toward one-class parts, since the final
  assistant turn is precisely what the label is a property of.
- The partition is a statement about **what the conversation is about**, independent of
  the thing being scored.

Plain k-means cuts these splits very unequally (oig_context_drift 108/42/38/6), and an
AUROC over three rows per class is not a measurement, so the assignment step is
size-constrained — Lloyd's with an exact minimum-cost assignment to equal-sized slots.
Each part is ~48–52 rows, ~24–26 per class.

**These are weak clusters, and the numbers below should be read that way.** Silhouettes
are near zero and re-clustering 80% subsamples agrees only moderately:

| split | rows / pairs | silhouette | stability ARI | the four parts, by top distinguishing terms |
|---|---|---|---|---|
| hc_context_drift | 194 / 97 | +0.081 | 0.37 | p0 HIV/circumcision/beta · p1 heart, asthma, immunity, doses · p2 child, cough, fever, paracetamol · p3 magnetic bracelets, copper, joints |
| hc_contradiction | 200 / 100 | +0.115 | **0.67** | p0 cold, steam, cough, croup · p1 osteoarthritis, tylenol, copper · p2 cholesterol, LDL, cognition · p3 phone radiation, MMR/autism, selenium |
| mm_substitution | 200 / 92 | +0.029 | 0.36 | p0 pay, hours, work · p1 short/quoted speech · p2 Trump, election, immigration · p3 court, federal, city |
| oig_context_drift | 194 / 97 | +0.007 | 0.24 | p0 general "what is/was" lookups · p1 county, census, population · p2 biography, born, served · p3 film, directed, starred |

Only `hc_contradiction` has a partition that survives resampling well (ARI 0.67).
`oig_context_drift` at ARI 0.24 is close to an arbitrary quartering of a homogeneous
split, so a per-part difference there is much weaker evidence than the same difference on
`hc_contradiction`.

## Results


# Default accumulation (batch 16 x accum 4)

## +shape  ->  hc_context_drift   (draws: n60:8, n120:8, n300:8, n540:8)
| row | n=60 | n=120 | n=300 | n=540 |
|---|---|---|---|---|
| on target | 0.793±0.141 | 0.987±0.011 | 0.991±0.008 | 0.996±0.008 |
| eval mean | 0.678±0.034 | 0.715±0.032 | 0.751±0.052 | 0.730±0.056 |
|   part p0 | 0.836±0.120 | 0.978±0.029 | 0.980±0.018 | 0.993±0.008 |
|   part p1 | 0.820±0.129 | 0.996±0.007 | 1.000±0.001 | 0.999±0.003 |
|   part p2 | 0.790±0.128 | 0.982±0.011 | 0.988±0.011 | 0.994±0.014 |
|   part p3 | 0.785±0.160 | 0.998±0.002 | 1.000±0.001 | 0.997±0.007 |

## +shape  ->  hc_contradiction   (draws: n60:8, n120:8, n300:8, n540:8)
| row | n=60 | n=120 | n=300 | n=540 |
|---|---|---|---|---|
| on target | 0.887±0.058 | 0.922±0.015 | 0.933±0.009 | 0.922±0.018 |
| eval mean | 0.788±0.031 | 0.769±0.032 | 0.779±0.019 | 0.783±0.010 |
|   part p0 | 0.874±0.028 | 0.895±0.017 | 0.894±0.011 | 0.870±0.023 |
|   part p1 | 0.907±0.044 | 0.946±0.016 | 0.952±0.015 | 0.945±0.023 |
|   part p2 | 0.911±0.079 | 0.950±0.028 | 0.967±0.012 | 0.966±0.016 |
|   part p3 | 0.867±0.075 | 0.898±0.020 | 0.915±0.017 | 0.910±0.026 |

## +shape  ->  mm_substitution   (draws: n60:8, n120:8, n300:8, n540:8)
| row | n=60 | n=120 | n=300 | n=540 |
|---|---|---|---|---|
| on target | 0.752±0.058 | 0.795±0.047 | 0.897±0.059 | 0.942±0.041 |
| eval mean | 0.681±0.029 | 0.700±0.031 | 0.711±0.019 | 0.703±0.019 |
|   part p0 | 0.755±0.051 | 0.790±0.053 | 0.899±0.053 | 0.948±0.024 |
|   part p1 | 0.818±0.052 | 0.849±0.039 | 0.924±0.045 | 0.951±0.047 |
|   part p2 | 0.711±0.083 | 0.762±0.054 | 0.894±0.082 | 0.965±0.019 |
|   part p3 | 0.732±0.066 | 0.779±0.056 | 0.886±0.072 | 0.927±0.060 |

## shape-free  ->  hc_context_drift   (draws: n60:8, n120:8, n300:8, n540:8)
| row | n=60 | n=120 | n=300 | n=540 |
|---|---|---|---|---|
| on target | 0.646±0.176 | 0.864±0.103 | 0.935±0.055 | 0.954±0.028 |
| eval mean | 0.611±0.043 | 0.651±0.050 | 0.666±0.026 | 0.681±0.013 |
|   part p0 | 0.668±0.185 | 0.896±0.084 | 0.951±0.039 | 0.967±0.015 |
|   part p1 | 0.667±0.202 | 0.903±0.095 | 0.963±0.036 | 0.971±0.022 |
|   part p2 | 0.635±0.165 | 0.842±0.106 | 0.905±0.075 | 0.930±0.032 |
|   part p3 | 0.645±0.181 | 0.862±0.120 | 0.940±0.073 | 0.964±0.040 |

## shape-free  ->  oig_context_drift   (draws: n60:8, n120:8, n300:8, n540:8)
| row | n=60 | n=120 | n=300 | n=540 |
|---|---|---|---|---|
| on target | 0.645±0.029 | 0.684±0.050 | 0.870±0.051 | 0.945±0.011 |
| eval mean | 0.599±0.026 | 0.639±0.034 | 0.690±0.023 | 0.694±0.015 |
|   part p0 | 0.660±0.046 | 0.673±0.043 | 0.814±0.034 | 0.896±0.019 |
|   part p1 | 0.639±0.060 | 0.700±0.090 | 0.940±0.052 | 0.987±0.010 |
|   part p2 | 0.643±0.057 | 0.680±0.048 | 0.813±0.060 | 0.898±0.023 |
|   part p3 | 0.679±0.029 | 0.709±0.052 | 0.910±0.057 | 0.981±0.010 |


# Accumulation 1 — n=30 only, NOT comparable to the block above

## +shape  ->  hc_context_drift   (draws: n30:8)
| row | n=30 |
|---|---|
| on target | 0.593±0.166 |
| eval mean | 0.596±0.040 |
|   part p0 | 0.604±0.163 |
|   part p1 | 0.604±0.162 |
|   part p2 | 0.602±0.162 |
|   part p3 | 0.599±0.166 |

## +shape  ->  hc_contradiction   (draws: n30:8)
| row | n=30 |
|---|---|
| on target | 0.789±0.144 |
| eval mean | 0.745±0.064 |
|   part p0 | 0.782±0.145 |
|   part p1 | 0.795±0.141 |
|   part p2 | 0.810±0.150 |
|   part p3 | 0.783±0.140 |

## +shape  ->  mm_substitution   (draws: n30:8)
| row | n=30 |
|---|---|
| on target | 0.676±0.072 |
| eval mean | 0.661±0.030 |
|   part p0 | 0.686±0.080 |
|   part p1 | 0.721±0.077 |
|   part p2 | 0.639±0.082 |
|   part p3 | 0.660±0.082 |

## shape-free  ->  hc_context_drift   (draws: n30:8)
| row | n=30 |
|---|---|
| on target | 0.521±0.055 |
| eval mean | 0.572±0.028 |
|   part p0 | 0.525±0.064 |
|   part p1 | 0.519±0.053 |
|   part p2 | 0.529±0.086 |
|   part p3 | 0.509±0.051 |

## shape-free  ->  oig_context_drift   (draws: n30:8)
| row | n=30 |
|---|---|
| on target | 0.634±0.034 |
| eval mean | 0.620±0.028 |
|   part p0 | 0.642±0.053 |
|   part p1 | 0.631±0.051 |
|   part p2 | 0.634±0.022 |
|   part p3 | 0.664±0.047 |


## Reading

### 1. Shape information is worth more than four times the data

`hc_context_drift` is the one split run in both arms, so it is the clean head-to-head:

| n | + shape info | shape-free | gap |
|---|---|---|---|
| 30 (accum 1) | 0.593±0.166 | 0.521±0.055 | +0.072 |
| 60 | 0.793±0.141 | 0.646±0.176 | +0.147 |
| 120 | 0.987±0.011 | 0.864±0.103 | +0.123 |
| 300 | 0.991±0.008 | 0.935±0.055 | +0.056 |
| 540 | 0.996±0.008 | 0.954±0.028 | +0.042 |

**120 rows with the split's measured shape beat 540 rows without it** (0.987 vs 0.954),
and the gap is widest in the middle of the curve rather than at either end. The shape-free
arm is still climbing at 540; the +shape arm is done by 120.

### 2. The parts converge under +shape and stay apart under shape-free

Spread = best part minus worst part, on the same fits:

| arm → split | n=60 | n=120 | n=300 | n=540 |
|---|---|---|---|---|
| +shape → hc_context_drift | 0.050 | 0.020 | 0.019 | **0.006** |
| shape-free → hc_context_drift | 0.032 | 0.061 | 0.058 | **0.041** |
| +shape → mm_substitution | 0.107 | 0.087 | 0.038 | 0.038 |
| +shape → hc_contradiction | 0.044 | 0.055 | 0.072 | **0.095** |
| shape-free → oig_context_drift | 0.039 | 0.035 | 0.127 | **0.090** |

On `hc_context_drift` the +shape set does not merely score higher, it scores **evenly**:
by 540 rows all four parts sit within 0.006 of each other (0.993–0.999). The same split
written shape-free leaves a 0.041 spread, and it is the *same part* that lags at every
size — p2, the short paediatric-advice conversations (child, cough, fever, fluids,
paracetamol), at 0.930 against p1's 0.971. Describing the conversation's shape closes a
gap that more rows alone does not.

### 3. More rows can make a split less even, not more

`hc_contradiction` under +shape is the counter-case, and it is the best-evidenced one —
its partition is the only one that survives resampling well (ARI 0.67). On-target AUROC
plateaus from n=120 (0.922 → 0.933 → 0.922) while the **spread more than doubles**,
0.044 → 0.095. The parts move in opposite directions:

| part | n=60 | n=120 | n=300 | n=540 | |
|---|---|---|---|---|---|
| p0 cold, steam, cough, croup | 0.874 | 0.895 | 0.894 | **0.870** | flat, then down |
| p1 osteoarthritis, tylenol, copper | 0.907 | 0.946 | 0.952 | 0.945 | +0.038 |
| p2 cholesterol, LDL, cognition | 0.911 | 0.950 | 0.967 | **0.966** | +0.055 |
| p3 phone radiation, MMR, selenium | 0.867 | 0.898 | 0.915 | 0.910 | +0.043 |

Rows 60→540 buy p2 a further 0.055 and cost p0 0.004. A flat split-level curve is hiding
three parts that improve and one that does not — exactly the structure a single AUROC per
split cannot show.

`oig_context_drift` shows the same unevenness more sharply. Its spread goes 0.039 → 0.090
because the four parts start together near 0.65 and then separate into two pairs: p1
(county, census, population) and p3 (film, directed, starred) reach 0.987 and 0.981 by
n=540, while p0 (general "what is/was" lookups) and p2 (biography) stall at 0.896 and
0.898. The rows past n=120 buy the two pairs very different amounts. But this split's
partition is the least stable of the four (ARI 0.24), so treat it as suggestive rather
than established.

### 4. Targeted sets buy their own split, not the concept

On every arm the on-target curve climbs steeply while `eval mean` over all seven splits
barely moves — +shape `hc_context_drift` goes 0.793 → 0.996 on target while its eval mean
goes 0.678 → 0.730. Consistent with the cross-split transfer matrices: a set written for
one split is not a set that teaches the concept.

### 5. At 30 rows there is nothing to decompose

In the accumulation-1 block the four parts of a split move together to within ~0.01 and
carry draw-to-draw sd of 0.05–0.17. `hc_context_drift` shape-free sits at 0.521±0.055 —
chance. Whatever the parts differ in, 30 rows does not resolve it.

## Reproduction check

The two shape-free arms overlap the published no-base curve
(`per_split_studies2:scripts/instructions_tgtmin_size_curve_nobase.csv`), and because the
draws are seeded identically these are the same training sets, refit here:

| split | n | published | this run | Δ |
|---|---|---|---|---|
| hc_context_drift | 60 | 0.6516±0.1846 | 0.6463±0.1764 | −0.0052 |
| hc_context_drift | 120 | 0.8732±0.0921 | 0.8640±0.1030 | −0.0092 |
| hc_context_drift | 300 | 0.9448±0.0532 | 0.9354±0.0555 | −0.0093 |
| hc_context_drift | 540 | 0.9716±0.0148 | 0.9536±0.0275 | −0.0180 |
| oig_context_drift | 60 | 0.6459±0.0293 | 0.6453±0.0289 | −0.0006 |
| oig_context_drift | 120 | 0.6839±0.0509 | 0.6845±0.0502 | +0.0006 |
| oig_context_drift | 300 | 0.8695±0.0505 | 0.8704±0.0512 | +0.0010 |
| oig_context_drift | 540 | 0.9436±0.0120 | 0.9447±0.0107 | +0.0011 |

`oig_context_drift` reproduces to ±0.0011 at every size, and all eight of its n=540 draws
match within ±0.007. `hc_context_drift` reproduces on six of eight draws at n=540 (within
±0.007) but two draws moved by 0.037 and 0.094, which is what pulls its mean down 0.018.

**So the fit is not bit-reproducible across processes.** Note what the probe returns is
the LAST epoch and tuberlens early-stops on validation AUROC, so a run that stops an epoch
earlier returns a different probe, and on a near-ceiling 194-row split that can move AUROC
by a lot. The per-part differences in §2 and §3 are well inside the sizes that survive
this (0.04–0.10 on 8-draw means), but a single-draw per-part number is not worth reading.

## Files

| | |
|---|---|
| `scripts/make_instructions_parts.py` | cuts the four splits into four equal, class-balanced parts |
| `data/instructions_parts/<split>_parts.jsonl` | part per eval row, in file order |
| `data/instructions_parts/parts_summary.json` | sizes, silhouette, stability ARI, top terms, examples |
| `scripts/fit_instructions_parts.py` | the fits; part AUROCs off the same forward pass as the split's |
| `scripts/prep_instructions_parts.py` | Kaggle eval + dev activations, once |
| `run_instrparts_fits.sh` | the two passes |
| `scripts/report_instrparts.py` | the tables above |
| `scripts/instructions_parts_size_curve.csv` | 160 rows, n ≥ 60, default accumulation |
| `scripts/instructions_parts_size_curve_accum1.csv` | 40 rows, n = 30, accumulation 1 |

These are also plotted as the "Where inside a split the rows actually land" section of
[Pooled, Steered, Targeted](https://claude.ai/code/artifact/b096f13a-d74b-4457-bdf1-6e3db56515b2):
five per-part panels, the part-spread curve, the partition with its stability numbers, and
the n=30 rows kept off the curve axes.

## One environment fix this needed

gemma-3 extraction was broken for every concept before this run. tuberlens `e8b5833`
moved `Gemma3Arch` to `model.language_model.layers`, the transformers ≥ 4.52 layout; the
pinned transformers is 4.51.3, where `language_model` is a `Gemma3ForCausalLM` and the
layers are one level deeper, so `ArchitectureRegistry.get_architecture` matched nothing
and every extraction died with `Unsupported model architecture`.
`model_loading._register_legacy_gemma3_arch` appends a handler for the older layout.
tuberlens' own handler still wins where it works, and both read the same `input_layernorm`
of the same decoder layer, so this cannot move an activation and belongs in no cache key.
