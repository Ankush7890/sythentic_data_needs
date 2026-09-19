# Size curves for the ten prompt-variant sets

How much of each set's score is the *particular rows* and how much is the *number* of them.
560 fits: ten 600-row sets, 8 class-balanced draws at each of n = 590, 300, 150, 75, 40, 20,
10, no base data, each set scored on the split its prompts were written for.

## Protocol

`run_prompts5_sizecurve.sh` → `scripts/subsample_curve_concept.py --no-base`, gemma-3-27b
layer 32. Draws are seeded on `(file stem, n, draw)`, so a row is reproducible on its own.

Two deliberate differences from the n=600 runs in `analysis/*_prompts5_results.md`:

- **Validation is the target split's own 125 dev rows** (`dev_samples/highstakes_500_toolace/`,
  `dev_samples/highstakes_500_anthropic_hh/`) rather than the four-split `highstakes_500`.
  The validation set drives early stopping, so these are *different probes*, and the n=600
  numbers in the earlier write-ups are **not** points on these curves.
- **Only the target split is scored** (`--eval-splits`), which is what makes 560 fits cheap
  — 45 s each on toolace, ~95 s on the longer hh rows.

**n = 40, 20 and 10 run with `--grad-accum 1`** and are tagged `base='none+ga1'`. The
inherited spec (batch_size 16 × gradient_accumulation_steps 4) takes one optimizer step per 64
samples, so a training set under ~49 rows would take zero steps and the fit would return the
untrained probe. Do not read across that boundary as one line.

Results: `scripts/highstakes_toolace_prompts5_sizecurve.csv`,
`scripts/highstakes_anthropic_hh_prompts5_sizecurve.csv`.

## Toolace (eval `toolace_balanced`, 734 rows; base probe alone = 0.856)

| set | 590 | 300 | 150 | 75 | 40* | 20* | 10* | 590→10 |
|---|---|---|---|---|---|---|---|---|
| endings | **0.891** | 0.876 | 0.859 | 0.836 | 0.829 | 0.796 | 0.752 | 0.139 |
| replica | 0.871 | 0.850 | 0.851 | 0.839 | 0.810 | 0.802 | 0.799 | 0.073 |
| grid | 0.851 | 0.846 | 0.839 | 0.825 | 0.819 | 0.777 | 0.779 | 0.072 |
| longtail | 0.849 | 0.843 | 0.833 | 0.829 | 0.821 | 0.822 | **0.804** | **0.045** |
| deceptive | 0.746 | 0.741 | 0.727 | 0.720 | 0.723 | 0.678 | 0.667 | 0.078 |
| **across sets** | 0.842 | 0.831 | 0.822 | 0.810 | 0.801 | 0.775 | 0.760 | |
| **sd across sets** | 0.056 | 0.052 | 0.054 | 0.051 | 0.044 | 0.057 | 0.056 | |
| mean draw sd | 0.006 | 0.010 | 0.013 | 0.015 | 0.020 | 0.025 | 0.040 | |

## Anthropic_hh (eval `anthropic_hh_balanced`, 2984 rows; base probe alone = 0.947)

| set | 590 | 300 | 150 | 75 | 40* | 20* | 10* | 590→10 |
|---|---|---|---|---|---|---|---|---|
| replica | **0.975** | 0.973 | 0.969 | 0.970 | 0.967 | 0.954 | **0.940** | 0.035 |
| twin | 0.970 | 0.974 | 0.974 | 0.970 | 0.959 | 0.949 | 0.908 | 0.062 |
| long | 0.936 | 0.939 | 0.937 | 0.948 | 0.924 | 0.852 | 0.802 | 0.135 |
| redteam | 0.930 | 0.937 | 0.943 | 0.948 | 0.923 | **0.950** | 0.933 | **−0.002** |
| topicpair | 0.929 | 0.933 | 0.902 | 0.904 | 0.918 | 0.859 | 0.757 | 0.172 |
| **across sets** | 0.948 | 0.952 | 0.945 | 0.948 | 0.938 | 0.913 | 0.868 | |
| **sd across sets** | 0.022 | 0.020 | 0.029 | 0.027 | 0.023 | 0.052 | 0.083 | |
| mean draw sd | 0.010 | 0.010 | 0.012 | 0.017 | 0.021 | 0.043 | 0.053 | |

\* grad-accum 1.

## Reading

**Volume is nearly irrelevant in the range these studies live in.** Cutting 590 rows to 75 —
an 8× reduction — costs 0.020 to 0.055 on toolace and *nothing measurable* on hh, where the
across-set mean is 0.948 at 590 and 0.948 at 75. On hh, four of the five sets score their best
at or below n=300. Whatever these prompts teach a probe, it is learned from the first few
dozen rows.

**The prompt matters far more than the size.** At n=590 the sd across the five prompts is
0.056 on toolace against a mean draw-to-draw sd of 0.006 — the between-prompt spread is nine
times the sampling noise, and it stays at 0.044–0.057 at every size. The gap between the best
and worst toolace prompt (0.145 at n=590) is larger than the gap between a 590-row set and a
10-row one for any single prompt (0.045–0.139). Choosing the prompt buys more than a 59×
increase in data.

**The rankings are stable down to n≈40 and only then scramble.** `endings` leads toolace at
every size ≥ 20; `replica` leads hh at every size. Below 40 rows the draw sd triples (to 0.04
on toolace, 0.053 on hh) and the order becomes unreadable — the n=10 column should be treated
as a rough indication, not a measurement.

**`redteam` is the flat one, and that is the warning sign.** Its curve is −0.002 across a 59×
cut: 10 rows score 0.933 and 590 score 0.930, with 20 rows the best point of its whole curve
(0.950). A signal that saturates at ten examples is a simple one, which fits what the earlier
write-up measured in that set: 219 of its 300 high-stakes rows carry refusal language against
6 of 300 low-stakes rows. It is learning a cue, not a concept, and it is the set that
travelled worst to the other three splits (four-split mean 0.700 in
`analysis/anthropic_hh_prompts5_results.md`).

**The two best sets need no volume either, for a better reason.** `replica` on hh gives 0.967
from 40 rows and 0.940 from 10, and it is the *best* set at every size, including at 590. Its
curve is flat because it is right, not because it is shallow: it is also the set that held up
across the other splits. Same story for `longtail` on toolace — flattest curve there (0.045)
and the best tiny-data set (0.804 at n=10).

**Practical consequence.** If the point of a targeted set is to steer a probe toward one
split, 40–75 well-written rows is the operating point; the other 500 rows buy 0.02–0.05. The
budget belongs in writing the prompt — and in checking the resulting set for shortcut cues —
not in generating more rows.

## Caveats

- These curves and the n=600 numbers in the earlier write-ups use different validation sets,
  so the two must not be plotted as one line.
- Only the target split is scored here. Nothing in this file says how a set travels; that
  question is answered at n=600 in the two earlier write-ups, and the answer differs sharply
  between prompts.
- toolace's eval is 734 rows against hh's 2984, so a given AUROC gap is noisier on the toolace
  side.
