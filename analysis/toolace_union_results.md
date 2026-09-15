# toolace: every specialist set pooled

Round 3 of `hard_split_experiments`. All eight specialist sets in one training set — round 1's four
600-row part sets ([`toolace_parts_results.md`](toolace_parts_results.md)) and round 2's four 300-row
sub-part sets ([`toolace_subparts_results.md`](toolace_subparts_results.md)) — **3600 rows, 1800/1800,
all unique**. Same protocol as both rounds: NO base data, gemma-3-27b-it L32 `linear_then_softmax`, dev
`dev_samples/highstakes_500`, full eval splits, seed 42. `scripts/highstakes_toolace_union.csv`.

**A single fit, no resampled draws** (by choice: one fit was what was asked for). Single full-set fits on
these branches have landed up to ~0.02 from their draw means, so compare it first against the other
arms' SINGLE full-set fits, and read the draw means as context.

The fit took 830 s, early-stopping at epoch 52 of 200.

## Result

| scored on | **union, 3600** | round-1 pool, 600 | round-1 ops, 600 | round-2 utility, 300 |
| --- | --- | --- | --- | --- |
| **toolace** | **0.876** | 0.841 (draws 0.860) | 0.857 (0.845) | 0.862 (0.859) |
| part ops | **0.915** | 0.902 (0.902) | 0.895 (0.882) | 0.871 (0.868) |
| part lookup | **0.838** | 0.748 (0.771) | 0.748 (0.747) | 0.823 (0.807) |
| part finance | **0.773** | 0.719 (0.760) | 0.648 (0.622) | 0.669 (0.681) |
| part roledef | 0.912 | 0.881 (0.888) | **0.922** (0.914) | 0.891 (0.893) |
| sub lookup_media | 0.787 | — | — | **0.808** (0.761) |
| sub lookup_utility | **0.842** | — | — | 0.826 (0.815) |
| sub finance_markets | **0.824** | — | — | 0.789 (0.793) |
| sub finance_money | 0.656 | — | — | 0.594 (0.616) |
| mean of 4 eval splits | 0.858 | **0.881** (0.875) | **0.881** (0.866) | 0.753 (0.760) |
| anthropic_hh | 0.802 | **0.872** (0.831) | 0.800 (0.809) | 0.628 (0.645) |
| mt | 0.843 | 0.905 (0.892) | **0.923** (0.866) | 0.695 (0.727) |
| mts | 0.911 | 0.904 (0.916) | **0.944** (0.942) | 0.826 (0.810) |

Parentheses: that arm's mean over its resampled draws (540 rows for round 1, 270 for round 2). Round-1
arms have no sub-part scores (their probes were not kept). The best round-2 arm for `finance_money` is
the money specialist (0.694 single fit, 0.690 draws), not shown.

## Reading

- **Best toolace result of the experiment: 0.876**, +0.014 over the best single fit of any other arm and
  +0.016 over the best draw mean (round-1 pool, 0.860). It also sets the best lookup part (0.838), finance
  part (0.773), ops part (0.915) and two of four sub-parts. It does not have the tightest per-part
  profile — roledef is 0.01 below round-1 ops, `lookup_media` 0.02 below the utility set's single fit.
- **The gain is concentrated where the round-2 sub-part sets added data**: lookup part +0.09 and finance
  part +0.05 over the round-1 pool's single fit, against +0.01 on ops and +0.03 on roledef.
- **It does not keep the other splits best.** Four-split mean 0.858 against 0.881 for the round-1 pool and
  ops single fits: `anthropic_hh` 0.802 and `mt` 0.843 both sit below the round-1 pool. The round-2 lookup
  sets, which drove each of those down to ~0.63–0.73 on their own, are a sixth of this pool and still
  pull on it.

## Caveats

- **One fit.** No sd; a difference under ~0.02 against any single arm is within what single fits have
  moved on these branches.
- **Unequal weighting by construction.** Every set enters whole, so round-1 parts carry 600 rows and
  round-2 sub-parts 300; lookup and finance are covered twice (round-1 set plus both halves).
- **The returned probe is the last epoch, not the best.** Training loss reached 0.0000 and early stopping
  fired at epoch 52 (patience 50), and this repo returns the final weights unless
  `PROBE_RESTORE_BEST_CHECKPOINT=1` — true of every fit in all three rounds, so the comparison is like for
  like, but at 3600 rows the probe is far past its best-dev epoch.
- Test-set information in the prompts and label noise in toolace, as in rounds 1 and 2.

## Eight random 600-row draws from the pool

The 3600 rows merged into one file (`data/toolace_parts/highstakes_deepseekv4pro_tgtnone_union8_3600.jsonl`),
then eight random class-balanced draws of 600 from it — over the merged pool, not per set, so a draw
holds on average ~100 rows of each round-1 set and ~50 of each round-2 set. Same protocol, no base.
`scripts/highstakes_toolace_union600.csv`.

| scored on | **pool draws, 600** (mean ±sd, min–max) | union 3600 (1 fit) | round-1 pool, 540 draws | round-1 ops, 540 draws | round-2 utility, 270 draws |
| --- | --- | --- | --- | --- | --- |
| **toolace** | 0.855 ±0.019 (0.821–0.877) | **0.876** | 0.860 | 0.845 | 0.859 |
| part ops | 0.901 ±0.017 | **0.915** | 0.902 | 0.882 | 0.868 |
| part lookup | 0.799 ±0.025 | **0.838** | 0.771 | 0.746 | 0.807 |
| part finance | **0.780** ±0.018 | 0.773 | 0.760 | 0.622 | 0.681 |
| part roledef | 0.877 ±0.020 | 0.912 | 0.888 | **0.914** | 0.893 |
| sub lookup_media | 0.747 ±0.028 | **0.787** | — | — | 0.761 |
| sub lookup_utility | 0.803 ±0.024 | **0.842** | — | — | 0.815 |
| sub finance_markets | 0.809 ±0.030 | **0.824** | — | — | 0.793 |
| sub finance_money | **0.708** ±0.048 | 0.656 | — | — | 0.616 |
| mean of 4 eval splits | **0.881** ±0.017 | 0.858 | 0.875 | 0.866 | 0.760 |
| anthropic_hh | **0.859** ±0.021 | 0.802 | 0.831 | 0.809 | 0.644 |
| mt | 0.884 ±0.050 | 0.843 | **0.892** | 0.866 | 0.727 |
| mts | **0.926** ±0.023 | 0.911 | 0.916 | 0.942 | 0.810 |

Against the round-1 pool at near-matched size — its draws are 540 rows, these 600 (difference of draw
means ± standard error):
toolace −0.005 ±0.008, lookup part **+0.027 ±0.010**, finance part +0.021 ±0.011, mean of 4 +0.006 ±0.008.

**Reading.**

- **At 600 rows the full pool is no better on toolace than round 1's four-set pool** (0.855 vs 0.860,
  within noise), and only the best of the eight draws (0.877) reaches the 3600-row fit's 0.876; the
  draw mean sits 0.021 below it. The union's toolace lead is therefore mostly VOLUME, not the
  composition — with the caveat that 0.876 is a single fit.
- **The composition does buy the parts round 2 targeted**: lookup +0.027 and finance +0.021 over the
  round-1 pool at the same size, paid for on ops (−0.001) and roledef (−0.011) as those shares shrink
  from 150 to ~100 rows.
- **It is the best arm off toolace.** Four-split mean 0.881 and `anthropic_hh` 0.859 are the highest of the
  experiment. The 3600-row fit lost 0.023 on that mean against these draws: at 3600 rows the probe gives
  back the other splits, at 600 it does not.
- **Draw-to-draw spread is the widest of any arm** (toolace sd 0.019, range 0.821–0.877), as expected
  when each set's share varies by chance; `mt` swings 0.790–0.942.
