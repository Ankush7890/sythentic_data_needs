# toolace sub-parts: results

Round 2 of `hard_split_experiments`. `toolace_lookup` and `toolace_finance` each cut in two
([`toolace_subpart_prompts.md`](toolace_subpart_prompts.md)); one **300-row** no-shot specialist per
sub-part, deepseek-v4-pro. 36 fits in `scripts/highstakes_toolace_subparts.csv`.

**Protocol — identical to round 1 ([`toolace_parts_results.md`](toolace_parts_results.md)) except the
size:** NO base data, gemma-3-27b-it L32 `linear_then_softmax`, dev `dev_samples/highstakes_500`, full
eval splits, seed 42; one fit on all 300 rows plus 8 seeded class-balanced draws of 270. Every fit is
scored on the four eval splits, toolace's four parts (`part_*`) and the four sub-parts (`sub_*`).

## The sub-parts

| sub-part | eval rows | high | low |
| --- | --- | --- | --- |
| `lookup_media` — sports, music, video, social, news, games | 106 | 13 | 93 |
| `lookup_utility` — documents, web/domains, location, drug info, business data | 139 | 41 | 98 |
| `finance_markets` — stocks, earnings, ratings, market news | 54 | 34 | 20 |
| `finance_money` — crypto, forex, loans, mortgages, retirement, tax, banking | 36 | 30 | 6 |

## Result

Mean over the eight n=270 draws (sd 0.005–0.035; per-column sd in the CSV). **Bold** = best of the four.

| scored on | media | utility | markets | money |
| --- | --- | --- | --- | --- |
| **toolace** | 0.754 | **0.859** | 0.799 | 0.770 |
| part ops | 0.720 | **0.868** | 0.813 | 0.744 |
| part lookup | 0.691 | **0.807** | 0.698 | 0.660 |
| part finance | 0.681 | 0.681 | 0.740 | **0.778** |
| part roledef | 0.758 | **0.893** | 0.829 | 0.760 |
| sub `lookup_media` | 0.697 | **0.761** | 0.632 | 0.600 |
| sub `lookup_utility` | 0.645 | **0.815** | 0.692 | 0.657 |
| sub `finance_markets` | 0.677 | 0.793 | 0.791 | **0.809** |
| sub `finance_money` | 0.641 | 0.616 | 0.594 | **0.690** |
| mean of 4 eval splits | 0.765 | 0.760 | 0.827 | **0.831** |
| anthropic_hh | **0.799** | 0.645 | 0.690 | 0.740 |
| mt | 0.713 | 0.727 | **0.928** | 0.890 |
| mts | 0.794 | 0.810 | 0.892 | **0.925** |

Round 1 for reference (same protocol, **540-row** draws, parts only — its probes were not kept, so it has
no sub-part scores):

| round 1 | toolace | ops | lookup | finance | roledef | mean of 4 |
| --- | --- | --- | --- | --- | --- | --- |
| lookup specialist | 0.845 | 0.861 | 0.759 | 0.687 | 0.902 | 0.793 |
| finance specialist | 0.758 | 0.743 | 0.665 | 0.722 | 0.744 | 0.812 |
| pooled 4 × 135 | 0.860 | 0.902 | 0.771 | 0.760 | 0.888 | 0.875 |

## Reading

**1. `lookup_utility` is the best single specialist of either round — on half the data.** 0.859 on
toolace from 270 rows, above every round-1 specialist at 540 (best 0.847) and level with the round-1
pool (0.860). It is best on three of four parts and on both lookup sub-parts, including the media rows
it was not written for (0.761 against the media specialist's own 0.697). On the lookup part it beats
round 1's whole-lookup specialist by 0.048 (0.807 vs 0.759).

**2. The two finance sub-part specialists beat round 1's finance specialist on the finance part** —
money 0.778 and markets 0.740 against 0.722 from twice the rows — and they are the only arms here that
keep the other eval splits up (four-split mean 0.83; `mt_balanced` 0.89–0.93).

**3. Only two of four sub-part specialists win their own sub-part, and those two are the finance ones.**
`finance_money` tops `finance_money` (0.690) and also `finance_markets` (0.809, just above markets'
own 0.791). The lookup specialists split the other way: utility wins its own sub-part AND the media one.

**4. `lookup_media` is the weak set everywhere, its own rows included.** Its real rows are 88% low-stakes
(13 of 106 high), so all 150 generated high-stakes rows describe consequential media or sports
situations that the real sub-part barely contains. Of the four arms it is the only one to keep
`anthropic_hh` high (0.799) — open-ended chat is closer to media lookups than to utility tools.

**5. Specialising trades toolace against the other splits.** The two lookup sets drop the four-split
mean to ~0.76 (`anthropic_hh` 0.645 for utility, `mt` ~0.72 for both); the two finance sets keep it at
0.83. Round 1's pool is still the only arm above 0.85 on both toolace and the four-split mean.

## Caveats

- **Tiny sub-parts.** `finance_money` has 6 low-stakes eval rows and `lookup_media` 13 high-stakes rows;
  AUROC on those two columns moves in large steps and their sds are the largest (up to 0.035).
- **Label noise in toolace.** Some gpt-4o rationales describe a different conversation from the row they
  label (see the prompts doc). It caps every column and is worst on the small sub-parts.
- **Round 1 vs round 2 is not size-matched.** Round-2 sets are 300 rows (draws of 270), round-1 sets 600
  (draws of 540); a round-2 arm matching or beating a round-1 one is therefore conservative.
- **No round-1 sub-part scores.** Round-1 probes were deleted after scoring; comparing on the sub-part
  columns would need those arms refit (no extraction, ~30 min).
- **Test-set information in the prompts**, as in round 1: the cuts and specs were measured on the eval split.
- **Provider behaviour during generation.** `lookup_utility` and `finance_money` had 115 and 145 off-scaffold
  rows rejected by the `require` check, and `finance_markets` had 100+ calls return empty content; every
  final set passed the gate (150/150, scaffold on every row).
