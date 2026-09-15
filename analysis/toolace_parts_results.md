# toolace in four parts: results

Four 600-row no-shot specialists, one per part of `toolace_balanced`, plus the four pooled.
deepseek-v4-pro throughout; 45 fits in `scripts/highstakes_toolace_parts.csv`, written by
[`scripts/fit_toolace_parts.py`](../scripts/fit_toolace_parts.py). The parts and the prompts are in
[`toolace_part_prompts.md`](toolace_part_prompts.md).

**Protocol — NO base data anywhere.** gemma-3-27b-it L32, `linear_then_softmax`, single probe
(architecture inherited from `probe_iter0.pkl`, weights never used); training rows are the drawn
generated rows alone, class-balanced; dev `dev_samples/highstakes_500`; eval the full highstakes
splits; transforms on; seed 42. Each set: one fit on all 600 rows plus 8 seeded draws of 540.
The pooled arm draws 135 rows (at n=540) or 150 (at n=600) from EACH of the four sets, so it
trains on the same number of rows as one specialist. This is the `base='none'` protocol of
`scripts/highstakes_gen90_dev500.csv`.

## The parts

| part | eval rows | high | low |
| --- | --- | --- | --- |
| `toolace_ops` — operational tasks, code-style functions | 290 | 191 | 99 |
| `toolace_lookup` — information lookups, API-marketplace functions | 245 | 54 | 191 |
| `toolace_finance` — financial data | 90 | 64 | 26 |
| `toolace_roledef` — "Role definition … Historical dialog" wrapper | 109 | 58 | 51 |

## Result

Mean ± sd over the eight n=540 draws. `part_*` is the AUROC over that part's toolace rows, from
the same scores as the `toolace` column.

| training set | toolace | ops | lookup | finance | roledef | mean of 4 eval splits |
| --- | --- | --- | --- | --- | --- | --- |
| ops specialist | 0.8447 ±0.006 | 0.8818 | 0.7465 | 0.6224 | **0.9143** | 0.8656 ±0.019 |
| lookup specialist | 0.8447 ±0.006 | 0.8608 | 0.7594 | 0.6870 | 0.9015 | 0.7934 ±0.018 |
| finance specialist | 0.7580 ±0.005 | 0.7429 | 0.6647 | 0.7223 | 0.7440 | 0.8120 ±0.007 |
| roledef specialist | 0.8469 ±0.010 | 0.8294 | **0.7786** | 0.6226 | 0.8882 | 0.7846 ±0.015 |
| **pooled, 4 × 135** | **0.8604 ±0.012** | **0.9024** | 0.7714 | **0.7597** | 0.8876 | **0.8748 ±0.013** |

The other eval splits, same fits:

| training set | anthropic_hh | mt | mts |
| --- | --- | --- | --- |
| ops | 0.8093 | 0.8661 | **0.9421** |
| lookup | 0.7880 | 0.7011 | 0.8399 |
| finance | 0.7652 | 0.8735 | 0.8512 |
| roledef | 0.7194 | 0.7153 | 0.8569 |
| pooled | **0.8308** | **0.8917** | 0.9164 |

Pooled against the best specialist on each column (difference of draw means, ± its standard error):

| column | best specialist | pooled − best |
| --- | --- | --- |
| toolace | roledef | **+0.014 ±0.005** |
| ops part | ops | **+0.021 ±0.004** |
| lookup part | roledef | −0.007 ±0.006 |
| finance part | finance | **+0.037 ±0.010** |
| roledef part | ops | **−0.027 ±0.005** |
| mean of 4 eval splits | ops | +0.009 ±0.008 |

## Reading

**1. Every arm but finance clears the earlier toolace specialists by ~0.14.** Under this same
no-base protocol the whole-split deepseek specialists scored 0.6931 (measured-shape, no shots) and
0.7113 (measured-shape, dev anchor) on toolace; three of the four part specialists land at
0.845–0.847 and the pool at 0.860. For scale, the best deepseek toolace result on record WITH a
50-row base was the `--kind` arm at 0.8111. What changed is the prompt, not the protocol: the
label defined as the stakes of the user's situation (the earlier spec taught acts-vs-retrieves,
which the gpt-4o ratings do not follow) and the ToolACE scaffold reproduced verbatim.

**2. Pooling is the best single training set.** At the same 540 rows it is top on toolace
(+0.014 over the best specialist), top on two of four parts, top on the four-split mean, and the
only arm without a weak split. The two parts it loses are lost to a DIFFERENT part's specialist
rather than to their own.

**3. Specialisation mostly does not land on its own part.** Only two of four specialists are best
on their own part (ops, finance). The lookup part is served best by the roledef set, and the
roledef part by the ops set — roledef's distinct wrapper format bought nothing on roledef rows.
The pooled result is consistent with the parts sharing most of what the probe needs, with each set
adding coverage the others lack rather than a part-specific skill.

**4. The finance part is where pooling pays most** (+0.037 over the finance specialist, which
trained on four times as much finance data), and the finance specialist is the only one that
hurts toolace overall (0.758): 90 of 734 rows, specialised on, costs more elsewhere than it gains.

**5. Narrow sets cost the other splits.** lookup and roledef drop `mt_balanced` to ~0.70 and the
four-split mean to ~0.79; ops and the pool keep it at 0.87–0.87.

## Caveats

- **Test-set information in the prompts.** The parts were cut from, and the specs measured on, the
  eval split itself (as every split-targeted arm on these branches was). Part scores are optimistic
  in that sense; the comparison between arms is fair, since all share it.
- **Small parts.** finance has 26 low-stakes rows and lookup 54 high-stakes rows, so those part
  AUROCs are the noisiest columns (sd up to 0.027).
- **One generation per set, one generator.** Draw-to-draw sd measures resampling of fixed sets, not
  generator variance.
- **Single full-set fits are noisy.** The pool's n=600 fit scored 0.841 on toolace against 0.860
  for its draw mean; read the draws, not the single fit.
- **Generation defect, fixed before any fit.** The first pass wrote ~220 plain-chat rows into
  lookup and roledef each; the sets were topped up under a scaffold check and the fit runner
  refuses to fit a set with any row lacking it (see `ad60ea0a`).
- **No reference arms were refit** under this protocol with a per-part breakdown; the 0.69–0.71
  comparison is whole-split only.
