# The toolace eval rows every probe gets wrong

Twenty probes — the five prompt-variant sets × 4 draws at n=590, no base data — scored on all
734 rows of `eval_sets/highstakes/toolace_balanced.jsonl`, to ask whether the residual error is
spread over the split or concentrated on particular rows.

## How it was done

`scripts/prompts5_toolace_agreement.py` refits the 20 draws (the size curve unlinks its
probes, so they had to be rebuilt; `draw_subset` is seeded on `(stem, n, draw)`, so these are
the same probes behind the n=590 rows of `scripts/highstakes_toolace_prompts5_sizecurve.csv`),
keeps them in `probes/prompts5_toolace_n590/`, and writes one probability per (probe, row) to
`scripts/prompts5_toolace_eval_scores.csv`. `scripts/prompts5_toolace_hard_rows.py` does the
analysis.

A row counts as misclassified under two rules, and the rows below fail **both**:

- **0.5**, the probe's own decision boundary — 90 rows are wrong for all 20 probes;
- **each probe's median score on this split**, the balanced-split threshold (the eval is
  exactly 367/367), which removes any constant calibration shift between probes — 27 rows.

**17 of 734 rows are wrong under both rules for all 20 probes.** For contrast, 173 rows are
right under both rules for all 20.

## What the 17 have in common

Every one of them is the same kind of error:

| | the 17 always wrong | other 350 high-stakes | 367 low-stakes |
|---|---|---|---|
| label | **all `high-stakes`** | high-stakes | low-stakes |
| stakes score | **all 8** (none 9 or 10) | 8–10 | 1–3 |
| labeller confidence | all 9 | median 9 | median 9 |
| mean probe probability | **0.001** | 0.369 | 0.041 |
| rationale mentions safety / health / legal / harm | **12%** | 55% | — |
| median characters | 3316 | 2746 | — |
| median functions listed | 4 | 2 | — |

**They are not rows the probes find hard. They are rows the probes are certain about.** Mean
probability 0.001, on a scale where 1 is high-stakes: the ensemble calls these *more*
confidently low-stakes than it calls the actual low-stakes rows (0.041). Nothing about them is
borderline from the probe's side.

**What they are, in content:** a sepia-tone image conversion, a weather forecast for a camping
trip, the latest tweets about the Super Bowl, news articles about Tesla and the World Cup,
property listings in San Francisco, exchange rates for USD and gold, historical crypto prices,
live betting odds, e-commerce product details, a Broadway performance schedule, a public-opinion
lookup, a travel itinerary. Read them cold and they are ordinary lookups — which is exactly
what twenty independently-trained probes say.

**The label rests on the rationale, not on the exchange.** These rows carry a stakes score of 8
justified by reasoning about the *domain's* general importance rather than anything at risk in
the conversation: the Super Bowl "involves advertising revenues and broadcasting rights";
property listings are "a significant financial decision"; news about Tesla covers "topics of
significant global interest". Only 2 of the 17 rationales mention safety, health, legal
exposure or harm, against 55% of the other high-stakes rows. Strip that abstract economic
framing and there is no risk in the text the probe reads.

**Two rationales describe a different conversation altogether.** Row 546 is a request for
climate-change news articles, and its rationale reads "the user is preparing for a critical
work presentation that determines the team's success, impacting job security". Row 713 is a
photo being converted to sepia, and its rationale reads "the user asked about the company's
policy on sending sensitive emails to external entities without encryption". Measured as word
overlap between rationale and conversation, those two score 0.00 and 0.05 against 0.29 for
other high-stakes rows. Those are labelling failures, not hard cases.

## What this is not

- **Not the read-versus-write distinction.** 80% of the 17 end in a read-only call list,
  against 85% of the other high-stakes rows and 91% of the low-stakes rows. The split's
  high-stakes class is mostly read-only anyway; that is the finding the whole prompt study rests
  on, and it does not separate these rows.
- **Not simply "rationale lacks safety words".** 158 other high-stakes rows also have no such
  word in their rationale, and the probes put them at 0.254, not 0.001. The 17 are the extreme
  tail, not a category defined by that feature.
- **Not a big share of the error.** 17 rows is 2.3% of the split — about 0.02 of AUROC if every
  one were fixed. The other 90-odd% of the residual error is spread over rows where the probes
  disagree with each other.

## What it means

The ceiling on this split is partly **label noise of a specific kind**: rows where the
scale-labelling model reasoned from the subject area ("finance", "markets", "news", "a major
event") to a score of 8, on conversations whose text contains nothing at stake. Twenty probes
trained on five deliberately different synthetic sets converge on calling them low-stakes, from
600 rows each, with near-total confidence and no exposure to these rows.

That bounds what better training data can buy here. No prompt variation will move these 17,
because agreeing with them means learning that "this is about money or news, therefore it is
high-stakes" — a rule the rest of the split contradicts, since its low-stakes class is full of
finance and sports lookups too.

The practical read: when comparing prompts on toolace, treat about 0.02 of AUROC as unreachable,
and do not tune against it. If the split is ever regenerated, rows whose rationale names no
concrete risk, and the two whose rationale does not match the conversation, are the ones to
re-label first.
