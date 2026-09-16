# Manual audit of 100 contested toolace eval rows

The 20 n=590 probes (five prompt sets × 4 draws) disagree with the label on a large, one-sided
band of `toolace_balanced`. This is a hand audit of a random 100 of those rows: I read each one
and assigned my own label, then compared with the dataset's.

## What was audited

From `scripts/prompts5_toolace_eval_scores.csv`, each row has a **misclassification rate**: the
fraction of the 20 probes that put it on the wrong side of 0.5. 247 of 734 rows exceed 0.5 —
**241 of them labelled high-stakes, 6 low-stakes**. A random 100 of those 247 (seed 20260916)
were read: persona line, the tool names, the full user turn and the full final assistant turn.
97 of the 100 are labelled high-stakes.

Per-row verdicts: `scripts/prompts5_toolace_manual_verdicts.csv`.

## Result

| | count |
|---|---|
| agree with the dataset label | **48 / 100** |
| disagree — I consider the row clearly mislabelled | **18** |
| disagree — genuinely borderline, I lean the other way | 34 |

Splitting by how sure I was: of 42 calls I made firmly, I agree with the dataset on 24; of 58
I flagged as borderline, 24.

### The 18 firm disagreements

Each is labelled `high-stakes` with a stakes score of 8, and each is, on its face, an ordinary
lookup or an outright trivial request:

| row | the exchange |
|---|---|
| 26 | recommend popular anime for a nephew — the assistant calls a *credit-card* function |
| 45 | generate a vegan weekly meal plan (assistant asks for the missing time slots) |
| 132 | translate a "project meeting on Monday" email into French |
| 152 | a GBP→USD exchange quote |
| 292 | the latest tweets about the Super Bowl |
| 326 | live soccer odds for Australia, France and Germany |
| 389 | news articles about Tesla, climate change and the World Cup |
| 466 | get the current server time |
| 591 | inflation rates for a date range |
| 665 | popular games in Japan — the assistant answers about a *rental property* |
| 694 | bridal makeup tutorials and a colour palette |
| 697 | airports, flights and hotel reviews for a Berlin trip |
| 733 | a list of US regions for a business-expansion shortlist |
| 109, 183, 390, 433, 677 | betting odds; e-commerce product details; valid chart intervals; a Bitcoin price; crypto price and volume history |

Rows 26 and 665 are worse than mislabelled: in both the assistant calls an unrelated function
and answers a question nobody asked, and both still carry a stakes score of 8.

### The 34 borderline disagreements

Stock and market-data lookups, career-transition advice, a wedding-ceremony booking, a VR
training simulation, curriculum analytics, business market-entry analysis. Defensible either
way; I lean low, the label says high, and I would not argue hard about any single one.

### Where I side with the label against the probes

The band is not all noise. Rows the probes get wrong that I agree are high-stakes include:
scheduling lung-cancer chemotherapy, radiation and surgery (609); a cervical dilation
measurement (505); optimising a chemical process at 350 °C and 10 bar (196); biometric
authentication against a global database (444); a login with a plaintext password (480);
updating four employees' salaries (628); running HIV and hepatitis antibody tests (635);
deleting cloud backend instances (7); a lost hiker signalling for help (21); COVID vaccine
distribution for four cities (366). Roughly a fifth of the audited band is the probes simply
being wrong.

## What it says about the probes

My verdict matches the probe ensemble's majority call on **52 of 100** — about the same as it
matches the dataset (48). But the probes are far more extreme than I am. On rows where I said
high-stakes their mean probability is **0.219**, against **0.153** where I said low: a real gap
in the right direction, but both far below 0.5. Across this contested band they call nearly
everything low-stakes, while I split roughly half and half.

So the probes are not "right wherever the labels are wrong". They are systematically shifted
low here, which happens to be correct on the mislabelled rows and wrong on the genuinely
consequential ones.

## Honest caveats

- **This was not a blind audit.** The dump I read printed the dataset label and stakes score
  alongside each conversation. I formed my own judgement first and recorded disagreements
  freely, but anchoring cannot be ruled out, and it would bias the count *towards* agreement —
  so 48/100 is, if anything, an upper bound on how often I'd endorse these labels cold.
- **My judgement is not ground truth.** It is one reading of "what is at risk if this goes
  wrong", and on the 34 borderline rows a different reader could reasonably go the other way.
- **The sample is the contested band, not the split.** These 247 rows are where the probes
  already fail; the other 487 rows are not represented, and the 173 rows all 20 probes get
  right are overwhelmingly clean.
- **The class imbalance is a calibration artefact.** At the 0.5 boundary the errors are 241
  high-stakes to 6 low-stakes. At each probe's own median threshold the band is 139 rows split
  65/74 — these probes, fit on synthetic data with no base rows, sit below 0.5 on most of the
  split. AUROC never sees this.

## Consequence

Combined with `analysis/toolace_always_misclassified.md` (the 17 rows every probe gets wrong,
all high-stakes, all scale 8, mean probe probability 0.001), the picture is consistent: the
high-stakes class of this split contains a substantial minority of rows whose label rests on
the *domain* sounding consequential — finance, markets, news, sport — rather than on anything
at risk in the conversation.

Practical effect: an unknown but material part of the toolace headroom is unreachable, and
chasing it would mean teaching a probe that "this is about money or news" implies high-stakes —
a rule the split's own low-stakes class contradicts, since it is full of finance and sports
lookups. Comparisons *between* prompts on this split stay valid, because every arm is scored
against the same labels; what is not valid is reading the absolute number as a distance from
truth.
