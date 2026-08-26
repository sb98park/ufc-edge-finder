# Parlays: graded, published, not staked

**Decided 2026-08-26, before a single slip has been settled.** That timing is
the point. Written afterwards, a threshold gets chosen to fit whatever the
first few cards happened to do.

## What was true when this was written

- The parlay ledger held **193 slips across two events and zero grading
  fields**. Nothing had ever recorded whether a published slip landed.
- The settlement rules already existed, in JavaScript, in `templates/site.html`
  (`gradeCondition`, `gradeLeg`, `slipState`). They ran live during a card and
  the answer was discarded at the final bell.
- The builder re-picked on every render. One card (Hernandez vs. Rodrigues)
  accumulated **51 distinct bankroll slips and 93 lotto slips**, twenty of the
  bankroll ones alive for a single five-minute build.
- A one-off grade of that card's slips returned **bankroll −12.04U on 44
  risked, lotto −58.77U on 72, moonshot −23.00U on 23**. The most-shown slip in
  each tier went 0 for 3. That is one card and proves nothing; it is recorded
  here so nobody later remembers it as better than it was.

## The decisions

1. **Slips are graded and the record is published. They are not staked.**
   Scored at **1U flat**, always stated as a hypothetical, never folded into
   the plays ledger or the bankroll. Precedent: props have 50 settled quotes
   against 894 recorded and still get no stake.

2. **Only the pinned slip is graded** (`src/parlay_pin.py`). One slip per card
   per tier. The pre-pin churn stays in the ledger, ungraded, forever.

3. **No backfill.** The past is not reopened.

4. **The graded record is FREE.** The upcoming slips stay member-only; whether
   a published read paid is evidence, and evidence on this site is public.
   Same footing as the ledger and the closing-line figure.

5. **It lives under "All calls", not "Bets"**, as its own block, and **outside
   that tab's count**. `All calls 11-3` has meant moneyline picks since the
   beginning and a five-leg slip is not a call in that sense.

6. **The copy is "published, not played"** -- the phrase the site already uses.
   Not "calibrating", not "under construction". Those promise this will become
   staked, and it may well not.

## The bar for staking them

**Not before 20 graded slips per tier, and not before 10 cards.**

At that point, the tier is staked only if **1U flat is positive over the whole
sample**. Not "close to break-even", not "positive excluding one bad card".

If it is negative, the tier stays published and unstaked, and this test does
not run again until another 20 slips have settled. Re-reading it card by card
is precisely what pre-registration exists to prevent.

### The thing most likely to sink it, named in advance

**AMENDED 2026-08-26, same day, before any slip was graded.** The original
text of this section named *leg correlation* as the likely culprit and cited
a 1.14 calibration ratio from `parlay_builder`'s docstring. Both were wrong,
and they were tested rather than argued about. What follows replaces them.

`scripts/replay_parlay_construction.py` drives the shipped builders over 400
real cards with exact settlement. Ratio is published hit rate over realised;
1.00 is honest.

| sigma | tier | slips | published | realised | ratio |
|---|---|---|---|---|---|
| 0.00 | bankroll | 400 | 46.7% | 50.7% | **0.92** |
| 0.00 | lotto | 362 | 7.8% | 9.9% | **0.78** |
| 0.83 | bankroll | 400 | 51.9% | 48.0% | **1.08** |
| 0.83 | lotto | 398 | 12.8% | 8.5% | **1.50** |

**At sigma = 0 the construction UNDERSTATES.** A perfectly calibrated model
produces slips that beat their own published hit rate on both tiers.
Multiplying marginals is conservative here, not optimistic -- so the
correlation penalty this file originally predicted would have pushed an
already-conservative number further in the wrong direction.

**The overstatement is entirely model error being selected for.** Lotto moves
0.78 -> 1.50 the moment sigma is realistic. Ranking candidates by combined
probability inside a payout band is algebraically ranking by the model's
claimed edge, so the search actively hunts the legs where the model is most
optimistic relative to truth. Lotto searches the largest space at the longest
prices and is hit hardest; bankroll searches a small one and stays near 1.00.

**Family concentration was tested too, and is not it.** Splitting the same
slips by whether every leg came from one market family:

| sigma 0.83 | tier | slips | ratio |
|---|---|---|---|
| all one family | bankroll | 186 | 1.03 |
| mixed | bankroll | 214 | 1.13 |
| all one family | lotto | 63 | 1.38 |
| mixed | lotto | 335 | 1.52 |

Concentrated slips are **better** calibrated on both tiers, and at sigma = 0
the two buckets are indistinguishable. Forcing diversity -- the second fix
proposed and dropped -- would have made calibration worse. The direction is
itself evidence for the selection story: mixing families opens a larger
combination space, and a larger space means more opportunity to select on
noise.

**So the expected finding for the forward record is the opposite of what this
file first said.** With the shipped 0.30 blend weight, the lotto tier's
published probability is roughly 1.5x its realised hit rate, and the fix
belongs in the objective function -- how much the model is shrunk toward the
market *before* the search ranks on it -- not in a dependence model.

If the blend weight changes on the strength of that, the slip count for this
test restarts. A record built on one construction does not describe another.

## AMENDED, same day: the lotto tier is retired

The blend sweep that followed the finding above settled it. Over 400 cards at
sigma = 0.83:

| ranking weight | 0.00 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|
| **lotto** | 0.96 | 1.22 | 1.48 | **1.50** | 2.75 |
| **bankroll** | 0.94 | 1.03 | 1.04 | **1.08** | 1.25 |

Lotto is honest only at weight 0.00, where the model plays no part in
choosing the slip. At every weight where the model contributes, the published
probability is 22-50% above what happens. **The tier is deleted**, and its
numbers and the argument for its deletion are recorded in
`src/parlay_builder.py` so it does not get rebuilt from the same reasoning.

Two consequences for this file:

- **The parlay ranking weight is now 0.10**, its own constant rather than the
  site-wide 0.30 that sizes single bets. Sizing one bet has nothing hunting
  its error; a search over thousands of combinations does.
- **The staking test above now applies to bankroll alone**, and its count
  starts from the first card built under the new weight. A record built on
  one construction does not describe another -- the rule this file already
  set for exactly this case.

Nothing was staked at any point, so nothing needs unwinding. The pinned lotto
slip for Nurmagomedov vs. Song was dropped before the card, and the ledger
keeps every historical lotto row untouched and ungraded.

## What would reverse this

A tier clearing the bar above. Or evidence that the correlation penalty, once
built, changes construction enough that the earlier record no longer describes
what the builder now does -- in which case the count restarts rather than the
threshold moving.
