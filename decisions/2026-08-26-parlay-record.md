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

**Leg correlation.** The two slips pinned for Nurmagomedov vs. Song are made
of *nothing but* "Over 1.5 rounds" legs -- two on the bankroll, five on the
lotto. Those are not independent: a card where several fights end early kills
every one of them at once. The builder multiplies marginal probabilities,
which cannot represent that dependence and therefore overstates the slip.

`src/parlay_builder.py` already half-says this -- measured against a perfect
model over 250 cards, the lotto tier publishes a hit rate 1.14x its realised
one, and the note is explicit that shrinking toward the market does not touch
the residual.

So the expected finding is that **graded results come in worse than the
model's own combined probability**, and the size of that gap is the most
useful thing this record will produce. If the gap is the whole story, the fix
is in construction -- penalising shared failure modes across legs -- not in
staking.

## What would reverse this

A tier clearing the bar above. Or evidence that the correlation penalty, once
built, changes construction enough that the earlier record no longer describes
what the builder now does -- in which case the count restarts rather than the
threshold moving.
