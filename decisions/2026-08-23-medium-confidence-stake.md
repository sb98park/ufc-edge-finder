# Medium Confidence: stake cut to 2U, and a pre-registered test

**Decided 2026-08-23. Effective 2026-08-25. Test resolves at n = 75 Medium picks.**

This file exists so the decision is made *before* the data arrives. After two more
cards the temptation will be to read whatever happened as confirmation of whichever
answer is convenient. Writing the rule down now is what stops that.

---

## What was observed

Seven cards, 88 graded picks, priced at the real market odds published at pick time.

| tier | n | win | model said | market implied | units | ROI |
|---|---|---|---|---|---|---|
| Lock of the Week | 9 | 100.0% | 83.6% | 69.5% | +46.96 | +52.2% |
| High Confidence | 11 | 90.9% | 80.3% | 73.2% | +11.80 | +21.5% |
| **Medium Confidence** | **35** | **57.1%** | **66.7%** | **60.8%** | **−7.01** | **−6.7%** |
| Low Confidence | 33 | 72.7% | 54.6% | 56.0% | +11.69 | +35.4% |

Calibration, all 88 picks:

| model says | n | actual | gap |
|---|---|---|---|
| 45–55% | 16 | 68.8% | +16.7pp |
| 55–60% | 17 | 76.5% | +19.4pp |
| **60–65%** | **14** | **50.0%** | **−12.6pp** |
| **65–70%** | **13** | **53.8%** | **−14.1pp** |
| 70–75% | 8 | 75.0% | +3.0pp |
| 75–85% | 14 | 92.9% | +13.4pp |
| 85–100% | 6 | 100% | +12.7pp |

Underconfident everywhere except one window in the middle, and the whole of Medium's
loss sits in it. The 60–65% slice alone is −9.83U over 14 picks; Medium at 0.65 and
above is +2.82U.

Bucketing every pick by model edge over the market instead, all five buckets are
profitable (+8.2%, +16.1%, +44.6%, +16.5%, +45.2%). The directional signal is not the
problem.

## What was NOT concluded

**The dip is not statistically significant, and nothing here treats it as though it
were.**

For the 60–70% band, n = 27 with 14 wins:

- P(≤ 14 wins | the model's own probabilities are correct) = **10.7%**
- P(≤ 14 wins | the *market's* implied probabilities are correct) = **34.8%**

The 60–65 slice alone: 23.9% and 46.9%. Medium's 3/11 against market underdogs, which
looked like a smoking gun: 42.7%. Medium's −7.01U is inside ordinary variance for 35
picks at a ~61% break-even, and may well revert.

That cuts both ways: at these sample sizes Low's +35.4%, High's 95% and 9/9 on Locks
are not established either. No tier structure should be rebuilt on any of them.

## What was decided, and why

**Medium's stake goes 3U → 2U, effective 2026-08-25, forward only.**

The defensible criticism is not "this tier loses money" — that is not known. It is that
the 5/3/1 ladder was set a priori from the labels' names and never from evidence, and
that Medium was carrying **three times** Low's stake while underperforming it on every
measure, at **37% of all units risked and 40% of all picks.**

Cutting to 2U removes a multiple that had no basis in the first place, keeps the ladder
monotonic with the model's stated confidence, and leaves the tier standing so the test
below can run.

Forward only is not negotiable. Restaking history would rewrite every published figure
on the site — and note the direction it would move them: the retroactive version of this
change reports **+65.77U instead of +63.44U**. A stake edit that quietly improves the
track record is precisely the thing this product cannot be caught doing. Enforced by
`scripts/check_stake_schedule.py`, which fails CI if any published figure moves.

## What was rejected

**Folding Medium into Low.** It averages the second-best cohort on the board into the
worst (merged: +13.8% ROI, against Low's +35.4% standalone). Almost all of the apparent
benefit is the stake change, not the merge — Medium alone restaked to 1U goes from
−7.01U to −2.33U, which is essentially the entire gain. It makes the question
permanently unanswerable. And the ledger holds 36 rows stamped `Medium Confidence` in an
append-only file, so a merge means either rewriting published history or carrying a
display mapping forever.

## The test

**At n = 75 graded Medium Confidence picks**, evaluated once, no peeking-and-deciding
in between:

| outcome at n = 75 | action |
|---|---|
| cumulative units **> 0** | restore 3U; the dip was variance |
| units **≤ 0** but ROI **> −5%** | hold at 2U, re-test at n = 125 |
| ROI **≤ −5%** | cut to 1U and stop publishing Medium as an actionable tier — show the pick, drop the stake |
| ROI **≤ −5%** *and* the 60–70% calibration gap is still **worse than −10pp** | the band is a model defect, not variance: fix the model, do not re-tier around it |

Measured on Medium picks **graded from 2026-08-25 onward** — the cohort that ran under
the new stake — reported alongside the full-history figure so neither can be
cherry-picked.

`compute_track_record()` prints a `[track_record] PRE-REGISTERED TEST DUE` line once the
threshold is reached, so this does not depend on anyone remembering.

## What is deliberately not being changed

- **The model.** No re-fit, no re-tuning of the 60–70% band. At p = 0.11 that would be
  fitting to noise, and it would also destroy the test.
- **The tier boundaries.** Raising Medium's floor from 0.60 to 0.65 would have turned
  −7.01U into +2.82U on the same data. That is fitting to n = 14.
- **The Medium-vs-market-underdog rule** (3/11, model said 65.4%, market implied 34.9%).
  The most interesting pattern in the data and the least supported. Revisit at n ≥ 30.
