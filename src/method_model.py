"""
Fight-level method-of-victory probabilities.

DIRECT FIT, NOT CHAINED. The previous version predicted a per-ROUND hazard
and multiplied it across a fight's scheduled rounds. Two failures compounded:
any per-round overstatement multiplied, and P(decision) was defined as
"survived every round" -- a product of five survival terms, the most fragile
quantity in the construction. On a real five-round main event it produced
65.0% submission and 8.7% decision against holdout base rates of 16.5% and
52.4%. Decision was wrong by a factor of six.

It had passed its earlier validation because that only scored KO and SUB via
Brier. Nobody scored the DECISION leg, and nothing compared mean prediction to
observed frequency -- the check that makes an 8.7% decision rate impossible to
miss. research_method_fightlevel.py now runs exactly that check.

VALIDATED (research_method_fightlevel.py), frozen 2019+ holdout, n=1743:

    holdout log-loss    base rates 1.0047   this model 0.9559

    calibration          base      model     observed
      KO/TKO            +3.3%     +2.4%       31.2%
      SUB               +2.0%     +1.1%       16.5%
      DEC               -5.3%     -3.5%       52.4%

    five-round subgroup (n=267)   predicted   actual
      KO/TKO                         36.6%    37.8%
      SUB                            17.5%    14.6%
      DEC                            45.9%    47.6%

Better calibrated than the base rates on EVERY class, better on log-loss, and
within 3pp on the five-round subgroup -- which is where the previous version
was off by 13pp and where the liquid markets are.

The residual -3.5% on decisions is era drift, not model error: decisions were
more common in the holdout (52.4%) than in training (47.1%), and nothing fit
on the training period can recover that. Expect decision probabilities to read
slightly low.

No MAIN_EVENT_SHRINK any more -- that existed to damp the chaining, and there
is no chaining left to damp.

Refit with research_method_fightlevel.py and paste new numbers below.
"""

import math

# NO `scheduled` FEATURE. Including it made the model 13.0% miscalibrated on
# five-round fights -- it learned "longer fight, more finishes" from 211
# training examples, while five-rounders actually go to decision 47.6% of the
# time against 53.3% for three-rounders. Dropping it cut the five-round error
# to 2.9% AND improved log-loss (0.9610 -> 0.9559): the feature was doing
# active harm, not adding signal.
# Fitting the two lengths separately was tested too and came out WORSE
# (13.2%) -- 211 fights is not enough to fit a second model on.
FEATURES = ["ko_press", "sub_press", "ko_rate_sum",
            "sub_rate_sum", "durability", "elo_gap"]

# Rows are outcomes in order: KO/TKO, SUB, DEC.
COEF = [
    [0.869430, -0.099895, 0.620604, -0.324009, 0.558473, 0.253277],
    [-0.751140, 1.501577, -0.064537, 0.777400, 0.226345, -0.233184],
    [-0.118290, -1.401682, -0.556067, -0.453391, -0.784818, -0.020093],
]
INTERCEPT = [-0.285880, -0.743323, 1.029203]

KO, SUB, DEC = 0, 1, 2


def method_probabilities(ko_press: float, sub_press: float, ko_rate_sum: float,
                         sub_rate_sum: float, durability: float, elo_gap: float,
                         scheduled_rounds: int = 3) -> dict | None:
    """
    P(KO/TKO), P(submission), P(decision) for a fight. Sums to 1 by
    construction -- it's one softmax over three outcomes, not three estimates
    reconciled after the fact.

    Returns None when an input is missing rather than substituting a default:
    a fabricated feature yields a confident-looking number with nothing behind
    it, and every caller already has a fallback.
    """
    vals = (ko_press, sub_press, ko_rate_sum, sub_rate_sum, durability, elo_gap)
    if any(v is None for v in vals):
        return None

    # scheduled_rounds is accepted but DELIBERATELY UNUSED -- see the note on
    # FEATURES. Kept in the signature so callers don't need changing, and so
    # removing it can't silently look like an oversight.
    x = [float(ko_press), float(sub_press),
         float(ko_rate_sum), float(sub_rate_sum), float(durability), float(elo_gap)]
    z = [INTERCEPT[k] + sum(c * v for c, v in zip(COEF[k], x)) for k in range(3)]
    m = max(z)                                  # subtract the max for stability
    e = [math.exp(v - m) for v in z]
    total = sum(e)
    return {"ko": e[KO] / total, "sub": e[SUB] / total, "decision": e[DEC] / total}
