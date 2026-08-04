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

# The 1st/99th percentile of each feature ACROSS THE TRAINING SET
# (research_method_fightlevel.py prints these). A logistic model extrapolates
# without bound, so an input beyond these produces a confident number built on
# no evidence -- which is exactly how a denominator mismatch turned into a
# 60% submission probability the model never produced once in 1,743 holdout
# fights. Clipping bounds the damage and the warning names the cause.
TRAINING_RANGE = {
    "ko_press": (0.000, 0.333),
    "sub_press": (0.000, 0.208),
    "ko_rate_sum": (0.000, 1.348),
    "sub_rate_sum": (0.000, 1.083),
    "durability": (0.000, 0.808),
    "elo_gap": (0.003, 0.469),
}


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
    raw = {"ko_press": float(ko_press), "sub_press": float(sub_press),
           "ko_rate_sum": float(ko_rate_sum), "sub_rate_sum": float(sub_rate_sum),
           "durability": float(durability), "elo_gap": float(elo_gap)}

    # Clip to the fitted region, and say so. A value well outside it usually
    # means the caller is computing the feature differently from the training
    # harness -- a train/serve skew, which no offline metric can detect
    # because the harness scores its own features.
    x = []
    for name in FEATURES:
        lo, hi = TRAINING_RANGE[name]
        v = raw[name]
        # Only warn on values FAR outside, and only above -- a feature
        # legitimately sitting at zero (two fighters with identical ratings)
        # is a real observation, not a definition mismatch. Warning on it
        # would train the reader to ignore the warning.
        if v > hi * 1.5:
            print(f"[method_model] {name}={v:.3f} is far outside the training "
                  f"range [{lo:.3f}, {hi:.3f}] -- clipping. If this fires often, "
                  f"the caller's feature definition has drifted from the harness.")
        x.append(min(max(v, lo), hi))
    z = [INTERCEPT[k] + sum(c * v for c, v in zip(COEF[k], x)) for k in range(3)]
    m = max(z)                                  # subtract the max for stability
    e = [math.exp(v - m) for v in z]
    total = sum(e)
    return {"ko": e[KO] / total, "sub": e[SUB] / total, "decision": e[DEC] / total}


def reconcile_fighter_methods(seed_a, seed_b, win_a, win_b, fight_dist, iters=80):
    """
    Per-fighter method probabilities that match BOTH known margins.

    Returns a 2x3 grid [[ko_a, sub_a, dec_a], [ko_b, sub_b, dec_b]] where each
    ROW sums to that fighter's win probability and each COLUMN sums to the
    fight-level probability for that method. The whole grid therefore sums
    to 1 -- the six outcomes are mutually exclusive and exhaustive.

    WHY THIS EXISTS. The per-fighter numbers were previously produced by
    multiplying a win probability by three independently-blended
    method-given-win rates that didn't sum to 1. Each fighter's methods
    overshot his own win probability, the six rows summed to 126.6% on a real
    card, and a submission row read 30.6% against a market implying 15.4% --
    an apparent 15-point edge that was arithmetic rather than signal.

    A first fix normalised only the model-only projection path, missing that
    edge_finder computes PRICED rows through a separate blend. The grid stayed
    incoherent at 119.9% because the two paths disagreed. Hence this shared
    function: one computation, both callers.

    The seeds carry the only unvalidated part -- a fighter's relative
    preference among methods. Iterative proportional fitting keeps that
    preference's SHAPE while forcing both margins to hold exactly.
    """
    grid = [[max(float(v), 1e-4) for v in seed_a],
            [max(float(v), 1e-4) for v in seed_b]]
    rows = [float(win_a), float(win_b)]
    cols = None
    if fight_dist:
        cols = [float(fight_dist["ko"]), float(fight_dist["sub"]), float(fight_dist["decision"])]

    for _ in range(iters):
        for i in range(2):
            tot = sum(grid[i]) or 1e-9
            grid[i] = [v * rows[i] / tot for v in grid[i]]
        if not cols:
            break                      # rows only: still coherent, no column target
        for j in range(3):
            tot = (grid[0][j] + grid[1][j]) or 1e-9
            f = cols[j] / tot
            grid[0][j] *= f
            grid[1][j] *= f
    return grid
