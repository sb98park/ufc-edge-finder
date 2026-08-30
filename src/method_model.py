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


# WHAT elo_gap CLIPPING ACTUALLY COSTS -- measured 2026-08-29, because the
# warning below sent an investigation looking for a serving bug that is not
# there.
#
# The warning used to say "the caller's feature definition has drifted from
# the harness." On this data it has not. Every bout that exceeded the ceiling
# was a genuine skill gap between two fighters with full records -- Abdul
# Hussein 16-2 (1724) vs Cody Gibson 22-13 (1424) at 0.750, Luke Riley 14-0
# vs Kai Kamaka III 18-8 at 0.655, four more of the same shape. This is not
# the Terrance Chatman failure, where a 0-0 roster row manufactured a 0.895
# gap out of missing data; that one was real and is fixed in power_rating.
#
# The served distribution IS wider than the declared range, and serving off
# build_effective_ratings rather than raw Elo accounts for about half of it
# (n=102 bouts, so these tails are noisy estimates):
#
#     declared 99th pct                  0.469
#     raw elo.ratings, served            0.548
#     effective ratings, served          0.655   (max 0.750)
#     bouts above the ceiling            6 of 102 (5.9%, not the ~1% implied)
#
# AND IT DOES NOT MATTER. elo_gap carries the smallest coefficients in the
# model -- DEC is -0.0201, effectively zero -- so it moves the KO/SUB split,
# not decision-vs-finish, which is what the rounds and distance markets are
# priced off. Clipping the most extreme value ever observed moves P(decision)
# by 0.0141; the 0.750 case above moves it by 0.0091.
#
# So: do NOT widen TRAINING_RANGE to silence this. That would extrapolate a
# logistic model past everything it was fitted on to buy at most 1.4pp on a
# feature the decision leg barely reads. The clip is the correct behaviour.
# The honest fix is a refit whose training features are built the way serving
# builds them -- and that needs research_method_fightlevel.py, which is named
# in the docstring above but is not in this repo.
ELO_GAP_CLIP_MEASURED = "2026-08-29: <=1.4pp on P(decision); clip is correct, do not widen"


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
                  f"range [{lo:.3f}, {hi:.3f}] -- clipping. See ELO_GAP_CLIP_MEASURED "
                  f"before treating this as a serving bug.")
        x.append(min(max(v, lo), hi))
    z = [INTERCEPT[k] + sum(c * v for c, v in zip(COEF[k], x)) for k in range(3)]
    m = max(z)                                  # subtract the max for stability
    e = [math.exp(v - m) for v in z]
    total = sum(e)
    p = [e[KO] / total, e[SUB] / total, e[DEC] / total]

    # CLAMPED AWAY FROM CERTAINTY. A softmax on extreme inputs will happily
    # return 1.000, and one card produced exactly that: decision 100.0%, which
    # then made P(finish) zero and collapsed every Under line to 0.0%.
    #
    # No fight is certain, and more to the point the training set contains no
    # certain fight -- the holdout's most confident decision bucket ran 60-70%
    # and hit 71.6%. A prediction at 1.0 is extrapolation past anything the
    # coefficients ever saw, not a strong opinion.
    #
    # The floor is per-class and small: it bounds the damage without
    # meaningfully moving a normal prediction. Renormalised after, so the
    # three still sum to 1.
    FLOOR = 0.015
    p = [max(v, FLOOR) for v in p]
    tot = sum(p)
    p = [v / tot for v in p]
    return {"ko": p[KO], "sub": p[SUB], "decision": p[DEC]}


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


# ---------------------------------------------------------------------------
# P(method | win) -- given a fighter wins, HOW does he win?
#
# This is the SEED for reconcile_fighter_methods(). It was previously a
# hand-weighted blend of divisional priors and career rates that had never
# been measured; the reconciliation guaranteed the totals were coherent while
# saying nothing about whether the shape being reconciled was any good.
#
# VALIDATED (research_method_given_win.py), frozen 2019+ holdout, n=1743 wins:
#
#     holdout log-loss   base rates 1.0047   this model 0.9372
#
#     calibration        base      model     observed
#       KO/TKO          +3.3%     +2.8%       31.2%
#       SUB             +2.0%     +0.7%       16.5%
#       DEC             -5.3%     -3.5%       52.4%
#
#     five-round wins (n=267)   predicted   actual
#       KO/TKO                     35.0%     37.8%
#       SUB                        16.8%     14.6%
#       DEC                        48.2%     47.6%
#
# NO `scheduled` FEATURE, for the same reason the fight-level model dropped
# it: including it put five-round calibration 13.2% out, because it learned
# "longer fight, more finishes" from few five-round wins. Removing it improved
# log-loss AND cut the subgroup error to 2.8%. Fitting the two lengths
# separately was tested and came out worse still.
#
# KNOWN RESIDUAL: in the top KO bucket (predicted >60%, n=40) the model reads
# 67.3% against an actual 77.5% -- UNDER-confident by 10pp. Small sample, and
# it errs toward understating a KO, which suppresses edges rather than
# inventing them. Worth rechecking as that bucket fills.
#
# DENOMINATORS ARE TOTAL FIGHTS, matching how the features were built in
# training. Dividing win-methods by wins and loss-methods by losses is the
# skew that once produced a 60% submission probability out of nothing.
MGW_FEATURES = ["own_ko_rate", "own_sub_rate", "opp_ko_lost", "opp_sub_lost",
                "ko_match", "sub_match", "elo_gap"]

MGW_COEF = [
    [1.144111, -0.634738, 0.984310, 0.078320, 0.987627, -0.467497, -0.582525],
    [-0.427732, 1.904605, 0.031644, 1.739280, -0.493505, 0.210939, 0.195236],
    [-0.716379, -1.269866, -1.015954, -1.817599, -0.494122, 0.256558, 0.387289],
]
MGW_INTERCEPT = [-0.163982, -0.912995, 1.076977]


def method_given_win(own_ko_rate, own_sub_rate, opp_ko_lost, opp_sub_lost,
                     elo_gap) -> list[float]:
    """
    [P(KO|win), P(SUB|win), P(DEC|win)] for one fighter against one opponent.

    Sums to 1 by construction -- one softmax over three outcomes.

    elo_gap is SIGNED and from this fighter's perspective
    ((own - opponent) / 400), unlike the fight-level model which uses the
    absolute gap. A favourite and an underdog finish differently, and the sign
    is what carries that.
    """
    ko_match = own_ko_rate * opp_ko_lost
    sub_match = own_sub_rate * opp_sub_lost
    x = [own_ko_rate, own_sub_rate, opp_ko_lost, opp_sub_lost,
         ko_match, sub_match, elo_gap]
    z = [MGW_INTERCEPT[k] + sum(c * v for c, v in zip(MGW_COEF[k], x)) for k in range(3)]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    tot = sum(e)
    return [v / tot for v in e]



# Share of FINISHES that land before each half-round mark, by scheduled
# length. Finishes are front-loaded, so these are cumulative and strictly
# increasing -- which is the property that matters.
#
# The previous form was a dict lookup keyed on the line with a default, and a
# 0.5 line wasn't in it: it fell through to the default and came out at 0.86,
# the same value as Under 2.5. A card showed "Under 0.5  51.3%" beside
# "Under 2.5  51.3%", which is impossible -- a fight ending in the first 150
# seconds cannot be as likely as one ending any time in three rounds.
#
# Built from a per-round distribution instead, so every line is derived from
# the same shape and monotonicity holds by construction rather than by the
# author remembering to check.
# MEASURED, not assumed (research_finish_timing.py):
#   3-round, n=4107 finishes
#   5-round, n=470 finishes
#
# The previous values were written to be "front-loaded", which was the right
# shape and the wrong magnitude -- badly so. Round 1 accounts for 54.7% of
# three-round finishes, not the 40% assumed, and round 3 for 14.4% rather than
# 27%. Fights that get finished get finished EARLY, far more than a plausible-
# looking curve suggested.
#
# The error ran one way on every line: Under 1.5 on a three-rounder was 13.6
# points understated, so the site showed negative edges on Under bets that
# were genuinely positive. P(finish) itself was never affected -- that comes
# from the validated fight-level model -- so this only ever misallocated
# finishes BETWEEN lines, which is precisely where round props are priced.
_ROUND_FINISH_SHARE = {
    3: [0.547, 0.308, 0.144],
    5: [0.368, 0.272, 0.181, 0.106, 0.072],
}

# ---------------------------------------------------------------------------
# DIVISION-CONDITIONED THREE-ROUND CURVE
#
# The constants above are one curve for every division, and measured on 3,945
# dated three-round finishes that is wrong in a specific, one-directional way:
#
#     men's divisions      R1 share 0.535   (n = 3,683)
#     women's divisions    R1 share 0.440   (n = 291, 95% CI 0.382-0.499)
#
# 0.534 -- what the pooled curve predicts -- sits OUTSIDE the women's interval
# (binomial p = 0.0015). Under 1.5 is priced directly off this number, so every
# women's fight on a card was being quoted a first-round finish share nine
# points too high. Across all eleven divisions the R1 share is ordered almost
# perfectly by weight (Spearman rho = 0.909, p = 0.0001; chi-square on the
# division x round table p = 0.0002).
#
# WHY THE PRIOR IS A LINE IN WEIGHT rather than the pooled mean. Cutting 3,945
# finishes eleven ways leaves the women's divisions on 80-110 observations, and
# shrinking those toward the pooled mean would erase the very fact the ordering
# establishes -- that a 115lb division belongs BELOW the middle, not at it.
# Shrinking toward the weight line keeps the gradient and discards only each
# division's idiosyncratic wiggle, which is what the sample size cannot
# support. Measured point-in-time it beat flat shrinkage at every K tried.
#
# WHAT THE HARNESS ACTUALLY SHOWED (validate_divisional_finish_curve.py, two
# windows, refit at every scored fight from strictly prior finishes):
#
#     arm            log loss   d vs pooled     p     n
#     shipped        1.01026      +0.00157   0.0147   2500
#     pooled (PIT)   1.00869           --      --
#     weight_k800    1.00612      -0.00257   0.1020
#
# Read this honestly. The AGGREGATE conditioning result is directionally
# consistent -- ten of ten arms improved on the pooled curve across both
# windows -- but it does not clear p < 0.05, because 71% of the sample sits in
# middle divisions where the correction is nearly nothing (Brier -0.0006). The
# gain is concentrated where the bias is: on women's fights the R1 Brier moves
# -0.0061, ten times the effect anywhere else.
#
# So this ships as a BIAS CORRECTION on a subgroup where the error is
# significant and pre-specified, not as an aggregate accuracy win. The separate
# finding that the frozen constant is beaten by a plain point-in-time refit IS
# significant in both windows (p = 0.049, p = 0.0147) -- the numbers above were
# fitted years ago and the sample has moved underneath them.
#
# FIVE-ROUND BOUTS STAY POOLED. There are 478 five-round finishes in the entire
# UFC record, about 43 per division. Nothing there is conditionable and
# pretending otherwise would fit noise.
_ROUND_SHARE_SHRINK_K = 800.0
_MIN_DIVISION_FINISHES = 20
_MIN_DIVISIONS_FOR_FIT = 4

# Nominal limits, used only as the REGRESSOR for the prior -- never as a
# lookup. A catchweight or an unlisted division falls out of the fit and
# resolves to the pooled curve rather than needing a weight invented for it.
_DIVISION_LBS = {
    "Women's Strawweight": 115, "Women's Flyweight": 125, "Women's Bantamweight": 135,
    "Women's Featherweight": 145,
    "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145, "Lightweight": 155,
    "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205, "Heavyweight": 265,
}

_division_round_shares_cache = None


def _norm3(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1 / 3] * 3


def _weight_line_prior(counts, pooled):
    """
    Per-division three-round curve predicted by a weighted least-squares line
    in division weight, fit across divisions on the round-1 and round-2 shares.

    Divisions enter weighted by their own finish count, so Heavyweight informs
    the slope more than Women's Featherweight. Round 3 is the remainder, which
    keeps the vector summing to one without a third fit and without a
    renormalisation that could reorder the first two.
    """
    pts = [(_DIVISION_LBS[d], sum(c), _norm3(c)) for d, c in counts.items()
           if d in _DIVISION_LBS and sum(c) >= _MIN_DIVISION_FINISHES]
    if len(pts) < _MIN_DIVISIONS_FOR_FIT:
        return {}
    coefs = []
    for idx in (0, 1):
        sw = sum(w for _, w, _ in pts)
        mx = sum(w * x for x, w, _ in pts) / sw
        my = sum(w * s[idx] for _, w, s in pts) / sw
        num = sum(w * (x - mx) * (s[idx] - my) for x, w, s in pts)
        den = sum(w * (x - mx) ** 2 for x, w, _ in pts)
        b = (num / den) if den > 0 else 0.0
        coefs.append((my - b * mx, b))
    out = {}
    for d, lbs in _DIVISION_LBS.items():
        r1 = min(max(coefs[0][0] + coefs[0][1] * lbs, 0.05), 0.90)
        r2 = min(max(coefs[1][0] + coefs[1][1] * lbs, 0.05), 0.90)
        out[d] = _norm3([r1, r2, max(1.0 - r1 - r2, 0.02)])
    return out


def _division_round_shares():
    """
    division -> three-round finish curve, shrunk to the weight line.

    Built once from data/ufc_fight_results.csv. Any failure to read or parse
    that file leaves this empty and every caller falls back to the pooled
    curve, which is the shipped behaviour -- a missing data file must never be
    able to take round props down with it.
    """
    global _division_round_shares_cache
    if _division_round_shares_cache is not None:
        return _division_round_shares_cache

    shares = {}
    try:
        import os

        import pandas as pd

        from .matchup_model import _division_from_bout_label

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "data", "ufc_fight_results.csv")
        d = pd.read_csv(path)
        d = d[d["TIME FORMAT"] == "3 Rnd (5-5-5)"]
        d = d[~d["METHOD"].str.contains("Decision", case=False, na=False)]
        d = d.assign(_r=pd.to_numeric(d["ROUND"], errors="coerce"),
                     _d=d["WEIGHTCLASS"].map(_division_from_bout_label))
        d = d.dropna(subset=["_r", "_d"])
        d = d[d["_r"].between(1, 3)]

        counts = {}
        for div, g in d.groupby("_d"):
            counts[div] = [int((g["_r"] == i).sum()) for i in (1, 2, 3)]
        if counts:
            # The pooled curve is computed over EVERY bout, including the
            # catchweights and the old tournament brackets, because it is
            # meant to be the all-UFC baseline. Only recognised divisions get
            # their own curve: _division_from_bout_label faithfully returns
            # things like 'Ultimate Fighter 1 Middleweight Tournament', and a
            # label seen once should resolve to the baseline rather than
            # acquire a curve of its own.
            total = [sum(c[i] for c in counts.values()) for i in range(3)]
            pooled = _norm3(total)
            prior = _weight_line_prior(counts, pooled)
            k = _ROUND_SHARE_SHRINK_K
            for div, c in counts.items():
                if div not in _DIVISION_LBS:
                    continue
                p = prior.get(div, pooled)
                shares[div] = _norm3([c[i] + k * p[i] for i in range(3)])
    except Exception:
        shares = {}

    _division_round_shares_cache = shares
    return shares


def finish_share_before(line: float, scheduled_rounds: int = 3, division=None) -> float:
    """
    Fraction of a fight's finishes that occur before `line` rounds elapse.

    "Under 2.5" means the fight ends before the midpoint of round 3: all
    finishes in rounds 1 and 2, plus roughly half of round 3's.

    On a three-round bout the curve is conditioned on `division` when one is
    known and recognised -- see the note above for why, and for the limits of
    what that conditioning was shown to buy. Five-round bouts and unknown
    divisions use the pooled curve.

    What the shares guarantee either way is ordering: P(Under 0.5) < P(Under
    1.5) < P(Under 2.5), which is arithmetic rather than a modelling claim and
    was being violated before this was built from a per-round distribution.
    """
    rounds = int(scheduled_rounds)
    shares = None
    if rounds == 3 and division:
        shares = _division_round_shares().get(str(division).strip())
    if shares is None:
        shares = _ROUND_FINISH_SHARE.get(rounds, _ROUND_FINISH_SHARE[3])
    full = int(line)                      # complete rounds below the line
    total = sum(shares[:full])
    # HALF A ROUND ONLY WHEN THE LINE IS ACTUALLY MID-ROUND. This was added
    # unconditionally, so f(1.0) and f(1.5) returned the same number and an
    # INTEGER line was priced as the Over beneath it. recommendations.py is
    # the one caller that passes an integer -- "does the fight reach round N"
    # -- so every published round-start leg understated by half a round's
    # finishes: 7-10 points on the current card, which is a threshold price
    # 40-90 American points too long.
    if line - full >= 0.5 and full < len(shares):
        total += shares[full] * 0.5
    return min(total, 1.0)
