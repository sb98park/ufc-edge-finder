"""
Sportsbook odds math: American odds <-> implied probability, and vig removal.

Sportsbooks bake in a margin (the "vig") so that implied probabilities on
both sides of a bet sum to more than 100%. To compare a model's true
probability estimate against what the book is "really" pricing, we need
to strip the vig out first.
"""


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds (e.g. -150, +130) to implied probability (0-1)."""
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)


def american_to_decimal(odds_american) -> float:
    """Convert a sportsbook's American odds into decimal (payout multiplier) odds."""
    odds_american = float(odds_american)
    if odds_american > 0:
        return 1.0 + odds_american / 100.0
    return 1.0 + 100.0 / abs(odds_american)


def decimal_to_american(decimal_odds_value: float) -> float:
    """Inverse of the above -- used after combining parlay legs to show a familiar American price."""
    if decimal_odds_value >= 2.0:
        return (decimal_odds_value - 1.0) * 100.0
    return -100.0 / (decimal_odds_value - 1.0)


def implied_prob_to_american(prob: float) -> float:
    """
    Inverse of the above, useful for sanity checks.

    Explicitly checks for NaN, not just the 0/1 bounds -- prob <= 0 and
    prob >= 1 both evaluate to False when prob is NaN (any comparison
    with NaN is False under IEEE 754), so a NaN probability used to sail
    straight through this guard, produce a NaN "American odds" value, and
    only fail much later and less clearly when something tried to format
    it for display. Raising here, at the actual source of the bad value,
    is what lets the caller's existing try/except around this function
    actually catch it.
    """
    if prob != prob or prob <= 0 or prob >= 1:  # prob != prob is true only for NaN
        raise ValueError("probability must be between 0 and 1")
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    else:
        return 100 * (1 - prob) / prob


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """
    Normalize two implied probabilities that sum to >1 (because of vig)
    back down to a fair, no-vig pair that sums to exactly 1.
    """
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("implied probabilities must be positive")
    return prob_a / total, prob_b / total


# Typical UFC moneyline overround on DraftKings/FanDuel. This is a rough,
# named ESTIMATE, not a per-book/per-fight measurement -- real books don't
# split their margin evenly or proportionally (favorite-longshot bias means
# they usually shade favorites more than dogs), so this deliberately simple
# proportional model won't reproduce any specific book's exact posted line.
# It's the user's explicit choice to show book-style (vig-included) odds
# instead of Polymarket's near-vig-free raw probability -- this constant is
# what makes that conversion look like a typical book line rather than a
# no-vig "fair" price. Reverse-engineered from one real DK moneyline as a
# sanity check (implied a ~4.3% overround) -- tune this if better data on
# actual UFC moneyline vig shows up later.
DEFAULT_BOOK_OVERROUND = 0.045


def add_estimated_vig(prob_a: float, prob_b: float, overround: float = DEFAULT_BOOK_OVERROUND) -> tuple[float, float]:
    """
    Inverse of remove_vig_two_way: takes a fair pair (should sum to ~1) and
    proportionally inflates both sides so they sum to (1 + overround),
    approximating what a real sportsbook's vig-inclusive prices would look
    like. Normalizes the input first (in case it doesn't sum to exactly 1
    due to snapshot staleness/noise), so the output always sums to exactly
    (1 + overround) regardless of minor input drift.
    """
    fair_a, fair_b = remove_vig_two_way(prob_a, prob_b)
    factor = 1.0 + overround
    return fair_a * factor, fair_b * factor


def format_american_odds(value) -> str:
    """+230 for underdogs, -280 for favorites -- never a bare decimal."""
    v = int(round(float(value)))
    return f"+{v}" if v > 0 else str(v)


def decimal_odds(prob: float) -> float:
    """Fair decimal payout odds implied by a probability."""
    if prob <= 0:
        return float("inf")
    return 1.0 / prob


def edge_percent(model_prob: float, book_fair_prob: float) -> float:
    """
    Edge = how much higher your model's probability is than what the book
    (after removing vig) is effectively pricing. Positive = value bet
    candidate. Negative = book is favored over your model.
    """
    return (model_prob - book_fair_prob) * 100.0


MARKET_BLEND_MODEL_WEIGHT = 0.30


def market_blended_prob(model_prob: float, book_fair_prob: float) -> float:
    """
    Shrinks the model's probability toward the market's de-vigged price
    for STAKE SIZING purposes (not for the displayed edge %, which by
    definition is the raw model-vs-book comparison).

    Why: the 2026 backtest of the Elo backbone over ~2,900 out-of-sample
    historical fights put its standalone log loss at 0.6825 vs. a coin
    flip's 0.6931 -- real signal, but far from sportsbook-closing-line
    quality. Sizing Kelly bets from the raw model probability treats the
    model as the sole truth and systematically overbets whenever the
    model and a sharp book disagree by a lot -- which is exactly when
    the model is most likely to be the wrong one. Blending toward the
    market is the standard fix.

    The 0.30 model weight is a deliberate, conservative HEURISTIC, not a
    fitted value -- fitting it properly needs a dataset of past model
    probabilities alongside closing odds and outcomes, which
    fight_history.csv doesn't contain (no odds column). predictions_log.csv
    is accumulating exactly that data going forward; revisit this weight
    once enough graded picks exist to fit it out-of-sample.
    """
    return MARKET_BLEND_MODEL_WEIGHT * model_prob + (1.0 - MARKET_BLEND_MODEL_WEIGHT) * book_fair_prob


def kelly_fraction(model_prob: float, american_odds: float, fraction: float = 0.10, max_stake_pct: float = 0.05) -> float:
    """
    Fractional Kelly stake sizing (as a fraction of bankroll).

    Uses tenth-Kelly, not half-Kelly -- and hard-caps the result at 5% of
    bankroll regardless. Quarter-Kelly was tried first but turned out too
    aggressive in practice: standout props are specifically the biggest
    edges on the board, and quarter-Kelly already exceeds 5% above roughly
    a 20-point edge -- meaning nearly every standout prop collapsed to the
    same 5% ceiling with no variation between a 6% edge and a 40% edge.
    Tenth-Kelly keeps that differentiation intact across the normal range,
    reserving the cap for genuinely extreme cases (~45+ point edges), which
    are themselves a signal of likely model overconfidence rather than a
    real edge that big -- a method-of-victory prop resting on a small
    career sample, a stat that hasn't caught up to recent injury or camp
    news, etc.
    """
    american_odds = float(american_odds)
    b = (american_odds / 100.0) if american_odds > 0 else (100.0 / -american_odds)
    q = 1 - model_prob
    edge = (model_prob * b) - q
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    return min(max(0.0, full_kelly * fraction), max_stake_pct)
