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
