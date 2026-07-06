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
    """Inverse of the above, useful for sanity checks."""
    if prob <= 0 or prob >= 1:
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


def kelly_fraction(model_prob: float, american_odds: float, fraction: float = 0.5) -> float:
    """
    Fractional Kelly stake sizing (as a fraction of bankroll).
    `fraction` defaults to 0.5 (half-Kelly) since full Kelly is aggressive
    and very sensitive to model error -- exactly the kind of thing worth
    being conservative about with real money.
    """
    american_odds = float(american_odds)
    b = (american_odds / 100.0) if american_odds > 0 else (100.0 / -american_odds)
    q = 1 - model_prob
    edge = (model_prob * b) - q
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    return max(0.0, full_kelly * fraction)
