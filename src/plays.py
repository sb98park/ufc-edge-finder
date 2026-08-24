"""
Which bets the model actually stakes, and how much.

THE PROBLEM THIS SOLVES. Confidence tiers alone cannot size a prop. The model
will call Over 0.5 rounds at 95% and be right, but the price is -1000 and the
edge is four points -- and four points at -1000 is fragile in a way twenty
points at +150 is not. Break-even at -1000 is 90.9%, so a two-point
calibration miss halves the edge; break-even at +150 is 40%, so the same miss
barely registers. Short prices amplify model error and long ones absorb it.

That is not hypothetical here. compute_track_record() currently prints a
calibration warning of its own: the 50-60% bin runs underconfident, 54.6%
predicted against 72.7% actual over 33 picks. The model is good and it is not
perfectly calibrated, and a staking rule has to survive the gap.

THE RULE. Require a minimum expected return per unit RISKED:

    EV per unit = p*b - (1-p)  >=  hurdle

which rearranges to  p >= implied * (1 + hurdle)  -- the model must beat the
price by a MULTIPLICATIVE margin. That single line replaces both a price floor
and an edge threshold, and it demands more absolute edge as the price shortens,
which is exactly the right direction:

    -1000  implied 90.9%  ->  needs 95.5%   (4.6 pts)
    -190   implied 65.5%  ->  needs 68.8%   (3.3 pts)
    +300   implied 25.0%  ->  needs 26.3%   (1.3 pts)

No banned-price list. A genuinely enormous edge at a short price still
qualifies; a marginal one cannot.

SIZING. Kelly for the shape, the published ladder for the ceiling. Raw Kelly is
unusable at short prices -- a 97% read at -1000 asks for 67% of a bankroll --
so quarter Kelly, expressed in units where 1U = 1% of bankroll, then clamped.
Quarter rather than half is not only caution: at full Kelly nearly every play
exceeds its cap, the cap binds every time, and the stake stops varying with the
edge at all. Quarter keeps most plays under their ceiling, which is what makes
the size mean something.

Everything here is a pure function of its arguments. No I/O, no clock, no
config file -- so the worked examples in the tests are the specification.
"""

from __future__ import annotations

# --- hurdles -------------------------------------------------------------
# Moneylines and props get different hurdles because we know different amounts
# about them. High confidence is 19-1 and 58.3% of priced positions beat the
# closing line, so the moneyline edge has evidence behind it. Method and
# duration markets have no comparable record, worse vig and thinner liquidity,
# and the hurdle should price that ignorance rather than pretend it away.
# At 10%, a -1000 prop needs a 100% read -- which is to say the short end of
# the props board is closed until there is a record to justify opening it.
HURDLE_MONEYLINE = 0.05
HURDLE_PROP = 0.10

# --- sizing --------------------------------------------------------------
KELLY_SCALE = 0.25          # quarter Kelly; see the module docstring
UNIT_AS_BANKROLL_PCT = 0.01  # 1U = 1% of bankroll, matching the published ladder

# Ceilings, not stake sizes. The ladder is what the record has always been
# published at; Kelly decides where inside it a given play lands.
TIER_CAP_UNITS = {
    "Lock of the Week": 10.0,
    "High Confidence": 5.0,
    "Medium Confidence": 2.0,
    "Low Confidence": 1.0,
}
PROP_CAP_UNITS = 3.0        # below every moneyline tier, deliberately
MIN_STAKE_UNITS = 1.0       # under this it is not worth publishing
STAKE_INCREMENT = 0.5       # so a card stays legible

# --- exposure ------------------------------------------------------------
# One play per fight per axis. Over 2.5 rounds, Goes to Decision and Under 1.5
# rounds are largely the same bet wearing different labels; publishing three at
# 3U each claims 9U of independent exposure while carrying nearer 4U of real
# risk, and one bad read loses all three. A units total that lies about risk is
# the one thing this product cannot afford.
AXIS_OUTCOME = "outcome"
AXIS_METHOD = "method"
AXIS_DURATION = "duration"

MAX_UNITS_PER_FIGHT = 12.0
MAX_UNITS_PER_CARD = 40.0


def decimal_odds(american: float) -> float:
    """American price to decimal. -150 -> 1.667, +150 -> 2.5."""
    a = float(american)
    if a == 0:
        raise ValueError("american odds of 0 are not a price")
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def implied_prob(american: float) -> float:
    """The price's own probability, vig included -- what you must beat."""
    return 1.0 / decimal_odds(american)


def ev_per_unit(p: float, american: float) -> float:
    """
    Expected return per unit RISKED, not per unit returned.

    Per unit risked is the comparable quantity across prices: it answers "what
    does a unit on this make on average", which is what a staking decision
    needs. Per unit returned would flatter every short price.
    """
    b = decimal_odds(american) - 1.0
    return p * b - (1.0 - p)


def required_prob(american: float, hurdle: float) -> float:
    """The model probability this price needs before it is worth staking."""
    return implied_prob(american) * (1.0 + hurdle)


def kelly_fraction(p: float, american: float) -> float:
    """
    Full-Kelly fraction of bankroll. Negative when the bet is bad.

    Exposed on its own because it is worth being able to see how violent raw
    Kelly is at short prices before it gets scaled and clamped.
    """
    b = decimal_odds(american) - 1.0
    if b <= 0:
        return 0.0
    return ev_per_unit(p, american) / b


def _round_to_increment(units: float) -> float:
    return round(units / STAKE_INCREMENT) * STAKE_INCREMENT


def size_play(p: float, american: float, tier: str, is_prop: bool = False,
              hurdle: float | None = None) -> dict:
    """
    Decide whether to play, and for how much.

    Returns a dict rather than a bare number so a rejected play can say WHY --
    a card that silently drops a fight is impossible to audit, and "why is this
    not on the card" is the first question anyone will ask.
    """
    if hurdle is None:
        hurdle = HURDLE_PROP if is_prop else HURDLE_MONEYLINE

    need = required_prob(american, hurdle)
    ev = ev_per_unit(p, american)
    kelly = kelly_fraction(p, american)
    cap = PROP_CAP_UNITS if is_prop else TIER_CAP_UNITS.get(tier, 0.0)

    out = {
        "play": False, "units": 0.0, "reason": None, "capped": False,
        "ev_per_unit": round(ev, 4), "implied": round(implied_prob(american), 4),
        "required_prob": round(need, 4), "kelly": round(kelly, 4), "cap": cap,
    }

    if cap <= 0:
        out["reason"] = f"no stake ladder for tier {tier!r}"
        return out
    if p < need:
        out["reason"] = (f"model {p:.1%} below the {need:.1%} this price needs "
                         f"at a {hurdle:.0%} hurdle")
        return out

    # Kelly is a fraction of bankroll; a unit is 1% of bankroll; so units are
    # the fraction divided by the unit size.
    raw_units = kelly * KELLY_SCALE / UNIT_AS_BANKROLL_PCT

    # THE FLOOR IS CHECKED BEFORE ROUNDING, not after. Rounding first lets a
    # play Kelly wanted at 0.83U round up to 1.0U and clear a 1U floor, which
    # makes the floor silently mean 0.75U. Rejecting on the raw number keeps
    # the constant honest: below one unit of genuine Kelly, we do not play.
    if raw_units < MIN_STAKE_UNITS:
        out["reason"] = (f"Kelly sizes this at {raw_units:.2f}U, below the "
                         f"{MIN_STAKE_UNITS:.0f}U floor")
        return out

    units = _round_to_increment(min(raw_units, cap))

    out["play"] = True
    out["units"] = units
    out["capped"] = raw_units > cap
    return out


def select_card(candidates: list[dict]) -> dict:
    """
    Turn a card's worth of qualifying plays into the ones actually staked.

    Each candidate needs: fight_id, axis, units, ev_per_unit, and whatever the
    caller wants carried through. Assumes size_play has already run -- this
    layer is only about correlation and exposure.

    Order matters and is by EV per unit risked, so when two plays collide on an
    axis the better bet keeps the slot rather than whichever happened to be
    listed first.
    """
    ranked = sorted(candidates, key=lambda c: -c.get("ev_per_unit", 0.0))
    taken: list[dict] = []
    dropped: list[dict] = []
    used_axis: set[tuple] = set()
    per_fight: dict = {}
    card_total = 0.0

    for c in ranked:
        fid, axis, units = c.get("fight_id"), c.get("axis"), float(c.get("units", 0.0))

        if (fid, axis) in used_axis:
            dropped.append({**c, "dropped": f"already have a {axis} play on this fight"})
            continue
        if per_fight.get(fid, 0.0) + units > MAX_UNITS_PER_FIGHT:
            dropped.append({**c, "dropped": f"would exceed {MAX_UNITS_PER_FIGHT:.0f}U on one fight"})
            continue
        if card_total + units > MAX_UNITS_PER_CARD:
            dropped.append({**c, "dropped": f"would exceed {MAX_UNITS_PER_CARD:.0f}U on the card"})
            continue

        used_axis.add((fid, axis))
        per_fight[fid] = per_fight.get(fid, 0.0) + units
        card_total += units
        taken.append(c)

    return {"plays": taken, "dropped": dropped, "total_units": round(card_total, 2)}
