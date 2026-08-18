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
# named ESTIMATE, not a per-book/per-fight measurement -- real books use
# some discretionary/round-number pricing on top of any formula, especially
# for very short favorites, so this won't reproduce any specific book's
# exact posted line. It's the user's explicit choice to show book-style
# (vig-included) odds instead of Polymarket's near-vig-free raw probability
# -- this constant is the TOTAL vig budget the power-method split (below)
# allocates asymmetrically between favorite and underdog. Reverse-engineered
# from real DK moneylines as a sanity check (~4.3% overround, consistent
# across both a moderate 62/38 fight and a lopsided 90.5/9.5 one) -- tune
# this if better data on actual UFC moneyline vig shows up later.
DEFAULT_BOOK_OVERROUND = 0.045


def add_estimated_vig(prob_a: float, prob_b: float, overround: float | None = None) -> tuple[float, float]:
    """
    Inverse of remove_vig_two_way: takes a fair pair (should sum to ~1) and
    inflates both sides so they sum to (1 + overround), approximating what a
    real sportsbook's vig-inclusive prices would look like.

    OMITTING `overround` NOW MEANS "whatever the books are actually charging",
    resolved at call time. It used to bind DEFAULT_BOOK_OVERROUND at import,
    so callers that passed nothing -- the rationale copy among them -- kept
    quoting a constant estimated from historical data even on a build where
    DraftKings and FanDuel were quoting the very card being described. Falls
    back to that same constant when nothing was measurable, so a
    Polymarket-only build behaves exactly as before.

    Uses the POWER METHOD, not simple proportional scaling: raises each
    probability to a shared exponent m < 1 (solved per-pair via bisection so
    the pair sums to exactly the target), rather than multiplying both by
    the same factor. This matters because of a real bug found in production:
    proportional scaling multiplies BOTH sides by the same factor, but the
    American-odds formula (-100*p/(1-p)) has a 1/(1-p) term that blows up
    non-linearly as p->1 -- so a fixed proportional bump to an already-huge
    favorite (e.g. 90%) shrinks its (1-p) denominator by a much larger
    RELATIVE amount than it does for a moderate favorite, producing wildly
    exaggerated odds (a real 90.5% favorite came out -1742 instead of a
    realistic ~-900). Real sportsbooks don't actually split vig this way --
    they exhibit "favorite-longshot bias": heavy favorites get barely any
    extra juice (nobody wants to bet a -1700 favorite, so books keep it
    near its fair price), while underdogs absorb almost all of the margin.
    x^m for m<1 is concave, so it inflates SMALL probabilities proportionally
    more than large ones -- which reproduces this real asymmetry naturally,
    without needing separate rules for "close" vs. "lopsided" fights.

    Verified against 2 real DraftKings lines: a moderate 62/38 fight landed
    within a couple points of the real -180/+150 line; a lopsided 90.5/9.5
    fight came out far closer to the real -900/+600 line than the old
    proportional model's wildly-inflated -1742/+907. Not exact for extreme
    favorites -- real books also use some discretionary/round-number pricing
    on very short prices that no smooth formula fully replicates -- but the
    right shape and much closer in magnitude.
    """
    if overround is None:
        measured = _MEASURED_OVERROUND.get("two_way")
        overround = (measured - 1.0) if measured is not None else DEFAULT_BOOK_OVERROUND

    fair_a, fair_b = remove_vig_two_way(prob_a, prob_b)
    target = 1.0 + overround
    # Degenerate edge cases: a probability already at/near 0 or 1 can't be
    # meaningfully exponentiated toward a higher sum (0^m stays 0 for any
    # m>0, and there's no valid solution) -- fall back to proportional
    # scaling for just that pair rather than looping forever or dividing by
    # zero. Vanishingly rare for real fight probabilities.
    if fair_a <= 0.0 or fair_a >= 1.0 or fair_b <= 0.0 or fair_b >= 1.0:
        factor = target
        return fair_a * factor, fair_b * factor
    lo, hi = 0.01, 1.0
    for _ in range(60):  # bisection converges to far more precision than needed well within 60 steps
        mid = (lo + hi) / 2.0
        total = fair_a ** mid + fair_b ** mid
        if total > target:
            lo = mid
        else:
            hi = mid
    m = (lo + hi) / 2.0
    return fair_a ** m, fair_b ** m


def format_american_odds(value, cap: int | None = 5000) -> str:
    """
    +230 for underdogs, -280 for favorites -- never a bare decimal.

    Capped at ±5000 by default: stress-testing found extreme probabilities
    (99%+) produce mathematically-correct but absurd American odds (-19900,
    even -223304 at 99.9%) that no real sportsbook would ever quote --
    books stop around the low thousands or delist the market entirely.
    Since every displayed odds value in the site flows through this one
    formatter, capping here fixes the display-realism issue everywhere
    at once without touching any underlying probability math (edges,
    parlays, and model internals all use the raw probabilities, never
    this formatted string).

    PASS cap=None FOR A COMBINED PARLAY PRICE. The realism argument above is
    about a SINGLE market: no book quotes -19900 on one fighter. It does not
    hold for a parlay, where +28000 is simply what eight legs multiply out
    to and is exactly the number the bettor is being sold.

    Leaving the cap on there was silently destroying the Moonshot tier.
    parlay_builder sets its floor at min_american=5000 with no ceiling, so
    the tier's floor WAS the display cap and all three slates rendered an
    identical "+5000" -- true prices +8191, +10772 and +28476. A $5 slip
    shown returning $255 actually returns $1,429. _select_spread exists
    specifically to stop tier-floor clustering; the formatter was
    reintroducing the exact bug that function was written to prevent.
    """
    v = int(round(float(value)))
    if cap is not None:
        v = max(-cap, min(cap, v))
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


# TYPICAL OVERROUND ON A METHOD / PROP MARKET, measured rather than assumed:
# across the 5,646 bouts in data/external_odds.csv carrying a complete
# six-cell method grid, the implied probabilities sum to a mean of 1.2003 and
# a median of 1.2178 (p5 1.133, p95 1.291). Two-way moneylines on the same
# file sum to 1.0353, which is why they get exact de-vigging and props cannot.
# ONE CONSTANT WOULD BE WRONG. The 1.20 above is the SIX-CELL method grid.
# The genuinely two-way prop markets carry nothing like it -- measured on the
# live book, TotalRounds Over/Under 2.5 sums to 1.048, Over/Under 1.5 to
# 1.055, and GoesTheDistance to 1.049. Applying a 20% correction to a 5%
# market would strip four times the margin that is actually there and zero
# out every stake on it.
# NOT RE-MEASURABLE FROM data/odds_snapshot.json. Both figures above come from
# real two-sided books. The Polymarket feed we ship SYNTHESISES every "No" side
# as 1 - price_a (polymarket_source: implied_prob_to_american(1 - price_a)), so
# every pair in that file sums to exactly 1.0000 by construction. Measuring
# these constants there returns 1.00 for everything and means nothing.
PROP_OVERROUND = 1.20          # six-cell method grid, n=5,646 bouts
TWO_WAY_OVERROUND = 1.05       # totals and distance, measured on the live book


# MEASURED OVERROUND, when the feed can supply one. The constants above were
# measured historically off data/external_odds.csv and are still the fallback,
# but DraftKings and FanDuel now quote live -- so what they ACTUALLY charge on
# this card beats a figure averaged over 5,646 old bouts. Populated once per
# build by set_measured_overrounds(); empty means nothing was measurable and
# the constants stand.
_MEASURED_OVERROUND: dict[str, float] = {}

# A book's margin is not stable enough across two quotes to be worth trusting,
# and one weird line should not move the number every consumer reads.
MIN_QUOTES_TO_TRUST_MEASURED = 4


def set_measured_overrounds(by_family: dict | None) -> None:
    """Install this build's measured margins. Call once, before pricing."""
    _MEASURED_OVERROUND.clear()
    for k, v in (by_family or {}).items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        # A margin below 1.0 is arbitrage, not vig, and above 1.5 is a broken
        # pair rather than a real market. Either means the measurement is
        # wrong, and a wrong margin is worse than a stale one.
        if 1.0 < v < 1.5:
            _MEASURED_OVERROUND[str(k)] = v


def measure_overrounds(rows) -> dict:
    """
    What the books are really charging, per market family, from two-sided
    vig-bearing quotes in the current feed.

    Only a single source's own two sides are ever summed. Pairing DraftKings
    against FanDuel produces a margin belonging to neither book and can even
    come out below 1.0 when they disagree, which is the arbitrage the guard in
    set_measured_overrounds refuses.

    Returns {"two_way": x, "prop": y, "_counts": {...}} with families omitted
    when too few quotes were found to trust the average.
    """
    groups: dict = {}
    for r in rows or []:
        if r.get("source_is_vig_free", True):
            continue                      # a vig-free price has no margin
        odds = r.get("odds_american")
        if odds is None:
            continue
        fam = "prop" if overround_family(r.get("market")) == "prop" else "two_way"
        # The line has to be part of the key or Over 1.5 gets summed with
        # Under 2.5 -- two different markets that both look two-sided.
        key = (r.get("fight_id"), r.get("market"), r.get("source"),
               str(r.get("selection_method") or ""))
        groups.setdefault((fam, key), []).append(float(odds))

    sums: dict = {}
    for (fam, _key), prices in groups.items():
        if len(prices) != 2:
            continue
        total = sum(american_to_implied_prob(p) for p in prices)
        sums.setdefault(fam, []).append(total)

    out: dict = {"_counts": {k: len(v) for k, v in sums.items()}}
    for fam, vals in sums.items():
        if len(vals) >= MIN_QUOTES_TO_TRUST_MEASURED:
            vals = sorted(vals)
            # Median, not mean: one mispriced pair on a thin undercard bout
            # should not drag the figure every stake on the site is sized from.
            mid = len(vals) // 2
            out[fam] = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
    return out


def overround_family(market: str | None) -> str:
    """Which margin regime a market string belongs to. See overround_for_market."""
    m = (market or "").lower()
    if "fight method" in m:
        return "two_way"
    if "method" in m or "round betting" in m:
        return "prop"
    return "two_way"


def overround_for_market(market: str | None) -> float:
    """
    The margin to strip for a given market string.

    Defaults to the two-way figure, which is the conservative direction: an
    under-correction leaves a stake slightly too large, an over-correction
    silently deletes the bet.
    """
    fam = overround_family(market)
    # THE MEASURED FIGURE WINS WHERE THERE IS ONE. Falls through to the
    # historical constant when this build saw too few book quotes to trust.
    measured = _MEASURED_OVERROUND.get(fam)
    if measured is not None:
        return measured
    m = (market or "").lower()
    # "Fight Method: ..." IS NOT THE SIX-CELL GRID, despite containing the word
    # "method". It is a fight-level binary -- "does this fight end by KO" vs
    # "Not KO/TKO" -- built exactly like "Fight Outcome: Goes The Distance":
    # polymarket_source synthesises the No side as implied_prob_to_american(
    # 1 - price_a) and emits the Yes/Not pair, so the two sides sum to 1.000 by
    # construction. Matching it on the bare "method" substring sent a two-way
    # binary to the 20% six-cell margin, a 4x over-correction -- and the
    # docstring above says what that does: it silently deletes the bet. Four of
    # five real Fight Method quotes in data/odds_snapshot.json collapsed to a
    # zero stake. Checked FIRST, because "fight method" also contains "method".
    if "fight method" in m:
        return TWO_WAY_OVERROUND
    if "method" in m or "round betting" in m:
        return PROP_OVERROUND
    return TWO_WAY_OVERROUND


def devig_single_sided(implied_prob: float, market: str | None = None,
                       overround: float | None = None) -> float:
    """
    Approximate fair probability for a prop quoted with no complement.

    remove_vig_two_way needs both sides. A single prop leg does not have one,
    so the raw implied probability carries the book's whole margin and is
    biased HIGH.

    Proportional de-vig: divide by that market type's overround. Crude
    compared to the two-way version, and stated as such, but the alternative
    is treating a number known to be biased high as if it were fair.

    WHERE THIS MATTERS AND WHERE IT DOES NOT. The displayed edge uses the raw
    implied probability deliberately, and is labelled "not devigged" at every
    call site: an inflated book probability makes the edge look SMALLER, which
    errs toward not betting. Staking runs the other way. market_blended_prob
    exists to shrink the model toward the book so Kelly does not overbet a
    disagreement, and feeding it an inflated book probability inflates the
    blend and therefore the stake -- turning a safety mechanism into the
    opposite. Same input, opposite consequences, so only the staking path
    changes.

    NaN IN, NaN OUT -- checked explicitly, for the same IEEE 754 reason
    implied_prob_to_american checks it (see that docstring). The clamp below
    is max(0.0, min(1.0, x)), and min(1.0, nan) returns 1.0: Python's min
    keeps its first argument whenever the comparison is False, and every
    comparison with NaN is False. So a MISSING price was being clamped UP to
    a 100%-certain book probability, which is the one direction that must
    never happen in a staking path -- market_blended_prob turned it into a
    0.79 blend and kelly_fraction sized the hard 5%-of-bankroll cap, the
    largest stake this system can emit, off a price nobody quoted.
    Propagating the NaN instead keeps "we do not know" as "we do not know";
    kelly_fraction turns it into a zero stake explicitly.
    """
    if implied_prob != implied_prob:  # true only for NaN
        return float("nan")
    o = overround if overround is not None else overround_for_market(market)
    if o <= 0:
        return implied_prob
    return max(0.0, min(1.0, implied_prob / o))


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

    AN UNKNOWN PROBABILITY IS A ZERO STAKE, said out loud rather than left to
    argument order. NaN already came out of here as 0.0, but only by accident:
    the return is min(max(0.0, full_kelly), max_stake_pct) and max(0.0, nan)
    happens to be 0.0 -- swap those two arguments and the same line returns
    NaN, which is a stake percentage that would render on the page. The guard
    below makes the intent explicit so a later edit cannot quietly undo it.
    """
    if model_prob != model_prob or american_odds != american_odds:  # NaN
        return 0.0
    american_odds = float(american_odds)
    b = (american_odds / 100.0) if american_odds > 0 else (100.0 / -american_odds)
    q = 1 - model_prob
    edge = (model_prob * b) - q
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    return min(max(0.0, full_kelly * fraction), max_stake_pct)
