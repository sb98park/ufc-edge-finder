"""
The leg inventory behind the slip builder.

WHY THIS EXISTS RATHER THAN ANOTHER RECOMMENDER. The site used to search
~10^6 combinations and publish the argmax. Two audits and a replay harness
agreed that the objective it searched on -- highest combined probability
inside a payout band -- is algebraically "maximum model-vs-market
disagreement", and this model's disagreements are where it is worst: 39.1%
on the 23 logged picks where it took the market underdog, against 85.2% on
the 61 where it agreed on the winner.

The reader was already hand-building slips from the site's own high-
confidence picks and ignoring the generated ones -- i.e. filtering to the
cohort the search was structurally biased against. This module serves that
behaviour instead of competing with it: it publishes every leg the card
offers, priced honestly, and lets the person choose. The arithmetic they
cannot do in their head -- compounding vig, shrinking the model toward the
market, turning a Polymarket mid into a DraftKings-equivalent price -- is
what gets automated. The judgement does not.

WHAT "HONESTLY" MEANS HERE, in three numbers per leg:

  p_model   the model's raw probability. Shown, never multiplied.
  p_blend   that probability shrunk toward the de-vigged market price. This
            is what a slip's combined probability is built from, because a
            product of raw model probabilities compounds the model's error
            multiplicatively -- measured at 1.80x on two legs and 5.70x on
            the longer tier before shrinkage was applied.
  p_book    the market price WITH a sportsbook's margin added back. Legs are
            sourced from Polymarket, which is peer-to-peer and vig-free, so
            its prices are not obtainable at DraftKings or FanDuel. Every
            payout and EV figure the builder shows is computed from this one,
            because it is the only price the reader can actually get.

THE VIG IS THE POINT OF THE WHOLE EXERCISE. A slip that is exactly fair on
Polymarket -- model perfectly right, market perfectly right -- is about
-6.5% at two legs and -23% at eight once placed at a real book, and worse on
method markets whose measured overround is 1.20 against 1.045 for a
moneyline. That compounding is invisible when a book quotes you one number
at the end, and it is the single thing a builder can show that a bet slip
cannot.

DOUBLE CHANCE IS DERIVED, NOT SOURCED. "Wins by KO/TKO or Submission" is a
market the reader bets and the feed does not carry. It is exactly the sum of
two cells the method grid already produces, so it is constructed here rather
than left out -- with its own vig applied on the way out, since the summed
fair price is not a book price either.
"""

from src.odds_utils import (american_to_decimal, add_estimated_vig, american_to_implied_prob,
                            implied_prob_to_american, market_blended_prob, overround_for_market)

# Markets the builder will offer. Anything else on the card is either already
# excluded upstream by is_pickable_market / price_is_fragile, or is a market
# the model does not produce a probability for -- and an unpriced leg in a
# builder is worse than an absent one, because it looks priced.
_FAMILY = {
    "Moneyline": "moneyline",
    "Method": "method",
    "Total Rounds": "length",
    "Fight Outcome": "length",
}


def _family(market: str) -> str:
    if market == "Moneyline":
        return "moneyline"
    if market.startswith("Method"):
        return "method"
    if market.startswith("Total Rounds") or market.startswith("Fight Outcome"):
        return "length"
    return "other"


def _book_price(p_fair: float, market: str) -> tuple[float, float]:
    """
    (book-implied probability, book American odds) for a fair probability.

    The overround is per-market and measured, not assumed: 1.045 on a
    moneyline, 1.20 on the six-cell method grid, 1.05 on two-way totals.
    add_estimated_vig uses the power method rather than proportional scaling,
    which matters here because a 0.90 favourite scaled proportionally comes
    out at roughly -1742 instead of a realistic -900.
    """
    o = overround_for_market(market)
    p_book, _ = add_estimated_vig(p_fair, 1.0 - p_fair, overround=o - 1.0)
    p_book = min(max(p_book, 1e-4), 0.9999)
    return p_book, implied_prob_to_american(p_book)


def _leg(row: dict, label: str, market: str, p_model: float, p_fair: float) -> dict | None:
    if p_model is None or p_fair is None or p_model != p_model or p_fair != p_fair:
        return None
    if not (0.0 < p_fair < 1.0) or not (0.0 < p_model < 1.0):
        return None
    # A REAL BOOK PRICE BEATS A RECONSTRUCTED ONE. _book_price exists because
    # every leg used to come from Polymarket, which is vig-free and therefore
    # not a price anyone can take -- so a sportsbook-equivalent had to be
    # synthesised from an assumed margin. Where DraftKings or FanDuel actually
    # quoted, that assumption is now redundant and strictly worse than the
    # number sitting in the row: it is the difference between "roughly what a
    # book would charge" and "what this book is charging".
    #
    # The synthesis stays for everything the books do not cover, which is
    # still most of the card and all of the method markets.
    if row.get("source_is_vig_free") is False and row.get("odds_american") is not None:
        am_book = float(row["odds_american"])
        p_book = american_to_implied_prob(am_book)
        book_is_real = True
    else:
        p_book, am_book = _book_price(p_fair, market)
        book_is_real = False
    if not (0.0 < p_book < 1.0):
        return None
    return {
        "fight_key": row.get("fight_key"),
        "fighter": row.get("fighter"),
        "market": market,
        "family": _family(market),
        "label": label,
        # Three probabilities, three jobs -- see the module docstring.
        "p_model": round(float(p_model), 4),
        "p_blend": round(float(market_blended_prob(float(p_model), float(p_fair))), 4),
        # The de-vigged market price. Kept because it is the ONLY input to the
        # zero-edge floor -- what the slip returns if the model adds nothing --
        # and that floor is the number that makes an extra leg feel expensive.
        "p_fair": round(float(p_fair), 4),
        "p_book": round(float(p_book), 4),
        "odds_book": round(float(am_book)),
        "odds_source": round(float(row.get("odds_american") or 0)),
        "decimal_book": round(1.0 / p_book, 6),
        # Whether the payout above is a quoted price or a reconstruction. A
        # builder that cannot tell the reader which is which is the failure
        # this module's docstring already calls the worst one available.
        "book_is_real": book_is_real,
        "book_source": row.get("source") if book_is_real else None,
        "conditions": row.get("conditions") or [],
    }


def build_leg_inventory(tracked_edges: list[dict]) -> list[dict]:
    """
    Every leg the card offers, grouped by fight, priced at book-equivalent
    odds. Returns [{fight_key, fight_label, legs: [...]}, ...].
    """
    from src.parlay_builder import _leg_conditions, _leg_label
    from src.card_matcher import is_pickable_market, price_is_fragile

    by_fight: dict = {}
    method_cells: dict = {}

    for row in tracked_edges or []:
        market = str(row.get("market") or "")
        fam = _family(market)
        if fam == "other":
            continue
        # METHOD LEGS ARE WITHHELD, and this is a provenance problem rather
        # than a pricing opinion.
        #
        # data/upcoming_props.csv carries no Method rows for the tracked card
        # -- the only ones in the feed belong to a different event -- yet 16
        # "Method: KO/TKO" legs reach this function with book_fair_prob
        # between 0.45 and 0.68. A specific fighter winning by a specific
        # method at 57% is not a price any book quotes, so whatever those
        # numbers are, they are not a quoted market, and edge_finder builds
        # Method rows down several paths (one of which explicitly derives a
        # missing side by subtraction) without a field that says which.
        #
        # A builder exists to tell someone what a combination is really worth.
        # Showing a DraftKings-equivalent price computed from a number that
        # may never have been a market price is the one failure that would
        # make it worse than no tool at all. Withheld until the source is
        # traced -- which also defers the derived Double Chance leg, since it
        # is the sum of two method cells.
        if fam == "method":
            continue
        key = row.get("fight_key")
        if not key:
            continue
        # THE SAME MARKET-QUALITY FENCES THE REST OF THE SITE APPLIES. Skipping
        # them offered legs everything else refuses -- Over/Under 0.5 rounds
        # was appearing here while PICKABLE_ROUND_LINES excludes it and
        # Favorite Picks would never show it. A builder that offers a leg the
        # site elsewhere calls unpickable is not a more permissive tool, it is
        # an inconsistent one.
        if not is_pickable_market(row) or price_is_fragile(row):
            continue
        p_model = row.get("model_prob")
        p_fair = row.get("book_fair_prob")
        if p_fair is None or p_fair != p_fair:
            # No market price means no honest book conversion and no blend.
            # A leg the builder cannot price is a leg it must not offer.
            continue
        r = dict(row)
        r["conditions"] = _leg_conditions(row)
        leg = _leg(r, _leg_label(row), market, p_model, p_fair)
        if not leg:
            continue
        by_fight.setdefault(key, []).append(leg)

        # Stash the KO and SUB cells so a Double Chance leg can be summed.
        if market.startswith("Method") and market.split(": ")[-1] in ("KO/TKO", "SUB"):
            method_cells.setdefault((key, row.get("fighter")), {})[market.split(": ")[-1]] = (
                float(p_model), float(p_fair), r)

    # DOUBLE CHANCE, derived. P(by KO/TKO or by SUB) is the sum of two
    # mutually exclusive cells of the same grid, so it needs no new model --
    # only its own vig on the way out, because a summed fair price is still a
    # fair price and not something a book would quote.
    for (key, fighter), cells in method_cells.items():
        if len(cells) != 2:
            continue
        p_model = sum(c[0] for c in cells.values())
        p_fair = sum(c[1] for c in cells.values())
        if not (0.0 < p_fair < 1.0):
            continue
        base = dict(cells["KO/TKO"][2])
        leg = _leg(base, f"{fighter} by KO/TKO or SUB", "Method: DoubleChance",
                   p_model, p_fair)
        if leg:
            # Grades as: this fighter won, by either finish method.
            leg["conditions"] = [{"kind": "winner", "fighter": fighter},
                                 {"kind": "method", "any_of": ["kotko", "submission"]}]
            leg["derived"] = True
            by_fight.setdefault(key, []).append(leg)

    out = []
    for key, legs in by_fight.items():
        parts = str(key).split("|")
        out.append({
            "fight_key": key,
            "fight_label": " vs ".join(parts) if len(parts) == 2 else str(key),
            # Moneyline first, then finish markets, then length -- the order
            # someone builds a slip in, rather than alphabetical.
            "legs": sorted(legs, key=lambda l: ({"moneyline": 0, "method": 1, "length": 2}
                                                .get(l["family"], 3), -l["p_blend"])),
        })
    out.sort(key=lambda f: f["fight_label"])
    return out


def zero_edge_return(legs: list[dict]) -> float:
    """
    What a slip returns if the model has NO edge at all -- every leg's fair
    price exactly right, every disagreement worth nothing.

    This is the floor the reader is betting against, and it is the number
    that makes leg count feel expensive rather than free: it is roughly
    -4.5% per moneyline leg and -20% per method leg, compounding. A book
    quotes one number at the end and never shows this; it is the main thing
    a builder can tell you that a bet slip cannot.
    """
    fair = 1.0
    book = 1.0
    for l in legs:
        fair *= l["p_fair"]
        book *= l["p_book"]
    return (fair / book - 1.0) if book else 0.0
