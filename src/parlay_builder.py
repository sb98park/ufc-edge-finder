"""
Builds multi-leg parlays out of the individual live-odds edges already
computed for the tracked card. Two flavors:

  - "Bankroll Builder": modest 2-3 leg combos landing around +100 to +300,
    built from legs the model actually likes (win probability > 50%).
  - "Lotto Parlays": longer-shot 3-5 leg combos at +1000 or higher, built
    from whichever legs give the best combined hit probability for that
    payout tier -- still long shots, just the least-long of the long shots.

Legs from the SAME fight CAN be combined -- e.g. "Fighter A Moneyline" +
"Under 2.5 Rounds" -- as long as they're not actually contradictory or
redundant. Two rules keep this sane:
  1. At most ONE "who wins / how" leg per fight (Moneyline or Method --
     never both, since Method already implies the Moneyline pick and
     combining them is either redundant, same fighter, or contradictory,
     different fighters).
  2. At most ONE "fight length" leg per fight (Total Rounds or Goes The
     Distance), with contradictions excluded (e.g. "wins by Decision"
     can't coexist with "Under 2.5 rounds" or "Ends In Finish").

Cross-fight legs are still assumed independent, which is a real
simplification -- fights on the same card aren't perfectly independent
in reality, but there's no clean way to quantify that from public data.
"""

import hashlib
import itertools
import os
import re

import pandas as pd

from src.odds_utils import (american_to_decimal, decimal_to_american, format_american_odds,
                            implied_prob_to_american, market_blended_prob)
from src.card_matcher import is_pickable_market, price_is_fragile
from src.ufc_method_rates import has_measured_method_rates


# WHERE A SLIP CAN ACTUALLY BE PLACED.
#
# The site carries two kinds of price and they do different jobs. DraftKings
# and FanDuel carry vig and are bettable, so they are the BOOK. Polymarket has
# no margin in it, which is exactly why it serves as the FAIR line the model
# measures edge against -- and exactly why it must not be quoted as a wager.
#
# Only a book can carry a parlay. If none of them can field one, the section
# is empty; see the note in _find_parlays.
BETTABLE_VENUES = frozenset({"DraftKings", "FanDuel", "BetMGM"})

# HOW MUCH MODEL GOES INTO THE RANKING. Deliberately NOT the site's
# MARKET_BLEND_MODEL_WEIGHT (0.30), which sizes single bets.
#
# Those are different jobs. Sizing one moneyline bet has nothing hunting its
# error. A parlay search ranks thousands of combinations by estimated joint
# probability, which is algebraically ranking by the model's CLAIMED edge --
# so it actively seeks the legs where the model is most optimistic relative
# to truth, and the estimate that wins is the one most inflated by noise.
# More model in the signal means more for the search to hunt.
#
# Swept over 400 real cards at sigma = 0.83, ratio = published hit rate over
# realised, 1.00 is honest:
#
#     weight   bankroll   lotto
#       0.00     0.94      0.96
#       0.10     1.03      1.22
#       0.20     1.04      1.48
#       0.30     1.08      1.50      <- what this used to be
#       0.50     1.25      2.75
#
# Monotone: calibration improves as the model is taken out of the ranking,
# and is only honest at 0.00 -- where the model plays no part in choosing a
# slip at all, which is a defensible measurement and not a product. 0.10 is
# the compromise: it keeps the model in the loop and takes bankroll from
# 1.08 to 1.03.
#
# THIS REPLACES A 250-CARD SWEEP recorded above that read 0.96 / 1.04 at this
# weight. It did not reproduce on the larger sample -- bankroll roughly held,
# lotto did not. Nothing should cite the old magnitudes. Re-measure at 400+
# after any threshold change, with scripts/replay_parlay_construction.py.
PARLAY_RANK_MODEL_WEIGHT = 0.10

# THE FLOOR A LEG HAS TO CLEAR, named so the PIN can enforce it too.
#
# It was a literal 0.50 inside build_bankroll_builder_parlays, which meant
# _find_parlays applied it when a slip was BUILT and nothing applied it when
# a pinned slip was RE-QUOTED. src/parlay_pin matches legs by identity --
# fight_key plus conditions -- and re-prices whatever it finds, so a leg the
# model has since turned against stayed in the slip at its new price, forever.
#
# Found on the live card for Nurmagomedov vs. Song: the pinned bankroll slip
# carried "Alex Perez vs Sumudaerji Under 2.5 rounds" at a model probability
# of 0.4282, while Over 2.5 -- the same market, the other side -- sat at
# 0.5718. The slip was holding the side the model had come to disagree with,
# below a floor it was supposed to have cleared.
BANKROLL_MIN_LEG_PROB = 0.50

# ON, MEASURED. scripts/replay_parlay_construction.py over 400 real cards,
# calibration ratio (published hit rate over realised; 1.00 is honest):
#
#     sigma      baseline        with the constraint
#     0.00       0.92  +2.8%     0.92  +2.8%     identical, as it must be
#     0.83       1.03  -5.2%     1.01  -3.7%
#
# Identical at sigma = 0 because with no model error there is no disagreement
# to flip a side. At realistic error it moves calibration slightly toward
# honest. That gain is small and inside the noise, and it is NOT the argument
# -- the measurement's job was to show the constraint costs nothing, and it
# does not. The argument is that a slip must never contain a side the model
# actively disbelieves.
#
# Env-overridable so the constraint can be measured back OFF without editing
# code: PARLAY_REQUIRE_MODEL_SIDE=0.
REQUIRE_MODEL_SIDE = os.environ.get("PARLAY_REQUIRE_MODEL_SIDE", "1") != "0"

def leg_still_eligible(piece: dict, min_leg_prob: float = BANKROLL_MIN_LEG_PROB) -> bool:
    """
    Whether a candidate leg is one this construction would take TODAY.

    ONE DEFINITION, TWO CALLERS. _find_parlays screens on this when a slip is
    built; src/parlay_pin screens on it when a pinned slip is re-quoted. They
    were separate, and the pin's copy checked only the probability floor -- so
    the model-side rule below applied to new slips and never to the pinned one
    already on the card, which is the slip a reader actually sees.
    """
    try:
        ranked = float(piece.get("model_prob"))
    except (TypeError, ValueError):
        return False
    if ranked < min_leg_prob:
        return False
    if REQUIRE_MODEL_SIDE:
        try:
            if float(piece.get("model_prob_raw", ranked)) < 0.50:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _ranking_prob(row: dict) -> float:
    """
    The probability a leg is RANKED and COMBINED on: the model shrunk toward
    the market's de-vigged price, exactly as edge_finder already does before
    sizing any single bet.

    WHY THIS EXISTS AT ALL. Every other consumer of these rows passes
    model_prob through market_blended_prob before kelly_fraction. This module
    read model_prob raw -- so parlays, the one product where errors compound
    multiplicatively, were the one product betting the unblended model.

    WHAT THAT COST, AND WHAT THIS BUYS. Measured by
    scripts/replay_parlay_construction.py, which drives THESE builders over
    250 real UFC cards with real book prices and exact settlement. Ratio is
    published hit rate divided by realised -- 1.00 is honest:

                             bankroll                lotto
                        published/realised      published/realised
        raw model        73.4% / 40.8%  1.80    45.6% /  8.0%   5.70
        blended (0.30)   52.1% / 46.8%  1.11    12.9% /  9.3%   1.39
        blended (0.10)   48.1% / 50.0%  0.96     9.0% /  8.6%   1.04

    THESE NUMBERS REPLACE AN EARLIER SET TAKEN ON 40 CARDS, which read 1.93 /
    10.37 raw and 1.41 / 2.97 blended. Those were not wrong measurements, they
    were a sample six times too small, and every magnitude in them was roughly
    two to three times overstated. The DIRECTION survived the larger sample --
    unblended is badly overstated, lotto is worse than bankroll, shrinkage
    fixes most of it -- but nothing that cites the old magnitudes should be
    trusted. Re-measure at 250+ cards after any threshold change.

    THE CONTROL SEPARATES TWO DIFFERENT FAULTS. Feeding the builders the
    market EXACTLY -- a perfectly calibrated model, no error to find:

        sigma = 0        bankroll 47.0% / 52.8%  0.89
                         lotto     7.9% /  6.9%  1.14

    At full sample the construction is close to honest on its own: bankroll is
    slightly CONSERVATIVE and lotto carries about 14% of residual bias, not
    the 51% the 40-card run suggested. So most of what is left after shrinking
    is model noise rather than construction, and the case for the exact
    same-fight joint rests on the arithmetic of leg dependence rather than on
    this measurement, which no longer shows much of it.

    Why noise is selected rather than merely tolerated: ranking by combined
    probability inside a payout band is algebraically ranking by the model's
    CLAIMED edge, so the search actively seeks the legs where the model is
    most wrong in the optimistic direction. Shrinking toward the market
    removes most of the error the search would otherwise be hunting.

    This repo's own evidence says that disagreement is mostly error:
    validate_market_blend.py records the model picking the market favourite
    40/47 (85%) and picking AGAINST the market 9/21 (43%) -- worse than a coin
    flip.

    A MISSING PRICE MEANS NO BLEND IS POSSIBLE. Model-only projected legs
    carry no book_fair_prob because no book quoted them; they are priced at
    implied_prob_to_american(model_prob), so their edge ratio is exactly 1.000
    by construction. They pass through unshrunk here, which is the honest
    handling of "there is no market to shrink toward" -- but it also means
    they can only ever inflate a slip's advertised payout, never its real
    value. Whether they belong in a parlay at all is a product question, left
    alone here.

    The 0.30 weight is inherited rather than chosen: it is the same constant
    the rest of the site sizes with, and its own docstring calls it a
    conservative heuristic rather than a fitted value. Fitting it -- and
    testing whether a logit-space blend beats this linear one, which matters
    most in the tails where lotto legs live -- is what
    scripts/replay_parlay_construction.py exists to do.
    """
    p = row.get("model_prob")
    book = row.get("book_fair_prob")
    if p is None or book is None or book != book:      # NaN-safe
        return p
    # PARLAY_RANK_MODEL_WEIGHT, not the site-wide constant. See below.
    w = PARLAY_RANK_MODEL_WEIGHT
    return w * float(p) + (1.0 - w) * float(book)

# Model-projected legs are priced at the model's own probability, so their
# edge ratio is 1.000 by construction. Kept as a named switch rather than
# deleted code so the reasoning at the call site stays attached to something
# real, and so re-enabling it is a deliberate act.
ALLOW_MODEL_ONLY_LEGS = False

WINNER_FAMILY = {"Moneyline"}  # "Method: X" markets are matched by prefix below
LENGTH_FAMILY_PREFIXES = ("Total Rounds", "Fight Outcome")


def _leg_label(row: dict) -> str:
    """Human-readable description of exactly what this leg is (odds shown separately, not embedded here)."""
    market = row["market"]
    if market == "Moneyline":
        return f"{row['fighter']} ML"
    elif market.startswith("Method"):
        method = market.replace("Method: ", "")
        return f"{row['fighter']} by {method}"
    elif market.startswith("Total Rounds"):
        line_desc = market.replace("Total Rounds ", "")
        return f"{row['fighter']} {line_desc} rounds"
    elif market.startswith("Fight Outcome"):
        outcome = market.replace("Fight Outcome: ", "")
        return f"{row['fighter']} — {outcome}"
    return f"{row['fighter']} — {market}"


# ESPN's own vocabulary, from result.name on the per-competition status
# object. Confirmed live against espn_method_probe.json: kotko, submission,
# decision---unanimous, decision---split, decision---majority.
#
# GRADE AGAINST THESE SLUGS, NEVER AGAINST THE DISPLAY STRING. _leg_label
# builds prose for humans ("Topuria by KO/TKO") and prose is not a protocol --
# reparsing it client-side would couple leg grading to a copy change.
# KEYED ON THE SITE'S OWN ABBREVIATIONS, which is what actually arrives here.
# upcoming_props.selection_method holds KO/TKO, SUB and DEC, and edge_finder
# interpolates that straight into "Method: {…}" -- so keying this on the
# spelled-out words matched KO/TKO by luck and silently made every submission
# and decision leg ungradeable. The long forms are kept as aliases in case a
# future feed spells them out.
_METHOD_SLUGS = {
    "KO/TKO": ["kotko"],
    "SUB": ["submission"],
    "DEC": ["decision---unanimous", "decision---split", "decision---majority"],
    "Submission": ["submission"],
    "Decision": ["decision---unanimous", "decision---split", "decision---majority"],
}


def _leg_conditions(row: dict) -> list[dict]:
    """
    Machine-gradeable predicates for one leg. The leg wins iff ALL of them are
    true; any single false condition kills it.

    A list rather than a single dict because a combined winner+length piece is
    ONE leg carrying TWO markets (see _build_candidate_pieces), and because a
    method bet is really two claims -- this fighter won, AND it ended that way.
    Splitting them matters for latency as much as correctness: the winner
    clause grades off the scoreboard within seconds, so a method leg whose
    fighter LOST dies immediately rather than waiting on the deeper method
    fetch that would only confirm what is already decided.

    Returns [] for a market this does not recognise, which the client treats
    as permanently ungradeable rather than guessing. Silence is the only safe
    failure here -- a leg wrongly shown as lost is worse than one shown as
    unknown.
    """
    market = str(row.get("market") or "")
    fighter = row.get("fighter")
    if market == "Moneyline":
        return [{"kind": "winner", "fighter": fighter}]
    if market.startswith("Method"):
        method = market.replace("Method: ", "").strip()
        slugs = _METHOD_SLUGS.get(method)
        if not slugs:
            return []
        return [{"kind": "winner", "fighter": fighter},
                {"kind": "method", "any_of": slugs}]
    if market.startswith("Total Rounds"):
        m = re.search(r"(Under|Over)\s*([\d.]+)", market, re.IGNORECASE)
        if not m:
            return []
        return [{"kind": "rounds", "op": m.group(1).lower(), "line": float(m.group(2))}]
    if market.startswith("Fight Outcome"):
        outcome = market.replace("Fight Outcome: ", "").strip().lower()
        if "distance" in outcome:
            return [{"kind": "distance", "value": True}]
        if "finish" in outcome:
            return [{"kind": "distance", "value": False}]
        return []
    return []


def _fight_key(row: dict) -> str | None:
    """
    'Fighter A|Fighter B' for the bout this leg belongs to.

    generate_site stamps fight_key onto every tracked edge, which is the only
    place the edge and its fight are both in scope. The fallback covers
    model-only projected rows, whose fight_id is already built in that shape
    -- and it deliberately does NOT try fighter + opponent, because
    fight-level markets set no opponent and would silently produce a
    half-formed key that matches nothing.
    """
    key = row.get("fight_key")
    if key:
        return str(key)
    fid = str(row.get("fight_id") or "")
    return fid if "|" in fid else None


def _slip_id(fight_ids, legs) -> str:
    """
    Stable DOM identity for a slip, across the three families and across
    rebuilds. Hashed over the legs' own labels rather than their order in the
    combination search, so the same slip keeps the same id when the ranking
    around it changes -- which is what makes a per-slip collapse state
    survivable from one refresh to the next.
    """
    parts = sorted(f"{fid}::{leg.get('label')}" for fid, leg in zip(fight_ids, legs))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _leg_family(market: str) -> str:
    if market == "Moneyline" or market.startswith("Method"):
        return "winner"
    if any(market.startswith(p) for p in LENGTH_FAMILY_PREFIXES):
        return "length"
    return "other"


def _is_contradiction(leg_a: dict, leg_b: dict) -> bool:
    """
    Blocks both real contradictions AND redundant pairs. A redundant pair
    isn't impossible to happen together -- it's the SAME claim stated twice
    (e.g. "wins by KO/TKO" already means "ends in finish"), so combining
    them double-counts one signal as if it were two independent risks.
    """
    markets = {leg_a["market"], leg_b["market"]}
    is_decision = any(m.startswith("Method: DEC") for m in markets)
    is_finish_method = any(m.startswith("Method: KO") or m.startswith("Method: SUB") for m in markets)
    is_under = any("Under" in m for m in markets)
    # "Over" was never tested, and that asymmetry cost real money. A decision
    # means the fight reached the final bell, so it went OVER every line a
    # book offers -- those lines sit strictly below the scheduled distance.
    # Measured on 5,645 three-round bouts: P(Over 2.5 | DEC) = 0.9997 and
    # P(Over 1.5 | DEC) = 1.0000, i.e. the same claim stated twice.
    # "Fighter A by Decision + Over 2.5 rounds" was therefore a legal piece
    # that multiplied two prices together and published roughly +410 for a bet
    # whose real probability is the decision probability alone, fair around
    # +199. The second leg added no risk and doubled the advertised payout --
    # exactly the double-count the four rules below already exist to prevent,
    # on the mirror-image case.
    is_over = any("Over" in m for m in markets)
    is_ends_in_finish = any("Ends In Finish" in m for m in markets)
    is_goes_distance = any("Goes The Distance" in m for m in markets)

    if is_decision and (is_under or is_ends_in_finish):
        return True  # winning by decision means it went the full distance
    if is_finish_method and is_goes_distance:
        return True  # a finish contradicts "goes the distance"
    if is_finish_method and is_ends_in_finish:
        return True  # redundant: "wins by KO/TKO or SUB" already IS "ends in finish"
    if is_decision and is_goes_distance:
        return True  # redundant: "wins by decision" already IS "goes the distance"
    if is_decision and is_over:
        return True  # redundant: a decision goes OVER every offered line -- see above
    return False


def _length_leg_has_measured_inputs(row: dict) -> bool:
    """
    A fight-length leg needs a method model that actually saw these two
    fighters. Winner legs are unaffected.

    WHY ONLY THE LENGTH FAMILY. Total Rounds and the Fight Outcome markets are
    priced straight off the method model: model_preview sets
    goes_distance_prob = method_distribution["decision"] and derives every
    rounds line from it. When a fighter has fewer than three UFC bouts,
    ufc_method_rates returns None and divisional_fallback_rates substitutes
    half the divisional prior -- the SAME numbers for every such fighter in
    that division. With both sides falling back, the model's inputs for that
    bout are the divisional base rate and nothing else, so any gap against the
    book's number is the book pricing information we do not have, not an edge.
    The moneyline does not go through this path and keeps its own inputs.

    Measured on the card that prompted this: 11 of 26 fighters were below the
    threshold. See has_measured_method_rates for what was and was not
    established there -- notably NOT a general claim that the method model
    lacks spread, which one card's 0.121 range does not support.

    Fails OPEN on an unparseable key. A leg whose fighters cannot be
    identified is already handled by the fences above; silently dropping it
    here would make this filter's effect impossible to reason about.
    """
    if not str(row.get("market") or "").startswith(LENGTH_FAMILY_PREFIXES):
        return True
    key = _fight_key(row)
    if not key or "|" not in key:
        return True
    names = [n.strip() for n in key.split("|") if n.strip()]
    if len(names) != 2:
        return True
    return all(has_measured_method_rates(n) for n in names)


def _build_candidate_pieces(tracked_edges: list[dict], model_only_by_fight: dict | None = None) -> list[dict]:
    """
    Builds the atomic units that can be cross-fight-combined: either a
    single leg, or a valid same-fight (winner + length) pairing. Each piece
    is tagged with its fight_id so the outer combination step still
    enforces "no two pieces from the same fight."

    Includes model-only projected legs (no live book price) when a fight
    has no live props at all -- e.g. a fighter with a standout stat like
    Terrance McKinney's first-round finish rate can still be a real parlay
    idea even without a live book line for it. These are clearly labeled
    "(model)" in the leg text rather than presented as a real bettable price.
    """
    # SAME MARKET-QUALITY FENCE THE REST OF THE SITE USES. This filtered on
    # nothing but "has a price and has a model probability", so every market
    # Favorite Picks refuses and the Edges tab flags as fragile was landing
    # in published parlays unchallenged -- 13 of 32 legs on one card were
    # Over 0.5 rounds.
    #
    # is_pickable_market drops 0.5 and 5.5 round lines and complement
    # markets; price_is_fragile drops legs resting on a near-certain outcome.
    # The reasoning is in card_matcher's fence comment and applies with MORE
    # force here, not less: a 95% leg contributes almost nothing to a slip's
    # payout while carrying real risk, and the apparent edge on it is a thin
    # Polymarket quote rather than alpha. The Edges tab keeps showing these
    # because it exists to show every disagreement; a parlay is a
    # recommendation to bet, which is a different claim.
    real_legs = [
        row for row in tracked_edges
        if row.get("odds_american") is not None and row.get("model_prob") is not None
        and is_pickable_market(row) and not price_is_fragile(row)
        and _length_leg_has_measured_inputs(row)
    ]

    # ONE CANDIDATE PER BOOK, NOT ONE PER MARKET.
    #
    # odds_american on an edge is the SHOPPED price -- the best of the books
    # that quoted it -- and best_book names the winner. That is right for a
    # single bet and wrong here, because a parlay is one ticket at one book:
    # keeping only the winner fragments the per-book boards. DraftKings may
    # quote every leg a slip wants, but if FanDuel beats it on one, that leg
    # leaves the DraftKings pool and the slip cannot be built anywhere.
    #
    # Measured on Nurmagomedov vs. Song: best_book split 150 two-way edges
    # into DraftKings 32 / FanDuel 26 / Polymarket 92, and after the leg
    # floor and the model-side rule no single book had enough left to form a
    # 2-fight slip in the +100 to +320 band. The card published no parlay at
    # all.
    #
    # book_prices carries the whole board (see _devig_and_shop._best), so a
    # market quoted by two books becomes two candidates at their own prices.
    # A row without it -- a model-only leg, a one-sided quote, a Polymarket
    # reference line -- passes through unchanged, so nothing regresses.
    expanded = []
    for row in real_legs:
        # isinstance, NOT `or {}`. These rows come back through pandas, which
        # fills a missing cell with float NaN -- and NaN is TRUTHY, so `or {}`
        # passes it straight through and len() raises. src/card_plays._venue
        # carries the same warning about best_book for the same reason; this
        # is that trap a second time, and it cost a build: the parlay block's
        # catch-all swallowed "object of type 'float' has no len()" and the
        # card published with no parlay section at all.
        prices = row.get("book_prices")
        if not isinstance(prices, dict) or len(prices) <= 1:
            expanded.append(row)
            continue
        for book, am in prices.items():
            expanded.append(dict(row, odds_american=am, source=book, best_book=book,
                                 books_quoting=1))
    real_legs = expanded

    by_fight: dict = {}
    for row in real_legs:
        by_fight.setdefault(row["fight_id"], []).append(row)

    pieces = []
    for fight_id, legs in by_fight.items():
        winner_legs = [l for l in legs if _leg_family(l["market"]) == "winner"]
        length_legs = [l for l in legs if _leg_family(l["market"]) == "length"]

        # single-leg pieces (either family alone)
        for leg in winner_legs + length_legs:
            pieces.append({
                "fight_id": fight_id,
                "label": _leg_label(leg),
                "odds_display": format_american_odds(leg["odds_american"]),
                # RANKED on the shrunk probability, DISPLAYED as the raw one.
                # The edge % the site shows is by definition model-vs-book, so
                # the raw number still has a job; it just must not be the one
                # that gets multiplied.
                "model_prob": _ranking_prob(leg),
                "model_prob_raw": leg["model_prob"],
                "decimal_odds": american_to_decimal(leg["odds_american"]),
                "is_model": False,
                "conditions": _leg_conditions(leg),
                "fight_key": _fight_key(leg),
                "source": leg.get("source"),
            })

        # combined winner+length pieces, skipping real contradictions
        for w in winner_legs:
            for l in length_legs:
                if _is_contradiction(w, l):
                    continue
                combined_decimal = american_to_decimal(w["odds_american"]) * american_to_decimal(l["odds_american"])
                # BOTH markets' conditions, concatenated. This is the case
                # the list return type exists for: one visible leg that only
                # wins if two separate claims both hold.
                w_c, l_c = _leg_conditions(w), _leg_conditions(l)
                pieces.append({
                    "fight_id": fight_id,
                    "label": f"{_leg_label(w)} + {_leg_label(l)}",
                    "odds_display": format_american_odds(decimal_to_american(combined_decimal)),
                    "model_prob": _ranking_prob(w) * _ranking_prob(l),
                    "model_prob_raw": w["model_prob"] * l["model_prob"],
                    "decimal_odds": combined_decimal,
                    "is_model": False,
                    # An unrecognised half makes the WHOLE leg ungradeable --
                    # grading it on the half we understand would report a
                    # verdict the leg has not actually earned.
                    "conditions": (w_c + l_c) if (w_c and l_c) else [],
                    "fight_key": _fight_key(w),
                    # Both halves come off the same fight, so they share a
                    # feed; if they somehow did not, the combination is not
                    # one a single book would price anyway.
                    "source": w.get("source") or l.get("source"),
                })

    # Model-only projected pieces -- only added for fights that had NO real
    # legs at all, so a fight with live data isn't diluted with unpriced
    # guesses when real prices already exist for it. Capped to the top 2
    # per fight (not all ~9 possible projections) -- with 11 fights on a
    # card, including every projection exploded the combinatorial search
    # space enough to hang the process (confirmed: caused an OOM kill).
    #
    # MAX_MODEL_LEG_JUICE excludes projected legs beyond -400 -- the same
    # category of problem the Favorite Picks -220 floor exists to prevent
    # (a leg that heavily juiced contributes almost nothing to a parlay's
    # payout while adding real risk, and it's worse here since there's no
    # real market price backing the number at all, just a model estimate).
    # Confirmed live: legs at -599 and -567 were slipping into real parlay
    # slates with nothing catching them.
    # MODEL-ONLY LEGS NO LONGER ENTER PARLAYS.
    #
    # These are priced at implied_prob_to_american(model_prob) -- the model's
    # own number turned into odds -- so their edge ratio is exactly 1.000 by
    # construction. They can never improve a slip's real value; they can only
    # inflate its advertised payout with a price nobody quoted.
    #
    # This was left in place through two audits that both recommended removing
    # it. What settled it was stamping provenance on every row: on the very
    # first build after that landed, ALL SEVEN published parlay legs came back
    # `source: model`. The site was publishing a full slate of parlays built
    # entirely from prices it had invented, rendered in American odds
    # indistinguishable from a real quote.
    #
    # It also cannot serve the point of the product. The premise is comparing
    # what a book offers against what the model thinks; a leg with no book
    # price has nothing on one side of that comparison.
    #
    # They remain on the Edges tab, where they are labelled as projections and
    # inform without pretending to be bettable. If a card has too little real
    # coverage to build a slip, the honest output is no slip.
    if ALLOW_MODEL_ONLY_LEGS and model_only_by_fight:
        MAX_MODEL_LEG_JUICE = -400
        for fight_id, rows in model_only_by_fight.items():
            if fight_id in by_fight:
                continue
            top_rows = sorted(rows, key=lambda r: r["model_prob"], reverse=True)[:2]
            for row in top_rows:
                try:
                    proj_odds = implied_prob_to_american(row["model_prob"])
                except (ValueError, ZeroDivisionError):
                    continue
                if proj_odds < 0 and proj_odds < MAX_MODEL_LEG_JUICE:
                    continue
                pieces.append({
                    "fight_id": fight_id,
                    "label": _leg_label(row),
                    "odds_display": format_american_odds(proj_odds),
                    "model_prob": _ranking_prob(row),
                    "model_prob_raw": row["model_prob"],
                    "decimal_odds": american_to_decimal(proj_odds),
                    "is_model": True,
                    "conditions": _leg_conditions(row),
                    "fight_key": _fight_key(row),
                    # Projected, not quoted. Naming a feed here would claim a
                    # price nobody posted.
                    "source": "model",
                })

    return pieces


def _combine(pieces: tuple[dict, ...]) -> dict:
    combined_decimal = 1.0
    combined_prob = 1.0
    combined_prob_raw = 1.0
    legs = []
    fight_ids = []
    for piece in pieces:
        combined_decimal *= piece["decimal_odds"]
        combined_prob *= piece["model_prob"]
        combined_prob_raw *= piece.get("model_prob_raw", piece["model_prob"])
        legs.append({
            "label": piece["label"],
            "odds_display": piece["odds_display"],
            "is_model": piece.get("is_model", False),
            # --- live grading payload ---
            # decimal_odds was already computed above and thrown away here.
            # It is what lets a VOIDED leg be divided back out of the slip's
            # price the way a book would, instead of the slip either dying or
            # quietly paying the wrong number.
            "decimal_odds": round(piece["decimal_odds"], 6),
            "fight_key": piece.get("fight_key"),
            "conditions": piece.get("conditions") or [],
            # Which feed priced this leg. With more than one source in play,
            # a slip can silently mix a vig-free peer-to-peer quote with a
            # vigged sportsbook one, and the ledger has to record which so a
            # graded result means something later.
            "source": piece.get("source"),
        })
        fight_ids.append(piece["fight_id"])
    combined_american = decimal_to_american(combined_decimal)
    return {
        "legs": legs,
        "has_model_legs": any(l["is_model"] for l in legs),
        "fight_ids": fight_ids,
        "slip_id": _slip_id(fight_ids, legs),
        "combined_decimal": round(combined_decimal, 6),
        "combined_american": round(combined_american),
        # UNCAPPED. This is the whole slip's price, not a single market --
        # see the cap note in format_american_odds. Individual legs above
        # keep the cap, where the realism argument does apply.
        "combined_american_display": format_american_odds(combined_american, cap=None),
        "combined_prob": round(combined_prob, 4),
        # The unshrunk product, kept so the gap between what the model claims
        # and what it is trusted for stays visible rather than being quietly
        # replaced.
        "combined_prob_raw": round(combined_prob_raw, 4),
        # Every leg shares this by construction -- see the venue split in
        # _find_parlays. Stamped on the slip so the page can say where the
        # price is, which is the other half of the same problem: a price with
        # no venue reads as "your book", and Polymarket is not your book.
        "venue": next((l.get("source") for l in legs if l.get("source")), None),
    }


def _record_reason(label: str, reason: str, detail: dict) -> None:
    """Write why this tier produced no slip into the committed health file.

    Keyed per tier under one `parlay` block, so bankroll and lotto can each
    say their own thing without overwriting the other -- and merged through
    src.source_health so neither erases anybody else's keys.
    """
    try:
        from src.source_health import record, PATH
        import json as _json
        block = {}
        try:
            with open(PATH, encoding="utf-8") as fh:
                block = (_json.load(fh) or {}).get("parlay") or {}
        except (OSError, ValueError):
            block = {}
        if not isinstance(block, dict):
            block = {}
        block[str(label)] = dict(detail, reason=reason)
        record("parlay", block)
    except Exception as exc:                      # noqa: BLE001
        print(f"[{label}] reason not recorded ({exc}) -- continuing")


def _find_parlays(
    pieces: list[dict],
    leg_counts: tuple[int, ...],
    min_american: float,
    max_american: float | None,
    min_leg_prob: float,
    max_results: int,
    label: str = "parlay",
) -> list[dict]:
    before_all = len(pieces)
    eligible = [p for p in pieces if leg_still_eligible(p, min_leg_prob)]
    if REQUIRE_MODEL_SIDE and before_all:
        _floor_only = [p for p in pieces if float(p.get("model_prob", 0)) >= min_leg_prob]
        if len(_floor_only) != len(eligible):
            print(f"[{label}] model-side filter dropped "
                  f"{len(_floor_only) - len(eligible)} of {len(_floor_only)} eligible piece(s)")


    # ONE VENUE PER SLIP, AND IT IS NOT A PREFERENCE. A parlay is a single
    # wager placed at a single book: legs from DraftKings and Polymarket
    # cannot be combined into one, so a mixed slip is not a worse
    # recommendation, it is an unplaceable one. Measured on the ledger when
    # this was added: 26 of 193 published slips drew from more than one
    # venue.
    #
    # The search therefore runs once PER VENUE and the winners compete at the
    # end. Filtering afterwards would not work -- the best mixed combination
    # is usually better than the best single-venue one, so a post-filter
    # would throw away the slate and return nothing.
    #
    # A PIECE WITH NO RECORDED VENUE IS DROPPED rather than pooled. Those
    # cannot be attributed to a book, so there is no way to know they can sit
    # on one ticket together -- exactly the property this is enforcing. The
    # ledger carries 318 such legs from before the source column existed;
    # this stops any more being created.
    by_venue: dict = {}
    unattributed = 0
    reference_only = 0
    for p in eligible:
        venue = p.get("source")
        if not venue or venue == "model":
            unattributed += 1
            continue
        # A REFERENCE LINE IS NOT A TICKET. Polymarket is carried precisely
        # because it has no margin in it -- that is what makes it a fair
        # yardstick for measuring edge, and it is also what makes it the
        # wrong thing to quote as a bet. Publishing a Polymarket parlay tells
        # a reader to go and take a price their sportsbook does not offer,
        # which is how a -270 that only exists there got read as a
        # DraftKings line.
        if venue not in BETTABLE_VENUES:
            reference_only += 1
            continue
        by_venue.setdefault(venue, []).append(p)
    if unattributed:
        print(f"[{label}] {unattributed} leg(s) dropped: no venue recorded, so they "
              f"cannot be shown to belong on one ticket")
    if not by_venue:
        # EMPTY BEATS UNPLACEABLE. A week where no book fields a full slip is
        # a real answer, and the section says so rather than substituting a
        # price from the yardstick.
        if reference_only:
            print(f"[{label}] no slip: {reference_only} leg(s) priced only on a "
                  f"reference line, which is not a book you can bet at")
        # THE REASON OUTLIVES THE LOG. The section renders "Not enough
        # live-priced or model-projected legs on this card yet" -- generic
        # where the real answer is specific, and the specific one existed only
        # as a print in a CI log nobody can read afterwards. A week with no
        # parlay is a legitimate outcome; the point is being able to tell that
        # apart from a feed being down.
        _record_reason(label, "no bettable venue", {
            "reference_only_legs": reference_only,
            "detail": (f"{reference_only} leg(s) priced only on a reference line, "
                       f"which is not a book you can bet at") if reference_only else
                      "no venue recorded on any leg",
        })
        return []
    if len(by_venue) > 1:
        print(f"[{label}] venues available: "
              + ", ".join(f"{v} ({len(ps)} legs)" for v, ps in sorted(by_venue.items())))

    # Hard safety cap: combinations of size 5 from a pool of even ~100
    # pieces is tens of millions of combos -- confirmed this can hang the
    # process. Capping the pool (keeping the most-likely pieces first)
    # keeps the search fast regardless of how large the input ever gets.
    MAX_POOL_SIZE = 30

    if len(by_venue) > 1:
        # Recurse once per venue, then take the best slate. Each inner call
        # sees a single-venue pool and so cannot mix by construction.
        per_venue = []
        for venue, ps in by_venue.items():
            got = _find_parlays(ps, leg_counts, min_american, max_american,
                                min_leg_prob, max_results, label=f"{label}/{venue}")
            for g in got:
                g["venue"] = venue
            per_venue.extend(got)
        if not per_venue:
            return []
        per_venue.sort(key=lambda x: x["combined_prob"], reverse=True)
        return per_venue[:max_results]

    venue, eligible = next(iter(by_venue.items()))
    if len(eligible) > MAX_POOL_SIZE:
        eligible = sorted(eligible, key=lambda p: p["model_prob"], reverse=True)[:MAX_POOL_SIZE]

    results = []
    # The closest near-miss, kept as (american, combo) and only turned into a
    # real parlay if the slate ends up empty and the log line needs it.
    best_miss_am, best_miss_combo = None, None

    for count in leg_counts:
        if len(eligible) < count:
            continue
        for combo in itertools.combinations(eligible, count):
            fight_ids = [p["fight_id"] for p in combo]
            if len(set(fight_ids)) != len(fight_ids):
                continue  # no two pieces from the same fight

            # AND NO TWO PIECES FROM THE SAME FIGHTER. Deduping on fight_id
            # alone is not enough: a fighter can appear on one card twice --
            # usually because a bout was rebooked or cancelled and its
            # replacement kept one corner -- and those are two DIFFERENT
            # fight_ids. Confirmed live: a published Moonshot slip carried
            # "Kody Steele vs Gauge Young Over 1.5 rounds" beside "Gauge Young
            # vs Stan Dorsainvil -- Ends In Finish", multiplying two legs that
            # cannot both happen as though they were independent risks.
            keys = [p.get("fight_key") for p in combo if p.get("fight_key")]
            fighters = [n.strip().lower() for k in keys for n in str(k).split("|")]
            if len(set(fighters)) != len(fighters):
                continue

            # BAND CHECK BEFORE _combine, not after. _combine builds nested
            # dicts and hashes a slip id, at ~7us a call, and it was running on
            # EVERY combination -- 8,656,906 of them for Moonshot, about 62
            # seconds of a 300-second rebuild cycle, almost all of it on slips
            # discarded microseconds later. The product of the decimals is the
            # only thing the band needs and it costs a multiply.
            _dec = 1.0
            for p in combo:
                _dec *= p["decimal_odds"]
            _am = decimal_to_american(_dec)
            if _am < min_american or (max_american is not None and _am > max_american):
                # best_miss exists only to explain an EMPTY slate in a log
                # line, so the losing combo is remembered as a tuple and built
                # into a parlay once, after the search. Materialising it here
                # put _combine back on the reject path -- which is most of the
                # search -- and gave back most of what moving the band check
                # bought.
                if best_miss_am is None or abs(_am - min_american) < abs(best_miss_am - min_american):
                    best_miss_am, best_miss_combo = _am, combo
                continue

            # Past the band check above, so this one is a keeper by
            # construction -- the three re-checks that used to live here are
            # gone with it, and the near-miss is only tracked on the reject path
            # where it means something.
            results.append(_combine(combo))

    if not results:
        distinct_fights = len({p["fight_id"] for p in eligible})
        best_miss = _combine(best_miss_combo) if best_miss_combo else None
        print(f"[{label}] no combos found: {len(eligible)} eligible pieces across {distinct_fights} distinct fights "
              f"(need >= {min(leg_counts)} distinct fights). "
              f"Closest miss: {best_miss['combined_american_display'] if best_miss else 'none tried'} "
              f"(target: {min_american:+.0f}{'+' if max_american is None else f' to {max_american:+.0f}'})")
        _record_reason(label, "no combination met the target", {
            "eligible_pieces": len(eligible),
            "distinct_fights": distinct_fights,
            "distinct_fights_needed": min(leg_counts),
            "closest_miss": (best_miss["combined_american_display"] if best_miss else None),
            "target": (f"{min_american:+.0f}"
                       + ("+" if max_american is None else f" to {max_american:+.0f}")),
        })
        return []

    # RECORD THE GOOD OUTCOME TOO, or last week's "no parlay" reason sits in
    # the health file beside a card that has one, which is worse than silence.
    _selected = _select_spread(results, max_results)
    _record_reason(label, "built", {"slips": len(_selected),
                                    "combinations_considered": len(results)})
    return _selected


def _select_spread(results: list[dict], max_results: int) -> list[dict]:
    """
    Picks results with genuine PAYOUT SPREAD across the range that was
    actually found, not just the top-N by probability. Sorting by
    probability alone naturally clusters near the minimum threshold, since
    the highest-probability combos that still clear the bar tend to be the
    ones that barely clear it (confirmed live: three "lotto" picks landing
    at +1000/+1000/+1003 instead of spanning the real range). Divides the
    payout-sorted pool into max_results buckets and takes the best
    (highest-probability, still-diverse) option from each, so a 3-slot
    slate spans low/mid/high instead of clustering at the floor.
    """
    if not results:
        return []
    if len(results) <= max_results:
        return _select_diverse(sorted(results, key=lambda p: p["combined_prob"], reverse=True), max_results)

    by_payout = sorted(results, key=lambda p: p["combined_american"])
    bucket_size = len(by_payout) / max_results

    def _fits(candidate, already_selected, max_shared):
        fight_id_set = set(candidate["fight_ids"])
        return all(
            len(fight_id_set & set(chosen["fight_ids"])) <= max_shared
            for chosen in already_selected
        )

    selected = []
    for i in range(max_results):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size) if i < max_results - 1 else len(by_payout)
        bucket = sorted(by_payout[start:end], key=lambda p: p["combined_prob"], reverse=True)

        picked = None
        for max_shared in range(0, 10):
            picked = next((c for c in bucket if _fits(c, selected, max_shared)), None)
            if picked:
                break
        if picked:
            selected.append(picked)

    # Any bucket that failed to yield a pick (e.g. too small/all-overlapping)
    # gets backfilled from the full payout-sorted pool rather than shorting
    # the result count.
    if len(selected) < max_results:
        for candidate in by_payout:
            if len(selected) >= max_results:
                break
            if candidate not in selected:
                selected.append(candidate)

    return sorted(selected, key=lambda p: p["combined_american"])


def _select_diverse(results: list[dict], max_results: int) -> list[dict]:
    """
    Picking the top N by raw probability tends to produce near-duplicates --
    if one leg has an unusually high individual probability, almost every
    top-ranked combo ends up including it. This tries progressively looser
    overlap tolerances (0 shared legs first, then 1, then 2...) rather than
    jumping straight from "zero overlap" to "no constraint at all" -- lotto
    in particular has a much smaller pool of genuinely long-shot-priced legs
    than bankroll does, so a hard 0-or-unlimited jump was reusing the same
    1-2 standout legs across every slot far more than necessary.
    """
    max_possible_overlap = max((len(r["fight_ids"]) for r in results), default=1)
    for max_shared in range(0, max_possible_overlap + 1):
        selected = []
        for parlay in results:
            fight_id_set = set(parlay["fight_ids"])
            too_similar = any(
                len(fight_id_set & set(chosen["fight_ids"])) > max_shared
                for chosen in selected
            )
            if not too_similar:
                selected.append(parlay)
            if len(selected) >= max_results:
                return selected
        if len(selected) >= max_results:
            return selected

    return selected


def build_bankroll_builder_parlays(tracked_edges: list[dict], model_only_by_fight: dict | None = None, max_results: int = 1) -> list[dict]:
    """
    2-3 piece combos landing roughly +100 to +300, from legs the model favors (>50%).

    ONE SLIP, NOT THREE. Publishing three at a unit each is a three-unit
    position, and if the ranking means anything then three units on the best
    slip beats one unit on each of the top three -- splitting stake across a
    best, a second-best and a third-best is a strictly worse allocation. If
    the ranking does NOT mean anything, publishing three does not rescue it
    either; it just spreads the same error over more bets.

    The three were also never three independent bets. They are drawn greedily
    from one ranked list over the same small pool, routinely share legs, and
    the diversity helpers widen their own overlap tolerance rather than return
    short -- so the effective sample was always closer to one.
    """
    pieces = _build_candidate_pieces(tracked_edges, model_only_by_fight)
    return _find_parlays(
        pieces, leg_counts=(2, 3), min_american=100, max_american=320,
        min_leg_prob=BANKROLL_MIN_LEG_PROB, max_results=max_results, label="bankroll",
    )


# THE LOTTO TIER IS RETIRED. Deleted 2026-08-26 on measurement rather than
# taste, and the reasoning is left here for the same reason the moonshot
# deletion's was: so nobody rebuilds it from the argument that justified it.
#
# It published one +1000-or-better slip, 2-5 legs. Swept over 400 real cards
# at sigma = 0.83 -- published hit rate over realised, 1.00 is honest:
#
#     ranking weight   0.00   0.10   0.20   0.30   0.50
#     lotto ratio      0.96   1.22   1.48   1.50   2.75
#     bankroll ratio   0.94   1.03   1.04   1.08   1.25
#
# Lotto is honest only at 0.00 -- where the model plays no part in choosing
# the slip, and the product reduces to "combine market favourites at the
# book's margin". At every weight where the model contributes anything, the
# number printed on the page is 22-50% higher than what actually happens.
# Bankroll over the same sweep stays inside 0.94-1.08 and is fine.
#
# WHY THIS TIER AND NOT THAT ONE. Both rank by estimated joint probability,
# which is algebraically ranking by the model's claimed edge, so both hunt the
# legs where the model is most optimistic. Lotto searches a far larger space
# at far longer prices and therefore selects far harder on that error. The
# deleted moonshot tier was worse still. The bias scales with the size of the
# search, and a tier defined by a long-shot payout target is by definition the
# one with the largest search.
#
# WHAT WOULD BRING IT BACK: a ranking objective that does not reward the
# model's own optimism -- a lower confidence bound, or a penalty in the number
# of candidates considered. Not a different payout band, and not a bigger
# sample. 400 cards is not where the uncertainty is.
