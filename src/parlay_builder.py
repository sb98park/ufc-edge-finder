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
import re

import pandas as pd

from src.odds_utils import (american_to_decimal, decimal_to_american, format_american_odds,
                            implied_prob_to_american, market_blended_prob)
from src.card_matcher import is_pickable_market, price_is_fragile


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
    scripts/replay_parlay_construction.py, which drives THESE builders over 40
    real UFC cards with real book prices and exact settlement. Ratio is
    published hit rate divided by realised -- 1.00 is honest:

                             bankroll                lotto
                        published/realised      published/realised
        raw model        64.2% / 33.3%  1.93    34.6% /  3.3%  10.37
        blended (0.30)   43.6% / 30.8%  1.41    11.1% /  3.7%   2.97

    The lotto column is the one to look at: unblended, the tier advertised a
    34.6% hit rate on slips that landed 3.3% of the time. Shrinking toward the
    market cuts that overstatement from 10.4x to 3.0x.

    THE CONTROL SEPARATES TWO DIFFERENT FAULTS. Feeding the builders the
    market EXACTLY -- a perfectly calibrated model, no error to find:

        sigma = 0        bankroll 40.4% / 39.2%  1.03
                         lotto     8.7% /  5.8%  1.51

    Bankroll is honest at 1.03, so all of its remaining bias is model noise.
    Lotto is 1.51 with NO model error at all, which is construction bias --
    dependence between legs the product-of-probabilities does not model, and
    it grows with leg count. Blending cannot fix that half and does not claim
    to; the exact same-fight joint would.

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
    return market_blended_prob(float(p), float(book))

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
    ]

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
    if model_only_by_fight:
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
    }


def _find_parlays(
    pieces: list[dict],
    leg_counts: tuple[int, ...],
    min_american: float,
    max_american: float | None,
    min_leg_prob: float,
    max_results: int,
    label: str = "parlay",
) -> list[dict]:
    eligible = [p for p in pieces if p["model_prob"] >= min_leg_prob]

    # Hard safety cap: combinations of size 5 from a pool of even ~100
    # pieces is tens of millions of combos -- confirmed this can hang the
    # process. Capping the pool (keeping the most-likely pieces first)
    # keeps the search fast regardless of how large the input ever gets.
    MAX_POOL_SIZE = 30
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
        return []

    return _select_spread(results, max_results)


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


def build_bankroll_builder_parlays(tracked_edges: list[dict], model_only_by_fight: dict | None = None, max_results: int = 3) -> list[dict]:
    """2-3 piece combos landing roughly +100 to +300, from legs the model favors (>50%)."""
    pieces = _build_candidate_pieces(tracked_edges, model_only_by_fight)
    return _find_parlays(
        pieces, leg_counts=(2, 3), min_american=100, max_american=320,
        min_leg_prob=0.50, max_results=max_results, label="bankroll",
    )


def build_lotto_parlays(tracked_edges: list[dict], model_only_by_fight: dict | None = None, max_results: int = 3) -> list[dict]:
    """+1000 or higher combos, 2-5 pieces -- leg count doesn't matter, only the payout does."""
    pieces = _build_candidate_pieces(tracked_edges, model_only_by_fight)
    return _find_parlays(
        pieces, leg_counts=(2, 3, 4, 5), min_american=1000, max_american=None,
        min_leg_prob=0.15, max_results=max_results, label="lotto",
    )


def build_moonshot_parlays(tracked_edges: list[dict], model_only_by_fight: dict | None = None, max_results: int = 3) -> list[dict]:
    """
    +5000 or higher, any leg count from 2 up to 8. This has essentially no
    business hitting -- it's the "why not" tier, built purely for fun. Even
    the longest of long shots still gets ranked by the model's best combined
    probability among everything that clears the bar, so it's the "best
    worst bet" rather than a totally random pile of legs.
    """
    pieces = _build_candidate_pieces(tracked_edges, model_only_by_fight)
    return _find_parlays(
        pieces, leg_counts=(2, 3, 4, 5, 6, 7, 8), min_american=5000, max_american=None,
        min_leg_prob=0.05, max_results=max_results, label="moonshot",
    )
