"""
Turning a card into the bets we actually place.

src/plays.py holds the staking ARITHMETIC and knows nothing about this
project's data. This module is the adapter: it reads the edge rows a card
produces, decides which are eligible, and hands them to select_card. Still no
I/O and no clock -- events in, plays out -- so the tests can pin real cards.

WHICH PROBABILITY GETS STAKED, and why it is not the model's.

Every edge row carries three numbers: model_prob (what the model thinks),
book_fair_prob (the de-vigged market) and blended_prob (30% model, 70%
market -- see odds_utils.market_blended_prob for why that weight). Staking
runs on the BLENDED one, for the same reason the site's EV column does:
"Raw-model EV is flattering by construction -- it is built from the same
disagreement the edge measures, so it would report the model's own optimism
as profit."

That is not a stylistic preference, it is the difference between a card and a
catastrophe. Measured on UFC Fight Night: Nurmagomedov vs. Song, staking on
model_prob qualifies 60 plays worth 167 units on a single card, nearly every
one pinned to its cap. The same card on blended_prob qualifies 17 before
correlation and exposure trim it further. A rule that wants to risk 167% of a
bankroll in one night has not found 60 edges; it has found one bad
assumption.

WHAT COUNTS AS A PRICE. Everything quoting this pipeline today is Polymarket,
which is peer-to-peer and carries no margin -- source_is_vig_free is true on
all 292 current edges. So the price a play is recorded at is a real,
executable price on a real venue, not a "fair line" reconstruction, and the
5% hurdle is not fighting phantom vig: it is demanding genuine edge over a
market that already agrees with itself. Each play records the venue it was
priced at, so when a sportsbook feed comes online the ledger stays readable
across the change rather than silently mixing two different kinds of number.

WE NEVER PUBLISH A PLAY AGAINST OUR OWN PICK. On a moneyline only the
model's favorite is considered. The value side of a fight the model has
called the other way is a real thing the edge table can show; staking it
would mean the site tips one fighter and bets the other.
"""

from __future__ import annotations

from src.plays import (
    size_play, select_card,
    AXIS_OUTCOME, AXIS_METHOD, AXIS_DURATION,
)

# --- what kind of risk is this? ------------------------------------------
# The axis is not "what market is it called", it is "what independent thing
# has to happen". Two rows on the same axis are the same bet wearing
# different labels, and select_card keeps only the better one.
#
# A PER-FIGHTER METHOD BET SITS ON THE OUTCOME AXIS, not the method one.
# "Song Yadong by KO/TKO" cannot win unless "Song Yadong to win" wins; it is
# a subset, not a second opinion. Filing it under method would let one fight
# publish both and claim two units of independent exposure for one.
# "Fight ends by KO/TKO" -- either fighter -- genuinely is independent of who
# wins, and stays on the method axis.
def axis_for_market(market: str) -> str | None:
    m = (market or "").strip()
    if m == "Moneyline":
        return AXIS_OUTCOME
    if m.startswith("Method: "):           # per-fighter: "Method: KO/TKO"
        return AXIS_OUTCOME
    if m.startswith("Fight Method: "):     # fight-level: "Fight Method: SUB"
        return AXIS_METHOD
    if m.startswith("Total Rounds ") or m.startswith("Round Betting: "):
        return AXIS_DURATION
    if m.startswith("Fight Outcome: "):    # finish vs distance == duration
        return AXIS_DURATION
    return None


def label_for(market: str, fighter: str | None, matchup: str) -> str:
    """
    What the play is called on the page. Written the way someone would say it
    out loud, because a play the reader has to decode is a play they will get
    wrong at the counter.
    """
    m = (market or "").strip()
    if m == "Moneyline":
        return f"{fighter} to win"
    if m.startswith("Method: "):
        return f"{fighter} by {m.split(': ', 1)[1]}"
    if m.startswith("Fight Method: "):
        sel = m.split(': ', 1)[1]
        if sel.startswith("Not "):
            return f"Does not end by {sel[4:]}"
        return f"Fight ends by {sel}"
    if m.startswith("Total Rounds "):
        return m.split("Total Rounds ", 1)[1] + " rounds"
    if m == "Fight Outcome: Ends In Finish":
        return "Fight ends inside the distance"
    if m == "Fight Outcome: Goes The Distance":
        return "Fight goes the distance"
    return m


def _venue(edge: dict) -> str:
    """
    Where the price was quoted, or "" when nothing said.

    best_book arrives from pandas as a float NaN on any row no book quoted --
    and NaN is TRUTHY, so `best_book or source` handed it straight through and
    every fight-level prop rendered "Polymarket" as the word "nan".
    """
    for key in ("best_book", "source"):
        value = edge.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _fight_key(fight: dict) -> str:
    return f"{fight.get('fighter_a')}|{fight.get('fighter_b')}"


def candidates_for_fight(fight: dict) -> tuple[list[dict], list[dict]]:
    """
    Every staked candidate this fight offers, and every priced row that was
    considered and refused with the reason why.

    The refusals are returned rather than dropped because "why is my lock not
    on the card" is the first question this section will ever be asked, and a
    card that silently omits a fight is indistinguishable from a card that
    forgot it.
    """
    preview = fight.get("preview") or {}
    tier = preview.get("confidence_label") or ""
    favorite = preview.get("favorite")
    matchup = f"{fight.get('fighter_a')} vs {fight.get('fighter_b')}"
    taken: list[dict] = []
    refused: list[dict] = []

    if fight.get("cancelled"):
        return taken, refused

    for edge in fight.get("edges") or []:
        market = edge.get("market") or ""
        axis = axis_for_market(market)
        if axis is None:
            continue

        is_moneyline = market == "Moneyline"
        # See the module docstring: the model's own side only.
        if is_moneyline and edge.get("fighter") != favorite:
            continue
        # Same rule one level down -- "X by KO/TKO" where X is the underdog
        # the model did not pick is still a bet against our own read.
        if market.startswith("Method: ") and edge.get("fighter") != favorite:
            continue

        p = edge.get("blended_prob")
        price = edge.get("odds_american")
        if p is None or price is None:
            continue
        try:
            p, price = float(p), float(price)
        except (TypeError, ValueError):
            continue

        sized = size_play(p, price, tier, is_prop=not is_moneyline)
        row = {
            "fight_key": _fight_key(fight),
            "fight_id": _fight_key(fight),
            "matchup": matchup,
            "weight_class": fight.get("weight_class"),
            "card_position": fight.get("card_position"),
            "axis": axis,
            "market": market,
            "selection": edge.get("fighter"),
            "label": label_for(market, edge.get("fighter"), matchup),
            "odds_american": round(price),
            "venue": _venue(edge),
            # THE TIER BELONGS TO THE PICK, NOT TO EVERY BET ON THE FIGHT.
            # "Over 2.5 rounds -- Low Confidence" is a category error: the
            # tier describes how sure the model is about WHO WINS, which a
            # duration bet does not ask. Carried on the row that it actually
            # governed the stake for, and null everywhere else.
            "tier": tier if is_moneyline else None,
            "fight_tier": tier,
            "is_lock": bool(fight.get("is_lock_of_week")),
            "is_prop": not is_moneyline,
            # See select_card: the moneyline is the market this site has a
            # graded record in (19-1, and 58.3% of priced positions beat the
            # close). The prop board has none, which is why its hurdle is
            # double -- and why, when the card cap binds, it yields.
            "priority": 1 if is_moneyline else 0,
            "model_prob": edge.get("model_prob"),
            "fair_prob": edge.get("book_fair_prob"),
            "blended_prob": round(p, 4),
            "units": sized["units"],
            "ev_per_unit": sized["ev_per_unit"],
            "required_prob": sized["required_prob"],
            "capped": sized["capped"],
            "reason": sized["reason"],
        }
        (taken if sized["play"] else refused).append(row)

    return taken, refused


def build_card_plays(event: dict | None, committed: list[dict] | None = None) -> dict:
    """
    The staked card. One event -- the one the reader can actually bet.

    `committed` is what the ledger has already published for this card. Those
    plays are real money at a real price and are never restated; they are
    passed to select_card so they keep spending the card's budget, and the
    only thing this call decides is what to ADD. See src/plays_ledger.

    Returns plays, the moneyline picks that did NOT make it (with reasons, so
    a tiered pick never vanishes unexplained), and the totals.
    """
    empty = {"event_name": None, "plays": [], "passed": [], "dropped": [],
             "total_units": 0.0, "new_units": 0.0, "fights_considered": 0}
    if not event or not event.get("fights"):
        return empty

    candidates: list[dict] = []
    refused: list[dict] = []
    considered = 0
    for fight in event["fights"]:
        if fight.get("cancelled"):
            continue
        considered += 1
        t, r = candidates_for_fight(fight)
        candidates.extend(t)
        refused.extend(r)

    card = select_card(candidates, committed=committed)

    # THE PICKS THAT DID NOT MAKE IT, moneyline only. A reader following the
    # confidence tiers needs to see that the model still likes Umar and the
    # PRICE is the reason there is no play -- otherwise the plays section and
    # the fight card look like they disagree about the same fight.
    played_fights = {p["fight_key"] for p in card["plays"] if p["axis"] == AXIS_OUTCOME}
    passed = [
        r for r in refused
        if r["market"] == "Moneyline"
        and r["fight_key"] not in played_fights
        and r["tier"] in ("Lock of the Week", "High Confidence", "Medium Confidence")
    ]
    passed.sort(key=lambda r: -(r.get("blended_prob") or 0))

    return {
        "event_name": event.get("event_name"),
        "event_date": event.get("event_date"),
        "plays": sorted(card["plays"], key=lambda p: (-p["units"], -p["ev_per_unit"])),
        "passed": passed,
        "dropped": card["dropped"],
        "total_units": card["total_units"],
        "new_units": card["new_units"],
        "fights_considered": considered,
    }
