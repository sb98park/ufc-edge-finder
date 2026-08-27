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

import json
import os
from datetime import datetime, timezone

from src.parlay_builder import BETTABLE_VENUES
from src.plays import (
    size_play, select_card, decimal_odds, ev_per_unit, required_prob,
    TIER_CAP_UNITS, HURDLE_MONEYLINE,
    AXIS_OUTCOME, AXIS_MANNER,
)

# THE TWO TIERS WITH A RECORD ARE PLAYED ON THE RECORD, NOT ON THE HURDLE.
#
# The EV hurdle was refusing them, and measured against every graded pick we
# have, it was refusing money:
#
#     Locks and high confidence the hurdle REFUSED   12 picks, 12-0, +21.10U
#     Locks and high confidence the hurdle CLEARED    8 picks,  7-1, +39.20U
#
# 12-0 is above expectation -- the market priced those at 79.9% and expected
# 9.6 winners, and the run has about a 6.6% chance of happening by luck -- so
# the +24.8% ROI is not the number to plan around. But that is not the point.
# The point is the EV: those refused picks averaged +1.02% per unit ON OUR OWN
# BLENDED PROBABILITY. They were not negative-expectation bets that got lucky.
# They were thin positive-expectation bets sitting under a threshold set for a
# different purpose.
#
# The hurdle exists to stop the props board selling us a 95% read at -1000,
# where four points of edge is fragile and the market is thin. On a moneyline
# the model has been graded 19-1 on, it was solving a problem that was not
# there -- and it broke something real in the process: the published record IS
# "every lock at 10U and every high-confidence pick at 5U", and a plays
# section that quietly skips half of them is no longer the system the landing
# page shows a curve of. A subscriber following the plays could not reproduce
# the record they subscribed for.
#
# So these two tiers ride the published ladder, unconditionally. Everything
# else -- Medium, Low, and every prop -- still has to earn its place on EV.
_LADDER_TIERS = ("Lock of the Week", "High Confidence")

# THE DISCRETIONARY LAYER IS OFF, AND THE BACKTEST IS WHY.
#
# Replaying every graded moneyline pick through each rule, 84 picks over the
# 7 cards with results:
#
#     ladder only (locks + high)     20 picks, 19-1, 150U staked, +60.30U (+40.2%)
#     ladder + discretionary         42 picks, 26-16, 184U staked, +55.94U (+30.4%)
#
# The layer I added returns LESS than not adding it: the 22 Medium and Low
# picks it selected went -4.36U on 34U staked, -12.8%. And it is worse than
# the naive alternative -- taking every Medium and Low at its ladder stake
# returns +7.99U over the same tiers. So the hurdle is not merely failing to
# add value there; it is selecting worse than not selecting at all.
#
# n=22 is far too small to call the rule broken rather than unlucky, and the
# effective sample is 7 CARDS. That cuts both ways: it is also far too small
# to justify risking a third of a bankroll on a layer with no record, when the
# layer beside it has one. The prop half cannot be judged at all -- 894 quotes
# recorded, 50 settled.
#
# So this is a pause, not a verdict. The prop ledger keeps recording, the
# selector keeps computing candidates, and everything that would have been
# played is still returned under `shelved` so the page can show what the rule
# WOULD have taken. Turn it back on when the props have a settled record, or
# when the discretionary moneylines have enough of one to argue with.
DISCRETIONARY_PLAYS = False

# HOW FAR THE MODEL MAY DISAGREE WITH A LIQUID MARKET BEFORE WE STOP CALLING
# IT AN EDGE.
#
# Measured on UFC Fight Night: Nurmagomedov vs. Song, the model sits BELOW the
# market on all seven market favourites (median -12.0 points) and ABOVE it on
# all four market dogs (median +20.7). That is not a scatter of independent
# disagreements; it is one systematic compression of the probability scale
# toward 50%, which is what an Elo backbone does. The staking rule is
# price-neutral -- the hurdle is exactly 5% expected return per unit risked at
# every price -- so the model's compression, and nothing else, is why every
# qualifying moneyline on that card was a plus-money underdog.
#
# The worst case was Aoriqileng: market 20.5%, model 53.5%. A 33-point
# disagreement with a market that has real money on both sides is evidence
# that the MODEL is wrong, not the market. Past this line we do not bet.
MAX_MODEL_DISAGREEMENT = 0.25

# MARKETS WE DO NOT STAKE, and why each one is here rather than merely
# unprofitable.
#
# "Does not end by SUB" at -102 is a near-certainty wearing a coinflip's
# price: it wins whenever the fight is a decision, a KO, a DQ or a doctor
# stoppage. The model can be right about it every week and the bet still adds
# nothing but variance to a bankroll, because the thing it is fading is rare
# and the price is not compensating for the times it lands. A negated prop is
# a bet against a tail, and we are not in that business.
_UNSTAKED_PREFIX = "Fight Method: Not "

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
    if m.startswith(_UNSTAKED_PREFIX):
        return None
    # EVERY OTHER PROP IS THE SAME QUESTION. "Fight ends by KO/TKO", "Over 2.5
    # rounds" and "Goes the distance" are three ways of asking how this fight
    # finishes, and they move together. One per bout.
    if (m.startswith("Fight Method: ") or m.startswith("Total Rounds ")
            or m.startswith("Round Betting: ") or m.startswith("Fight Outcome: ")):
        return AXIS_MANNER
    return None


def label_for(market: str, fighter: str | None, matchup: str) -> str:
    """
    What the play is called on the page. Written the way someone would say it
    out loud, because a play the reader has to decode is a play they will get
    wrong at the counter.
    """
    m = (market or "").strip()
    if m == "Moneyline":
        # "Moneyline", not "to win". This is the name the market has at the
        # book the reader is about to place it at, and a play they have to
        # translate is a play they get wrong at the counter.
        return f"{fighter} Moneyline"
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


def _disagreement(edge: dict) -> float | None:
    """How far the raw model sits from the de-vigged market, in probability."""
    try:
        return abs(float(edge["model_prob"]) - float(edge["book_fair_prob"]))
    except (KeyError, TypeError, ValueError):
        return None


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


HEALTH_PATH = "data/source_health.json"


def _record_incoherent(fighter, market, model, fair, blend, edge=None,
                       source=None, odds=None, path: str = HEALTH_PATH) -> None:
    """
    Leave the refusal somewhere a person can read without CI log access.

    The guard below prints, and print goes to the Actions log, which is the
    one place this repo cannot get at from a laptop -- there is no gh here.
    source_health.json is committed by the refresh job, so a build that
    refuses to stake says so in the diff instead of only in a log nobody
    fetches.

    Keyed by fighter and market rather than appended, so the file describes
    what is wrong NOW instead of growing a line every five minutes; `at`
    is there to tell a live entry from one a later fix already cleared.

    Written where the refusal happens rather than batched at the end,
    because a build that dies mid-card is exactly when the reason matters.
    Every source_health writer runs during the props fetch, long before card
    selection, so this cannot land under one of them.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    entries = payload.get("incoherent_blends")
    if not isinstance(entries, dict):
        entries = {}
    # EDGE_PCT IS THE DIAGNOSTIC, and it is why this records more than the
    # three numbers that failed the check. edge_pct and blended_prob are
    # produced together by _two_numbers from one (model, fair) pair, so
    # whether the EDGE still agrees with the stored pair says which pass went
    # stale:
    #
    #   edge agrees, blend does not -> blended_prob alone is from another
    #                                  pass; something rewrote it, or rewrote
    #                                  the fair it was derived from
    #   neither agrees              -> model_prob or book_fair_prob was
    #                                  overwritten AFTER _two_numbers ran and
    #                                  the whole triple is stale together
    #
    # Two local reproductions of the multi-source path -- with and without the
    # rounds-reconciliation fix -- failed to produce this, so the next real
    # occurrence is the only evidence available and it should arrive already
    # narrowed rather than as a bare flag.
    expected_edge = round((model - fair) * 100.0, 2)
    entry = {
        "model_prob": round(model, 4), "fair_prob": round(fair, 4),
        "blended_prob": round(blend, 4),
        "source": str(source or ""), "odds_american": odds,
        "edge_pct_stored": edge,
        "edge_pct_from_stored_pair": expected_edge,
        "edge_agrees": (edge is not None
                        and abs(float(edge) - expected_edge) < 0.05),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entry["reads_as"] = (
        "blended_prob alone is from another pass" if entry["edge_agrees"]
        else "model_prob or book_fair_prob was overwritten after the blend")
    entries[f"{fighter}|{market}"] = entry
    payload["incoherent_blends"] = entries
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except OSError as exc:
        print(f"[card_plays] could not record the refusal ({exc}) -- continuing")


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

        on_ladder = is_moneyline and tier in _LADDER_TIERS
        if on_ladder:
            # THE PUBLISHED SYSTEM, EXECUTED. No hurdle, no Kelly -- the stake
            # is the one this tier's whole record was built at. See
            # _LADDER_TIERS for the measurement behind that.
            sized = {
                "play": True, "units": TIER_CAP_UNITS[tier], "reason": None,
                "capped": False, "ev_per_unit": round(ev_per_unit(p, price), 4),
                "implied": None, "kelly": None, "cap": TIER_CAP_UNITS[tier],
                "required_prob": round(required_prob(price, HURDLE_MONEYLINE), 4),
            }
        else:
            sized = size_play(p, price, tier, is_prop=not is_moneyline)

            # SANITY BEFORE SIZE. A disagreement this large is a model failure
            # wearing an edge's clothes -- see MAX_MODEL_DISAGREEMENT. Not
            # applied to the ladder tiers: there the tier's own graded record
            # is the evidence, and it is a better one than this heuristic.
            gap = _disagreement(edge)
            if sized["play"] and gap is not None and gap > MAX_MODEL_DISAGREEMENT:
                sized = dict(sized, play=False, units=0.0, reason=(
                    f"the model is {gap * 100:.0f} points off a market with money on "
                    f"both sides, which is a disagreement we distrust rather than an edge"))

        # A STAKE NEEDS A PRICE YOU CAN PLACE.
        #
        # Every play on this card was priced at Polymarket, which is carried
        # precisely because it is VIG-FREE -- the fair line the model measures
        # edge against. Staking there means the units, the ROI and the
        # bankroll are all computed at prices better than any book offers, so
        # the record would run flatteringly high forever while describing bets
        # a subscriber cannot make. Umar at -388 and Rei at -545 were
        # Polymarket numbers; a DraftKings bettor pays vig on both.
        #
        # So a reference price can still produce a PICK -- it appears in the
        # track record like every other call -- but it cannot produce a STAKE.
        # The cost is real and was chosen knowingly: on a week when the books
        # quote thinly, fewer plays are staked. A thin week is honest; a
        # record measured at unobtainable prices is not.
        _venue_name = _venue(edge)
        if sized["play"] and _venue_name not in BETTABLE_VENUES:
            sized = dict(sized, play=False, units=0.0, reason=(
                f"priced at {_venue_name or 'no named venue'}, which is a reference "
                f"line rather than a book this can be placed at"))
        # A BLEND THAT IS NOT BETWEEN ITS OWN INPUTS DID NOT COME FROM THEM.
        #
        # blended_prob is market_blended_prob(model_prob, book_fair_prob) -- a
        # convex combination, so whatever the weight is, the result has to sit
        # between the two. When it does not, the row is carrying numbers from
        # two different passes: one field was recomputed after a price or a
        # model probability moved and the others kept their old values.
        #
        # This is not hypothetical. Both moneylines staked for Nurmagomedov
        # vs. Song on 2026-08-26 were written with model 0.593 / fair 0.839 /
        # blend 0.483 and model 0.445 / fair 0.808 / blend 0.367. No weight in
        # [0, 1] produces either. The same fights rebuilt from the live
        # pipeline gave 0.785 / 0.845 / 0.827 and 0.751 / 0.825 / 0.803.
        # Because EV is quoted off the blend, the two rows reported -0.448 and
        # -0.564 units per unit and were staked five units each anyway: the
        # ladder branch above sizes off the tier, not off EV, so nothing
        # downstream was ever going to look at those numbers and object.
        #
        # The invariant is checked rather than the arithmetic re-run, because
        # re-deriving the blend here would paper over whichever upstream pass
        # is stale and hand back a confident number built on one stale input.
        # An incoherent row still PUBLISHES as a pick -- the model made a call
        # and the call is on the record -- it just cannot carry money.
        if sized["play"]:
            _m, _f = edge.get("model_prob"), edge.get("book_fair_prob")
            try:
                _m, _f = float(_m), float(_f)
            except (TypeError, ValueError):
                _m = _f = None
            if _m is not None and not (min(_m, _f) - 1e-6 <= p <= max(_m, _f) + 1e-6):
                print(f"[card_plays] REFUSING TO STAKE {edge.get('fighter')} {market}: "
                      f"blended {p:.4f} is outside [{min(_m, _f):.4f}, {max(_m, _f):.4f}] "
                      f"-- model {_m:.4f}, fair {_f:.4f}. Stale derived field upstream.")
                _record_incoherent(edge.get("fighter"), market, _m, _f, p,
                                   edge=edge.get("edge_pct"),
                                   source=_venue_name, odds=price)
                sized = dict(sized, play=False, units=0.0, reason=(
                    "the blended probability on this row is not between the model's "
                    "number and the market's, so at least one of the three was "
                    "computed in a different pass than the others"))

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
            "venue": _venue_name,
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
            # 2 for the ladder tiers, so a full card can never let a round
            # total crowd out a lock; 1 for other moneylines; 0 for props.
            "priority": (2 if on_ladder else 1) if is_moneyline else 0,
            "on_ladder": on_ladder,
            # See select_card: a cap invented after the fact must not be able
            # to cancel a bet the published record is made of.
            "caps_exempt": on_ladder,
            "model_prob": edge.get("model_prob"),
            "fair_prob": edge.get("book_fair_prob"),
            "blended_prob": round(p, 4),
            "units": sized["units"],
            # WHAT IT RETURNS IF IT LANDS. Profit, not total return -- a
            # bettor thinks "2 to win 4.5", and printing the 6.5 that comes
            # back would read as a bigger win than it is.
            "to_win": round(sized["units"] * (decimal_odds(price) - 1.0), 2),
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
             "shelved": [], "discretionary_on": DISCRETIONARY_PLAYS,
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

    # See DISCRETIONARY_PLAYS. Held back rather than never computed, so the
    # section can be honest about what it is not betting and why.
    shelved = []
    if not DISCRETIONARY_PLAYS:
        shelved = [c for c in candidates if not c.get("on_ladder")]
        candidates = [c for c in candidates if c.get("on_ladder")]

    card = select_card(candidates, committed=committed)

    # THE PICKS THAT DID NOT MAKE IT, moneyline only. A reader following the
    # confidence tiers needs to see that the model still likes Umar and the
    # PRICE is the reason there is no play -- otherwise the plays section and
    # the fight card look like they disagree about the same fight.
    # COMMITTED PLAYS COUNT AS PLAYED. This read card["plays"] alone, which
    # holds only what THIS render added -- so once a moneyline was on the
    # board, any later render where its price no longer cleared the hurdle
    # listed it under "Picked, not played" while the bet was still live. Denise
    # Gomes appeared as a 2U play and as an unplayed pick on the same screen.
    played_fights = {p["fight_key"] for p in card["plays"] if p["axis"] == AXIS_OUTCOME}
    played_fights |= {c.get("fight_id") for c in (committed or [])
                      if c.get("axis") == AXIS_OUTCOME}
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
        "shelved": sorted(shelved, key=lambda c: -c["ev_per_unit"]),
        "discretionary_on": DISCRETIONARY_PLAYS,
        "fights_considered": considered,
    }
