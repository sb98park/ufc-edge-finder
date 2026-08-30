"""
The plays ledger: every bet the model said it was placing, written down when
it said it.

SEPARATE FROM data/predictions_log.csv, AND THAT IS THE POINT. The predictions
log is the published moneyline record -- 19-1 on High Confidence, 9-0 on
Locks, all of it staked from the confidence ladder and none of it ever to be
restated. This file starts empty on the day it ships and answers a different
question: not "was the pick right" but "did the bet make money". A pick can be
right at a price that lost you nothing worth having, which is the entire
reason src/plays.py exists.

SET ONCE, LIKE pick_odds, AND FOR THE SAME REASON. A card is republished every
few minutes for a week. If price, stake and probability moved with each render,
the ledger would record what we would have bet knowing what we know now, and a
record assembled with hindsight is not a record. So the first publication of a
(fight, market, selection) fixes:

    odds_american   the price we said we were taking
    units           the stake Kelly wanted AT THAT PRICE
    model/fair/blended_prob, ev_per_unit, required_prob
    venue           where that price was quoted -- everything today is
                    Polymarket, which is peer-to-peer and vig-free, and when a
                    sportsbook feed comes online the ledger has to stay
                    readable across the change rather than silently mixing two
                    different kinds of number
    published_at

and nothing afterwards may change them. Later renders only advance last_seen
and closing_odds.

WHICH MEANS THE CARD BUDGET IS SPENT AS IT IS COMMITTED. A play published on
Tuesday is money already down. Wednesday's render cannot un-bet it because the
price moved, and cannot pretend the card still has a full 20 units of room --
so committed plays are fed back into select_card, where they occupy their
(fight, axis) slot and spend their share of every cap. Without that, a card
republished 2,000 times over a fight week gets a fresh 20U of room on each one
and "20 units on a card" describes nothing at all.

CLOSING PRICE IS RECORDED BUT NEVER STAKED. It is what makes CLV computable on
plays the same way it already is on picks, and it carries the same guard: a
market does not delist when the horn sounds, it prints 0.9995 while it
resolves, and writing that into closing_odds grades the model against its own
scoreboard.

THE RESULT IS NOT WRITTEN HERE. Grading happens from data/fight_results.csv
through src/play_settlement, so the render path stays incapable of inventing an
outcome.
"""

from __future__ import annotations

import csv
import datetime as dt
import os

from src.odds_utils import american_to_implied_prob, american_to_decimal
from src.play_settlement import settle_play

LEDGER_PATH = "data/plays_ledger.csv"

FIELDNAMES = [
    "play_id", "event_name", "event_date", "fighter_a", "fighter_b",
    "card_position", "weight_class",
    "axis", "market", "selection", "label", "tier", "is_lock", "is_prop",
    "odds_american", "venue", "units", "to_win",
    "model_prob", "fair_prob", "blended_prob", "ev_per_unit", "required_prob",
    "published_at", "last_seen", "closing_odds",
    "result", "units_result", "graded_at", "void_reason",
]

# HOW LONG A PLAY MAY STAY OPEN AFTER ITS CARD.
#
# A card grades over about four hours, and results land as the fetcher picks
# them up, so "the event has some results" is not evidence that a particular
# fight has happened. Two days is well past the last prelim and well short of
# anything a human would call slow.
#
# Past this line an open play is not waiting, it is stuck -- and a stuck play
# is worse than a lost one, because it never settles, never reaches the
# bankroll, and (since summarise_by_event is settled-only) silently vanishes
# from the card instead of showing as a hole.
VOID_AFTER_DAYS = 2

# Set on first publication and never rewritten. Everything not in here is
# either advanced by later renders (last_seen, closing_odds) or filled in by
# grading, and everything in here is a claim about a moment that has passed.
_SET_ONCE = (
    "odds_american", "venue", "units", "to_win", "model_prob", "fair_prob",
    "blended_prob", "ev_per_unit", "required_prob", "published_at",
    "event_name", "event_date", "fighter_a", "fighter_b", "card_position",
    "weight_class", "axis", "market", "selection", "label", "tier",
    "is_lock", "is_prop",
)

# Same threshold and same reasoning as track_record._is_settled_price.
_SETTLED_PROB = 0.97


def _is_settled_price(american) -> bool:
    try:
        p = american_to_implied_prob(float(american))
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return p is None or p >= _SETTLED_PROB or p <= (1.0 - _SETTLED_PROB)


def fight_key(fighter_a, fighter_b) -> frozenset:
    """The pipeline's fight key: order-insensitive, case-folded."""
    return frozenset({str(fighter_a or "").strip().lower(),
                      str(fighter_b or "").strip().lower()})


def play_id(event_name: str, fighter_a: str, fighter_b: str,
            market: str, selection: str | None) -> str:
    """
    Stable across rebuilds, which is what makes the merge a merge.

    Includes the EVENT because a rematch is a different bet, and the selection
    because one fight can carry a moneyline and a method play at once.
    """
    return "|".join([str(event_name or ""), str(fighter_a or ""), str(fighter_b or ""),
                     str(market or ""), str(selection or "")])


# EVERYTHING COMES BACK TYPED, because CSV gives strings and record_plays
# hands back a mixture: rows written this render carry real numbers, rows read
# from disk carry "2.5". A template formatting one of those and comparing the
# other is a bug waiting for the second week of a card, so the coercion
# happens once, here, at the only door into this file. A blank stays None
# rather than becoming 0.0 -- an ungraded play has no result, and zero is a
# result.
_NUMERIC = ("odds_american", "closing_odds", "units", "to_win", "model_prob", "fair_prob",
            "blended_prob", "ev_per_unit", "required_prob", "units_result")
_INTEGER = ("odds_american", "closing_odds")
_BOOLEAN = ("is_lock", "is_prop")


def _coerce(row: dict) -> dict:
    out = dict(row)
    for k in _NUMERIC:
        v = out.get(k)
        if v is None or v == "":
            out[k] = None
            continue
        try:
            out[k] = int(round(float(v))) if k in _INTEGER else float(v)
        except (TypeError, ValueError):
            out[k] = None
    for k in _BOOLEAN:
        out[k] = str(out.get(k) or "").strip() not in ("", "0", "False", "false")
    return out


def _serialise(row: dict) -> dict:
    """
    The inverse of _coerce, and it MUST exist.

    Without it this file quietly destroyed itself on the second render.
    _coerce turns is_prop into a Python bool and an empty closing_odds into
    None; the writer then wrote those straight back, so "1" became "True" and
    "" became "None". Every row read, rewritten, and no longer equal to itself
    -- which is precisely what check_plays_ledger.py calls a restated play, so
    the gate failed the build and the site froze on the last good payload.

    The gate was right. One door in, one door out, and this is the way out.
    """
    out = dict(row)
    for k in _BOOLEAN:
        out[k] = "1" if out.get(k) in (True, 1, "1") else "0"
    for k in FIELDNAMES:
        if out.get(k) is None:
            out[k] = ""
    return {k: out.get(k, "") for k in FIELDNAMES}


def load(path: str = LEDGER_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [_coerce(r) for r in csv.DictReader(fh)]


def committed_for(event_name: str | None, rows: list[dict],
                  event_date: str | None = None) -> list[dict]:
    """
    What select_card needs to know about money already on this card.

    Deliberately the minimum -- fight_id, axis, units -- because this is only
    ever used to spend budget and hold a slot, never to render anything. A
    voided play frees its budget back up; nothing is riding on it any more.

    MATCHED ON THE DATE TOO, NOT THE NAME ALONE. card_discovery renames an
    event automatically whenever the lineup changes ("same date, different
    name -- replacing the old entry"), and play_id embeds event_name. After a
    rename this returned NOTHING for the card, so select_card saw a fresh full
    budget and wrote a SECOND set of plays for fights already backed -- at that
    day's price, under new play_ids. grade_rows keys on the fighter pair alone,
    so both rows then graded, and bankroll.apply_settled compounded both
    because the ids differ. Reproduced on a copy of the real ledger: one 5U bet
    at -625 became 10U staked returning +2.36U instead of +0.80U, and
    check_plays_ledger passed throughout, because nothing was rewritten -- rows
    were merely added.

    A card's DATE does not change when its name does, so it is the stable half
    of the identity. The name still matches on its own for any row written
    before this file carried event_date.
    """
    out = []
    for r in rows:
        if event_name and r.get("event_name") != event_name:
            same_date = (event_date and str(r.get("event_date") or "").strip() == str(event_date).strip())
            if not same_date:
                continue
        if (r.get("result") or "").lower() == "void":
            continue
        try:
            units = float(r.get("units") or 0)
        except (TypeError, ValueError):
            continue
        out.append({"fight_id": f"{r.get('fighter_a')}|{r.get('fighter_b')}",
                    "axis": r.get("axis"), "units": units})
    return out


def _row_from_play(play: dict, event_name: str | None, event_date, now: str) -> dict:
    a, b = (play.get("fight_key") or "|").split("|", 1)
    return {
        "play_id": play_id(event_name, a, b, play.get("market"), play.get("selection")),
        "event_name": event_name or "", "event_date": event_date or "",
        "fighter_a": a, "fighter_b": b,
        "card_position": play.get("card_position") or "",
        "weight_class": play.get("weight_class") or "",
        "axis": play.get("axis") or "", "market": play.get("market") or "",
        "selection": play.get("selection") or "", "label": play.get("label") or "",
        "tier": play.get("tier") or "", "is_lock": "1" if play.get("is_lock") else "0",
        "is_prop": "1" if play.get("is_prop") else "0",
        "odds_american": play.get("odds_american"),
        "venue": play.get("venue") or "", "units": play.get("units"),
        # Stored rather than derived at render time: both inputs are set once,
        # so this cannot drift from them, and it makes the file readable by a
        # person auditing what was risked and what it returned.
        "to_win": play.get("to_win"),
        "model_prob": play.get("model_prob"), "fair_prob": play.get("fair_prob"),
        "blended_prob": play.get("blended_prob"),
        "ev_per_unit": play.get("ev_per_unit"),
        "required_prob": play.get("required_prob"),
        "published_at": now, "last_seen": now, "closing_odds": "",
        "result": "", "units_result": "", "graded_at": "",
    }


def record_plays(card: dict, now: str, live_prices: dict | None = None,
                 path: str = LEDGER_PATH) -> list[dict]:
    """
    Merge this render's plays into the ledger and return every row for the
    event, published order preserved.

    `live_prices` maps play_id -> current American price, used only to advance
    closing_odds. Pass the whole card's prices, including for plays that are no
    longer being selected: a bet placed on Tuesday still has a closing line.
    """
    rows = load(path)
    by_id = {r["play_id"]: r for r in rows}
    order = [r["play_id"] for r in rows]
    event_name = card.get("event_name")

    for play in card.get("plays") or []:
        row = _row_from_play(play, event_name, card.get("event_date"), now)
        prior = by_id.get(row["play_id"])
        if prior is None:
            by_id[row["play_id"]] = row
            order.append(row["play_id"])
            continue
        # ALREADY PUBLISHED. Every set-once field keeps the value it was first
        # written with; the only thing this render has to say is that the play
        # is still on the board.
        prior["last_seen"] = now

    for pid, price in (live_prices or {}).items():
        prior = by_id.get(pid)
        if prior is None or price is None:
            continue
        if _is_settled_price(price):
            continue        # the result wearing a price -- see the docstring
        prior["closing_odds"] = round(float(price))
        prior["last_seen"] = now

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for pid in order:
            w.writerow(_serialise(by_id[pid]))

    return [_coerce(by_id[pid]) for pid in order
            if by_id[pid].get("event_name") == event_name]


def void_stale(rows: list[dict], now: str, days: int = VOID_AFTER_DAYS) -> list[dict]:
    """
    Void every play still open more than `days` after its card, and say why.

    WHAT THIS IS ACTUALLY CATCHING: opponent changes. Measured across every
    graded card in predictions_log, nine fights never joined to a result and
    six of them had ONE fighter present under a different pairing -- a late
    replacement, which in the UFC is routine rather than exceptional.

    A book voids the whole market when that happens. Not just the props: the
    moneyline too, because the bet was on a matchup and the matchup no longer
    exists. New opponent, new lines, old tickets refunded. So this does not
    try to be clever about settling the fighter we backed against whoever
    turned up -- there was no bet to settle.

    Also catches a fight scrapped outright, which resolves the same way.

    Returns the rows it changed, so the caller can report them rather than
    letting a silent void look like a clean card.
    """
    changed = []
    for r in rows:
        if (r.get("result") or "").strip():
            continue
        try:
            when = dt.date.fromisoformat(str(r.get("event_date"))[:10])
        except (TypeError, ValueError):
            continue
        try:
            today = dt.date.fromisoformat(str(now)[:10])
        except (TypeError, ValueError):
            continue
        if (today - when).days <= days:
            continue
        r["result"] = "void"
        r["units_result"] = "0.0"
        r["graded_at"] = now
        r["void_reason"] = ("no result for this pairing " + str(days + 1) + "+ days after the card "
                            "-- opponent change or scratched fight, which voids the market")
        changed.append(r)
    return changed


def grade_rows(rows: list[dict], results_by_fight: dict, now: str) -> int:
    """
    Settle every ungraded play we now have a result for. Returns how many
    changed.

    `results_by_fight` is keyed on frozenset({fighter_a.lower(),
    fighter_b.lower()}) -- the convention the rest of the pipeline already
    uses, and order-insensitive, which matters because a card can be
    re-scraped with the corners the other way round.

    A play this cannot settle stays ungraded rather than being guessed at, and
    a cancelled fight voids at zero rather than losing -- money that was never
    at risk did not lose.
    """
    changed = 0
    for r in rows:
        if r.get("result"):
            continue
        res = results_by_fight.get(fight_key(r.get("fighter_a"), r.get("fighter_b")))
        if not res:
            continue
        if res.get("cancelled"):
            r["result"], r["units_result"], r["graded_at"] = "void", "0.0", now
            r["void_reason"] = "fight cancelled"
            changed += 1
            continue
        outcome = settle_play(r.get("market"), r.get("selection"), res)
        if outcome is None:
            continue
        try:
            units = float(r.get("units") or 0)
            price = float(r.get("odds_american"))
        except (TypeError, ValueError):
            continue
        won = bool(outcome)
        r["result"] = "won" if won else "lost"
        r["units_result"] = round(units * (american_to_decimal(price) - 1.0), 2) if won else -units
        r["graded_at"] = now
        changed += 1
    return changed


def write_graded(rows: list[dict], path: str = LEDGER_PATH) -> None:
    """
    Persist rows that grading has just changed.

    Separate from record_plays on purpose: that function is the RENDER path
    and must never be able to write a result, and this one is the grading path
    and must never be able to write a price. Two doors, so neither can do the
    other's job by accident.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(_serialise(r))


def summarise_by_event(rows: list[dict]) -> dict:
    """
    Per-card totals, for the track record's Bets tab. SETTLED PLAYS ONLY.

    Keyed on event_name because that is what the results grouping uses. A card
    with no rows here simply is not in the dict, which is how the template
    tells a card graded before the plays ledger existed from one graded after
    -- no changeover date to hardcode and get wrong.

    THE SETTLED FILTER IS A PAYWALL BOUNDARY, NOT A TIDINESS ONE. This output
    is classified free in src/tiering, because a graded card's bets are the
    public record. An UNGRADED card's rows are the opposite: label, price and
    stake on fights that have not happened, which is the model layer exactly.
    Filtering here is what makes the free classification true, rather than
    true-for-now-because-no-template-renders-it. An earlier version of this
    function returned open rows and carried a comment claiming a tier check
    that was never written; the check is this line.
    """
    out: dict = {}
    for r in rows:
        name = r.get("event_name")
        if not name:
            continue
        if (r.get("result") or "").strip() == "":
            continue
        e = out.setdefault(name, {"plays": [], "won": 0, "lost": 0, "void": 0,
                                  "units": 0.0, "staked": 0.0, "settled": 0})
        e["plays"].append(r)
        result = (r.get("result") or "").lower()
        try:
            units = float(r.get("units") or 0)
        except (TypeError, ValueError):
            units = 0.0
        if result == "void":
            e["void"] += 1
            continue
        e["settled"] += 1
        e["staked"] += units
        try:
            e["units"] += float(r.get("units_result") or 0)
        except (TypeError, ValueError):
            pass
        e["won" if result == "won" else "lost"] += 1
    for e in out.values():
        e["units"] = round(e["units"], 2)
        e["staked"] = round(e["staked"], 1)
        e["roi_pct"] = round(e["units"] / e["staked"] * 100, 1) if e["staked"] else None
        # Biggest stake first, the way a slip reads, then by what it returned.
        e["plays"].sort(key=lambda p: (-(p.get("units") or 0),
                                       -(p.get("units_result") or 0)))
    return out


def summarise(rows: list[dict]) -> dict:
    """
    The record, and only from settled rows. Void plays count in neither the
    W-L nor the staked total: nothing was at risk.
    """
    won = lost = 0
    units = staked = 0.0
    for r in rows:
        result = (r.get("result") or "").lower()
        if result not in ("won", "lost"):
            continue
        try:
            units += float(r.get("units_result") or 0)
            staked += float(r.get("units") or 0)
        except (TypeError, ValueError):
            continue
        if result == "won":
            won += 1
        else:
            lost += 1
    return {
        "won": won, "lost": lost, "settled": won + lost,
        "units": round(units, 2), "staked": round(staked, 2),
        "roi_pct": round(units / staked * 100, 1) if staked else None,
    }
