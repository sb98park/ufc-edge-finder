"""
Did a published play win?

WHY THIS IS ITS OWN MODULE. The predicates below already existed inside
scripts/grade_prop_prices.py, which grades the recorded derivable-market
quotes. The plays ledger needs the same questions answered -- was it a
decision, did it go past 2.5 rounds -- about the same fights, from the same
results file. Two copies of "is this a KO" is how two ledgers end up
disagreeing about one night, so the script now imports these rather than
keeping its own.

WHAT IS DIFFERENT HERE is the vocabulary, not the logic. prop_ledger records
markets as "TotalRounds" / "GoesTheDistance"; an edge row calls the same
things "Total Rounds Over 2.5" / "Fight Outcome: Goes The Distance". This
module settles the edge-row spelling and adds the one market prop_ledger has
no concept of: the moneyline.

THE CLOCK COUNTS UP. data/fight_results.csv stores elapsed time within the
round, not time remaining. That is measured rather than assumed -- joining it
to ufcstats data matched on 21 of 21 bouts for count-up and 0 of 21 for
count-down. A round-2 finish at 0:30 and one at 4:30 fall on opposite sides
of an Over 1.5 line, so this is not a detail.

UNKNOWN IS A REAL ANSWER. Anything this cannot settle returns None and stays
ungraded rather than being guessed at. A ledger that resolves a bet it does
not understand is worse than one with a gap in it.
"""

from __future__ import annotations

import re
import unicodedata

ROUND_SECONDS = 300

WON, LOST, UNKNOWN = True, False, None


def fold(s) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def clock_seconds(value):
    m = re.fullmatch(r"(\d+):([0-5]\d)", str(value).strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def is_decision(method) -> bool:
    return "dec" in fold(method)


def is_ko(method) -> bool:
    m = fold(method)
    return "ko" in m or "tko" in m


def is_sub(method) -> bool:
    return "sub" in fold(method)


def elapsed_rounds(end_round, end_time, count_up: bool = True):
    """Total rounds elapsed at the stoppage, under the stated convention."""
    secs = clock_seconds(end_time)
    if secs is None or end_round in (None, ""):
        return None
    try:
        rnd = int(float(end_round))
    except (TypeError, ValueError):
        return None
    within = secs if count_up else (ROUND_SECONDS - secs)
    return (rnd - 1) + within / ROUND_SECONDS


def settle_play(market: str, selection: str | None, result: dict,
                count_up: bool = True):
    """
    WON / LOST / UNKNOWN for one play against one fight result.

    `result` needs winner, method, end_round, end_time -- the columns of
    data/fight_results.csv. `market` is the edge-row spelling.
    """
    m = (market or "").strip()
    sel = str(selection or "")
    winner = result.get("winner")
    method = result.get("method")

    if m == "Moneyline":
        if not winner:
            return UNKNOWN
        return WON if fold(sel) == fold(winner) else LOST

    if m == "Fight Outcome: Goes The Distance":
        return WON if is_decision(method) else LOST
    if m == "Fight Outcome: Ends In Finish":
        return LOST if is_decision(method) else WON

    if m.startswith("Fight Method: "):
        want = m.split(": ", 1)[1]
        negated = want.startswith("Not ")
        base = fold(want[4:] if negated else want)
        if base in ("ko/tko", "ko", "tko"):
            hit = is_ko(method)
        elif base in ("sub", "submission"):
            hit = is_sub(method)
        elif base.startswith("dec"):
            hit = is_decision(method)
        else:
            return UNKNOWN
        return LOST if (hit == negated) else WON

    if m.startswith("Method: "):
        # "<fighter> by <method>" -- BOTH halves must hold. A pick that wins
        # the fight the wrong way is a losing bet, which is exactly why these
        # sit on the outcome axis rather than the method one.
        if not winner or fold(sel) != fold(winner):
            return LOST
        want = fold(m.split(": ", 1)[1])
        if want in ("ko/tko", "ko", "tko"):
            return WON if is_ko(method) else LOST
        if want in ("sub", "submission"):
            return WON if is_sub(method) else LOST
        if want.startswith("dec"):
            return WON if is_decision(method) else LOST
        return UNKNOWN

    if m.startswith("Total Rounds "):
        mt = re.match(r"(Over|Under)\s+([\d.]+)", m.split("Total Rounds ", 1)[1])
        if not mt:
            return UNKNOWN
        side, line = mt.group(1).lower(), float(mt.group(2))
        if is_decision(method):
            # A decision clears every offered line by construction, so the
            # scheduled distance never has to be known.
            return WON if side == "over" else LOST
        total = elapsed_rounds(result.get("end_round"), result.get("end_time"), count_up)
        if total is None:
            return UNKNOWN          # never assume where in the round it ended
        return (total > line) if side == "over" else (total < line)

    return UNKNOWN
