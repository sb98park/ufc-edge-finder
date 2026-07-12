"""
Estimates approximate start times for each fight on a card, anchored to
real known segment-start times rather than a single event-start time plus
a guessed uniform per-fight duration.

Uniform-duration modeling was tried first and was meaningfully wrong: it
put the UFC 329 main event around 8:56 PM, when the real expected time
(confirmed) is closer to 11:15 PM -- a two-hour-plus miss, because a big
main event with multiple undercard fights, walkouts, and ad breaks before
it genuinely takes far longer per slot than an early prelim. Anchoring to
known segment start times and distributing fights evenly BETWEEN anchors
is a much better fit to how these cards actually run.

This is still explicitly an ESTIMATE, not a live feed -- real fights run
early or late constantly (decisions run long, first-round finishes run
short, doctor stoppages, replay reviews). It exists to give the "is a
fight roughly live right now" determination something to work from
client-side, using the visitor's own clock via JS, rather than requiring
the site to regenerate every few minutes to track a real live feed no
free data source provides.

Card order in fight_cards.csv is listed Main Event first -- the REVERSE of
actual chronological fight order (early prelims happen first in real time).
This sorts back to true chronological order before assigning estimated times.
"""

import datetime as dt
import json
import os

_SEGMENT_ORDER = {"Early Prelims": 0, "Prelims": 1, "Main Card": 2, "Co-Main Event": 3, "Main Event": 4}

# Known/typical segment start anchors (ET). Main Card's own anchor is used
# as the START of the main-card block; Main Event's anchor is used as the
# END of that block (main card undercard + co-main get evenly distributed
# across the gap), since that's the one point in the night where "start +
# uniform slots" breaks down hardest -- a stacked main card with a big
# walkout-heavy main event runs meaningfully longer per fight than earlier
# in the night.
DEFAULT_SEGMENT_START = {
    "Early Prelims": "17:15",
    "Prelims": "19:00",
    "Main Card": "21:00",
}
DEFAULT_MAIN_EVENT_START = "23:15"
_MAIN_EVENT_FALLBACK_DURATION_MIN = 30


def _parse(event_date: str, time_str: str) -> dt.datetime:
    hour, minute = map(int, time_str.split(":"))
    return dt.datetime.fromisoformat(f"{event_date}T{hour:02d}:{minute:02d}:00")


def _fmt(d: dt.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S-04:00")


def build_fight_schedule(
    fights: list[dict], event_date: str, event_start_time_et: str,
    segment_starts: dict | None = None, main_event_start_et: str | None = None,
) -> list[dict]:
    """
    Returns fights in true chronological order, each annotated with
    estimated_start_iso and estimated_end_iso. segment_starts /
    main_event_start_et let a specific card override the defaults with
    verified real anchor times (as UFC 329's were) rather than the generic
    broadcast-standard guesses.
    """
    segment_starts = {**DEFAULT_SEGMENT_START, **(segment_starts or {})}
    main_event_start_str = main_event_start_et or DEFAULT_MAIN_EVENT_START

    chronological = sorted(fights, key=lambda f: _SEGMENT_ORDER.get(f.get("card_position"), 2))

    early_prelims = [f for f in chronological if f.get("card_position") == "Early Prelims"]
    prelims = [f for f in chronological if f.get("card_position") == "Prelims"]
    main_block = [f for f in chronological if f.get("card_position") in ("Main Card", "Co-Main Event", "Main Event")]
    main_event_fights = [f for f in main_block if f.get("card_position") == "Main Event"]
    main_block_undercard = [f for f in main_block if f.get("card_position") != "Main Event"]

    schedule = []

    def _distribute(group: list[dict], start: dt.datetime, end: dt.datetime):
        if not group:
            return
        span_minutes = max((end - start).total_seconds() / 60, len(group))
        slot = span_minutes / len(group)
        cursor = start
        for fight in group:
            slot_end = cursor + dt.timedelta(minutes=slot)
            schedule.append({
                "fighter_a": fight["fighter_a"], "fighter_b": fight["fighter_b"],
                "card_position": fight.get("card_position"),
                "estimated_start_iso": _fmt(cursor), "estimated_end_iso": _fmt(slot_end),
            })
            cursor = slot_end

    ep_start = _parse(event_date, segment_starts.get("Early Prelims", event_start_time_et))
    prelims_start = _parse(event_date, segment_starts["Prelims"])
    main_card_start = _parse(event_date, segment_starts["Main Card"])
    main_event_start = _parse(event_date, main_event_start_str)

    _distribute(early_prelims, ep_start, prelims_start)
    _distribute(prelims, prelims_start, main_card_start)
    _distribute(main_block_undercard, main_card_start, main_event_start)

    cursor = main_event_start
    for fight in main_event_fights:
        slot_end = cursor + dt.timedelta(minutes=_MAIN_EVENT_FALLBACK_DURATION_MIN)
        schedule.append({
            "fighter_a": fight["fighter_a"], "fighter_b": fight["fighter_b"],
            "card_position": fight.get("card_position"),
            "estimated_start_iso": _fmt(cursor), "estimated_end_iso": _fmt(slot_end),
        })
        cursor = slot_end

    return schedule


SCHEDULE_STATE_PATH = "data/schedule_state.json"
# Typical real gap between one fight ending (scorecards read / ref waves it
# off) and the next actually starting (cage reset, walkouts, introductions).
INTER_FIGHT_GAP_MIN = 13


def apply_live_corrections(
    schedule: list[dict], finished_keys: set[frozenset], now: dt.datetime | None = None,
) -> tuple[list[dict], str | None]:
    """
    Self-correction: the pre-card estimate above is necessarily static, and
    real fights run early or late constantly -- without this, a single
    early stoppage or a slow decision compounds across the rest of a
    14-fight card and the "live now" guess drifts increasingly wrong as
    the night goes on (confirmed: this was the actual complaint).

    Fights with a confirmed result are removed from the schedule entirely
    (they're not an estimate anymore, they're a fact -- rendered via the
    real result elsewhere). The moment the count of confirmed results
    increases, "now" becomes a trusted real anchor: the remaining fights
    are shifted, preserving their relative spacing, so the next one is
    expected INTER_FIGHT_GAP_MIN after that real confirmation rather than
    wherever the original static guess placed it.

    Returns (remaining_schedule_with_corrected_times, last_confirmed_at_iso).
    The small state file persists only "how many are confirmed so far" and
    "when that count last increased" -- just enough to know a correction
    anchor exists, without needing to guess elapsed time.
    """
    now = now or dt.datetime.now(dt.timezone(dt.timedelta(hours=-4)))

    remaining = [
        f for f in schedule
        if frozenset({f["fighter_a"].strip().lower(), f["fighter_b"].strip().lower()}) not in finished_keys
    ]
    confirmed_count = len(schedule) - len(remaining)

    state = {"confirmed_count": 0, "last_confirmed_at": None}
    if os.path.exists(SCHEDULE_STATE_PATH):
        try:
            with open(SCHEDULE_STATE_PATH) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    if confirmed_count != state.get("confirmed_count", 0):
        # Count moved in EITHER direction -- forward (a new result just
        # landed) or backward (a new event started and confirmed_count
        # reset lower than a stale state file from the last card). Either
        # way "now" is the freshest trustworthy anchor; a backward reset
        # additionally clears last_confirmed_at since it no longer applies
        # to this card.
        state = {
            "confirmed_count": confirmed_count,
            "last_confirmed_at": now.isoformat() if confirmed_count > 0 else None,
        }
        try:
            with open(SCHEDULE_STATE_PATH, "w") as f:
                json.dump(state, f)
        except OSError:
            pass

    if state.get("last_confirmed_at") and remaining:
        last_confirmed_at = dt.datetime.fromisoformat(state["last_confirmed_at"])
        corrected_next_start = last_confirmed_at + dt.timedelta(minutes=INTER_FIGHT_GAP_MIN)
        original_next_start = dt.datetime.fromisoformat(remaining[0]["estimated_start_iso"])
        shift = corrected_next_start - original_next_start
        for f in remaining:
            f["estimated_start_iso"] = _fmt(dt.datetime.fromisoformat(f["estimated_start_iso"]) + shift)
            f["estimated_end_iso"] = _fmt(dt.datetime.fromisoformat(f["estimated_end_iso"]) + shift)

    return remaining, state.get("last_confirmed_at")
