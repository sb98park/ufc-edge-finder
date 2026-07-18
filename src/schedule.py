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

import pandas as pd

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

# Fight_cards.csv's row order is DISPLAY order (billing order -- Main
# Event first, most notable fights first within a segment), matching
# every fight-card site's convention (Google, ESPN, etc.) and what the
# template renders directly. That's DELIBERATELY separate from true
# chronological fight order, which this scheduling logic needs
# internally to estimate "what's live right now" -- confusing the two
# once already changed the VISIBLE card order to match chronological
# time, which broke the display (confirmed via screenshot comparison
# against Google's own UFC card listing).
#
# This maps (fighter_a, fighter_b) -> its real position in fight order,
# used ONLY to re-sort within a segment for scheduling purposes; the
# fights list returned to the template for display is untouched. Only
# populated where actually verified (news recaps, Tapology's "fight N of
# 14" billing data), not guessed -- segments without an entry here keep
# using file order as the chronology assumption, same as before.
VERIFIED_CHRONOLOGICAL_ORDER = {
    ("King Green", "Terrance McKinney"): 1,
    ("Brandon Royval", "Lone'er Kavanagh"): 2,
    ("Cory Sandhagen", "Mario Bautista"): 3,
}

# Real, sportsbook-confirmed start times (FanDuel, verified by the user
# directly against the live odds board) for UFC Fight Night: Du Plessis
# vs. Usman -- used in place of the generic evenly-distributed estimate
# for these specific fights, since an actual anchor beats a guess.
VERIFIED_FIGHT_TIMES = {
    ("Dione Barbosa", "Anna Melisano"): "17:10",
    ("Alvin Hines", "RJ Harris"): "17:35",
    ("Alden Coria", "Stewart Nicoll"): "18:00",
    ("Felipe Franco", "Levi Rodrigues Jr."): "18:25",
    ("Jean-Paul Lebosnoyani", "Seokhyeon Ko"): "18:50",
    ("Austin Bashi", "Jose Delgado"): "19:15",
    ("Tabatha Ricci", "Fatima Kline"): "19:40",
    ("Tommy McMillen", "Alberto Montes"): "20:45",
    ("Chase Hooper", "Mitch Ramirez"): "21:15",
    ("Jared Cannonier", "Christian Leroy Duncan"): "21:45",
    ("Dricus Du Plessis", "Kamaru Usman"): "22:45",
}


def _fight_key(f: dict) -> tuple:
    return (f["fighter_a"], f["fighter_b"])


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

    early_prelims = [f for f in chronological if f.get("card_position") == "Early Prelims"][::-1]
    prelims = [f for f in chronological if f.get("card_position") == "Prelims"][::-1]
    main_block = [f for f in chronological if f.get("card_position") in ("Main Card", "Co-Main Event", "Main Event")]
    main_event_fights = [f for f in main_block if f.get("card_position") == "Main Event"]
    main_block_undercard = [f for f in main_block if f.get("card_position") != "Main Event"]
    # Re-sort by verified real chronology where we have it (Main Card
    # tier); anything unlisted (Co-Main) keeps its relative file-order
    # position via a large fallback key, which naturally keeps it last,
    # immediately before the Main Event -- correct without needing an
    # explicit entry for it.
    main_block_undercard = sorted(
        main_block_undercard,
        key=lambda f: VERIFIED_CHRONOLOGICAL_ORDER.get(_fight_key(f), 999),
    )

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

    # Swap in real, sportsbook-confirmed start times wherever one exists,
    # in place of the generic evenly-distributed estimate above -- an
    # actual anchor beats a guess. Only the start time was confirmed
    # (not a fight-specific end time), so estimated_end_iso uses a fixed
    # 20-minute display window here rather than reusing the distributed
    # segment's own end-of-slot value, which wouldn't line up with the
    # now-corrected start.
    for entry in schedule:
        verified = VERIFIED_FIGHT_TIMES.get((entry["fighter_a"], entry["fighter_b"]))
        if verified:
            start = _parse(event_date, verified)
            entry["estimated_start_iso"] = _fmt(start)
            entry["estimated_end_iso"] = _fmt(start + dt.timedelta(minutes=20))

    return schedule


def promote_card_if_stale(
    cards_df: pd.DataFrame, future_cards_df: pd.DataFrame, today: dt.date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    "This Weekend" should keep showing the card that just happened through
    the following day (so Sunday still shows Saturday's results, not an
    empty or stale-feeling page) -- then automatically hand off to the
    next tracked future event starting the day after that, rather than
    someone needing to manually move a card from future_cards.csv to
    fight_cards.csv every week.

    Returns (current_cards_df, future_cards_df, days_since_event).
    days_since_event is 0 both for a same-day card and immediately after
    a promotion (the newly-current event hasn't happened yet either way,
    so the same "not stale" handling applies to both).
    """
    if cards_df.empty:
        return cards_df, future_cards_df, 0

    today = today or dt.datetime.now(dt.timezone(dt.timedelta(hours=-4))).date()
    event_date = dt.date.fromisoformat(str(cards_df["event_date"].iloc[0]))
    days_since = (today - event_date).days

    # 0 = event day itself, 1 = the day after (still show it, wrap-up
    # framing) -- 2+ means it's been sitting stale for a full extra day,
    # time to hand off to what's next.
    if days_since >= 2 and not future_cards_df.empty:
        next_event_name = future_cards_df["event_name"].iloc[0]
        new_current = future_cards_df[future_cards_df["event_name"] == next_event_name].reset_index(drop=True)
        new_future = future_cards_df[future_cards_df["event_name"] != next_event_name].reset_index(drop=True)
        return new_current, new_future, 0

    return cards_df, future_cards_df, max(days_since, 0)


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
