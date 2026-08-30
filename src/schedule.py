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
from zoneinfo import ZoneInfo
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

# The times above describe a STANDARD US-PRIMETIME card, and were previously
# used as absolute wall-clock times for every event. That silently broke every
# INTERNATIONAL card: a European or Middle Eastern event with 10:00 ET prelims
# still rendered a 21:00 main card, because only Early Prelims ever consulted
# the event's real start time. Real symptom: the countdown (which reads
# event_start_time_et directly) showed the right target while the banner's
# "Main Card · 9 PM ET" line, derived from this schedule, was eight hours out.
#
# So they're now treated as OFFSETS from the event's own prelims start rather
# than fixed clock times. With a 19:00 anchor every offset reproduces the old
# values exactly, so a normal US card is bit-for-bit unchanged; an early card
# simply shifts wholesale.
_SEGMENT_OFFSET_MIN = {
    "Early Prelims": -105,   # 17:15 vs a 19:00 prelims start
    "Prelims": 0,
    "Main Card": 120,        # 21:00
}
_MAIN_EVENT_OFFSET_MIN = 255  # 23:15


def _shift(hhmm: str, minutes: int) -> str:
    """Clock arithmetic on an HH:MM string, wrapping within the day."""
    try:
        h, m = (int(x) for x in hhmm.split(":"))
    except (ValueError, AttributeError):
        return hhmm
    total = (h * 60 + m + minutes) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"
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


ET = ZoneInfo("America/New_York")


def et_now() -> dt.datetime:
    """Now, in real Eastern wall-clock -- EDT or EST as the date requires."""
    return dt.datetime.now(ET)


def _fmt(d: dt.datetime) -> str:
    """
    Stamp an ET wall-clock time with the offset that date ACTUALLY has.

    This hardcoded -04:00. Eastern is -04:00 only from the second Sunday in
    March to the first Sunday in November; for the rest of the year it is
    -05:00, so every estimate published between November and March claimed to
    be an hour earlier than it was. The browser parses these as absolute
    instants, so on a winter card the countdown reaches zero an hour before
    first bell and every fight window opens an hour early -- which pushes
    "LIVE NOW" roughly one fight ahead of reality for the whole night.

    Not hypothetical: UFC Fight Night Bonfim vs. Brady is already tracked for
    2026-11-07, six days after DST ends.

    event_start_time_et itself was always correct -- card_discovery converts
    through ZoneInfo -- so the bug was purely in re-stamping that correct
    wall-clock with a fixed offset.
    """
    if d.tzinfo is None:
        d = d.replace(tzinfo=ET)
    return d.strftime("%Y-%m-%dT%H:%M:%S%z")[:-2] + ":" + d.strftime("%z")[-2:]


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
    # Derive each segment from THIS event's start time, then let any explicit
    # segment_starts argument win -- a caller passing real published times
    # should always beat a derived estimate.
    derived = {seg: _shift(event_start_time_et, off) for seg, off in _SEGMENT_OFFSET_MIN.items()}
    segment_starts = {**DEFAULT_SEGMENT_START, **derived, **(segment_starts or {})}
    if main_event_start_et is None:
        main_event_start_et = _shift(event_start_time_et, _MAIN_EVENT_OFFSET_MIN)
    main_event_start_str = main_event_start_et or DEFAULT_MAIN_EVENT_START

    # CANCELLED FIGHTS NEVER ENTER THE SCHEDULE. They occupy no slot on the
    # night, so including them shifted every later fight's estimate by one
    # slot AND made each subsequent confirmation look like this bout had been
    # skipped by the results fetcher -- which is what produced the recurring
    # "still unconfirmed despite a later result already landing" warning on a
    # deliberately-cancelled bout, on every ~5-minute refresh.
    # Filtering HERE rather than downstream is the point: the schedule entries
    # built below are fresh dicts carrying only fighter names, card_position
    # and timestamps, so `cancelled` does not survive into them and any later
    # filter would be testing a key that is always absent. That is exactly how
    # the first version of this fix silently did nothing.
    def _is_cancelled(f) -> bool:
        v = f.get("cancelled")
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() == "true"

    fights = [f for f in fights if not _is_cancelled(f)]

    chronological = sorted(fights, key=lambda f: _SEGMENT_ORDER.get(f.get("card_position"), 2))

    early_prelims = [f for f in chronological if f.get("card_position") == "Early Prelims"][::-1]
    prelims = [f for f in chronological if f.get("card_position") == "Prelims"][::-1]
    main_block = [f for f in chronological if f.get("card_position") in ("Main Card", "Co-Main Event", "Main Event")]
    main_event_fights = [f for f in main_block if f.get("card_position") == "Main Event"]
    # MAIN CARD IS REVERSED, exactly like the two segments above it. A card
    # file lists fights in BILLING order -- main event first -- and the night
    # runs the opposite way, which is why early prelims and prelims both take
    # [::-1]. The main card was the one segment that didn't, so it inherited
    # billing order as its chronology and came out backwards.
    #
    # Live consequence, observed during UFC 330: the main card's LAST fight
    # (Barboza vs Ribovics) was handed the FIRST main-card slot and its first
    # fight (Turner vs Fernandes) the last. Confirmed fights are dropped from
    # the schedule the JS sees, so once the real Barboza and Mansur bouts were
    # recorded, the only pending main-card fight left was Turner -- carrying
    # an estimated start that had already passed hours earlier. The banner's
    # "first fight whose start is still in the future" then skipped straight
    # past both him and the co-main and announced the main event as next.
    #
    # Co-Main is NOT reversed with it. It is a segment of one that always
    # runs immediately before the Main Event, so it stays pinned to the end.
    main_card_only = [f for f in main_block if f.get("card_position") == "Main Card"][::-1]
    co_main = [f for f in main_block if f.get("card_position") == "Co-Main Event"]
    main_block_undercard = main_card_only + co_main
    # Verified real chronology still wins where we have it. sorted() is
    # stable, so anything unlisted keeps the corrected order established
    # above rather than falling back to billing order.
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

    today = today or et_now().date()
    event_date = dt.date.fromisoformat(str(cards_df["event_date"].iloc[0]))
    days_since = (today - event_date).days

    # 0 = event day itself, 1 = the day after (still show it, wrap-up
    # framing) -- 2+ means it's been sitting stale for a full extra day,
    # time to hand off to what's next.
    if days_since >= 2 and not future_cards_df.empty:
        # Pick the SOONEST future event, not whichever happens to be in the
        # first row. Real bug: future_cards.csv is in discovery/append order,
        # and the dedupe/resync helpers rewrite it by concatenating groups
        # without ever sorting by date -- so row 0 is routinely NOT the next
        # event. In production this promoted a card 19 days out over one
        # happening that same weekend.
        #
        # Prefer the earliest event still to come; if every tracked future
        # card somehow sits in the past (stale data), fall back to the
        # earliest overall so the choice stays chronological and
        # deterministic rather than arbitrary.
        dated = future_cards_df.copy()
        dated["_d"] = pd.to_datetime(dated["event_date"], errors="coerce")
        dated = dated.dropna(subset=["_d"])
        if dated.empty:
            next_event_name = future_cards_df["event_name"].iloc[0]
        else:
            upcoming = dated[dated["_d"].dt.date >= today]
            pool = upcoming if not upcoming.empty else dated
            next_event_name = pool.loc[pool["_d"].idxmin(), "event_name"]
            if next_event_name != future_cards_df["event_name"].iloc[0]:
                print(f"[schedule] promoting '{next_event_name}' (soonest by date) rather than "
                      f"'{future_cards_df['event_name'].iloc[0]}' (merely first in the file)")
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
    now = now or et_now()

    # Cancelled fights were already removed in build_fight_schedule, before
    # these entries were constructed -- see the note there.
    scheduled = schedule
    remaining = [
        f for f in scheduled
        if frozenset({f["fighter_a"].strip().lower(), f["fighter_b"].strip().lower()}) not in finished_keys
    ]
    confirmed_count = len(scheduled) - len(remaining)

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
        # Real bug hit in production: this used to shift EVERY remaining
        # fight by one correction derived from remaining[0], assuming
        # "remaining" is always a clean contiguous block of not-yet-
        # happened fights. That broke when ONE early fight's result
        # specifically failed to get picked up (a name-match miss, a
        # source gap -- anything results_fetcher can hit for a single
        # pairing) while LATER fights confirmed normally: that one stuck
        # fight stayed first in `remaining`, and the shift (anchored to
        # the LATEST confirmation, hours later) dragged its estimated
        # time deep into the night -- producing exactly the observed
        # "LIVE NOW" on a fight that ended 30 minutes ago, next-fight
        # countdown showing 200+ minutes for a fight that was actually
        # FIRST on the card.
        #
        # Fix: a fight is "stuck" (results-pending, not upcoming) if a
        # LATER fight -- by true chronological position in the original
        # schedule -- has already confirmed. That positional gap can only
        # happen from a results-fetch miss on this specific pairing; it
        # can NEVER happen to a genuinely-currently-live fight (even on a
        # card running hours behind), since nothing chronologically after
        # the live fight could possibly have confirmed yet. Stuck fights
        # are dropped from live/next consideration entirely rather than
        # dragged forward into a fabricated future time.
        chrono_index = {id(f): i for i, f in enumerate(scheduled)}
        latest_confirmed_index = max(
            (i for i, f in enumerate(scheduled)
             if frozenset({f["fighter_a"].strip().lower(), f["fighter_b"].strip().lower()}) in finished_keys),
            default=-1,
        )
        stuck = [f for f in remaining if chrono_index[id(f)] < latest_confirmed_index]
        genuinely_upcoming = [f for f in remaining if f not in stuck]
        if stuck:
            print(f"[schedule] {len(stuck)} fight(s) still unconfirmed despite a later result already "
                  f"landing -- flagged results-pending (not shown as live/next): "
                  f"{[(f['fighter_a'], f['fighter_b']) for f in stuck]}")
        if genuinely_upcoming:
            corrected_next_start = last_confirmed_at + dt.timedelta(minutes=INTER_FIGHT_GAP_MIN)
            original_next_start = dt.datetime.fromisoformat(genuinely_upcoming[0]["estimated_start_iso"])
            shift = corrected_next_start - original_next_start
            for f in genuinely_upcoming:
                f["estimated_start_iso"] = _fmt(dt.datetime.fromisoformat(f["estimated_start_iso"]) + shift)
                f["estimated_end_iso"] = _fmt(dt.datetime.fromisoformat(f["estimated_end_iso"]) + shift)
        # FLAGGED, NOT DELETED -- and the distinction is the whole bug.
        #
        # This used to return genuinely_upcoming alone, dropping stuck fights
        # out of the schedule entirely. `schedule` is the ONLY lookup table
        # the browser has, so a fight missing from it cannot be found by
        # applyLiveScoreboard (its live result is never painted, the row shows
        # "VS" through the fight and after it), cannot be matched by the ESPN
        # live override, and cannot be resolved by gradeLeg -- so any parlay
        # leg on it stays pending forever while the slip reports itself alive,
        # which the client's own comment calls the one failure mode that
        # actively misleads. insideCardWindow also reads schedule[0] and
        # schedule[-1], so deleting the prelims moved the poll window start
        # hours later and the browser stopped polling ESPN entirely for that
        # stretch.
        #
        # Measured on the 2026-08-29 card, replaying its real confirmation
        # sequence: Liu Ce's result landed alone at 07:12 (a name mismatch
        # kept the rest from being fetched -- see _canon in results_fetcher),
        # 14 minutes before the prelim batch. That ONE out-of-order
        # confirmation deleted EIGHT unconfirmed fights and compressed the
        # four survivors ~30 minutes early, which is how the site showed
        # Yan Xiaonan live while Sumudaerji was the fight actually next.
        #
        # The original intent stands and is preserved: a stuck fight must not
        # be presented as live or next, and must not have its estimate dragged
        # into the future by a shift anchored hours later. Both are achieved
        # by the flag -- it keeps its ORIGINAL estimate (no shift applied
        # above) and consumers skip it when choosing live/next.
        for f in stuck:
            f["results_pending"] = True
        remaining = sorted(stuck + genuinely_upcoming, key=lambda f: chrono_index[id(f)])

    return remaining, state.get("last_confirmed_at")
