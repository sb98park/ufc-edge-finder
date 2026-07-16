"""
Auto-discovers upcoming UFC cards from ESPN's scoreboard calendar and
appends their full fight card to future_cards.csv. This file was
previously 100% manually maintained -- someone had to research each
card by hand and type it in -- which is why only 2 cards were tracked
despite many more being confirmed. Runs as part of every
generate_site.py call, same pattern as results_fetcher.py, but only
ever ADDS events not already present (matched by event_name). Never
touches or overwrites an existing row, so manually-verified cards
already in the file are never at risk from this.

*** HONEST CAVEAT, SAME SPIRIT AS results_fetcher.py ***
ESPN's scoreboard does not expose an explicit card-position label --
there is no "Main Event" / "Prelims" field anywhere in the response.
card_position is INFERRED from two real but indirect signals, not read
directly:
  - Fights on the same card cluster into 1-3 distinct start times.
    The latest cluster is treated as the main-card block; earlier
    clusters become Prelims (or Early Prelims / Prelims if there are
    two earlier clusters, oldest first).
  - Within the main-card block, array order is assumed to reflect
    broadcast order (least notable first, main event last): the last
    fight in that block is Main Event, the one before it Co-Main
    Event, everything earlier in the block is Main Card. Cross-checked
    where possible against format.regulation.periods == 5, a reliable
    independent signal for the Main Event specifically (UFC schedules
    ALL main events for 5 rounds, title fight or not) -- a mismatch
    between the two signals is logged, not silently overridden by one
    or the other.
This is weaker than a direct label would be. It's the best available
from this source, and it's disclosed here rather than presented as
certain. If it gets a Co-Main wrong, the fight itself and its details
are still correct -- only the segment label could be off.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT, ESPN_SCOREBOARD_URL

FUTURE_CARDS_COLUMNS = [
    "event_name", "event_date", "card_position", "weight_class",
    "fighter_a", "fighter_b", "event_start_time_et", "is_womens_division", "event_location",
]


def _weight_class_from_espn(abbreviation: str) -> tuple[str, bool]:
    """ESPN prefixes women's divisions with "W " (e.g. "W Strawweight");
    our schema spells it out ("Women's Strawweight"). Returns
    (weight_class, is_womens_division)."""
    if abbreviation.startswith("W "):
        return f"Women's {abbreviation[2:]}", True
    return abbreviation, False


def _infer_card_positions(competitions: list[dict]) -> dict[int, str]:
    """
    Returns {competition_index: card_position} for one event's fights,
    using the clustering + array-order heuristic documented in this
    module's docstring. competition_index refers to the position in
    the ORIGINAL competitions list passed in, so callers can map results
    back without re-sorting.
    """
    if not competitions:
        return {}

    # Group by exact start-time value -- fights sharing a broadcast
    # segment share a start time in ESPN's data.
    times = sorted({c.get("date") for c in competitions if c.get("date")})
    time_rank = {t: i for i, t in enumerate(times)}
    n_clusters = len(times)

    def segment_for_rank(rank: int) -> str:
        if rank == n_clusters - 1:
            return "__MAIN_BLOCK__"  # resolved to Main Event/Co-Main/Main Card below
        remaining_earlier = n_clusters - 1 - rank  # clusters strictly after this one, excluding the main block
        if remaining_earlier >= 2:
            return "Early Prelims"
        return "Prelims"

    positions: dict[int, str] = {}
    main_block_indices: list[int] = []
    for i, comp in enumerate(competitions):
        t = comp.get("date")
        rank = time_rank.get(t, n_clusters - 1)
        seg = segment_for_rank(rank)
        if seg == "__MAIN_BLOCK__":
            main_block_indices.append(i)
        else:
            positions[i] = seg

    if not main_block_indices:
        return positions

    # Within the main block: last = Main Event, second-to-last = Co-Main,
    # everything earlier = Main Card. Cross-check against the 5-round
    # signal where available -- logged, not used to silently override.
    five_round_indices = [
        i for i in main_block_indices
        if competitions[i].get("format", {}).get("regulation", {}).get("periods") == 5
    ]
    main_event_idx = main_block_indices[-1]
    if five_round_indices and five_round_indices[-1] != main_event_idx:
        print(f"[card_discovery] main-event inference mismatch: array-order picked index {main_event_idx}, "
              f"5-round signal points at {five_round_indices}. Keeping array-order pick, flagging the disagreement.")

    for pos_in_block, i in enumerate(main_block_indices):
        if i == main_event_idx:
            positions[i] = "Main Event"
        elif pos_in_block == len(main_block_indices) - 2:
            positions[i] = "Co-Main Event"
        else:
            positions[i] = "Main Card"
    return positions


def _fetch_espn_full_card(event_name: str, event_date: str) -> list[dict]:
    """Fetches one event's complete fight card from ESPN's scoreboard and
    transforms it into future_cards.csv's row schema. Never raises --
    returns [] on any failure, same convention as the rest of this
    project's external-data code."""
    try:
        date_param = pd.Timestamp(event_date).strftime("%Y%m%d")
        resp = requests.get(ESPN_SCOREBOARD_URL, params={"dates": date_param}, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[card_discovery] ESPN fetch failed for {event_name!r}: {e}")
        return []

    matched = next((ev for ev in data.get("events", []) if ev.get("name") == event_name), None)
    if matched is None:
        # Fall back to a single-event response, since a date-scoped query
        # usually returns exactly one UFC event.
        events = data.get("events", [])
        matched = events[0] if len(events) == 1 else None
    if matched is None:
        print(f"[card_discovery] ESPN: could not match {event_name!r} on {event_date}")
        return []

    competitions = matched.get("competitions", [])
    positions = _infer_card_positions(competitions)

    venue = (competitions[0].get("venue") if competitions else {}) or {}
    address = venue.get("address", {})
    location_parts = [p for p in [venue.get("fullName"), address.get("city"), address.get("state") or address.get("country")] if p]
    event_location = ", ".join(location_parts)

    prelims_time_et = None
    rows = []
    for i, comp in enumerate(competitions):
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue
        by_order = sorted(competitors, key=lambda c: c.get("order", 99))
        a_name = by_order[0].get("athlete", {}).get("fullName")
        b_name = by_order[1].get("athlete", {}).get("fullName")
        if not a_name or not b_name:
            continue

        weight_class, is_womens = _weight_class_from_espn(comp.get("type", {}).get("abbreviation", "Unknown"))
        card_position = positions.get(i, "Main Card")

        start_iso = comp.get("date")
        if start_iso:
            try:
                utc_dt = dt.datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                et_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
                if card_position in ("Prelims", "Early Prelims") and (prelims_time_et is None or et_dt.strftime("%H:%M") < prelims_time_et):
                    prelims_time_et = et_dt.strftime("%H:%M")
            except (ValueError, TypeError):
                pass

        rows.append({
            "event_name": event_name, "event_date": event_date, "card_position": card_position,
            "weight_class": weight_class, "fighter_a": a_name, "fighter_b": b_name,
            "event_start_time_et": None,  # filled in below once the earliest prelims time is known
            "is_womens_division": is_womens, "event_location": event_location,
        })

    # event_start_time_et reflects the Prelims start (existing convention
    # in this file, matching src/schedule.py's DEFAULT_SEGMENT_START usage) --
    # falls back to the earliest fight's time if no fight was classified
    # as Prelims/Early Prelims (e.g. a very short, main-card-only event).
    if prelims_time_et is None and rows:
        all_times = [c.get("date") for c in competitions if c.get("date")]
        if all_times:
            try:
                earliest = dt.datetime.fromisoformat(min(all_times).replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
                prelims_time_et = earliest.strftime("%H:%M")
            except (ValueError, TypeError):
                prelims_time_et = "19:00"
    for r in rows:
        r["event_start_time_et"] = prelims_time_et or "19:00"

    return rows


def discover_and_append_new_cards(future_cards_path: str = "data/future_cards.csv",
                                   current_event_name: str | None = None,
                                   days_ahead: int = 60) -> int:
    """
    Entry point called from generate_site.py. Returns the number of new
    rows appended -- never raises. Reads ESPN's scoreboard calendar
    (covers the full year), finds UFC events within `days_ahead` that
    aren't already in future_cards.csv or currently the active card
    (current_event_name), fetches each one's full fight card, and
    appends it. Existing rows are read but never modified or removed.
    """
    try:
        existing = pd.read_csv(future_cards_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing = pd.DataFrame(columns=FUTURE_CARDS_COLUMNS)

    known_event_names = set(existing["event_name"].unique()) if "event_name" in existing.columns else set()
    if current_event_name:
        known_event_names.add(current_event_name)

    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[card_discovery] could not fetch ESPN calendar: {e}")
        return 0

    leagues = data.get("leagues", [])
    calendar = leagues[0].get("calendar", []) if leagues else []
    if not calendar:
        print("[card_discovery] ESPN calendar was empty or unavailable this run")
        return 0

    today = dt.datetime.now(dt.timezone.utc).date()
    cutoff = today + dt.timedelta(days=days_ahead)

    new_rows = []
    added_events = []
    for entry in calendar:
        label = entry.get("label")
        start = entry.get("startDate")
        if not label or not start or label in known_event_names:
            continue
        try:
            event_date = dt.datetime.fromisoformat(start.replace("Z", "+00:00")).date()
        except (ValueError, TypeError):
            continue
        if not (today <= event_date <= cutoff):
            continue

        card_rows = _fetch_espn_full_card(label, event_date.isoformat())
        if card_rows:
            new_rows.extend(card_rows)
            added_events.append(label)
            known_event_names.add(label)  # avoid double-adding if the calendar lists it twice

    if not new_rows:
        return 0

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(future_cards_path, index=False)
    print(f"[card_discovery] added {len(added_events)} new card(s): {', '.join(added_events)} ({len(new_rows)} fights)")
    return len(new_rows)
