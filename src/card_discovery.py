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

from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT, ESPN_SCOREBOARD_URL, is_placeholder_fighter_name

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


def _sort_into_billing_order(rows: list[dict]) -> list[dict]:
    """
    Sorts one event's fight rows into billing order (Main Event first),
    matching the established convention in the manually-curated
    fight_cards.csv -- display order is billing order, not chronological
    order. Groups fights by segment and orders the SEGMENTS by billing
    order; within a segment, the relative order of the input is
    preserved as-is rather than reversed or otherwise rearranged.

    Deliberately idempotent: calling this on its own output must be a
    true no-op, since normalize_existing_card_order below runs on every
    generate_site.py call (every 5 minutes via cron). An earlier version
    of this function unconditionally reversed each segment's fights
    every call -- correct once, but flipped already-correct data back on
    the very next call, which would have made the displayed order within
    each segment visibly flip-flop on every scheduled refresh. Caught
    directly by testing a second call on the first call's own output
    before this shipped, not left to surface in production.
    """
    billing_order = {"Main Event": 0, "Co-Main Event": 1, "Main Card": 2, "Prelims": 3, "Early Prelims": 4}
    segments: dict[str, list[dict]] = {}
    for r in rows:
        segments.setdefault(r["card_position"], []).append(r)
    ordered_rows = []
    for segment in sorted(segments.keys(), key=lambda s: billing_order.get(s, 99)):
        ordered_rows.extend(segments[segment])
    return ordered_rows


def normalize_existing_card_order(future_cards_path: str = "data/future_cards.csv") -> int:
    """
    Self-healing pass: re-sorts every already-tracked event's rows into
    billing order. Exists because the fix to _fetch_espn_full_card only
    prevents the chronological-order bug for events discovered AFTER
    that fix landed -- any event already in future_cards.csv from
    before it (discover_and_append_new_cards doesn't re-fetch an event
    already present by name) would otherwise stay in the wrong order
    indefinitely. Runs every generate_site.py call; a no-op read+
    compare+skip-write on every run after the first one that actually
    needs it. Returns the number of events whose order was corrected.
    Never raises.
    """
    try:
        df = pd.read_csv(future_cards_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0
    if df.empty or "card_position" not in df.columns:
        return 0

    corrected = 0
    reordered_frames = []
    for event_name, group in df.groupby("event_name", sort=False):
        rows = group.to_dict("records")
        ordered = _sort_into_billing_order(rows)
        if [r["fighter_a"] for r in ordered] != [r["fighter_a"] for r in rows]:
            corrected += 1
            print(f"[card_discovery] corrected fight order for {event_name!r} (was not in billing order)")
        reordered_frames.append(pd.DataFrame(ordered))

    if corrected:
        pd.concat(reordered_frames, ignore_index=True).to_csv(future_cards_path, index=False)
    return corrected


def deduplicate_tracked_fights(future_cards_path: str = "data/future_cards.csv") -> int:
    """
    Self-healing pass: merges rows that represent the same real fight
    but ended up as two separate rows because of a name-format mismatch
    between data sources (e.g. "Jose Delgado" vs "Jose Miguel Delgado"
    for the same person). Confirmed happening in production: the
    original exact-name-only matching in an earlier version of
    resync_tracked_card_order below orphaned the short-name row while
    also adding a "new" row under the fuller name, creating a real
    duplicate fight on the live site. resync_tracked_card_order's own
    loose-match fallback (see its docstring) stops this specific
    failure mode from creating any NEW duplicate going forward, but
    fixing that matching logic does nothing to retroactively clean up
    a duplicate a previous, buggier run already wrote to disk -- this
    function is what actually removes it.

    Uses the same loose-name match (first + last word of each fighter)
    as resync_tracked_card_order. When two rows for the same event
    turn out to be the same fight under that looser comparison, keeps
    the first one encountered and discards the other -- arbitrary but
    deterministic, since there's no reliable signal here for which
    name variant is more correct -- and logs exactly which two rows
    were merged, so this is visible and auditable rather than data
    silently vanishing.

    Runs every generate_site.py call, before resync_tracked_card_order,
    so that function always operates on already-deduplicated data. A
    no-op read+compare+skip-write once nothing needs merging. Never
    raises.
    """
    try:
        df = pd.read_csv(future_cards_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0
    if df.empty or "event_name" not in df.columns:
        return 0

    def _loose_name(name: str) -> tuple:
        parts = str(name).strip().lower().split()
        return (parts[0], parts[-1]) if parts else (str(name).strip().lower(),)

    def _loose_key(row: dict) -> frozenset:
        return frozenset({_loose_name(row["fighter_a"]), _loose_name(row["fighter_b"])})

    removed = 0
    kept_groups = []
    for event_name, group in df.groupby("event_name", sort=False):
        rows = group.to_dict("records")
        seen: dict = {}
        deduped = []
        for r in rows:
            key = _loose_key(r)
            if key in seen:
                removed += 1
                kept = seen[key]
                print(f"[card_discovery] merging duplicate fight for {event_name!r}: "
                      f"{r['fighter_a']!r} vs {r['fighter_b']!r} looks like the same fight as "
                      f"already-kept {kept['fighter_a']!r} vs {kept['fighter_b']!r} -- keeping the "
                      f"first-seen row, dropping this one")
                continue
            seen[key] = r
            deduped.append(r)
        kept_groups.append(pd.DataFrame(deduped))

    if removed:
        pd.concat(kept_groups, ignore_index=True).to_csv(future_cards_path, index=False)
    return removed


def resync_tracked_card_order(future_cards_path: str = "data/future_cards.csv") -> int:
    """
    Self-healing pass, complementary to normalize_existing_card_order
    above: that function only re-sorts by SEGMENT (Main Event before
    Co-Main before Main Card, etc.), deliberately preserving whatever
    within-segment order the data already has, since blindly reordering
    within a segment with no real data to sort by caused a worse bug
    once already (see _sort_into_billing_order's docstring -- an
    earlier version that reversed each segment unconditionally flipped
    already-correct order back on the very next run). This function is
    what supplies the real data that makes within-segment reordering
    safe: it re-fetches each tracked event's CURRENT full card from
    ESPN and uses ESPN's own, live competitions order -- already
    trusted elsewhere in this file as the ground truth for which
    segment a fight belongs to -- to catch a fight that's been
    genuinely promoted or demoted since it was first added. Concretely:
    a fight added early (e.g. announced right when a card was first
    booked) that later gets moved up the actual lineup would otherwise
    stay wherever it was first placed forever, since
    discover_and_append_new_cards never re-fetches an event already
    present by name.

    Matches fights between the existing tracked rows and the fresh ESPN
    fetch by fighter pair (order-independent, since fighter_a/b
    assignment isn't guaranteed identical between two separate
    fetches), with a loose fallback (first + last word of each name)
    for when the two data sources disagree on exactly how a name is
    written -- confirmed in production: the originally-tracked "Jose
    Delgado" didn't exact-match ESPN's fresher "Jose Miguel Delgado",
    so the exact-only matching orphaned the old row AND added a "new"
    one under the fuller name, producing a real duplicate fight on the
    live site. A loose match still preserves the EXISTING row's own
    fighter_a/fighter_b spelling rather than adopting ESPN's, since the
    rest of the pipeline (fighters.csv, schedule.py, model lookups)
    all key off whatever name was already tracked. For a matched fight, only card_position is corrected and
    its position in the list is taken from the fresh order -- every
    other column (backfilled fighter research, manually-verified data,
    etc.) is preserved exactly as already tracked, never overwritten by
    the fresh fetch. A previously-tracked fight absent from the fresh
    fetch is kept, not dropped -- could be a transient ESPN gap rather
    than a real cancellation, same conservative default this file uses
    elsewhere for removing anything. A fight the fresh fetch reveals
    that wasn't tracked yet is added in its correct position, inheriting
    this event's own event_start_time_et from an existing row rather
    than the fresh fetch's independently-computed value, since that
    field is meant to be identical across every row of the same event.

    Runs every generate_site.py call, before normalize_existing_card_order,
    which still runs after as a final, cheap segment-order safety net.
    A no-op read+compare+skip-write for an event whose order hasn't
    actually changed. Never raises -- any single event's fetch failing
    just leaves that event's existing order untouched, rather than
    losing or scrambling data because of a network hiccup.
    """
    try:
        df = pd.read_csv(future_cards_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0
    if df.empty or "event_name" not in df.columns:
        return 0

    def _key(row: dict) -> frozenset:
        return frozenset({str(row["fighter_a"]).strip().lower(), str(row["fighter_b"]).strip().lower()})

    def _loose_name(name: str) -> tuple:
        # First + last word only, e.g. "Jose Miguel Delgado" -> ("jose", "delgado") --
        # matches "Jose Delgado" too. Deliberately not fuzzy string matching: two
        # different real fighters sharing an exact first AND last word on the same
        # card is a much rarer collision than a middle name being present in one
        # data source and missing in another, which is a real, confirmed case in
        # this project's own data (see this function's docstring).
        parts = str(name).strip().lower().split()
        return (parts[0], parts[-1]) if parts else (str(name).strip().lower(),)

    def _loose_key(row: dict) -> frozenset:
        return frozenset({_loose_name(row["fighter_a"]), _loose_name(row["fighter_b"])})

    corrected = 0
    reordered_groups = []
    for event_name, group in df.groupby("event_name", sort=False):
        rows = group.to_dict("records")
        event_date = rows[0].get("event_date")
        if not event_date:
            reordered_groups.append(pd.DataFrame(rows))
            continue

        fresh_rows = _fetch_espn_full_card(str(event_name), str(event_date))
        if not fresh_rows:
            # Fetch failed, or matched nothing usable -- leave this
            # event's existing order untouched rather than act on no
            # data at all.
            reordered_groups.append(pd.DataFrame(rows))
            continue

        existing_by_key = {_key(r): r for r in rows}
        existing_by_loose_key = {_loose_key(r): r for r in rows}
        before_snapshot = [(_key(r), r["card_position"]) for r in rows]
        matched_keys = set()
        new_order = []
        for fresh in fresh_rows:
            key = _key(fresh)
            existing_row = existing_by_key.get(key)
            if existing_row is None:
                existing_row = existing_by_loose_key.get(_loose_key(fresh))
            if existing_row is not None:
                matched_keys.add(_key(existing_row))
                existing_row["card_position"] = fresh["card_position"]
                new_order.append(existing_row)
            else:
                fresh["event_start_time_et"] = rows[0].get("event_start_time_et")
                new_order.append(fresh)

        orphaned = [r for k, r in existing_by_key.items() if k not in matched_keys]
        if orphaned:
            print(f"[card_discovery] {len(orphaned)} previously-tracked fight(s) for {event_name!r} "
                  f"not found in ESPN's current card -- keeping them, appended, rather than dropping "
                  f"data that might just be a transient gap")
        new_order.extend(orphaned)

        if [(_key(r), r["card_position"]) for r in new_order] != before_snapshot:
            corrected += 1
            print(f"[card_discovery] re-synced fight order for {event_name!r} against ESPN's current card")

        reordered_groups.append(pd.DataFrame(new_order))

    if corrected:
        pd.concat(reordered_groups, ignore_index=True).to_csv(future_cards_path, index=False)
    return corrected


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
        if is_placeholder_fighter_name(a_name) or is_placeholder_fighter_name(b_name):
            print(f"[card_discovery] skipping not-yet-confirmed bout ({a_name} vs {b_name}) -- "
                  f"one side is still a placeholder, nothing real to track yet")
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

    # Reversed before billing-order sorting: display order within a given
    # segment (prelims or main) should run from the soonest-to-happen
    # fight at the top down to the earliest -- explicit user preference,
    # not a guess. `rows` here is always freshly fetched from ESPN in
    # forward-chronological order (fight 1 first), never already-reversed
    # stored data, so reversing it on every call is safe and stable --
    # unlike reversing inside _sort_into_billing_order itself, which is
    # ALSO called on existing stored future_cards.csv rows elsewhere
    # (normalize_existing_card_order) that could already be in the
    # correct order; unconditionally reversing there caused a real
    # flip-flopping bug once already (see that function's own docstring).
    # A plain list reversal here correctly flips order WITHIN each
    # segment while leaving segment-level billing order (Main Event
    # first, etc.) untouched, since _sort_into_billing_order re-groups by
    # segment regardless of what order it's handed.
    return _sort_into_billing_order(rows[::-1])


def discover_and_append_new_cards(future_cards_path: str = "data/future_cards.csv",
                                   current_event_name: str | None = None,
                                   days_ahead: int = 60) -> int:
    """
    Entry point called from generate_site.py. Returns the number of rows
    added or removed -- never raises. Reads ESPN's scoreboard calendar
    (covers the full year), finds UFC events within `days_ahead` that
    aren't already in future_cards.csv or currently the active card
    (current_event_name), fetches each one's full fight card, and adds
    it. A tracked event whose booking changed (a fighter dropped out,
    the event got renamed for the replacement) is detected by matching
    on event_date against the calendar's current name for that date, and
    the stale entry is replaced -- covering both the case where the new
    name needs fetching fresh, and the case where it's already separately
    tracked (both old and new names present as two "different" events).
    Any other existing row is read but never modified.
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

    existing_dates_to_names: dict = {}
    if "event_name" in existing.columns and "event_date" in existing.columns:
        for name, date_str in existing[["event_name", "event_date"]].drop_duplicates().itertuples(index=False):
            existing_dates_to_names.setdefault(str(date_str), name)

    # Self-healing: the calendar is ground truth for "what UFC currently
    # calls the event on date X." If an existing tracked event's name
    # doesn't match that, it's a stale pre-replacement name -- including
    # the case where the CURRENT name is already separately tracked too
    # (both "vs. Rountree Jr." and "vs. Guskov" present as two different
    # events), which the discovery loop below can't catch on its own since
    # it skips any calendar entry whose name is already known, before ever
    # checking whether that name replaced something else on the same date.
    calendar_current_name_by_date: dict = {}
    for entry in calendar:
        label, start = entry.get("label"), entry.get("startDate")
        if not label or not start:
            continue
        try:
            d = dt.datetime.fromisoformat(start.replace("Z", "+00:00")).date().isoformat()
        except (ValueError, TypeError):
            continue
        calendar_current_name_by_date[d] = label

    stale_event_names = set()
    for date_str, tracked_name in existing_dates_to_names.items():
        current_name = calendar_current_name_by_date.get(date_str)
        # Only remove the old entry here if the replacement is ALREADY a
        # separately-tracked event with real data -- safe, since nothing is
        # lost. If the replacement isn't tracked yet, leave this alone and
        # let the main loop below fetch it first; that loop only marks an
        # old entry as replaced after confirming the fetch actually
        # succeeded, which this pre-pass must not bypass.
        if current_name and current_name != tracked_name and current_name in known_event_names:
            stale_event_names.add(tracked_name)
            print(f"[card_discovery] '{tracked_name}' is stale -- ESPN now calls this date's event "
                  f"'{current_name}', which is already tracked separately. Removing the stale entry.")

    new_rows = []
    added_events = []
    replaced_event_names = set()
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

        # Same date, different name than something already tracked -- almost
        # certainly the same booking after a lineup change (a fighter dropped
        # out, a replacement was announced, and the event got renamed for it),
        # not a second, different event UFC happens to be running the same
        # day. Replace the old entry rather than keeping both -- but only
        # once the replacement card is confirmed to actually have fights;
        # marking the old entry for removal before that would lose the event
        # entirely if this fetch fails or comes back empty.
        old_name = existing_dates_to_names.get(event_date.isoformat())

        card_rows = _fetch_espn_full_card(label, event_date.isoformat())
        if card_rows:
            if old_name and old_name != label:
                print(f"[card_discovery] '{old_name}' appears to have become '{label}' (same date, {event_date}) "
                      f"-- likely a lineup change, replacing the old entry rather than tracking both")
                replaced_event_names.add(old_name)
            new_rows.extend(card_rows)
            added_events.append(label)
            known_event_names.add(label)  # avoid double-adding if the calendar lists it twice

    all_removed = stale_event_names | replaced_event_names
    if not new_rows and not all_removed:
        return 0

    if all_removed:
        existing = existing[~existing["event_name"].isin(all_removed)]

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else existing
    combined.to_csv(future_cards_path, index=False)
    if added_events:
        print(f"[card_discovery] added {len(added_events)} new card(s): {', '.join(added_events)} ({len(new_rows)} fights)")
    return len(new_rows) + len(all_removed)
