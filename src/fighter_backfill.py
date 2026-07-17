"""
Backfills fighters.csv from ESPN for fighters appearing on tracked
future cards -- addressing several confirmed, measured gaps: 28 of 58
future-card fighters were missing from fighters.csv entirely (breaking
model preview generation for 14 of 29 future fights), and even fighters
already in the roster were frequently missing stance/country/reach --
and, per a later, direct user report, most backfilled fighters were
also showing blank KO/TKO, Submission, and Decision win counts, blank
age, and a blank last fight, because the original Pass 1 (below) never
attempted those fields at all.

Passes are kept DELIBERATELY separate because they rest on very
different levels of confidence -- see each function's own docstring:

  Pass 1 (in backfill_fighters' main loop): name, country, and overall
  win-loss record. Built on data this project directly observed and
  verified live during development (the scoreboard's competitor.athlete
  and competitor.records fields) -- same confidence tier as
  results_fetcher.py and card_discovery.py's core functionality.

  Pass 2 (_fetch_espn_athlete_detail): height, reach, stance, age.
  Height/reach/stance were directly confirmed via real production logs
  after shipping. Age was a new, unverified extension of that same
  already-confirmed endpoint when first added -- since confirmed
  working too, per real production data showing correctly-populated
  ages across multiple fighters.

  KO/TKO, Submission, and Decision win-count breakdown was attempted by
  reusing the same records array Pass 1 already fetches, and REMOVED
  (July 2026) after real production logs showed, across roughly 80
  fighters with zero exceptions, that ESPN's records array here always
  contains only a single 'overall' entry -- no method breakdown exists
  in this data source at all. Confirmed absent, not a parsing bug.

  Pass 3 (_fetch_espn_last_fight_info): last fight date. Follows a link
  (eventLog) found on the athlete-detail response. Originally guessed
  the list of past events would be under a key named 'items' -- real
  production logs showed the actual key is 'events', which is now
  fixed. Still capped at one additional request with no further
  cascading, since it remains unconfirmed whether the events in that
  list carry the date inline or are themselves further links requiring
  yet another fetch each.

All passes only ever fill gaps -- an empty cell for a fighter already
in the roster, or a wholly new row for one who's missing entirely.
Never overwrites a non-null value already in fighters.csv. Runs as
part of every generate_site.py call, same pattern as the rest of this
project's ESPN integration. Never raises.
"""

import datetime as dt
import re

import pandas as pd
import requests

from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT, ESPN_SCOREBOARD_URL, is_placeholder_fighter_name

FIGHTERS_COLUMNS_MINIMAL = ["name", "weight_class", "country", "wins", "losses"]

# Only accepted if a parsed value falls in this range -- guards against
# a field-name guess in Pass 2 matching something that isn't actually
# what it looks like.
_PLAUSIBLE_HEIGHT_IN = (55, 90)
_PLAUSIBLE_REACH_IN = (55, 95)
_PLAUSIBLE_AGE = (18, 55)
_KNOWN_STANCES = {"Orthodox", "Southpaw", "Switch"}


def _parse_record(summary: str) -> tuple[int | None, int | None]:
    """"8-4-0" -> (8, 4). Draws are dropped -- fighters.csv has no draws
    column. Returns (None, None) if the string doesn't match the
    expected W-L-D shape, rather than guessing."""
    if not summary:
        return None, None
    m = re.fullmatch(r"(\d+)-(\d+)-(\d+)", summary.strip())
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _fetch_espn_athlete_detail(athlete_id: str) -> tuple[dict, str | None]:
    """
    Pass 2 -- see this module's docstring for the confidence tier of
    each field. Fetches sports.core.api.espn.com's athlete-detail
    endpoint once and mines it for everything plausible in a single
    call: height, reach, and stance (whose field names -- 'height',
    'reach', 'stance' -- were directly confirmed via real production
    logs after this shipped: an actual run logged
    "no field passed validation... Top-level keys: [...'height'...
    'reach'...'stance'...]", meaning the fields exist under exactly
    these names but happened to fail this function's OWN validation
    that one time -- the field names themselves are confirmed, even
    though no run has yet logged a fighter where the values passed).
    Age is a new, unverified extension of the same call -- no field
    literally named 'age' was seen in that same confirmed key list, so
    this tries it and the more likely 'dateOfBirth'/'birthDate'
    fields, but logs for diagnosis rather than assuming either exists.

    Returns (fields_found, eventlog_ref) -- fields_found only includes
    values that both matched a plausible field name AND passed a sanity
    check (plausible human height/reach/age range; stance matching a
    known value). eventlog_ref is the raw $ref URL string from the
    response's 'eventLog' field if present (also seen, unexplored, in
    that same confirmed key list) -- passed along for the separate,
    far-less-certain last-fight lookup in _fetch_espn_last_fight_info,
    or None if absent. Returns ({}, None) on any failure.
    """
    url = f"https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{athlete_id}"
    try:
        resp = requests.get(url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[fighter_backfill] athlete-detail fetch failed for id={athlete_id}: {e}")
        return {}, None

    result = {}

    # Height: try a raw-inches numeric field first, then a "6' 4\"" display string.
    height_candidates = [data.get("height"), data.get("displayHeight")]
    for c in height_candidates:
        if isinstance(c, (int, float)) and _PLAUSIBLE_HEIGHT_IN[0] <= c <= _PLAUSIBLE_HEIGHT_IN[1]:
            result["height_in"] = float(c)
            break
        if isinstance(c, str):
            m = re.match(r"(\d+)'\s*(\d+)", c)
            if m:
                inches = int(m.group(1)) * 12 + int(m.group(2))
                if _PLAUSIBLE_HEIGHT_IN[0] <= inches <= _PLAUSIBLE_HEIGHT_IN[1]:
                    result["height_in"] = float(inches)
                    break

    reach_candidates = [data.get("reach"), data.get("displayReach")]
    for c in reach_candidates:
        if isinstance(c, (int, float)) and _PLAUSIBLE_REACH_IN[0] <= c <= _PLAUSIBLE_REACH_IN[1]:
            result["reach_in"] = float(c)
            break
        if isinstance(c, str):
            m = re.match(r'(\d+)"?$', c.strip())
            if m and _PLAUSIBLE_REACH_IN[0] <= int(m.group(1)) <= _PLAUSIBLE_REACH_IN[1]:
                result["reach_in"] = float(m.group(1))
                break

    stance_candidates = [data.get("stance"), (data.get("stance") or {}).get("text") if isinstance(data.get("stance"), dict) else None]
    for c in stance_candidates:
        if isinstance(c, str) and c.strip().title() in _KNOWN_STANCES:
            result["stance"] = c.strip().title()
            break

    # Age: unconfirmed field names, tried against the same already-fetched response.
    age_val = data.get("age")
    if isinstance(age_val, int) and _PLAUSIBLE_AGE[0] <= age_val <= _PLAUSIBLE_AGE[1]:
        result["age"] = age_val
    else:
        for dob_field in ("dateOfBirth", "birthDate"):
            dob_str = data.get(dob_field)
            if not isinstance(dob_str, str):
                continue
            try:
                dob = dt.datetime.fromisoformat(dob_str.replace("Z", "+00:00")).date()
                computed_age = (dt.date.today() - dob).days // 365
                if _PLAUSIBLE_AGE[0] <= computed_age <= _PLAUSIBLE_AGE[1]:
                    result["age"] = computed_age
                    break
            except (ValueError, TypeError):
                continue

    eventlog_field = data.get("eventLog")
    eventlog_ref = eventlog_field.get("$ref") if isinstance(eventlog_field, dict) else None

    if not result:
        print(f"[fighter_backfill] athlete-detail for id={athlete_id}: no field passed validation. "
              f"Top-level keys in response, for diagnosing the real schema: {sorted(data.keys())}")

    return result, eventlog_ref


def _fetch_espn_last_fight_info(eventlog_ref: str) -> dict:
    """
    Pass 4 -- the most experimental piece in this module. 'eventLog' was
    seen exactly once, as an unexplored field name, in a real
    production log from Pass 2's validation-failure diagnostic -- it
    was never fetched, and its contents have never been observed in any
    form. This function is the first attempt to actually follow it.

    Capped at exactly one additional HTTP request, no matter what comes
    back -- if the response turns out to itself be a list of further
    $ref links requiring more requests to reach any usable date, this
    logs that finding and stops, rather than cascading into an
    open-ended chain of speculative fetches per fighter. Returns {} on
    any failure or unrecognized shape, logging the raw response for
    diagnosis.
    """
    try:
        resp = requests.get(eventlog_ref, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[fighter_backfill] eventLog fetch failed: {e}")
        return {}

    # Confirmed via real production logs (July 2026): the actual field
    # name is 'events', not the originally-guessed 'items' -- every
    # eventLog response observed so far has had exactly ['$ref', 'events']
    # as its top-level keys. Kept the 'items' check too as a harmless
    # fallback in case a different athlete or a future schema change uses it.
    items = data.get("events") if isinstance(data, dict) and "events" in data else (
        data.get("items") if isinstance(data, dict) else (data if isinstance(data, list) else None)
    )
    if not items or not isinstance(items, list) or not isinstance(items[0], dict):
        print(f"[fighter_backfill] eventLog response wasn't a recognizable list of events. "
              f"Top-level shape for diagnosis: {sorted(data.keys()) if isinstance(data, dict) else type(data).__name__}")
        return {}

    most_recent = items[0]
    if set(most_recent.keys()) == {"$ref"}:
        print(f"[fighter_backfill] eventLog items are themselves bare $ref links needing another fetch each "
              f"-- not cascading further this run. Sample item: {most_recent}")
        return {}

    result = {}
    date_val = most_recent.get("date")
    if isinstance(date_val, str) and len(date_val) >= 10:
        result["last_fight_date"] = date_val[:10]

    if not result:
        print(f"[fighter_backfill] eventLog's most recent item had no recognizable date field. "
              f"Raw item, for diagnosing the real schema: {most_recent}")

    return result


def _safe_set_cell(df: pd.DataFrame, row_idx, col: str, val):
    """
    Sets df.at[row_idx, col] = val, upcasting the column to object dtype
    first if the direct assignment would fail. Discovered in testing: a
    column that's all-null across the whole roster (e.g. last_fight_date,
    for a fighter set with no non-null dates anywhere) gets inferred by
    pandas as float64, and assigning a string date into it raises
    TypeError -- a real risk here specifically, not just a test
    artifact, since these are exactly the columns this module exists to
    fill gaps in, which are more likely than most to be all-null in a
    given roster snapshot. Returns the DataFrame (may be a new object if
    upcasting was needed).
    """
    try:
        df.at[row_idx, col] = val
        return df
    except (TypeError, ValueError):
        df[col] = df[col].astype(object)
        df.at[row_idx, col] = val
        return df


def backfill_fighters(fighters_path: str = "data/fighters.csv",
                       future_cards_path: str = "data/future_cards.csv",
                       attempt_athlete_detail: bool = True) -> int:
    """
    Entry point called from generate_site.py. Returns the number of
    fighters newly added or filled in -- never raises. For every
    fighter on a tracked future card: creates a minimal roster row
    (Pass 1) if they're missing from fighters.csv entirely, or fills
    just the empty cells (all passes) if they're already present with
    gaps. Existing non-null values are never touched.
    """
    try:
        fighters = pd.read_csv(fighters_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"[fighter_backfill] could not read {fighters_path} -- skipping this run")
        return 0
    try:
        future = pd.read_csv(future_cards_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0
    if future.empty:
        return 0

    roster_names = set(fighters["name"])
    future_fighters = {n for n in (set(future["fighter_a"]) | set(future["fighter_b"])) if not is_placeholder_fighter_name(n)}
    needs_basic = future_fighters - roster_names
    gap_cols = ["stance", "country", "reach_in", "height_in", "age", "last_fight_date"]
    needs_gap_fill = set(
        fighters[fighters["name"].isin(future_fighters) & fighters[gap_cols].isna().any(axis=1)]["name"]
    )
    if not needs_basic and not needs_gap_fill:
        return 0

    weight_class_by_fighter = {}
    for _, r in future.iterrows():
        weight_class_by_fighter.setdefault(r["fighter_a"], r.get("weight_class"))
        weight_class_by_fighter.setdefault(r["fighter_b"], r.get("weight_class"))

    filled_count = 0
    new_rows = []
    for event_name, event_date in future[["event_name", "event_date"]].drop_duplicates().itertuples(index=False):
        target_names = {n for n in (needs_basic | needs_gap_fill)
                         if n in set(future[future["event_name"] == event_name]["fighter_a"])
                         or n in set(future[future["event_name"] == event_name]["fighter_b"])}
        if not target_names:
            continue

        try:
            date_param = pd.Timestamp(event_date).strftime("%Y%m%d")
            resp = requests.get(ESPN_SCOREBOARD_URL, params={"dates": date_param}, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            print(f"[fighter_backfill] ESPN fetch failed for {event_name!r}: {e}")
            continue

        matched = next((ev for ev in data.get("events", []) if ev.get("name") == event_name), None)
        if matched is None:
            events = data.get("events", [])
            matched = events[0] if len(events) == 1 else None
        if matched is None:
            continue

        for comp in matched.get("competitions", []):
            for c in comp.get("competitors", []):
                name = c.get("athlete", {}).get("fullName")
                if name not in target_names:
                    continue

                country = c.get("athlete", {}).get("flag", {}).get("alt")
                wins, losses = None, None
                for rec in c.get("records", []):
                    if rec.get("name") == "overall":
                        wins, losses = _parse_record(rec.get("summary", ""))
                        break

                physical = {}
                eventlog_ref = None
                athlete_id = c.get("athlete", {}).get("id") or c.get("id")
                if attempt_athlete_detail and athlete_id:
                    physical, eventlog_ref = _fetch_espn_athlete_detail(athlete_id)

                # Method-of-victory breakdown was attempted here (parsing the same
                # records array Pass 1 already fetches) and removed after real
                # production logs (July 2026) showed, across roughly 80 fighters
                # with zero exceptions, that ESPN's records array here contains
                # only a single 'overall' entry -- no KO/TKO, Submission, or
                # Decision breakdown exists in this data at all. Not a parsing
                # bug to fix; a confirmed absence in the source itself.

                if attempt_athlete_detail and eventlog_ref:
                    physical.update(_fetch_espn_last_fight_info(eventlog_ref))

                if name in needs_basic:
                    row = {col: None for col in fighters.columns}
                    row.update({
                        "name": name, "weight_class": weight_class_by_fighter.get(name),
                        "country": country, "wins": wins, "losses": losses,
                    })
                    row.update(physical)
                    new_rows.append(row)
                    filled_count += 1
                    print(f"[fighter_backfill] new roster entry: {name} ({country}, {wins}-{losses}"
                          f"{', ' + str(len(physical)) + ' extra field(s)' if physical else ''})")
                elif name in needs_gap_fill:
                    idx = fighters.index[fighters["name"] == name]
                    if len(idx) == 0:
                        continue
                    i = idx[0]
                    updated_fields = []
                    if pd.isna(fighters.at[i, "country"]) and country:
                        fighters = _safe_set_cell(fighters, i, "country", country)
                        updated_fields.append("country")
                    for col, val in physical.items():
                        if pd.isna(fighters.at[i, col]):
                            fighters = _safe_set_cell(fighters, i, col, val)
                            updated_fields.append(col)
                    if updated_fields:
                        filled_count += 1
                        print(f"[fighter_backfill] filled gap(s) for {name}: {', '.join(updated_fields)}")

    if new_rows:
        fighters = pd.concat([fighters, pd.DataFrame(new_rows)], ignore_index=True)
    if filled_count:
        fighters.to_csv(fighters_path, index=False)
    return filled_count
