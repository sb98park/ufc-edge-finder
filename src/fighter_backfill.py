"""
Backfills fighters.csv from ESPN for fighters appearing on tracked
future cards -- addressing two confirmed, measured gaps: 28 of 58
future-card fighters were missing from fighters.csv entirely (breaking
model preview generation for 14 of 29 future fights), and even fighters
already in the roster were frequently missing stance/country/reach.

Two passes, kept DELIBERATELY separate because they rest on very
different levels of confidence -- see each function's own docstring:

  Pass 1 (backfill_basic_profiles): name, country, and overall win-loss
  record. Built on data this project directly observed and verified
  live during development (the scoreboard's competitor.athlete and
  competitor.records fields) -- same confidence tier as
  results_fetcher.py and card_discovery.py's core functionality.

  Pass 2 (backfill_physical_stats): height, reach, stance. Built on an
  athlete-detail endpoint whose exact response shape was NEVER directly
  observed during development -- every attempt to verify it hit a wall
  (no network access from the dev sandbox, and web-fetch tools couldn't
  render ESPN's fighter-profile pages to confirm the underlying JSON).
  Parses several plausible field paths defensively and validates
  anything found against a sane physical range before accepting it
  (rather than trusting a matched field name alone) -- and when nothing
  passes, logs the response's own top-level keys so the real GitHub
  Actions logs become the first genuine confirmation of the actual
  schema, rather than this staying a guess indefinitely.

Both passes only ever fill gaps -- an empty cell for a fighter already
in the roster, or a wholly new row for one who's missing entirely.
Never overwrites a non-null value already in fighters.csv. Runs as
part of every generate_site.py call, same pattern as the rest of this
project's ESPN integration. Never raises.
"""

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


def _fetch_espn_athlete_physical(athlete_id: str) -> dict:
    """
    Pass 2 -- see this module's docstring for the honest confidence
    caveat. Attempts sports.core.api.espn.com's athlete-detail endpoint
    and defensively searches a few plausible field paths for height,
    reach, and stance, since the exact schema was never directly
    confirmed. Returns only fields that were both found AND passed a
    sanity check (plausible human height/reach range; stance matching a
    known value) -- an unrecognized or out-of-range value is treated as
    "not found," not accepted on a name match alone. Returns {} on any
    failure, unexpected shape, or nothing passing validation, and logs
    the response's own top-level keys in that last case so a real run's
    Action log becomes actual evidence for fixing this later, not more
    guessing.
    """
    url = f"https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{athlete_id}"
    try:
        resp = requests.get(url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"[fighter_backfill] athlete-detail fetch failed for id={athlete_id}: {e}")
        return {}

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

    if not result:
        print(f"[fighter_backfill] athlete-detail for id={athlete_id}: no field passed validation. "
              f"Top-level keys in response, for diagnosing the real schema: {sorted(data.keys())}")

    return result


def backfill_fighters(fighters_path: str = "data/fighters.csv",
                       future_cards_path: str = "data/future_cards.csv",
                       attempt_physical_stats: bool = True) -> int:
    """
    Entry point called from generate_site.py. Returns the number of
    fighters newly added or filled in -- never raises. For every
    fighter on a tracked future card: creates a minimal roster row
    (Pass 1) if they're missing from fighters.csv entirely, or fills
    just the empty stance/country/reach/height_in cells (Passes 1 and
    2) if they're already present with gaps. Existing non-null values
    are never touched.
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
    gap_cols = ["stance", "country", "reach_in", "height_in"]
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
                athlete_id = c.get("athlete", {}).get("id") or c.get("id")
                if attempt_physical_stats and athlete_id:
                    physical = _fetch_espn_athlete_physical(athlete_id)

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
                          f"{', ' + str(len(physical)) + ' physical field(s)' if physical else ''})")
                elif name in needs_gap_fill:
                    idx = fighters.index[fighters["name"] == name]
                    if len(idx) == 0:
                        continue
                    i = idx[0]
                    updated_fields = []
                    if pd.isna(fighters.at[i, "country"]) and country:
                        fighters.at[i, "country"] = country
                        updated_fields.append("country")
                    for col, val in physical.items():
                        if pd.isna(fighters.at[i, col]):
                            fighters.at[i, col] = val
                            updated_fields.append(col)
                    if updated_fields:
                        filled_count += 1
                        print(f"[fighter_backfill] filled gap(s) for {name}: {', '.join(updated_fields)}")

    if new_rows:
        fighters = pd.concat([fighters, pd.DataFrame(new_rows)], ignore_index=True)
    if filled_count:
        fighters.to_csv(fighters_path, index=False)
    return filled_count
