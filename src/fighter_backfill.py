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

from src.card_matcher import _normalize_name
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT, ESPN_SCOREBOARD_URL, WIKIPEDIA_OPENSEARCH_URL, is_placeholder_fighter_name

# Sentinel distinguishing "this source rate-limited us" from a genuine
# "no data here" (None) or "here's the data" (dict) result -- lets the
# caller trip a per-source circuit breaker specifically on 429s rather
# than on every ordinary miss, which would be both wrong (a miss isn't
# a rate-limit signal) and pointless (there'd be nothing to break).
RATE_LIMITED = "RATE_LIMITED"

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


def _fetch_method_breakdown_from_combat_edge(name: str) -> dict | None:
    """
    Career-wide KO/submission/decision win-and-loss breakdown, via
    Combat Edge -- tried before the Wikipedia fallback below since it
    has two real advantages, both verified directly rather than assumed:
    (1) a plain-HTTP, JS-free A-Z fighter directory (unlike Sherdog,
    FightMatrix, or ufcstats.com, none of which exposed a working
    plain-GET search this session), so name-to-URL discovery doesn't
    depend on Wikipedia's search working or covering this fighter at
    all; (2) each profile directly labels "N Wins by knockout" etc. as
    plain text, not a template parameter name that has to be guessed
    correctly -- confirmed this closes a real, specific Wikipedia gap
    (a fighter with no Wikipedia article at all still had a full,
    correct breakdown here).

    Returns None, not a guessed zero, if the fighter isn't listed on
    the matching A-Z page, the profile page doesn't have this section,
    or any request fails -- same "don't guess" principle as the
    Wikipedia path.
    """
    first_letter = name.strip()[:1].lower()
    if not first_letter.isalpha():
        return None

    try:
        directory_resp = requests.get(
            f"https://combat-edge.com/fighters/a-z/{first_letter}/",
            headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT,
        )
        directory_resp.raise_for_status()
        directory_html = directory_resp.text
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print(f"[fighter_backfill] combat-edge rate-limited (429) fetching directory for {name!r} -- "
                  f"backing off this source for the rest of this run")
            return RATE_LIMITED
        print(f"[fighter_backfill] combat-edge directory fetch failed for {name!r} (letter {first_letter!r}): {e}")
        return None
    except Exception as e:
        print(f"[fighter_backfill] combat-edge directory fetch failed for {name!r} (letter {first_letter!r}): {e}")
        return None

    # Directory entries link fighter name text directly to their profile
    # URL: <a href="/fighter/luke-riley-9437/">Luke Riley</a>. Match the
    # exact name (case-insensitive) to its href.
    link_match = re.search(
        rf'href="(/fighter/[^"]+)"[^>]*>\s*{re.escape(name.strip())}\s*<',
        directory_html, re.IGNORECASE,
    )
    if not link_match:
        print(f"[fighter_backfill] combat-edge: {name!r} not found on the {first_letter!r} directory page")
        return None
    profile_url = "https://combat-edge.com" + link_match.group(1)

    try:
        profile_resp = requests.get(profile_url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        profile_resp.raise_for_status()
        profile_html = profile_resp.text
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print(f"[fighter_backfill] combat-edge rate-limited (429) fetching profile for {name!r} -- "
                  f"backing off this source for the rest of this run")
            return RATE_LIMITED
        print(f"[fighter_backfill] combat-edge profile fetch failed for {name!r} ({profile_url}): {e}")
        return None
    except Exception as e:
        print(f"[fighter_backfill] combat-edge profile fetch failed for {name!r} ({profile_url}): {e}")
        return None

    def _extract(label: str) -> int | None:
        match = re.search(rf"(\d+)\s*{label}", profile_html, re.IGNORECASE)
        return int(match.group(1)) if match else None

    breakdown = {
        "ko_wins": _extract("Wins by knockout"), "sub_wins": _extract("Wins by submission"), "dec_wins": _extract("Wins by decision"),
        "ko_losses": _extract("Loss by knockout"), "sub_losses": _extract("Loss by submission"), "dec_losses": _extract("Loss by decision"),
    }
    parsed_count = sum(1 for v in breakdown.values() if v is not None)
    if parsed_count == 0:
        print(f"[fighter_backfill] combat-edge: {profile_url} found but no win/loss-by-method fields matched")
        return None
    print(f"[fighter_backfill] combat-edge method-breakdown: {name!r} -> {parsed_count}/6 fields parsed")
    return breakdown


def _fetch_method_breakdown_from_wikipedia(name: str) -> dict | None:
    """
    Career-wide KO/submission/decision win-and-loss breakdown, via
    Wikipedia -- since ESPN has no method-of-victory data at all
    (confirmed elsewhere in this codebase: its records array only ever
    has a single "overall" entry, no breakdown by method exists in that
    source). Wikipedia's {{Infobox martial artist}} template carries
    this as named, structured fields, consistently sourced from Sherdog
    across the fighters checked while building this.

    Fetches the page's raw wikitext (not rendered HTML) specifically so
    parsing relies on named template parameters, not on inferring which
    number means what from visual position -- far more robust against
    the page's exact layout/wording varying between fighters.

    Tries multiple plausible parameter-name variants per field (e.g.
    "mma_kowin" and "mmakowins") -- different mirrors of this template's
    own documentation disagree on exact naming, so a single hardcoded
    name risks silently matching nothing on a real page. Also tolerates
    an inline HTML comment sitting between the pipe and the parameter
    name, since real wikitext sometimes has editor annotations there
    that a plain-whitespace-only regex would fail to skip past.

    Returns None, not a guessed zero, when no Wikipedia page exists for
    this name or the page doesn't use this template -- many real,
    active fighters (especially newer/lesser-known ones) genuinely
    don't have their own Wikipedia article even though Sherdog tracks
    them, and that's a real "we don't know," not an error to paper over.
    """
    wiki_headers = {**BASE_HEADERS, "User-Agent": "OctaneAlpha/1.0 (personal MMA analytics project; contact via GitHub repo) fighter-backfill"}
    try:
        search_params = {"action": "opensearch", "search": name, "namespace": "0", "limit": "1", "format": "json"}
        resp = requests.get(WIKIPEDIA_OPENSEARCH_URL, params=search_params, headers=wiki_headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        results = resp.json()
        if len(results) < 4 or not results[3]:
            print(f"[fighter_backfill] wikipedia method-breakdown: no page match for {name!r}")
            return None
        page_title = results[3][0].rsplit("/", 1)[-1]
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print(f"[fighter_backfill] wikipedia rate-limited (429) searching for {name!r} -- "
                  f"backing off this source for the rest of this run")
            return RATE_LIMITED
        print(f"[fighter_backfill] wikipedia method-breakdown search failed for {name!r}: {e}")
        return None
    except Exception as e:
        print(f"[fighter_backfill] wikipedia method-breakdown search failed for {name!r}: {e}")
        return None

    try:
        raw_resp = requests.get(
            "https://en.wikipedia.org/w/index.php",
            params={"title": page_title, "action": "raw"},
            headers=wiki_headers, timeout=REQUEST_TIMEOUT,
        )
        raw_resp.raise_for_status()
        wikitext = raw_resp.text
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print(f"[fighter_backfill] wikipedia rate-limited (429) fetching {page_title!r} for {name!r} -- "
                  f"backing off this source for the rest of this run")
            return RATE_LIMITED
        print(f"[fighter_backfill] wikipedia method-breakdown fetch failed for {name!r} ({page_title!r}): {e}")
        return None
    except Exception as e:
        print(f"[fighter_backfill] wikipedia method-breakdown fetch failed for {name!r} ({page_title!r}): {e}")
        return None

    if "infobox martial artist" not in wikitext.lower():
        print(f"[fighter_backfill] wikipedia method-breakdown: {page_title!r} has no martial artist infobox")
        return None  # not an MMA fighter page, or doesn't use this template

    # Comment-tolerant, multi-variant field extraction. \|(?:<!--.*?-->)?\s*
    # lets an inline editor comment sit between the pipe and the param
    # name; each field tries several real-world naming variants in turn.
    variant_groups = {
        "ko_wins": ["mma_kowin", "mmakowins", "mma_ko_win"],
        "sub_wins": ["mma_subwin", "mmasubwins", "mma_sub_win"],
        "dec_wins": ["mma_decwin", "mmadecwins", "mma_dec_win"],
        "ko_losses": ["mma_koloss", "mmakolosses", "mma_ko_loss"],
        "sub_losses": ["mma_subloss", "mmasublosses", "mma_sub_loss"],
        "dec_losses": ["mma_decloss", "mmadeclosses", "mma_dec_loss"],
    }

    def _extract(variants: list[str]) -> int | None:
        for param in variants:
            match = re.search(rf"\|(?:<!--.*?-->)?\s*{param}\s*=\s*(\d+)", wikitext, re.IGNORECASE | re.DOTALL)
            if match:
                return int(match.group(1))
        return None

    breakdown = {field: _extract(variants) for field, variants in variant_groups.items()}
    parsed_count = sum(1 for v in breakdown.values() if v is not None)
    if parsed_count == 0:
        print(f"[fighter_backfill] wikipedia method-breakdown: {page_title!r} has the infobox template "
              f"but none of the known KO/SUB/DEC field name variants matched -- template naming may have "
              f"changed, worth checking a live wikitext sample directly")
        return None
    # Once the infobox genuinely carries method data (confirmed by at least
    # one field parsing), a sibling field that didn't parse is a real,
    # confirmed zero, not still-unknown -- Wikipedia's own editing
    # convention omits a genuinely-zero category's parameter entirely
    # rather than writing it as 0 (directly confirmed: a fighter's
    # rendered Losses section skips "By submission" entirely rather than
    # showing "By submission: 0" when that category is truly zero).
    zero_filled = [field for field, v in breakdown.items() if v is None]
    breakdown = {field: (v if v is not None else 0) for field, v in breakdown.items()}
    print(f"[fighter_backfill] wikipedia method-breakdown: {page_title!r} -> {parsed_count}/6 fields parsed"
          f"{f', {len(zero_filled)} inferred as zero (omitted from infobox)' if zero_filled else ''}")
    return breakdown


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
    for col in ("combat_edge_checked", "wikipedia_checked"):
        if col not in fighters.columns:
            fighters[col] = False
        fighters[col] = fighters[col].fillna(False).astype(bool)
    method_cols_for_migration = ["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]
    stuck_on_old_bug = fighters["wikipedia_checked"] & fighters[method_cols_for_migration].isna().any(axis=1)
    if stuck_on_old_bug.any():
        print(f"[fighter_backfill] one-time migration: resetting wikipedia_checked for "
              f"{int(stuck_on_old_bug.sum())} fighter(s) checked under the pre-fix zero-inference bug, "
              f"so they get one fresh, now-correct re-check: {sorted(fighters.loc[stuck_on_old_bug, 'name'])}")
        fighters.loc[stuck_on_old_bug, "wikipedia_checked"] = False
    try:
        future = pd.read_csv(future_cards_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0
    if future.empty:
        return 0

    roster_names = set(fighters["name"])
    future_fighters = {n for n in (set(future["fighter_a"]) | set(future["fighter_b"])) if not is_placeholder_fighter_name(n)}
    needs_basic = future_fighters - roster_names
    gap_cols = ["stance", "country", "reach_in", "height_in", "age", "last_fight_date",
                "ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]
    needs_gap_fill = set(
        fighters[fighters["name"].isin(future_fighters) & fighters[gap_cols].isna().any(axis=1)]["name"]
    )
    if not needs_basic and not needs_gap_fill:
        return 0

    # Circuit breakers, tripped for the rest of THIS run the first time
    # a source returns a 429 -- retrying a source that just told us to
    # back off only makes the block worse, and every other fighter in
    # this same run would almost certainly hit the same wall anyway.
    combat_edge_rate_limited = False
    wikipedia_rate_limited = False
    # Real production logs (July 2026) showed this backlog crossing 80+
    # fighters in one run (a one-time bootstrap after gap_cols grew to
    # include the 6 method-breakdown columns, which instantly made
    # nearly the whole pre-existing roster eligible for gap-fill at
    # once) -- both Combat Edge and Wikipedia hit real rate limits well
    # before the run finished. A cap on new method-breakdown lookups per
    # run spreads that one-time backlog across several 5-minute-interval
    # runs instead of bursting through it all at once.
    #
    # The circuit breakers above are what actually protect against
    # rate-limit abuse (confirmed directly: Combat Edge correctly
    # stopped after exactly 1 request once it 429'd) -- this cap only
    # needs to be a generous safety net against a source misbehaving in
    # a way that doesn't trip RATE_LIMITED (e.g. 200 with garbage data),
    # not the primary defense. A follow-up production log showed 15 was
    # too tight for that role: it cut off mid-card, exactly 15 fighters
    # into a single event's own competitor list, well before reaching
    # fighters later in that same card's billing order who still
    # genuinely needed backfill and whose sources were never even
    # attempted as a result. A full UFC card can run 24-28 fighters.
    METHOD_BREAKDOWN_CAP_PER_RUN = 60
    method_breakdown_attempts_this_run = 0
    cap_reached_logged = False

    weight_class_by_fighter = {}
    for _, r in future.iterrows():
        weight_class_by_fighter.setdefault(r["fighter_a"], r.get("weight_class"))
        weight_class_by_fighter.setdefault(r["fighter_b"], r.get("weight_class"))

    filled_count = 0
    any_checked_flag_changed = False
    new_rows = []
    event_order = future[["event_name", "event_date"]].drop_duplicates().copy()
    event_order["_sort_date"] = pd.to_datetime(event_order["event_date"], errors="coerce")
    event_order = event_order.sort_values("_sort_date", na_position="last")
    for event_name, event_date in event_order[["event_name", "event_date"]].itertuples(index=False):
        target_names = {n for n in (needs_basic | needs_gap_fill)
                         if n in set(future[future["event_name"] == event_name]["fighter_a"])
                         or n in set(future[future["event_name"] == event_name]["fighter_b"])}
        if not target_names:
            continue
        # Match ESPN's spelling against ours by normalized form, not exact
        # string equality -- accents, punctuation, and transliteration
        # differences between ESPN and whatever originally populated
        # fighters.csv/future_cards.csv otherwise cause this whole block
        # to silently skip a real, tracked fighter (same category of bug
        # already fixed elsewhere in this codebase for cross-source
        # matching). Canonical (our) spelling wins, per that same
        # established precedent, so target_names_by_normalized maps back
        # to the name already on record rather than ESPN's variant.
        target_names_by_normalized = {_normalize_name(n): n for n in target_names}
        matched_target_names = set()

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
            espn_event_names = [ev.get("name") for ev in data.get("events", [])]
            print(f"[fighter_backfill] event name mismatch for {event_name!r} on {date_param} -- "
                  f"ESPN returned {espn_event_names!r}, {len(target_names)} tracked name(s) for this "
                  f"event will not be backfilled this run: {sorted(target_names)}")
            continue

        for comp in matched.get("competitions", []):
            for c in comp.get("competitors", []):
                espn_name = c.get("athlete", {}).get("fullName")
                name = target_names_by_normalized.get(_normalize_name(espn_name or ""))
                if name is None:
                    continue
                matched_target_names.add(name)

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

                existing_row = fighters[fighters["name"] == name]
                method_cols = ["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]
                already_exhausted = (
                    not existing_row.empty
                    and bool(existing_row["combat_edge_checked"].iloc[0])
                    and bool(existing_row["wikipedia_checked"].iloc[0])
                )
                needs_method_data = not already_exhausted and (
                    name in needs_basic or (
                        not existing_row.empty and existing_row[method_cols].isna().any(axis=1).iloc[0]
                    )
                )
                if needs_method_data and method_breakdown_attempts_this_run >= METHOD_BREAKDOWN_CAP_PER_RUN:
                    if not cap_reached_logged:
                        print(f"[fighter_backfill] method-breakdown cap ({METHOD_BREAKDOWN_CAP_PER_RUN}) reached "
                              f"for this run at {name!r} -- remaining fighters needing this will retry next run")
                        cap_reached_logged = True
                    needs_method_data = False  # cap reached -- leave for a later run rather than risk worsening a rate limit
                already_checked = {
                    "combat_edge": not existing_row.empty and bool(existing_row["combat_edge_checked"].iloc[0]),
                    "wikipedia": not existing_row.empty and bool(existing_row["wikipedia_checked"].iloc[0]),
                }
                if needs_method_data:
                    method_breakdown_attempts_this_run += 1
                    breakdown = None
                    if combat_edge_rate_limited and wikipedia_rate_limited:
                        print(f"[fighter_backfill] {name}: both method-breakdown sources already rate-limited "
                              f"this run -- skipped, will retry next run")
                    if not combat_edge_rate_limited and not already_checked["combat_edge"]:
                        breakdown = _fetch_method_breakdown_from_combat_edge(name)
                        if breakdown == RATE_LIMITED:
                            combat_edge_rate_limited = True
                            breakdown = None
                        else:
                            physical["combat_edge_checked"] = True  # genuine attempt happened, regardless of outcome
                    if not breakdown and not wikipedia_rate_limited and not already_checked["wikipedia"]:
                        breakdown = _fetch_method_breakdown_from_wikipedia(name)
                        if breakdown == RATE_LIMITED:
                            wikipedia_rate_limited = True
                            breakdown = None
                        else:
                            physical["wikipedia_checked"] = True  # genuine attempt happened, regardless of outcome
                    if breakdown:
                        physical.update(breakdown)

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
                        if col in ("combat_edge_checked", "wikipedia_checked"):
                            if not bool(fighters.at[i, col]):  # only a genuine False->True change is worth writing
                                fighters = _safe_set_cell(fighters, i, col, val)
                                any_checked_flag_changed = True
                        elif pd.isna(fighters.at[i, col]):
                            fighters = _safe_set_cell(fighters, i, col, val)
                            updated_fields.append(col)
                    if updated_fields:
                        filled_count += 1
                        print(f"[fighter_backfill] filled gap(s) for {name}: {', '.join(updated_fields)}")

        unmatched = target_names - matched_target_names
        if unmatched:
            print(f"[fighter_backfill] {event_name!r}: {len(unmatched)} tracked name(s) never matched an "
                  f"ESPN competitor even with normalized comparison, still unbackfilled: {sorted(unmatched)}")

    if new_rows:
        fighters = pd.concat([fighters, pd.DataFrame(new_rows)], ignore_index=True)
    if filled_count or any_checked_flag_changed:
        fighters.to_csv(fighters_path, index=False)
    return filled_count
