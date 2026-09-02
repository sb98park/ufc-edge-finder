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
import os
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



# Completion STATES that ESPN returns where a method belongs. Shared by both
# last-fight extraction paths so they can never disagree about what counts as
# a real method.
STATUS_WORDS = {"final", "final/ot", "ft", "completed", "complete",
                "status_final", "end", "ended"}

SITE_ATHLETE_URL = "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/athletes/{}"


def _parse_wl(display_value):
    """'10-1' or '10-1-0' -> (10, 1). Returns (None, None) on anything else."""
    try:
        parts = [int(x) for x in str(display_value).strip().split("-")[:2]]
        return (parts[0], parts[1]) if len(parts) == 2 else (None, None)
    except (ValueError, AttributeError, IndexError):
        return (None, None)


def _fetch_espn_method_records(athlete_id: str) -> dict:
    """
    Method splits from ESPN's SITE athlete endpoint.

    WHY THIS EXISTS DESPITE AN EARLIER 'CONFIRMED ABSENCE'. A previous attempt
    parsed the SCOREBOARD's `records` array, found only an 'overall' entry
    across ~80 fighters, and concluded the breakdown wasn't in ESPN at all.
    That generalised one endpoint to the whole API. It IS published, under
    athlete.statsSummary.statistics on the site endpoint -- confirmed from a
    real probe: 'wins-losses-draws' -> '14-3-0' alongside 'tkos-tkoLosses'.
    That endpoint was never being called.

    Matching is by SUBSTRING on the entry name/displayName rather than an
    exact key, because only the TKO key was visible in the probe output and
    guessing the submission key exactly is how the last wrong conclusion got
    made. Anything unrecognised is logged, so a missed split self-diagnoses
    on the next run instead of silently staying blank.
    """
    out = {}
    try:
        resp = requests.get(SITE_ATHLETE_URL.format(athlete_id), headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return out
        stats = (resp.json().get("athlete") or {}).get("statsSummary", {}).get("statistics", [])
    except Exception as exc:
        print(f"[fighter_backfill] method-records fetch failed for {athlete_id}: {exc}")
        return out

    unmatched = []
    for entry in stats or []:
        name = f"{entry.get('name','')} {entry.get('displayName','')}".lower()
        w, l = _parse_wl(entry.get("displayValue"))
        if w is None:
            continue
        if "draw" in name or name.strip().startswith("wins-losses"):
            out["_total_w"], out["_total_l"] = w, l          # for the DEC remainder
        elif "knockout" in name or "tko" in name or name.startswith("ko"):
            out["ko_wins"], out["ko_losses"] = w, l
        elif "submission" in name or "sub" in name:
            out["sub_wins"], out["sub_losses"] = w, l
        elif "decision" in name:
            out["dec_wins"], out["dec_losses"] = w, l
        else:
            unmatched.append(f"{entry.get('name')}={entry.get('displayValue')}")

    # DEC by subtraction when ESPN doesn't publish it directly. This is
    # arithmetic, not inference: 14-3 total with KO 10-1 and SUB 4-1 leaves
    # DEC 0-1 exactly. Only applied when BOTH other splits are known, so a
    # missing one can never be silently absorbed into decisions.
    if "dec_wins" not in out and "_total_w" in out \
            and "ko_wins" in out and "sub_wins" in out:
        dw = out["_total_w"] - out["ko_wins"] - out["sub_wins"]
        dl = out["_total_l"] - out["ko_losses"] - out["sub_losses"]
        if dw >= 0 and dl >= 0:
            out["dec_wins"], out["dec_losses"] = dw, dl
    if unmatched:
        print(f"[fighter_backfill] unrecognised stat entries for {athlete_id}: {unmatched[:5]}")
    # Hand the career W-L back to the caller. It was being dropped after
    # deriving the decision remainder, even though it is the most reliable
    # career record ESPN publishes.
    if "_total_w" in out:
        out["_career_w"], out["_career_l"] = out["_total_w"], out["_total_l"]
    out.pop("_total_w", None)
    out.pop("_total_l", None)
    return out


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

    # DIAGNOSTIC (July 2026): user reported reach_in displaying identical to
    # height_in for multiple fighters (e.g. both showing 75.0). Height and
    # reach are extracted independently above from separate ESPN field names
    # ('height'/'displayHeight' vs 'reach'/'displayReach') -- there is no
    # fallback in this code that copies one into the other. So if they come
    # out exactly equal, it's either a real coincidence, or ESPN's own API is
    # returning matching values for both fields (a known pattern on sites
    # that don't track reach separately and mirror height as a placeholder).
    # Log the raw source values whenever this happens so it's confirmed one
    # way or the other from a live run, rather than guessing at which.
    if (result.get("height_in") is not None and result.get("reach_in") is not None
            and result["height_in"] == result["reach_in"]):
        print(f"[fighter_backfill] DIAGNOSTIC: height_in == reach_in ({result['height_in']}) for "
              f"athlete_id={athlete_id}. Raw ESPN fields -- height={data.get('height')!r}, "
              f"displayHeight={data.get('displayHeight')!r}, reach={data.get('reach')!r}, "
              f"displayReach={data.get('displayReach')!r}")

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


_LAST_FIGHT_METHOD_LOGGED = 0



def _fetch_last_fight_from_events_map(athlete_id: str, fighter_name: str | None = None) -> dict:
    """
    Last fight from the SITE athlete endpoint's eventsMap.

    WHY THIS REPLACES THE eventLog PATH. The core api's eventLog marks some
    SCHEDULED bouts as played=true, which gave 45 fighters a "last fight"
    dated up to five weeks in the FUTURE with a fabricated result attached.
    Rejecting those left many fighters with nothing at all, even though
    espn.com renders their full history -- because that page is backed by
    eventsMap, which we were never reading.

    Each eventsMap value carries gameDate, gameResult, opponent and status,
    so a completed bout is identifiable without trusting a played flag.

    Field SHAPES are handled defensively (dict or string) and anything
    unrecognised is logged rather than silently dropped -- the probe showed
    the keys but not every value type, and guessing shapes has been the
    single most expensive habit in this codebase.
    """
    out = {}
    try:
        resp = requests.get(SITE_ATHLETE_URL.format(athlete_id), headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return out
        events_map = resp.json().get("eventsMap") or {}
    except Exception as exc:
        print(f"[fighter_backfill] eventsMap fetch failed for {fighter_name or athlete_id}: {exc}")
        return out

    today = dt.date.today().isoformat()
    candidates = []
    for rec in events_map.values():
        if not isinstance(rec, dict):
            continue
        date = str(rec.get("gameDate") or "")[:10]
        if not date or date > today:
            continue                      # scheduled, or today's card mid-event
        result = rec.get("gameResult")
        if isinstance(result, dict):
            result = result.get("displayName") or result.get("abbreviation")
        result = str(result or "").strip().upper()
        # No W/L means the bout hasn't resolved, whatever status claims.
        if result not in ("W", "L", "D", "WIN", "LOSS", "DRAW"):
            continue
        candidates.append((date, rec, result))

    if not candidates:
        return out
    date, rec, result = max(candidates, key=lambda t: t[0])

    opp = rec.get("opponent")
    if isinstance(opp, dict):
        opp = opp.get("displayName") or opp.get("shortDisplayName") or opp.get("name")
    opp = str(opp).strip() if opp else None

    out["last_fight_date"] = date
    out["last_fight_result"] = "W" if result.startswith("W") else ("L" if result.startswith("L") else "D")
    if opp:
        out["last_fight_opponent"] = opp

    # Method, only if it's a real one. status.type.detail returns "Final" --
    # a completion state -- which is what produced "W by Final against X".
    status = rec.get("status")
    if isinstance(status, dict):
        stype = status.get("type") if isinstance(status.get("type"), dict) else status
        method = stype.get("detail") or stype.get("description") or stype.get("shortDetail")
        if isinstance(method, str) and method.strip().lower() not in STATUS_WORDS:
            out["last_fight_method"] = method.strip()
    return out


def _fetch_espn_last_fight_info(eventlog_ref: str, athlete_id=None, fighter_name=None) -> dict:
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
        print(f"[fighter_backfill] eventLog fetch failed for {fighter_name or athlete_id}: {e}")
        return {}

    # Confirmed via real production logs (July 2026): the actual field
    # name is 'events', not the originally-guessed 'items' -- every
    # eventLog response observed so far has had exactly ['$ref', 'events']
    # as its top-level keys. Kept the 'items' check too as a harmless
    # fallback in case a different athlete or a future schema change uses it.
    items = data.get("events") if isinstance(data, dict) and "events" in data else (
        data.get("items") if isinstance(data, dict) else (data if isinstance(data, list) else None)
    )

    # REAL PARSER (July 2026), written against a fully-confirmed schema from 4
    # rounds of production diagnostics. Confirmed structure:
    #   eventLog -> {'$ref','events'}
    #   events -> {count, items:[...], pageCount, pageIndex, pageSize}  (container)
    #   events.items[N] -> {competition:{$ref}, competitor:{$ref}, event:{$ref}, played:bool}
    #   item.event   -> {date, name, ...}                (date lives here)
    #   item.competition -> {competitors:[2], status:{$ref}, type:{...}, date, ...}
    #     competition.competitors[i] -> {athlete:{$ref}, winner:bool, order, id, ...}
    #     competition.type.text = WEIGHT CLASS (e.g. 'Lightweight') -- NOT the method
    #     competition.status -> {$ref} -> {type:{completed, description, detail, name, ...}}
    # Confirmed ORDERING: items[0] is the fighter's UPCOMING (played=False) bout;
    # all later items are played=True history. So we must filter played=True and
    # pick the most recent by event date -- never items[0] blindly.
    if isinstance(items, dict):
        inner = items.get("items")
        items = inner if isinstance(inner, list) else None

    if not items or not isinstance(items, list):
        print(f"[fighter_backfill] eventLog: no usable events list for {fighter_name or athlete_id}; skipping last-fight.")
        return {}

    # Keep only completed bouts. 'played' is embedded on each item (no fetch).
    played_items = [it for it in items if isinstance(it, dict) and it.get("played") is True]
    if not played_items:
        print(f"[fighter_backfill] eventLog: no played bouts found for {fighter_name or athlete_id} (only scheduled) -- likely a UFC debut with no prior UFC-tracked history; nothing to fill.")
        return {}

    # We only have this one page of items (pageIndex 1). That's fine: the most
    # recent completed bout is what we want, and page 1 holds the newest slice.
    # We still don't KNOW the page ordering for certain, so rather than trust
    # position we fetch each played item's event date and take the true max --
    # but bounded: cap at the first 6 played items to avoid a request storm,
    # since the most-recent bout is near the top of page 1 regardless of
    # asc/desc (page 1 can't exclude it when this fighter has >0 completed).
    BOUNDED = played_items[:6]

    def _get(ref):
        try:
            r = requests.get(ref, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            return None

    # Resolve (event_date, item) for each bounded played item, then pick latest.
    dated = []
    for it in BOUNDED:
        ev = it.get("event")
        ev_ref = ev.get("$ref") if isinstance(ev, dict) else None
        ev_obj = _get(ev_ref) if ev_ref else None
        d = ev_obj.get("date") if isinstance(ev_obj, dict) else None
        if isinstance(d, str) and len(d) >= 10:
            # A "last fight" cannot be in the FUTURE. ESPN's eventLog marks
            # some SCHEDULED bouts as played=true, so that flag alone let an
            # upcoming card through -- fighters were showing a last fight
            # dated a week ahead of today, with a fabricated result attached.
            # Comparing against today is cheap and doesn't depend on trusting
            # the flag.
            if d[:10] > dt.date.today().isoformat():
                print(f"[fighter_backfill] eventLog: skipping FUTURE-dated bout {d[:10]} "
                      f"for {fighter_name or athlete_id} -- ESPN flagged a scheduled fight as played.")
                continue
            dated.append((d[:10], it))
    if not dated:
        print(f"[fighter_backfill] eventLog: couldn't resolve any event dates for {fighter_name or athlete_id} -- skipping last-fight.")
        return {}
    dated.sort(key=lambda t: t[0], reverse=True)
    last_date, last_item = dated[0]

    result = {"last_fight_date": last_date}

    # Follow this bout's competition for opponent + result + method.
    comp = last_item.get("competition")
    comp_ref = comp.get("$ref") if isinstance(comp, dict) else None
    comp_obj = _get(comp_ref) if comp_ref else None
    if not isinstance(comp_obj, dict):
        # Date alone is still worth returning.
        return result

    competitors = comp_obj.get("competitors")
    if isinstance(competitors, list) and len(competitors) == 2:
        # Identify which competitor is OUR fighter via athlete_id (passed in).
        def _athlete_id(c):
            ath = c.get("athlete") if isinstance(c, dict) else None
            # athlete is a {$ref}; the id is in the ref URL's last path segment,
            # or we can match by following it. Cheapest: parse the ref URL.
            ref = ath.get("$ref") if isinstance(ath, dict) else None
            if isinstance(ref, str):
                # .../athletes/NNNN?lang=... -> grab the digits after /athletes/
                m = re.search(r"/athletes/(\d+)", ref)
                if m:
                    return m.group(1)
            return None

        ours = them = None
        for c in competitors:
            aid = _athlete_id(c)
            if athlete_id and aid == str(athlete_id):
                ours = c
            else:
                them = c
        # Fallback: if id-matching failed, we can't safely tell who's who --
        # leave result/opponent out rather than guess and risk inverting them.
        if ours is not None and them is not None:
            # RESULT: our fighter's winner boolean. NOT a two-way street --
            # in a draw or no contest ESPN sets `winner` False on BOTH
            # competitors, so reading ours alone reported a defeat for each of
            # them. Ours False AND theirs False is nobody's win.
            won, theirs = ours.get("winner"), them.get("winner")
            if won is True:
                result["last_fight_result"] = "W"
            elif won is False and theirs is True:
                result["last_fight_result"] = "L"
            elif won is False and theirs is False:
                result["last_fight_result"] = "NC"
            # OPPONENT: follow the other competitor's athlete for displayName.
            them_ath = them.get("athlete")
            them_ref = them_ath.get("$ref") if isinstance(them_ath, dict) else None
            them_obj = _get(them_ref) if them_ref else None
            if isinstance(them_obj, dict):
                opp_name = them_obj.get("displayName") or them_obj.get("fullName")
                if opp_name:
                    result["last_fight_opponent"] = opp_name

    # METHOD: from competition.status -> type. The exact value field for a
    # COMPLETED bout hasn't been directly observed yet (every diagnostic landed
    # on the scheduled items[0]), so read the most specific available field and
    # LOG the raw status.type the first few times so the mapping is verifiable
    # rather than blind. 'detail'/'shortDetail' typically read like "Decision -
    # Unanimous" or "KO/TKO"; 'description' is the human label.
    status = comp_obj.get("status")
    status_ref = status.get("$ref") if isinstance(status, dict) else None
    status_obj = _get(status_ref) if status_ref else None
    if isinstance(status_obj, dict):
        stype = status_obj.get("type")
        if isinstance(stype, dict):
            # Log raw values once per run (bounded via module-level flag) so the
            # real method-string mapping can be confirmed from a live log.
            global _LAST_FIGHT_METHOD_LOGGED
            if _LAST_FIGHT_METHOD_LOGGED < 5:
                print(f"[fighter_backfill] last-fight method raw status.type: "
                      f"{ {k: stype.get(k) for k in ('name','description','detail','shortDetail','completed')} }")
                _LAST_FIGHT_METHOD_LOGGED += 1
            method = stype.get("detail") or stype.get("shortDetail") or stype.get("description")
            # REJECT completion-status labels. status.type here reports whether
            # the bout has FINISHED, not how it ended -- "Final" / "FT" /
            # "Completed" are states, and writing one into last_fight_method is
            # what produced "W by Final against X" on every fighter card.
            # ESPN does publish real methods, but only via the CORE api (see
            # results_fetcher._fetch_espn_core_results). Until this path is
            # wired to that, storing nothing is strictly better than storing a
            # word that isn't a method: the template already renders the bare
            # "W against X", which is true.
            if isinstance(method, str) and method.strip() \
                    and method.strip().lower() not in STATUS_WORDS:
                result["last_fight_method"] = method.strip()

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


def _combat_edge_get(url: str, headers: dict, timeout: float):
    """
    Fetches a combat-edge.com URL, optionally routed through a Cloudflare
    Worker relay (see cloudflare-worker/combat-edge-relay.js) instead of
    hitting combat-edge.com directly. Only routes through the relay when
    BOTH COMBAT_EDGE_RELAY_URL and COMBAT_EDGE_RELAY_TOKEN environment
    variables are set (e.g. as GitHub Actions repo secrets) -- with
    neither set, this behaves exactly like a plain requests.get() call,
    so existing behavior is completely unchanged until the relay is
    actually deployed and configured. This is a genuinely unverified
    attempt at working around the GitHub-Actions-IP-range block (see the
    Worker file's own docstring for the full reasoning) -- not a
    guaranteed fix, just the most promising lead found so far.
    """
    relay_url = os.environ.get("COMBAT_EDGE_RELAY_URL", "").strip()
    relay_token = os.environ.get("COMBAT_EDGE_RELAY_TOKEN", "").strip()
    if relay_url and relay_token:
        relayed = f"{relay_url}?token={requests.utils.quote(relay_token)}&url={requests.utils.quote(url, safe='')}"
        return requests.get(relayed, timeout=timeout)
    return requests.get(url, headers=headers, timeout=timeout)


def _fetch_method_breakdown_from_combat_edge(name: str, known_wins: int | None = None) -> dict | None:
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

    # A realistic browser User-Agent specific to this source, not the
    # shared BASE_HEADERS' "personal research script" string -- that
    # string doesn't match any real browser's format and is a plausible
    # contributing factor to this specific source's aggressive blocking,
    # confirmed via a direct fetch from different network infrastructure
    # succeeding cleanly with real data for fighters the GitHub Actions
    # runner's shared IP range has never once gotten through for.
    ce_headers = {**BASE_HEADERS, "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}

    try:
        directory_resp = _combat_edge_get(
            f"https://combat-edge.com/fighters/a-z/{first_letter}/",
            headers=ce_headers, timeout=REQUEST_TIMEOUT,
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
        profile_resp = _combat_edge_get(profile_url, headers=ce_headers, timeout=REQUEST_TIMEOUT)
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

    # THE CAREER RECORD, which this function fetched all along and threw
    # away. The profile header carries "Record: 5-1-0" in plain text, and we
    # are already holding the page.
    #
    # It matters because ESPN is the only source for wins/losses and it has
    # nothing at all for some debutants -- no athlete page, no scoreboard
    # entry. Terrance Chatman went into UFC Fight Night 2026-08-22 recorded
    # as 0-0 when he is 5-1, which made his 7-0 opponent Lock of the Week.
    # Combat Edge had "Record: 5-1-0" for him the whole time.
    #
    # Draws are captured too even though fighters.csv has no draws column,
    # so the caller can tell "5-1-0" from a malformed parse rather than
    # inferring it from two numbers.
    record_wins = record_losses = None
    rec_match = re.search(r"Record:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", profile_html, re.IGNORECASE)
    if rec_match:
        record_wins, record_losses = int(rec_match.group(1)), int(rec_match.group(2))

    # Confirmed via real production diagnostics (July 2026): Combat Edge's
    # current template includes a sentence under a "How does X usually win?"
    # heading reading "X's N recorded wins include A knockouts, B
    # submissions, and C decisions, with P% ending before the final bell."
    # -- digit-based (not word-numbers like "seven"), verified against 2 real
    # fighters where A+B+C == N exactly. This is now the PRIMARY win-side
    # extractor; the old _extract() label regexes stay as a fallback in case
    # some pages still use the pre-redesign wording. Sums are sanity-checked
    # before being trusted -- a mismatch means don't guess, fall through.
    win_sentence = re.search(
        r"(\d+)\s*recorded wins include\s*(\d+)\s*knockouts?,?\s*(\d+)\s*submissions?,?\s*and\s*(\d+)\s*decisions?",
        profile_html, re.IGNORECASE,
    )
    ko_wins = sub_wins = dec_wins = None
    if win_sentence:
        total, ko, sub, dec = (int(x) for x in win_sentence.groups())
        if ko + sub + dec != total:
            print(f"[fighter_backfill] combat-edge: {name!r} win-sentence internally inconsistent "
                  f"({ko}+{sub}+{dec} != {total}) -- not trusting it, falling through to old extractor")
        elif known_wins is not None and total != known_wins:
            # Caught via a real audit run (July 2026): internal consistency
            # alone isn't enough -- Combat Edge's own stated total can
            # genuinely disagree with our already-recorded wins count (a
            # real cross-source discrepancy, not a parsing bug -- e.g. one
            # source not yet updated after a recent fight). Don't silently
            # write method data that would contradict our own wins field
            # and reintroduce exactly the kind of mismatch the audit tool
            # exists to catch.
            print(f"[fighter_backfill] combat-edge: {name!r} win-sentence total ({total}) disagrees with "
                  f"our recorded wins ({known_wins}) -- real cross-source discrepancy, not trusting it")
        else:
            ko_wins, sub_wins, dec_wins = ko, sub, dec

    breakdown = {
        "ko_wins": ko_wins if ko_wins is not None else _extract("Wins by knockout"),
        "sub_wins": sub_wins if sub_wins is not None else _extract("Wins by submission"),
        "dec_wins": dec_wins if dec_wins is not None else _extract("Wins by decision"),
        "ko_losses": _extract("Loss by knockout"), "sub_losses": _extract("Loss by submission"), "dec_losses": _extract("Loss by decision"),
    }
    parsed_count = sum(1 for v in breakdown.values() if v is not None)

    if parsed_count == 0:
        print(f"[fighter_backfill] combat-edge: {profile_url} found but no win/loss-by-method fields matched")
    else:
        print(f"[fighter_backfill] combat-edge method-breakdown: {name!r} -> {parsed_count}/6 fields parsed")

    # Loss-side diagnostic: fires whenever any loss field is still missing,
    # independent of whether the win side just succeeded via the confirmed
    # sentence pattern above -- Campbell and Pinto's real pages (round 2)
    # only ever showed a win-side "How does X usually win?" sentence, never
    # a parallel loss-side one, so we genuinely don't know yet whether one
    # exists (low-loss-count fighters might just not get that paragraph at
    # all) or uses different wording. Search broadly for "recorded loss" AND
    # any "usually lose"/"how does...lose" heading, uncapped-enough windows,
    # so a real occurrence can't be missed by a shared cap with win-side hits.
    if breakdown["ko_losses"] is None and breakdown["sub_losses"] is None and breakdown["dec_losses"] is None:
        loss_hits = 0
        for i, m in enumerate(re.finditer(r"recorded loss|usually lose|typically lose", profile_html, re.IGNORECASE)):
            if i >= 4:
                break
            loss_hits += 1
            start = max(0, m.start() - 150)
            end = min(len(profile_html), m.end() + 150)
            snippet = re.sub(r"\s+", " ", profile_html[start:end]).strip()
            print(f"[fighter_backfill] DIAGNOSTIC: {name!r} loss-side context #{i+1}: ...{snippet}...")
        if loss_hits == 0:
            print(f"[fighter_backfill] DIAGNOSTIC: {name!r} -- no loss-side breakdown language found at all "
                  f"(page length {len(profile_html)} chars). May genuinely not exist for this fighter "
                  f"(low loss count), or uses wording this search doesn't cover yet.")

    # The career record rides along even when NO method fields parsed. Those
    # are separate facts from separate parts of the page, and a fighter whose
    # method wording this parser doesn't cover can still have a perfectly
    # readable "Record: 5-1-0" in the header -- which is the single most
    # valuable field here for a debutant ESPN has never heard of.
    if record_wins is not None and record_losses is not None:
        breakdown["record_wins"] = record_wins
        breakdown["record_losses"] = record_losses
        parsed_count += 1

    if parsed_count == 0:
        return None
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
    # NOTE: a one-time migration used to live here, resetting wikipedia_checked
    # for any fighter where (checked=True AND still has a method-data gap).
    # That was a real, correctly-scoped fix for a specific bug (the Wikipedia
    # zero-inference parsing bug), confirmed resolved and working via a real
    # production log earlier this session. Removed because its trigger
    # condition can't distinguish "affected by that old, fixed bug" from "has
    # a permanent, genuinely unfillable gap" (no Wikipedia page exists at
    # all) -- so it kept re-firing every single run on any such fighter
    # forever, wastefully re-attempting Wikipedia for names it will never
    # find a page for, defeating the whole point of the checked-flags system.
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

                if attempt_athlete_detail and athlete_id:
                    method_recs = _fetch_espn_method_records(athlete_id)
                    # POP THE PRIVATE KEYS BEFORE MERGING, not after.
                    #
                    # _career_w/_career_l are internal signalling, not roster
                    # columns. This update() ran BEFORE the pop below, so they
                    # rode into `physical` and then into the gap-fill loop,
                    # which does fighters.at[i, col] for every key it holds --
                    # KeyError: '_career_w'.
                    #
                    # generate_site wraps this whole call in try/except and
                    # logs "fighter backfill failed ... continuing", so it
                    # never surfaced as a failure. It has been aborting the
                    # ENTIRE enrichment pass on every build, for both card
                    # files, which is why Terrance Chatman sat at 0-0 with no
                    # stance, country, reach or last-fight date: the code that
                    # would have filled them died before reaching him.
                    cw, cl = method_recs.pop("_career_w", None), method_recs.pop("_career_l", None)
                    physical.update(method_recs)
                    # PREFER THE ATHLETE ENDPOINT'S CAREER RECORD over the
                    # scoreboard's. The scoreboard's records[] entry named
                    # "overall" is the fighter's record WITHIN THAT PROMOTION,
                    # not their career -- so a regional fighter debuting in the
                    # UFC showed 2-1 against a true career mark of 16-2. That
                    # understates a debutant's experience, which is exactly the
                    # input the model leans on, and it silently inflated
                    # confidence in his opponent.
                    # athlete.statsSummary's "wins-losses-draws" IS the career
                    # figure (the same block the method splits come from), so
                    # when the two disagree the athlete endpoint wins.
                    if cw is not None and cl is not None:
                        if wins is not None and (cw != wins or cl != losses):
                            print(f"[fighter_backfill] {name}: scoreboard record {wins}-{losses} "
                                  f"disagrees with career {cw}-{cl}; using career")
                        wins, losses = cw, cl

                # eventsMap FIRST. It's the source behind espn.com's own
                # fight-history table, it distinguishes scheduled from
                # completed by whether a W/L exists, and it carries the
                # opponent and method in the same record. The eventLog path
                # below stays as a fallback for fighters whose site-endpoint
                # response lacks eventsMap.
                if attempt_athlete_detail and athlete_id:
                    em = _fetch_last_fight_from_events_map(athlete_id, name)
                    if em:
                        physical.update(em)

                if attempt_athlete_detail and eventlog_ref and not physical.get("last_fight_date"):
                    physical.update(_fetch_espn_last_fight_info(eventlog_ref, athlete_id, name))

                existing_row = fighters[fighters["name"] == name]
                method_cols = ["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]
                already_exhausted = (
                    not existing_row.empty
                    and bool(existing_row["combat_edge_checked"].iloc[0])
                    and bool(existing_row["wikipedia_checked"].iloc[0])
                )
                # AN ALL-ZERO ROW IS NOT DATA. A fighter recorded as 0-0 with
                # zero wins by every method has never been looked up -- those
                # zeros are the shape missing data takes here, not
                # measurements. isna() cannot see them, so the lookup that
                # would fix the row was gated off by the very emptiness it
                # exists to fill.
                #
                # Terrance Chatman: 0-0, all six method columns 0.0, so
                # needs_method_data was False and Combat Edge -- which has him
                # at 5-1 with a full win breakdown -- was never called.
                record_empty = (
                    not existing_row.empty
                    and pd.notna(existing_row["wins"].iloc[0])
                    and pd.notna(existing_row["losses"].iloc[0])
                    and float(existing_row["wins"].iloc[0]) == 0
                    and float(existing_row["losses"].iloc[0]) == 0
                )
                # AND THE SAME TRAP ON THE WIN SIDE. A fighter with 5 wins
                # and 0 KO + 0 sub + 0 dec is not a fighter whose wins came
                # some other way -- the splits cannot sum to less than the
                # wins, so the row is arithmetically impossible and those
                # zeros are placeholders. This is a consistency check, not a
                # heuristic.
                #
                # It matters because the zeros cost real rating points:
                # compute_stats_rating reads a 0 finish rate as 150 * (0 -
                # 0.4) = -60, so an unlooked-up fighter is docked for
                # finishing nobody. Chatman is 4 KO and 1 decision from 5
                # wins -- an 80% finish rate scored as 0%.
                win_splits_impossible = False
                if not existing_row.empty and pd.notna(existing_row["wins"].iloc[0]):
                    _w = float(existing_row["wins"].iloc[0])
                    _splits = [existing_row[c].iloc[0] for c in ("ko_wins", "sub_wins", "dec_wins")]
                    if _w > 0 and all(pd.notna(s) for s in _splits) and sum(float(s) for s in _splits) == 0:
                        win_splits_impossible = True
                needs_method_data = not already_exhausted and (
                    name in needs_basic or record_empty or win_splits_impossible or (
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
                METHOD_FIELDS = ["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]

                def _is_complete(bd):
                    return bool(bd) and all(bd.get(f) is not None for f in METHOD_FIELDS)

                if needs_method_data:
                    method_breakdown_attempts_this_run += 1
                    breakdown = None
                    if combat_edge_rate_limited and wikipedia_rate_limited:
                        print(f"[fighter_backfill] {name}: both method-breakdown sources already rate-limited "
                              f"this run -- skipped, will retry next run")
                    if not combat_edge_rate_limited and not already_checked["combat_edge"]:
                        known_wins = int(existing_row["wins"].iloc[0]) if not existing_row.empty and pd.notna(existing_row["wins"].iloc[0]) else None
                        breakdown = _fetch_method_breakdown_from_combat_edge(name, known_wins=known_wins)
                        if breakdown == RATE_LIMITED:
                            combat_edge_rate_limited = True
                            breakdown = None
                        else:
                            physical["combat_edge_checked"] = True  # genuine attempt happened, regardless of outcome
                    # Was "if not breakdown" -- a real bug once Combat Edge could return
                    # a PARTIAL dict (some fields real, some None): a non-empty dict is
                    # truthy in Python regardless of what its values are, so any partial
                    # Combat Edge success was permanently blocking Wikipedia from ever
                    # being tried for the still-missing fields (the loss side, in
                    # practice, since the confirmed win-sentence pattern only covers
                    # wins). Now checks genuine completeness instead, and merges both
                    # sources' fields rather than letting one replace the other -- keeps
                    # Combat Edge's real win values even if Wikipedia's own dict has
                    # None for those same fields.
                    if not _is_complete(breakdown) and not wikipedia_rate_limited and not already_checked["wikipedia"]:
                        wiki_breakdown = _fetch_method_breakdown_from_wikipedia(name)
                        if wiki_breakdown == RATE_LIMITED:
                            wikipedia_rate_limited = True
                        else:
                            physical["wikipedia_checked"] = True  # genuine attempt happened, regardless of outcome
                            if wiki_breakdown:
                                merged = dict(breakdown) if breakdown else {}
                                for f in METHOD_FIELDS:
                                    if merged.get(f) is None and wiki_breakdown.get(f) is not None:
                                        merged[f] = wiki_breakdown[f]
                                breakdown = merged
                    if breakdown:
                        # RECORD FALLBACK, and the only place a non-ESPN
                        # source is allowed to set wins/losses.
                        #
                        # record_wins/record_losses are NOT roster columns --
                        # they are Combat Edge's reading of the career record
                        # off the profile header. Popped before the update so
                        # they can never leak into fighters.csv as columns of
                        # their own.
                        #
                        # ESPN stays authoritative whenever it answered. This
                        # only fires when ESPN gave us nothing at all, which
                        # for a regional debutant means no athlete page and no
                        # scoreboard entry. Terrance Chatman went into UFC
                        # Fight Night 2026-08-22 recorded as 0-0 -- treated by
                        # the rating as "lost every fight" -- when Combat Edge
                        # had him at 5-1 the whole time.
                        record_w = breakdown.pop("record_wins", None)
                        record_l = breakdown.pop("record_losses", None)
                        if record_w is not None and record_l is not None:
                            espn_gave_nothing = (
                                wins is None or losses is None
                                or (int(wins) == 0 and int(losses) == 0)
                            )
                            if espn_gave_nothing:
                                print(f"[fighter_backfill] {name}: no career record from ESPN "
                                      f"-- using combat-edge {record_w}-{record_l}")
                                wins, losses = record_w, record_l
                                # Written explicitly rather than via `physical`,
                                # because the gap-fill path below only touches
                                # columns that are NaN and an existing 0-0 row
                                # is not NaN -- which is exactly the row this
                                # needs to correct.
                                physical["wins"] = record_w
                                physical["losses"] = record_l
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
                    # EVALUATED ONCE, BEFORE THE LOOP WRITES ANYTHING. The
                    # first version tested this inside the per-column loop,
                    # which re-read the row after its own write: "wins" was
                    # set to 5, then "losses" re-checked, saw 5-0, concluded
                    # the record was no longer absent, and skipped -- storing
                    # 5-0 for a 5-1 fighter. A guard must not depend on state
                    # its own action changes.
                    _rec_w, _rec_l = fighters.at[i, "wins"], fighters.at[i, "losses"]
                    record_was_absent = (
                        pd.isna(_rec_w) or pd.isna(_rec_l)
                        or (float(_rec_w) == 0 and float(_rec_l) == 0)
                    )
                    if pd.isna(fighters.at[i, "country"]) and country:
                        fighters = _safe_set_cell(fighters, i, "country", country)
                        updated_fields.append("country")
                    for col, val in physical.items():
                        if col in ("combat_edge_checked", "wikipedia_checked"):
                            if not bool(fighters.at[i, col]):  # only a genuine False->True change is worth writing
                                fighters = _safe_set_cell(fighters, i, col, val)
                                any_checked_flag_changed = True
                        # WINS/LOSSES CAN OVERWRITE AN EXISTING 0-0, unlike
                        # every other column here, which is strictly
                        # fill-if-empty. 0-0 is not empty -- it is a WRONG
                        # value that compute_stats_rating reads as "lost every
                        # fight", and it is exactly the row a record fallback
                        # exists to correct. Terrance Chatman sat at 0-0
                        # against a true 5-1 and no fill-if-empty rule would
                        # ever have touched him.
                        #
                        # Strictly bounded: only an all-zero record is
                        # replaceable. A real record, even 1-0, is never
                        # overwritten by this path.
                        elif col in ("wins", "losses") and val is not None:
                            if record_was_absent:
                                fighters = _safe_set_cell(fighters, i, col, val)
                                updated_fields.append(col)
                        # THE WRITE SIDE OF THE SAME TRAP. Reaching here means
                        # the row claimed wins with every win-split at zero --
                        # arithmetically impossible, so those zeros are
                        # placeholders. The fill-if-NaN rule below cannot see
                        # them, so a freshly fetched breakdown would be
                        # discarded on arrival and the fighter would keep a 0%
                        # finish rate worth -60 rating points.
                        # Bounded by win_splits_impossible, evaluated before
                        # this loop ran, so a legitimate zero on a fighter
                        # whose other splits are non-zero is never touched.
                        elif (col in ("ko_wins", "sub_wins", "dec_wins")
                              and val is not None and win_splits_impossible):
                            fighters = _safe_set_cell(fighters, i, col, val)
                            updated_fields.append(col)
                        # Was missing "and val is not None" -- a merged partial
                        # breakdown (e.g. wins found, losses not) still has
                        # None entries for the still-missing fields. Without
                        # this check, writing None over an already-empty cell
                        # is a genuine no-op (pandas treats it as NaN either
                        # way) but was being reported as "filled" -- a real,
                        # confusing log inaccuracy caught from a live run
                        # where every "filled ko_losses/sub_losses/dec_losses"
                        # message was actually reporting nothing happening.
                        elif pd.isna(fighters.at[i, col]) and val is not None:
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


def fill_missing_last_fights(fighters_path: str = "data/fighters.csv",
                             card_paths: tuple[str, ...] = ("data/fight_cards.csv", "data/future_cards.csv")) -> int:
    """
    Fill last-fight data for tracked fighters the main backfill missed.

    WHY THIS IS SEPARATE. backfill_fighters() only ever reaches a fighter who
    appears as a competitor inside a scoreboard event it fetched BY NAME, and
    it returns early when it judges there's nothing to do. Fighters on a
    future card kept ending up with null last-fight fields even though every
    individual step worked when called by hand -- verified with
    scripts/diagnose_last_fight.py, which resolved the id and pulled a real
    bout for a fighter the pipeline left empty.

    This resolves ids from the scoreboard BY EVENT DATE, which is the route
    that diagnostic proved, and reports what it found EVERY time. The
    previous version printed only on success, so "skipped" and "worked" were
    indistinguishable -- the same silent-skip failure this whole area keeps
    producing.
    """
    try:
        fighters = pd.read_csv(fighters_path)
    except FileNotFoundError:
        return 0
    if "last_fight_date" not in fighters.columns:
        return 0

    card_names, dates = set(), set()
    for path in card_paths:
        try:
            d = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if {"fighter_a", "fighter_b"} <= set(d.columns):
            card_names |= set(d["fighter_a"].dropna()) | set(d["fighter_b"].dropna())
        if "event_date" in d.columns:
            dates |= {str(x)[:10] for x in d["event_date"].dropna()}

    need = [n for n in fighters[fighters["last_fight_date"].isna()]["name"] if n in card_names]
    if not need:
        print("[last_fight] every tracked fighter already has a last fight on file.")
        return 0

    id_map = {}
    for d in sorted(dates):
        try:
            r = requests.get(ESPN_SCOREBOARD_URL, params={"dates": d.replace("-", "")},
                             headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            for ev in r.json().get("events", []):
                for comp in ev.get("competitions", []):
                    for c in comp.get("competitors", []):
                        ath = c.get("athlete") or {}
                        aid = ath.get("id") or c.get("id")
                        if ath.get("fullName") and aid:
                            id_map[_normalize_name(ath["fullName"])] = str(aid)
        except requests.RequestException:
            continue

    filled, no_id, no_bouts = 0, [], []
    for nm in need:
        aid = id_map.get(_normalize_name(nm))
        if not aid:
            no_id.append(nm)
            continue
        got = _fetch_last_fight_from_events_map(aid, nm)
        if not got:
            no_bouts.append(nm)
            continue
        idx = fighters.index[fighters["name"] == nm]
        if len(idx) == 0:
            continue
        i = idx[0]
        for col, val in got.items():
            if col in fighters.columns and pd.isna(fighters.at[i, col]) and val is not None:
                fighters = _safe_set_cell(fighters, i, col, val)
                filled += 1

    print(f"[last_fight] {len(need)} fighter(s) missing a last fight | "
          f"{len(id_map)} espn id(s) resolved from {len(dates)} card date(s) | "
          f"filled {filled} field(s)")
    if no_id:
        print(f"[last_fight] no ESPN id for: {sorted(no_id)[:8]}")
    if no_bouts:
        print(f"[last_fight] id found but no completed bouts for: {sorted(no_bouts)[:8]}")
    if filled:
        fighters.to_csv(fighters_path, index=False)
    return filled


def _fuzzy_espn_match(target_norm: str, id_map: dict) -> str | None:
    """
    Last-resort name match, for spelling variants accent folding can't reach.

    ESPN lists "Manoel Sousa" where the card says "Manuel Sosa" -- two real
    spelling differences, not diacritics. Exact and folded matching both miss,
    and the fighter ends up with no roster row and no model preview at all.

    Guarded two ways so this can never quietly match the WRONG fighter, which
    would be far worse than no match: the best candidate must clear 0.80
    similarity AND beat the runner-up by 0.15. On the real case that was 0.870
    against 0.435 -- a margin of 0.435, nowhere near ambiguous. A card with two
    genuinely similar names produces a small margin and is correctly rejected.
    """
    from difflib import SequenceMatcher
    scored = sorted(((SequenceMatcher(None, target_norm, k).ratio(), k)
                     for k in id_map), reverse=True)
    if not scored:
        return None
    best_score, best_key = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.80 and (best_score - runner_up) >= 0.15:
        print(f"[roster]   fuzzy-matched {target_norm!r} -> {best_key!r} "
              f"(similarity {best_score:.2f}, next best {runner_up:.2f})")
        return id_map[best_key]
    if best_score >= 0.80:
        print(f"[roster]   {target_norm!r} looks like {best_key!r} ({best_score:.2f}) "
              f"but {scored[1][1]!r} is close behind ({runner_up:.2f}) -- too "
              f"ambiguous to match automatically")
    return None


# ESPN tags every bout in an athlete's eventsMap with a league. The UFC (and
# Dana White's Contender Series, which it runs and whose graduates enter the
# same rating pool) is league 3321, slug "ufc"; everything else -- Levels
# Fight League, FightStar, Contenders, KSW -- is another id with no slug.
# Verified on Mario Pinto: 4 bouts at 3321/ufc, 8 at 3359 including three on
# 2023-03-11, a one-night regional tournament that is structurally impossible
# in the UFC.
_UFC_LEAGUE_IDS = {"3321"}
_LEAGUE_IN_UID = re.compile(r"~l:(\d+)")
_LEAGUE_IN_HREF = re.compile(r"/league/([a-z0-9_-]+)")


def _promotion_of(rec: dict) -> str:
    """"" for a UFC bout, else a label that ufc_only() will exclude.

    BLANK MEANS UFC because every pre-existing row in fight_history.csv is
    blank and src/elo.ufc_only reads it that way; inverting the default would
    drop the entire spine out of the rating graph. So the burden is on this
    function to LABEL the non-UFC ones, and a bout it cannot classify stays
    blank -- the same failure mode we already had, not a worse one.
    """
    name = str(rec.get("name") or "").strip()
    # THE ULTIMATE FIGHTER IS TWO DIFFERENT THINGS under one title, and ESPN
    # files both outside league 3321. A Tournament Finale is a real UFC event
    # on a UFC card and belongs in the graph; the house bouts -- "Semifinal",
    # "Quarterfinal", "Elimination" -- are exhibitions that do not appear on
    # official records at all, so excluding them is not merely acceptable, it
    # is more correct than the blanket UFC treatment they had before.
    if "ultimate fighter" in name.lower():
        return "" if "finale" in name.lower() else name

    for link in (rec.get("links") or []):
        m = _LEAGUE_IN_HREF.search(str(link.get("href") or ""))
        if m:
            return "" if m.group(1) == "ufc" else (name or m.group(1))
    m = _LEAGUE_IN_UID.search(str(rec.get("uid") or ""))
    if m:
        return "" if m.group(1) in _UFC_LEAGUE_IDS else (name or f"league {m.group(1)}")
    return ""


def fetch_espn_fight_history(athlete_id: str, fighter_name: str) -> list[dict]:
    """
    EVERY completed bout for a fighter, from the site athlete endpoint.

    _fetch_last_fight_from_events_map reads the same payload and keeps only
    the most recent bout. The rest is thrown away -- and it's exactly what
    fight_history.csv is missing.

    WHY THAT MATTERS. fight_history.csv comes from a raw UFC dataset that lags
    by weeks, and merge_results_into_history.py can only add fights the SITE
    itself watched. Anything between those two windows is invisible: Quillan
    Salkilld's May 2026 KO of Beneil Dariush is in neither, so the model saw
    a 4-fight streak against a real 5 -- below the threshold for a fight fact,
    and one streak-bonus step short in the rating.

    Returns rows shaped for fight_history.csv. Scheduled bouts are excluded by
    requiring a W/L, the same test used for last-fight -- ESPN marks some
    upcoming fights as played, so a flag alone can't be trusted.
    """
    out = []
    try:
        resp = requests.get(SITE_ATHLETE_URL.format(athlete_id),
                            headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return out
        events_map = resp.json().get("eventsMap") or {}
    except Exception as exc:
        print(f"[history] eventsMap fetch failed for {fighter_name}: {exc}")
        return out

    today = dt.date.today().isoformat()
    for rec in events_map.values():
        if not isinstance(rec, dict):
            continue
        date = str(rec.get("gameDate") or "")[:10]
        if not date or date > today:
            continue
        result = rec.get("gameResult")
        if isinstance(result, dict):
            result = result.get("displayName") or result.get("abbreviation")
        result = str(result or "").strip().upper()
        # A draw or no contest is a fight that happened, and dropping it is
        # why the spine held zero winnerless rows across 11,861 fights. It is
        # kept with an empty winner. The exact token ESPN uses here has not
        # been observed on a real no contest -- the accepted set is a guess,
        # and an unrecognised one is printed rather than swallowed so the real
        # spelling can be added the first time one comes through.
        no_contest = result in ("D", "DRAW", "NC", "NO CONTEST", "TIE")
        if result not in ("W", "L", "WIN", "LOSS") and not no_contest:
            if result:
                print(f"[fighter_backfill] unrecognised gameResult {result!r} "
                      f"for {fighter_name} on {date} -- row skipped")
            continue
        opp = rec.get("opponent")
        if isinstance(opp, dict):
            opp = opp.get("displayName") or opp.get("shortDisplayName") or opp.get("name")
        if not opp:
            continue
        method = ""
        status = rec.get("status")
        if isinstance(status, dict):
            stype = status.get("type") if isinstance(status.get("type"), dict) else status
            m = stype.get("detail") or stype.get("description")
            if isinstance(m, str) and m.strip().lower() not in STATUS_WORDS:
                method = m.strip()
        won = result.startswith("W")
        if no_contest:
            out.append({
                "date": date, "fighter_a": fighter_name,
                "fighter_b": str(opp).strip(), "winner": "",
                "method": method or ("Draw" if result in ("D", "DRAW", "TIE") else "NC"),
                "promotion": _promotion_of(rec),
            })
            continue
        out.append({
            "date": date,
            "fighter_a": fighter_name if won else str(opp).strip(),
            "fighter_b": str(opp).strip() if won else fighter_name,
            "winner": fighter_name if won else str(opp).strip(),
            "method": method,
            # THE ROW THAT MADE ufc_only() A NO-OP. This path writes most of
            # the spine and emitted no promotion key at all, so a fighter's
            # KSW and regional-tournament bouts landed blank and read as UFC.
            # 17 of 11,925 rows carried a promotion; every published streak was
            # a career streak wearing a UFC label.
            "promotion": _promotion_of(rec),
        })
    return out


def ensure_roster_rows(fighters_path: str = "data/fighters.csv",
                       card_paths: tuple[str, ...] = ("data/fight_cards.csv", "data/future_cards.csv")) -> int:
    """
    Create fighters.csv rows for card fighters the main backfill never matched.

    WHY THIS IS NEEDED. backfill_fighters() finds fighters by matching names
    against the competitors of a scoreboard event it looks up BY EVENT NAME.
    When that lookup misses, the fighter is logged as unmatched and dropped --
    no roster row is created, so build_fight_preview() has no stats and the
    whole fight renders with no model preview, no tale of the tape, nothing.
    Four fighters on one card hit this.

    ESPN knows them perfectly well: resolving athlete ids from the scoreboard
    BY EVENT DATE finds them immediately (verified with
    scripts/diagnose_last_fight.py -- Miles Johns resolved to 4010864 with a
    full fight history). That is the same route fill_missing_last_fights()
    uses, and the same one that fixed the last-fight gap.

    Creates a row from the athlete endpoint's physicals, method splits and
    career record. Only fills what ESPN actually returns -- a fighter with
    partial data gets a partial row, which the preview can still use, rather
    than a fabricated complete one.
    """
    try:
        fighters = pd.read_csv(fighters_path)
    except FileNotFoundError:
        return 0

    roster = set(fighters["name"].astype(str))
    roster_norm = {_normalize_name(n) for n in roster}

    wanted, dates = {}, set()
    for path in card_paths:
        try:
            d = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if "event_date" in d.columns:
            dates |= {str(x)[:10] for x in d["event_date"].dropna()}
        for col in ("fighter_a", "fighter_b"):
            if col not in d.columns:
                continue
            for n in d[col].dropna().astype(str):
                if n in roster or _normalize_name(n) in roster_norm:
                    continue
                if is_placeholder_fighter_name(n):
                    continue
                wanted[_normalize_name(n)] = n

    if not wanted:
        print("[roster] every card fighter already has a row.")
        return 0

    print(f"[roster] {len(wanted)} card fighter(s) missing from fighters.csv: "
          f"{sorted(wanted.values())[:8]}")

    id_map = {}
    for d in sorted(dates):
        try:
            r = requests.get(ESPN_SCOREBOARD_URL, params={"dates": d.replace("-", "")},
                             headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            for ev in r.json().get("events", []):
                for comp in ev.get("competitions", []):
                    for c in comp.get("competitors", []):
                        ath = c.get("athlete") or {}
                        aid = ath.get("id") or c.get("id")
                        if ath.get("fullName") and aid:
                            id_map[_normalize_name(ath["fullName"])] = str(aid)
        except requests.RequestException:
            continue

    new_rows, unresolved = [], []
    for norm, name in wanted.items():
        aid = id_map.get(norm) or _fuzzy_espn_match(norm, id_map)
        if not aid:
            unresolved.append(name)
            continue
        physical, _ = _fetch_espn_athlete_detail(aid)
        recs = _fetch_espn_method_records(aid)
        last = _fetch_last_fight_from_events_map(aid, name)

        row = {"name": name}
        row.update(physical or {})
        cw, cl = recs.pop("_career_w", None), recs.pop("_career_l", None)
        if cw is not None:
            row["wins"], row["losses"] = cw, cl
        row.update(recs)
        row.update(last or {})
        new_rows.append(row)
        print(f"[roster]   created {name} (espn id {aid}, "
              f"{row.get('wins', '?')}-{row.get('losses', '?')})")

    if unresolved:
        print(f"[roster] no ESPN id for: {sorted(unresolved)}")
    if not new_rows:
        return 0

    fighters = pd.concat([fighters, pd.DataFrame(new_rows)], ignore_index=True)
    fighters.to_csv(fighters_path, index=False)
    print(f"[roster] added {len(new_rows)} row(s) to {fighters_path}")
    return len(new_rows)


def fill_last_fight_methods(fighters_path: str = "data/fighters.csv",
                            history_path: str = "data/fight_history.csv") -> int:
    """
    Fill last_fight_method from fight_history.csv.

    WHY LOCAL DATA IS THE PRIMARY SOURCE HERE. ESPN's eventsMap carries the
    date, opponent and result but reports status as "Final" -- a completion
    state, not a method. That string is correctly filtered out (it once
    produced "W by Final against X"), which left the field empty and rendered
    as "L by None".

    fight_history.csv already knows the method for 8,400+ fights, costs no
    network call, and can't be rate-limited. It should have been consulted
    first; ESPN is the fallback for anything history hasn't caught up on.

    Matches on an unordered accent-folded name pair plus date -- the same key
    used by every other join here, because the two sources disagree about
    which fighter is listed first.
    """
    try:
        fighters = pd.read_csv(fighters_path)
        history = pd.read_csv(history_path)
    except FileNotFoundError:
        return 0
    if "last_fight_method" not in fighters.columns:
        fighters["last_fight_method"] = None

    lookup = {}
    for r in history.itertuples(index=False):
        method = str(getattr(r, "method", "") or "").strip()
        if not method or method.lower() in STATUS_WORDS:
            continue
        key = (frozenset({_normalize_name(r.fighter_a), _normalize_name(r.fighter_b)}),
               str(r.date)[:10])
        lookup[key] = method

    # Also index by fighter + date, so a HALF-RECORD can be completed.
    # Some rows arrive with a last_fight_date and nothing else -- no result,
    # no opponent -- which my first version silently skipped because it
    # couldn't build a name-pair key. Oliveira, Strickland and Umar
    # Nurmagomedov were all in that state: a date on screen with no fight
    # attached to it.
    by_fighter_date = {}
    for r in history.itertuples(index=False):
        d = str(r.date)[:10]
        winner = _normalize_name(str(getattr(r, "winner", "") or ""))
        for own, other in ((r.fighter_a, r.fighter_b), (r.fighter_b, r.fighter_a)):
            own_n = _normalize_name(str(own))
            by_fighter_date[(own_n, d)] = {
                "opponent": str(other),
                "result": "W" if winner == own_n else ("L" if winner else None),
                "method": str(getattr(r, "method", "") or "").strip() or None,
            }

    filled, completed, still_missing = 0, 0, 0
    for i, row in fighters.iterrows():
        date = row.get("last_fight_date")
        if not isinstance(date, str) or not date.strip():
            continue
        name_n = _normalize_name(str(row["name"]))
        opp = row.get("last_fight_opponent")
        have_method = isinstance(row.get("last_fight_method"), str) and row["last_fight_method"].strip()
        have_opp = isinstance(opp, str) and opp.strip()
        have_result = isinstance(row.get("last_fight_result"), str) and row["last_fight_result"].strip()
        if have_method and have_opp and have_result:
            continue

        rec = by_fighter_date.get((name_n, str(date)[:10]))
        if rec:
            wrote = False
            if not have_opp and rec["opponent"]:
                fighters.at[i, "last_fight_opponent"] = rec["opponent"]; wrote = True
            if not have_result and rec["result"]:
                fighters.at[i, "last_fight_result"] = rec["result"]; wrote = True
            if not have_method and rec["method"] and rec["method"].lower() not in STATUS_WORDS:
                fighters.at[i, "last_fight_method"] = rec["method"]; filled += 1; wrote = True
            if wrote:
                completed += 1
            continue

        # Fall back to the name-pair key when the date doesn't line up but the
        # opponent is known -- the two sources sometimes differ by a day.
        if have_opp and not have_method:
            method = lookup.get((frozenset({name_n, _normalize_name(opp)}), str(date)[:10]))
            if method:
                fighters.at[i, "last_fight_method"] = method
                filled += 1
                continue
        still_missing += 1

    if filled or completed:
        fighters.to_csv(fighters_path, index=False)
    print(f"[methods] filled {filled} method(s), completed {completed} partial "
          f"last-fight record(s) from history; {still_missing} still unknown")
    return filled


ESPN_ID_MAP_PATH = "data/espn_athlete_ids.csv"


def _espn_id_map(path: str = ESPN_ID_MAP_PATH) -> dict:
    """Folded fighter name -> ESPN athlete id, or {} if unreadable."""
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
        return {}
    if "name" not in df.columns or "espn_id" not in df.columns:
        return {}
    return {_normalize_name(str(r["name"])): str(r["espn_id"]).strip()
            for _, r in df.iterrows() if str(r.get("espn_id") or "").strip()}


def _record_is_unknown(row) -> bool:
    """
    True when this roster row carries NO usable career record.

    0-0 COUNTS AS UNKNOWN, and that is the whole point. A fighter with a
    genuine 0-0 professional record cannot be booked on a UFC card, so the
    pair only ever means "nobody filled this in" -- but it is stored as two
    real integers, so every consumer reads it as a fact. thinner_record goes
    to 0 and caps the fight's confidence; `max(int(wins), 1)` turns into a
    denominator of 1 and every method rate blows up.

    This is the third appearance of the same failure. Terrance Chatman went
    into 2026-08-22 recorded 0-0 when he was 5-1, which crowned his opponent
    Lock of the Week. Pavel Andrusca went onto the 2026-09-05 card recorded
    0-0 when ESPN had him 8-0 with the id in our own map.
    """
    for col in ("wins", "losses"):
        v = row.get(col)
        if v is None or (isinstance(v, float) and v != v):
            return True
    return int(row.get("wins") or 0) == 0 and int(row.get("losses") or 0) == 0


def fill_from_espn_id_map(fighters_path: str = "data/fighters.csv",
                          card_paths: tuple[str, ...] = ("data/fight_cards.csv", "data/future_cards.csv"),
                          id_map_path: str = ESPN_ID_MAP_PATH) -> int:
    """
    Last resort for a booked fighter the scoreboard pass never reached.

    WHY THIS IS SEPARATE, same reason fill_missing_last_fights is:
    backfill_fighters only sees a fighter who turns up as a competitor inside
    a scoreboard event it matched BY NAME, and its `unmatched` branch does
    nothing but print. A LATE REPLACEMENT is exactly the fighter that branch
    reports and exactly the fighter nobody looks at -- he is added after the
    card stopped being a future card, so ESPN's scoreboard entry for that
    event may not carry him at all.

    data/espn_athlete_ids.csv already holds 2,700+ name-to-id pairs and
    src/ never once read it -- only scripts/ did. So the id needed to fix
    Andrusca was sitting in the repo while the card published him as 0-0.

    ONLY FILLS WHAT IS MISSING, with one deliberate exception: a 0-0 record
    is treated as missing (see _record_is_unknown) and IS overwritten. Every
    other non-null cell is left exactly as it is. Never raises.
    """
    try:
        fighters = pd.read_csv(fighters_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
        return 0
    booked = set()
    for p in card_paths:
        try:
            c = pd.read_csv(p)
        except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
            continue
        for col in ("fighter_a", "fighter_b"):
            if col in c.columns:
                booked |= {str(n) for n in c[col].dropna() if not is_placeholder_fighter_name(n)}
    if not booked:
        return 0

    ids = _espn_id_map(id_map_path)
    if not ids:
        return 0

    filled = 0
    for name in sorted(booked):
        match = fighters[fighters["name"].astype(str) == name]
        if match.empty:
            continue
        i = match.index[0]
        row = fighters.loc[i]
        if not _record_is_unknown(row):
            continue
        aid = ids.get(_normalize_name(name))
        if not aid:
            print(f"[fighter_backfill] {name!r} is booked with no usable record and no ESPN id in "
                  f"{id_map_path} -- cannot fill from this source")
            continue
        try:
            physical, _ = _fetch_espn_athlete_detail(aid)
            methods = _fetch_espn_method_records(aid)
        except Exception as e:
            print(f"[fighter_backfill] ESPN id-map lookup failed for {name!r} (id {aid}): {e}")
            continue

        w, l = methods.pop("_career_w", None), methods.pop("_career_l", None)
        vals = {}
        if w is not None and l is not None and (w or l):
            vals["wins"], vals["losses"] = int(w), int(l)
        for k, v in (methods or {}).items():
            if v is not None:
                vals[k] = v
        # Physical only where we hold nothing; the record above is the one
        # thing allowed to correct an existing value.
        for k, v in (physical or {}).items():
            cur = row.get(k)
            if v is not None and (cur is None or (isinstance(cur, float) and cur != cur)):
                vals[k] = v
        if not vals:
            continue
        for col, val in vals.items():
            if col not in fighters.columns:
                fighters[col] = pd.NA
            fighters = _safe_set_cell(fighters, i, col, val)
        filled += 1
        print(f"[fighter_backfill] filled {name!r} from ESPN id {aid} "
              f"(scoreboard never matched him): {', '.join(sorted(vals))}")

    if filled:
        fighters.to_csv(fighters_path, index=False)
    return filled
