"""
Find the endpoint behind ESPN's fighter STATS page.

    espn.com/mma/fighter/stats/_/id/3068125/mateusz-gamrot

That page lists per-fight striking and grappling detail (SSL/SSA, TDL/TDA,
target breakdown, clinch/ground/distance splits, knockdowns, advances). If it
is reachable as JSON it replaces ufcstats entirely -- and gives strictly more
than ufcstats ever did, because it is PER FIGHT rather than a career average,
which is what recency weighting needs.

WHY THIS SCRIPT EXISTS RATHER THAN A DIRECT PATCH. I probed
athlete.statsSummary.statistics, saw three win-loss entries, and concluded
"ESPN does not publish MMA rate stats". That generalised ONE endpoint to the
whole API -- the identical mistake already recorded in
_fetch_espn_method_records' docstring, where an earlier attempt read the
scoreboard's `records` array and wrongly declared the method breakdown absent
from ESPN. Twice now. So: probe every candidate, print what comes back, and
write the parser against whatever is actually there.

Candidates, in order of how useful they'd be:
  1. athlete gamelog  -- one call per fighter, every fight. Ideal.
  2. athlete stats    -- career aggregate; still enough for accuracy columns.
  3. CORE per-competitor statistics -- richest, but needs event + competition
     ids, so it costs a traversal per fight.

Usage:
    python3 scripts/probe_espn_fight_stats.py 3068125
"""

import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT  # noqa: E402

CANDIDATES = [
    ("gamelog", "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/athletes/{id}/gamelog"),
    ("stats", "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/athletes/{id}/stats"),
    ("splits", "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/athletes/{id}/splits"),
    ("core-statistics", "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/statistics"),
]


def show(label, url, athlete_id):
    u = url.format(id=athlete_id)
    print(f"\n--- {label} ---\n{u}")
    try:
        r = requests.get(u, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f"  request failed: {e}")
        return None
    print(f"  HTTP {r.status_code}")
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        print(f"  not JSON: {r.text[:200]}")
        return None
    print(f"  top-level keys: {list(data.keys())[:15]}")
    # Print enough structure to write a parser from, without dumping megabytes.
    blob = json.dumps(data)
    print(f"  payload size: {len(blob)} chars")
    for probe_key in ("names", "labels", "displayNames", "categories", "statistics", "events", "seasonTypes", "filters"):
        if probe_key in data:
            print(f"  {probe_key}: {json.dumps(data[probe_key])[:900]}")
    return data


def core_walk(athlete_id):
    """The CORE chain: athlete eventlog -> an event -> competition -> competitor statistics."""
    print("\n--- CORE walk: eventlog -> competition -> competitor statistics ---")
    ev_url = f"https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{athlete_id}/eventlog"
    try:
        r = requests.get(ev_url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        log = r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  eventlog failed: {e}")
        return
    items = (log.get("events") or {}).get("items") or []
    print(f"  eventlog entries: {len(items)}")
    if not items:
        return
    # Most recent first in ESPN's ordering; take one and follow it down.
    entry = items[0]
    print(f"  sample entry keys: {list(entry.keys())}")
    # The eventlog entry carries no statistics ref of its own -- it hangs off
    # the COMPETITOR one level down, which is the shape the public API docs
    # describe: .../events/{e}/competitions/{c}/competitors/{a}/statistics
    comp_ref = entry.get("competitor", {}).get("$ref") if isinstance(entry.get("competitor"), dict) else None
    if not comp_ref:
        print(f"  no competitor ref; full entry: {json.dumps(entry)[:500]}")
        return
    stat_ref = comp_ref.split("?")[0].rstrip("/") + "/statistics"
    print(f"  competitor ref: {comp_ref}")
    print(f"  derived statistics ref: {stat_ref}")
    try:
        sr = requests.get(stat_ref, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        sr.raise_for_status()
        stats = sr.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  statistics fetch failed: {e}")
        return
    cats = (stats.get("splits") or {}).get("categories") or stats.get("categories") or []
    print(f"  categories: {len(cats)}")
    for c in cats:
        names = [s.get("name") for s in (c.get("stats") or [])]
        print(f"    {c.get('name')}: {names[:25]}")
        for s in (c.get("stats") or [])[:6]:
            print(f"       {s.get('name')}={s.get('displayValue')}  ({s.get('description')})")


def main():
    ids = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not ids:
        print(__doc__)
        sys.exit(1)
    for athlete_id in ids:
        print(f"\n{'=' * 72}\nATHLETE {athlete_id}")
        for label, url in CANDIDATES:
            show(label, url, athlete_id)
        core_walk(athlete_id)
    print(f"\n{'=' * 72}\nPaste this back. Whichever endpoint returns real stat "
          f"names is what the backfill should call.")


if __name__ == "__main__":
    main()
