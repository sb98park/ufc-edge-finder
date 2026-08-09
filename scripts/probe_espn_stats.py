"""
Dump EVERY entry ESPN publishes under athlete.statsSummary.statistics.

WHY. _fetch_espn_method_records already fetches this exact payload, but its
loop starts with:

    w, l = _parse_wl(entry.get("displayValue"))
    if w is None:
        continue

Anything that isn't a win-loss pair is dropped there -- and that `continue`
sits BEFORE the `unmatched` list, so single-value stats (striking accuracy,
takedown accuracy, takedown defence, strikes per minute) were never logged as
unrecognised either. They have been arriving and being silently discarded,
which is why strike_accuracy_pct / td_accuracy_pct / td_defense_pct sit at
roughly 29% roster coverage and 0/25 on a current card, for established
fighters as much as debutants.

This prints the raw entries so the parser can be written against ESPN's ACTUAL
key names instead of guessed ones. Guessing is what produced two wrong
conclusions about the ESPN 403 before a single experiment settled it.

Get an athlete id from the ESPN URL: espn.com/mma/fighter/_/id/<ID>/<name>

Usage:
    python3 scripts/probe_espn_stats.py 4887310
    python3 scripts/probe_espn_stats.py 4887310 2704326 3152929   # several at once
"""

import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT  # noqa: E402

SITE_ATHLETE_URL = "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/athletes/{}"


def probe(athlete_id: str):
    print(f"\n{'=' * 70}\nathlete id {athlete_id}")
    try:
        resp = requests.get(SITE_ATHLETE_URL.format(athlete_id),
                            headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f"  request failed: {e}")
        return
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}")
        return

    athlete = resp.json().get("athlete") or {}
    print(f"  name: {athlete.get('displayName')}")

    stats = (athlete.get("statsSummary") or {}).get("statistics") or []
    if not stats:
        print("  statsSummary.statistics is EMPTY for this athlete.")
    else:
        print(f"  statsSummary.statistics -- {len(stats)} entries:")
        for e in stats:
            print(f"    name={e.get('name')!r:<28} display={e.get('displayName')!r:<34} "
                  f"value={e.get('displayValue')!r}")

    # Other places ESPN sometimes puts per-fight rate stats. Printed shallowly
    # so a stat living somewhere else than statsSummary still turns up here
    # rather than needing a second round trip to discover.
    for key in ("statistics", "splits", "categories"):
        if key in athlete:
            blob = json.dumps(athlete[key])[:400]
            print(f"  athlete.{key} present -> {blob}...")


def main():
    ids = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not ids:
        print(__doc__)
        sys.exit(1)
    for athlete_id in ids:
        probe(athlete_id)
    print(f"\n{'=' * 70}")
    print("Paste this output back. The key names above are what the parser "
          "needs to match on -- substring matching, as _fetch_espn_method_records "
          "already does, so a renamed key degrades instead of breaking.")


if __name__ == "__main__":
    main()
