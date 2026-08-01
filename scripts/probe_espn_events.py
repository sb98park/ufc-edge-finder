"""
Probe: can we read a fighter's FIGHT HISTORY off the site athlete endpoint?

WHY. Last-fight data currently comes from the CORE api's eventLog, and that
path has two problems: it marks some SCHEDULED bouts as played=true (which
produced last fights dated a week in the future for 45 fighters), and once
those are correctly rejected, many fighters end up with nothing at all.

But espn.com plainly renders a full fight history on every athlete page, so
the data exists. The earlier method-records probe showed the SITE athlete
endpoint returns top-level keys including 'events' and 'eventsMap' -- which
is almost certainly what backs that table, and which we have never read.

This dumps their structure so the parser can be written against the real
shape. Guessing shapes has cost several rounds this session; one probe is
cheaper.

Usage:
    python3 scripts/probe_espn_events.py 5351430
"""

import json
import sys

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SITE_ATHLETE = "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/athletes/{}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    aid = sys.argv[1]
    r = requests.get(SITE_ATHLETE.format(aid), headers=HEADERS, timeout=20)
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        return
    data = r.json()

    for key in ("events", "eventsMap"):
        blob = data.get(key)
        if blob is None:
            print(f"\n{key}: absent")
            continue
        print(f"\n{'='*66}\n{key}: {type(blob).__name__}")
        if isinstance(blob, dict):
            print("  top-level keys:", list(blob.keys())[:10])
            # Dig one level for the first list of fight-like records.
            for k, v in list(blob.items())[:3]:
                if isinstance(v, list) and v:
                    print(f"\n  {key}['{k}'] -- {len(v)} entries, first entry:")
                    print("   ", json.dumps(v[0])[:900])
                    break
                if isinstance(v, dict):
                    print(f"\n  {key}['{k}'] keys:", list(v.keys())[:14])
                    print("   ", json.dumps(v)[:700])
                    break
        elif isinstance(blob, list) and blob:
            print(f"  {len(blob)} entries, first entry:")
            print("   ", json.dumps(blob[0])[:900])

    print(f"\n{'='*66}")
    print("WHAT I'M AFTER: a per-fight record carrying an opponent name, a")
    print("date, a W/L result and a method. If those are present, last-fight")
    print("comes from here instead of the eventLog and the future-dated")
    print("problem disappears at the source.")


if __name__ == "__main__":
    main()
