"""
Find how ESPN reports FIGHT LENGTH, so per-round rates can be computed.

WHY LENGTH IS THE PREREQUISITE. ESPN gives per-fight TOTALS -- 47 significant
strikes landed, 2 takedowns -- not rates. A total conflates two different
things: how fast a fighter works, and how long his fights last. A striker
whose last two bouts were a round-one knockout and a five-round decision has a
career average that describes neither. Projecting from that average would be
worst exactly where it matters, on finishers, whose totals are suppressed by
the very trait that makes them dangerous.

The fix needs duration:

    per-round rate  =  fight total / rounds actually fought

and then the projection recombines that with the round-survival grid the
method model already produces:

    E[strikes] = SUM over rounds  P(reaches round r) x per-round rate

That is a real structural advantage over reading career averages off a stats
page, and it is the whole reason this is worth building.

WHAT THIS PROBE SETTLES, and why it is not skipped. The formula depends
entirely on what `displayClock` MEANS. If it is time ELAPSED in the final
round, duration = (period - 1) * 5 + clock. If it is time REMAINING, it is
(period - 1) * 5 + (5 - clock). Those differ by minutes on every finish, and
getting it backwards would invert the correction for exactly the fighters the
correction exists for -- with no error anywhere, just quietly wrong numbers.
This project has already twice paid for a guess about an ESPN field that
looked obvious.

Cross-check it against a fight whose result you can look up: a known
"2:56 of round 2" finish should come back as ~7.9 minutes, not ~12.1.

Usage:
    python3 scripts/probe_fight_duration.py 3068125        # Gamrot
"""

import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT  # noqa: E402

EVENTLOG = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/eventlog"


def get(url):
    try:
        r = requests.get(url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f"    fetch failed: {e}")
        return None
    return r.json() if r.status_code == 200 else None


def main():
    ids = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not ids:
        print(__doc__)
        sys.exit(1)

    log = get(EVENTLOG.format(id=ids[0]))
    if not log:
        print("eventlog fetch failed")
        sys.exit(1)
    items = [e for e in (log.get("events") or {}).get("items") or [] if e.get("played")]
    print(f"{len(items)} played fights; inspecting the first 5\n")

    for entry in items[:5]:
        comp_ref = (entry.get("competitor") or {}).get("$ref")
        if not comp_ref:
            continue
        competition = comp_ref.split("/competitors/")[0].split("?")[0]

        # The event, for a date to anchor the fight in a record book.
        ev = get((entry.get("event") or {}).get("$ref", ""))
        date = (ev or {}).get("date", "")[:10]
        name = (ev or {}).get("name", "")

        print(f"--- {date}  {name}")
        status = get(competition + "/status")
        if not status:
            print("    no /status on this competition")
            continue
        # Print every scalar so nothing relevant is missed by looking only
        # where I expect it to be.
        for k, v in status.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                print(f"    {k}: {v!r}")
        t = status.get("type")
        if isinstance(t, dict):
            print(f"    type: {json.dumps({k: v for k, v in t.items() if not isinstance(v, (dict, list))})}")
        res = status.get("result")
        if isinstance(res, dict):
            print(f"    result: {json.dumps({k: v for k, v in res.items() if not isinstance(v, (dict, list))})[:300]}")

        period, clock = status.get("period"), status.get("displayClock")
        if period is not None and clock:
            try:
                mm, ss = str(clock).split(":")
                c = int(mm) + int(ss) / 60
                print(f"    -> if ELAPSED:   {(period - 1) * 5 + c:.2f} min")
                print(f"    -> if REMAINING: {(period - 1) * 5 + (5 - c):.2f} min")
            except ValueError:
                pass
        print()

    print("Compare against the real results. A finish reported as "
          "'2:56 of round 2' is 7.93 minutes; whichever line matches is the "
          "correct reading of displayClock.")


if __name__ == "__main__":
    main()
