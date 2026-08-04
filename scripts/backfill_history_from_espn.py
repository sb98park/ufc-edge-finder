"""
Fill gaps in fight_history.csv from ESPN, for fighters on tracked cards.

THE GAP THIS CLOSES. fight_history.csv has two sources and neither covers
recent months:

  etl_fight_history.py    a raw UFC dataset that lags by weeks
  merge_results_into_history.py   only fights the SITE itself watched

Anything between those windows is invisible. Quillan Salkilld's May 2026 KO
of Beneil Dariush -- a top-ten opponent -- is in neither, so the model read a
4-fight streak against a real 5. That is below the fight-fact threshold and
one step short in the streak bonus, on the exact fighter whose profile
prompted building that bonus.

ESPN's athlete endpoint has each fighter's full history, and the site already
fetches that payload for last-fight data. This reads the rest of it.

SAFE BY CONSTRUCTION:
  - append only
  - skips fights already present, matched on an unordered accent-folded name
    pair plus date, so fighter order can't create a duplicate
  - skips future-dated and unresolved bouts
  - only fetches fighters on tracked cards, not the whole roster

Usage:
    python3 scripts/backfill_history_from_espn.py            # dry run
    python3 scripts/backfill_history_from_espn.py --apply
"""

import os
import sys
import time
import unicodedata

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fighter_backfill import (fetch_espn_fight_history, BASE_HEADERS,  # noqa: E402
                                  REQUEST_TIMEOUT)
from src.results_fetcher import ESPN_SCOREBOARD_URL  # noqa: E402

HISTORY = "data/fight_history.csv"


def fold(n):
    s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def key(a, b, d):
    return (frozenset({fold(a), fold(b)}), str(d)[:10])


def main():
    apply = "--apply" in sys.argv
    hist = pd.read_csv(HISTORY)
    have = {key(r["fighter_a"], r["fighter_b"], r.get("date")) for _, r in hist.iterrows()}

    print("resolving athlete ids by card date:")
    blocked = 0
    names, dates = set(), set()
    for f in ("data/fight_cards.csv", "data/future_cards.csv"):
        try:
            d = pd.read_csv(f)
        except FileNotFoundError:
            continue
        names |= set(d["fighter_a"].dropna()) | set(d["fighter_b"].dropna())
        if "event_date" in d.columns:
            dates |= {str(x)[:10] for x in d["event_date"].dropna()}

    # Athlete ids by event DATE -- the route that has worked every time the
    # event-name lookup has not.
    # Reports per date. A silent "0 ids" is indistinguishable from a bad
    # date format, a non-200, or an empty payload -- and this exact approach
    # resolves ids elsewhere, so the difference has to be visible.
    id_map = {}
    for d in sorted(dates):
        param = str(d).replace("-", "")
        try:
            r = None
            # 403 means ESPN has temporarily blocked the IP, usually after a
            # burst of calls -- it is NOT a bad request, and retrying
            # immediately just extends the block. Back off, then give up
            # loudly rather than reporting "nothing to add", which reads
            # like the data was already complete.
            for attempt, wait in enumerate((0, 5, 20)):
                if wait:
                    print(f"   {d}: blocked, waiting {wait}s before retry {attempt}")
                    time.sleep(wait)
                r = requests.get(ESPN_SCOREBOARD_URL, params={"dates": param},
                                 headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
                if r.status_code != 403:
                    break
            if r is not None and r.status_code == 403:
                print(f"   {d} (as {param}): HTTP 403 -- ESPN is rate-limiting this IP.")
                blocked += 1
                continue
            if r.status_code != 200:
                print(f"   {d} (as {param}): HTTP {r.status_code}")
                continue
            payload = r.json()
            evs = payload.get("events", [])
            found = 0
            for ev in evs:
                for comp in ev.get("competitions", []):
                    for c in comp.get("competitors", []):
                        ath = c.get("athlete") or {}
                        aid = ath.get("id") or c.get("id")
                        if ath.get("fullName") and aid:
                            id_map[fold(ath["fullName"])] = str(aid)
                            found += 1
            print(f"   {d} (as {param}): {len(evs)} event(s), {found} competitor(s)")
        except requests.RequestException as e:
            print(f"   {d} (as {param}): {type(e).__name__}")
            continue

    print(f"\n{len(names)} card fighters | {len(id_map)} espn ids from {len(dates)} card date(s)")
    if blocked:
        print()
        print(f"  {blocked} of {len(dates)} date(s) returned 403. ESPN throttles after a")
        print("  burst of calls; this usually clears within an hour or two. Nothing")
        print("  was written -- re-run then. This is a BLOCK, not missing data.")
        print()
        return
    if not id_map and names:
        print("  (no ids -- check the per-date lines above for the reason)")
    print()

    new_rows, no_id = [], []
    for n in sorted(names):
        aid = id_map.get(fold(n))
        if not aid:
            no_id.append(n)
            continue
        for row in fetch_espn_fight_history(aid, n):
            k = key(row["fighter_a"], row["fighter_b"], row["date"])
            if k in have:
                continue
            have.add(k)
            new_rows.append(row)
        time.sleep(0.2)

    if no_id:
        print(f"no ESPN id for: {sorted(no_id)[:8]}\n")
    print(f"history rows : {len(hist)}")
    print(f"NEW to add   : {len(new_rows)}")
    for r in sorted(new_rows, key=lambda x: x["date"], reverse=True)[:15]:
        print(f"   {r['date']}  {r['fighter_a']} vs {r['fighter_b']}  -> {r['winner']}")
    if len(new_rows) > 15:
        print(f"   ... and {len(new_rows) - 15} more")

    if not new_rows:
        print("\nNothing to add.")
        return
    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return
    out = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True).sort_values("date")
    out.to_csv(HISTORY, index=False)
    print(f"\nWritten: {len(hist)} -> {len(out)} rows.")
    print("Re-run generate_site.py -- ratings, streaks and facts all read this.")


if __name__ == "__main__":
    main()
