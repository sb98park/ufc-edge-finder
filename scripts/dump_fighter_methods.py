"""
Fight-by-fight method dump for one fighter, from the ESPN cache.

WHAT THIS IS FOR. diagnose_split_mismatch narrowed the problem sharply: for
every affected fighter the WINS reconcile exactly and only the LOSSES fall
short, by one to three. A population mismatch (pre-UFC fights missing) would
shorten both sides equally, so that is ruled out -- and ESPN's own fight count
matches the record, so the fights exist.

That leaves a loss whose METHOD does not map to KO/SUB/DEC. Disqualification,
technical decision, overturned results and doctor stoppages are all plausible
and all invisible in a three-bucket split.

This prints every fight ESPN holds with its raw result string, so the
unmapped ones are visible by name rather than inferred. Run it on a fighter
from the mismatch list and compare the loss count here against ko+sub+dec
losses in fighters.csv.

Usage:
    python3 scripts/dump_fighter_methods.py "Sumudaerji"
    python3 scripts/dump_fighter_methods.py "Kevin Borjas"
"""

import datetime as dt
import hashlib
import json
import os
import re
import sys
import unicodedata

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT

CACHE_DIR = "data/.espn_cache"
EVENTLOG = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/eventlog"

# What the split columns can represent. Anything outside this is a loss (or
# win) the three-bucket model has nowhere to put.
MAPPED = {"ko/tko", "ko", "tko", "submission", "sub",
          "decision", "decision - unanimous", "decision - split",
          "decision - majority", "u dec", "s dec", "m dec"}


def fold(v):
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def cached(url):
    p = os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fetch(url):
    """
    Cache first, then the network.

    The first run of this script reported "(no result recorded)" for thirteen
    of Sumudaerji's twenty-four fights, which reads as ESPN having no method
    for them. It was a CACHE MISS: the stats backfill only ever cached the
    fights it needed, so an unfetched fight and a methodless fight looked
    identical. A diagnostic that cannot tell "we did not look" from "there is
    nothing there" answers the wrong question -- the same failure as the card
    discovery bug this project just fixed.
    """
    hit = cached(url)
    if hit is not None:
        return hit
    try:
        r = requests.get(url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]

    ids = {fold(r["name"]): str(r["espn_id"])
           for _, r in pd.read_csv("data/espn_athlete_ids.csv").iterrows()}
    aid = ids.get(fold(target))
    if not aid:
        print(f"No ESPN id for {target!r} in data/espn_athlete_ids.csv")
        sys.exit(1)

    f = pd.read_csv("data/fighters.csv")
    row = f[f["name"].apply(fold) == fold(target)]
    if not row.empty:
        r = row.iloc[0]
        def n(c):
            v = r.get(c)
            try:
                v = float(v)
                return 0 if v != v else int(v)
            except (TypeError, ValueError):
                return 0
        print(f"fighters.csv: record {n('wins')}-{n('losses')}   "
              f"wins {n('ko_wins')}KO/{n('sub_wins')}SUB/{n('dec_wins')}DEC = {n('ko_wins')+n('sub_wins')+n('dec_wins')}   "
              f"losses {n('ko_losses')}KO/{n('sub_losses')}SUB/{n('dec_losses')}DEC = {n('ko_losses')+n('sub_losses')+n('dec_losses')}\n")

    log = fetch(EVENTLOG.format(id=aid))
    if not log:
        print("No cached eventlog -- run scripts/backfill_espn_fight_stats.py first.")
        sys.exit(1)

    # ALL PAGES -- the eventlog paginates at 25 and the oldest fights fall
    # off the end, which is exactly where an early-career loss lives.
    ev = (log.get("events") or {})
    items = list(ev.get("items") or [])
    try:
        pages = int(ev.get("pageCount") or 1)
    except (TypeError, ValueError):
        pages = 1
    for pg in range(2, pages + 1):
        more = fetch(EVENTLOG.format(id=aid) + f"?page={pg}")
        items += ((more or {}).get("events") or {}).get("items") or []
    items = [e for e in items if e.get("played")]
    print(f"ESPN eventlog: {len(items)} played fight(s)\n")
    print(f"  {'date':<12}{'W/L':<5}{'result (raw from ESPN)':<34}{'mapped?'}")
    print("  " + "-" * 62)

    wins = losses = 0
    unmapped = []
    for e in items:
        cr = (e.get("competitor") or {}).get("$ref")
        er = (e.get("event") or {}).get("$ref")
        ev = fetch(er) if er else None
        date = (ev or {}).get("date", "")[:10] or "?"

        # The eventlog item does NOT carry the outcome -- `won` is absent, so
        # the first version reported "?" for every fight and a 0W-0L total.
        # It lives on the competitor object the item points at.
        comp = fetch(cr) if cr else None
        won = (comp or {}).get("winner")
        wl = "W" if won is True else ("L" if won is False else "?")

        method = ""
        if cr:
            st = fetch(cr.split("/competitors/")[0].split("?")[0] + "/status")
            res = (st or {}).get("result") or {}
            method = res.get("displayName") or res.get("name") or ""

        norm = fold(method).replace("---", " - ")
        ok = any(m in norm for m in ("ko", "tko", "submission", "decision"))
        if wl == "W":
            wins += 1
        elif wl == "L":
            losses += 1
        if not ok and method:
            unmapped.append((date, wl, method))
        elif not method:
            unmapped.append((date, wl, "(no result recorded)"))
        print(f"  {date:<12}{wl:<5}{(method or '(none)'):<34}{'yes' if ok else 'NO'}")

    print(f"\n  ESPN totals: {wins}W-{losses}L")
    lost_unmapped = [u for u in unmapped if u[1] == "L"]
    won_unmapped = [u for u in unmapped if u[1] == "W"]
    print(f"  unmapped: {len(lost_unmapped)} on the LOSS side, {len(won_unmapped)} on the win side")
    if unmapped:
        print(f"\n  {len(unmapped)} fight(s) whose method does NOT fit KO/SUB/DEC:")
        for d, wl, m in unmapped:
            print(f"     {d}  {wl}  {m}")
        print("\n  These are the fights missing from the split columns. If they are all")
        print("  losses, that is exactly the one-directional bias the audit found --")
        print("  and it is a MAPPING gap, not missing data.")
    else:
        print("\n  Every fight maps cleanly. The gap is therefore NOT a method-mapping")
        print("  issue for this fighter -- look at whatever writes the split columns.")


if __name__ == "__main__":
    main()
