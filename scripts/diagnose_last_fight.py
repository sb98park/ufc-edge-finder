"""
Why is a specific fighter's last-fight data missing?

The backfill is silent about fighters it never selects, so an empty field
looks identical to a failed fetch. This walks the chain and reports where it
stops:

    1. is the fighter on a tracked card at all?
    2. do they qualify for backfill (a NULL in a gap column)?
    3. can an athlete_id be resolved from ESPN's scoreboard?
    4. does eventsMap return usable bouts?

Usage:
    python3 scripts/diagnose_last_fight.py "Gamrot"
"""

import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fighter_backfill import _fetch_last_fight_from_events_map, BASE_HEADERS, REQUEST_TIMEOUT  # noqa: E402
from src.results_fetcher import ESPN_SCOREBOARD_URL  # noqa: E402
from src.card_matcher import _normalize_name  # noqa: E402

GAP_COLS = ["stance", "country", "reach_in", "height_in", "age", "last_fight_date",
            "ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    needle = sys.argv[1].lower()

    fighters = pd.read_csv("data/fighters.csv")
    match = fighters[fighters["name"].str.lower().str.contains(needle, na=False)]
    if match.empty:
        print(f"{needle!r} is not in fighters.csv at all.")
        return
    row = match.iloc[0]
    name = row["name"]
    print(f"fighter: {name}\n")

    # 1. on a tracked card?
    on_card, card_dates = [], []
    for f in ("data/fight_cards.csv", "data/future_cards.csv"):
        try:
            d = pd.read_csv(f)
        except FileNotFoundError:
            continue
        hit = d[(d["fighter_a"] == name) | (d["fighter_b"] == name)]
        if not hit.empty:
            on_card.append(f)
            card_dates.extend(str(x) for x in hit.get("event_date", pd.Series()).dropna().unique())
    print(f"1. on a tracked card : {on_card or 'NO -- backfill never considers them'}")
    if card_dates:
        print(f"   event date(s)     : {sorted(set(card_dates))}")

    # 2. qualifies for gap fill?
    nulls = [c for c in GAP_COLS if c in fighters.columns and pd.isna(row.get(c))]
    print(f"2. null gap columns  : {nulls or 'NONE -- backfill SKIPS them entirely'}")

    # 3. athlete id from the scoreboard for those dates
    aid = None
    for d in sorted(set(card_dates)):
        try:
            r = requests.get(ESPN_SCOREBOARD_URL, params={"dates": d.replace("-", "")},
                             headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                print(f"   scoreboard {d}: HTTP {r.status_code}")
                continue
            found = 0
            for ev in r.json().get("events", []):
                for comp in ev.get("competitions", []):
                    for c in comp.get("competitors", []):
                        ath = c.get("athlete") or {}
                        found += 1
                        if ath.get("fullName") and _normalize_name(ath["fullName"]) == _normalize_name(name):
                            aid = str(ath.get("id") or c.get("id"))
            print(f"   scoreboard {d}: {found} competitor(s) listed")
        except Exception as e:
            print(f"   scoreboard {d}: {e}")
    print(f"3. athlete_id        : {aid or 'NOT RESOLVED -- eventsMap can never be called'}")

    # 4. eventsMap
    if aid:
        got = _fetch_last_fight_from_events_map(aid, name)
        print(f"4. eventsMap result  : {got or 'EMPTY -- no completed bouts with a W/L'}")
    else:
        print("4. eventsMap result  : skipped (no athlete_id)")


if __name__ == "__main__":
    main()
