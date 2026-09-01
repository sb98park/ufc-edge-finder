"""
Stamp `promotion` onto spine rows that ESPN says are not UFC bouts.

WHY. src/elo.ufc_only() reads a blank promotion as UFC, and until now
fetch_espn_fight_history -- the path that writes most of
data/fight_history.csv -- emitted no promotion key at all. 17 of 11,925 rows
carried one, so the filter was a 0.14% no-op and every rendered "win streak"
was a CAREER streak wearing a UFC label: Mario Pinto's published 12-fight run
is 4 UFC bouts and 8 regional ones, three of them on a single night at a
Levels Fight League tournament. The same blank rows are inside the Elo graph,
where a regional opponent with no other results sits at the 1500 default.

fetch_espn_fight_history now stamps it going forward. This backfills what is
already on disk.

HOW IT REACHES FIGHTERS WHO ARE NOT ON A CARD. ESPN has no athlete name
search, so the first version could only resolve ids from the scoreboards of
cards we track -- 158 fighters, leaving 5,111 of 5,269 in the spine
unclassified. But every eventsMap record carries its OPPONENT's athlete id,
so the fight graph can be walked: fetch a fighter, classify their bouts,
enqueue everyone they fought. The scoreboards are only the seed.

WHY THAT IS WORTH DOING. Elo is a graph. A carded fighter's UFC opponent
carrying eight unclassified regional wins is rated too highly, and that flows
straight into the carded fighter's own number through the bout they shared.
Measured as an upper bound -- drop every unlabelled bout between two
off-roster fighters -- the 2026-09-05 card moves up to 8.24pp, and the main
event 5.20pp. That bound is deliberately too aggressive to act on, which is
exactly why the real classification is worth fetching rather than guessing.

BOUNDED AND RESUMABLE. --max-hops limits how far from the roster the crawl
walks (1 covers the 3,567 fighters who share a bout with someone on it);
--budget caps requests so a run ends predictably. Visited ids persist to
data/promotion_crawl_state.json, so the next run continues rather than
repeating. Rows for fighters never reached keep their blank promotion, which
is the state they are in today: this can only improve the file.

RE-RUNNABLE. Only ever writes rows it can positively classify as non-UFC.
A row already labelled is left alone.

Usage:
    python3 scripts/backfill_promotion_from_espn.py            # dry run
    python3 scripts/backfill_promotion_from_espn.py --apply
"""

import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_matcher import fight_key                                  # noqa: E402
from src.fighter_backfill import (BASE_HEADERS, REQUEST_TIMEOUT,        # noqa: E402
                                  SITE_ATHLETE_URL, _promotion_of)

HISTORY = "data/fight_history.csv"
CARDS = ("data/fight_cards.csv", "data/future_cards.csv")
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates={}"
# Where the crawl remembers what it has already fetched, so a bounded run can
# be resumed instead of restarted.
STATE = "data/promotion_crawl_state.json"


def athlete_ids() -> dict:
    """name -> ESPN athlete id, resolved from the scoreboards of tracked cards."""
    dates = set()
    for path in CARDS:
        try:
            c = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError):
            continue
        dates |= {str(d)[:10].replace("-", "") for d in c.get("event_date", []) if str(d)[:4].isdigit()}
    found = {}
    for d in sorted(dates):
        try:
            r = requests.get(SCOREBOARD.format(d), headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            for ev in (r.json().get("events") or []):
                for comp in (ev.get("competitions") or []):
                    for c in (comp.get("competitors") or []):
                        nm = ((c.get("athlete") or {}).get("displayName") or "").strip()
                        if nm and c.get("id"):
                            found.setdefault(nm, str(c["id"]))
        except requests.RequestException as exc:
            print(f"  scoreboard {d}: {exc}")
        time.sleep(0.4)
    return found


def _load_state() -> dict:
    try:
        with open(STATE, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    try:
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError as exc:
        print(f"  could not persist crawl state: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-hops", type=int, default=1,
                    help="how far from the roster to walk (1 = fighters who "
                         "share a bout with someone on a tracked card)")
    ap.add_argument("--budget", type=int, default=600,
                    help="maximum athlete fetches this run")
    args = ap.parse_args()

    hist = pd.read_csv(HISTORY)
    if "promotion" not in hist.columns:
        hist["promotion"] = ""

    state = _load_state()
    done = set(state.get("visited") or [])
    seeds = athlete_ids()
    print(f"seeded {len(seeds)} athlete id(s) from tracked card scoreboards; "
          f"{len(done)} already crawled in earlier runs")

    index = {}
    dates = pd.to_datetime(hist["date"], errors="coerce")
    for i in hist.index:
        if pd.notna(dates.loc[i]):
            index[fight_key(hist.at[i, "fighter_a"], hist.at[i, "fighter_b"], dates.loc[i])] = i

    # (athlete_id, name, hop). Seeds are hop 0.
    #
    # THE FRONTIER PERSISTS, NOT JUST THE VISITED SET. Saving only `visited`
    # made the crawl a one-shot dressed as a resumable one: the second run
    # rebuilt the queue from the seeds, found all 158 already visited, skipped
    # every one before it could expand them, and reported "0 fetched, 0
    # queued" over a frontier of 1,659. Whatever is still owed has to be
    # written down, or resuming means restarting from a wall.
    queue = [tuple(x) for x in (state.get("frontier") or [])]
    queued = {aid for aid, _, _ in queue}
    for nm, aid in sorted(seeds.items()):
        if aid not in queued and aid not in done:
            queue.append((aid, nm, 0))
            queued.add(aid)
    stamped, unresolved, failed, fetched = {}, 0, [], 0

    while queue and fetched < args.budget:
        aid, name, hop = queue.pop(0)
        if aid in done:
            continue
        try:
            r = requests.get(SITE_ATHLETE_URL.format(aid), headers=BASE_HEADERS,
                             timeout=REQUEST_TIMEOUT)
            fetched += 1
            if r.status_code != 200:
                failed.append(f"{name} HTTP {r.status_code}")
                time.sleep(3.0 if r.status_code in (429, 403) else 0.4)
                continue
            events = r.json().get("eventsMap") or {}
        except requests.RequestException as exc:
            failed.append(f"{name} {type(exc).__name__}")
            continue
        done.add(aid)
        time.sleep(0.35)

        for rec in events.values():
            if not isinstance(rec, dict):
                continue
            opp = rec.get("opponent") if isinstance(rec.get("opponent"), dict) else {}
            opp_name = opp.get("displayName") or opp.get("shortDisplayName")
            opp_id = str(opp.get("id") or "")
            if opp_id and hop + 1 <= args.max_hops and opp_id not in queued and opp_id not in done:
                queued.add(opp_id)
                queue.append((opp_id, opp_name or opp_id, hop + 1))

            promo = _promotion_of(rec)
            if not promo:
                continue                      # UFC, or unclassifiable: leave blank
            when = pd.to_datetime(str(rec.get("gameDate") or "")[:10], errors="coerce")
            if pd.isna(when) or not opp_name:
                continue
            hit = None
            for off in (-1, 0, 1):
                k = fight_key(name, opp_name, (when + dt.timedelta(days=off)).date().isoformat())
                if k in index:
                    hit = index[k]
                    break
            if hit is None:
                unresolved += 1
                continue
            existing = hist.at[hit, "promotion"]
            if pd.notna(existing) and str(existing).strip():
                continue
            stamped[hit] = promo

    if failed:
        print(f"athlete fetch FAILED for {len(failed)}: {failed[:6]}"
              f"{' ...' if len(failed) > 6 else ''}")
    print(f"fetched {fetched} athlete(s) this run; {len(queue)} still queued")
    print(f"rows to stamp non-UFC : {len(stamped)}")
    print(f"ESPN bouts not in the spine (nothing to stamp): {unresolved}")
    by_promo = {}
    for v in stamped.values():
        by_promo[v] = by_promo.get(v, 0) + 1
    for k, v in sorted(by_promo.items(), key=lambda kv: -kv[1])[:10]:
        print(f"   {v:4d}  {k}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return 0
    for i, promo in stamped.items():
        hist.at[i, "promotion"] = promo
    hist.to_csv(HISTORY, index=False)
    state["visited"] = sorted(done)
    state["frontier"] = [list(x) for x in queue]
    _save_state(state)
    print(f"\nWritten: {len(stamped)} row(s) labelled non-UFC in {HISTORY}; "
          f"crawl state has {len(done)} athlete(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
