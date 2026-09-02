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
import datetime as dt
import sys
import time
import unicodedata

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fighter_backfill import (fetch_espn_fight_history, BASE_HEADERS,  # noqa: E402
                                  REQUEST_TIMEOUT)
from src.results_fetcher import ESPN_SCOREBOARD_URL  # noqa: E402
from src.card_matcher import fight_key  # noqa: E402
from src.names import _normalize_name  # noqa: E402

HISTORY = "data/fight_history.csv"


# Bout identity comes from src/card_matcher.fight_key -- ONE definition for
# the whole spine. This file's own fold ignored punctuation, which is how
# "Benoit Saint-Denis" and "Benoit Saint Denis" ended up as two bouts.
def key(a, b, d):
    return fight_key(a, b, d)


# A NAME fold, which is a different question from a BOUT key and needs its own
# call. Deleting this file's local fold() and pointing bout identity at
# fight_key left the two name-folding call sites below still calling the
# deleted name -- so this script raised NameError on EVERY run from 2026-08-31,
# and CI invokes it with `|| true`, so nothing said a word for two days while
# fight_history quietly stopped being topped up.
#
# _normalize_name rather than a fresh local fold: it is the project's one
# name fold and it resolves NAME_ALIASES, so a fighter carrying a middle name
# in ESPN's payload resolves to the spelling the roster uses.
def fold(n):
    return _normalize_name(str(n))


# A ONE-DAY WINDOW, WHICH fight_key's OWN DOCSTRING ASKS CALLERS TO USE:
# "Callers that need a tolerance window should compare against fight_key(a, b,
# date +/- 1 day) rather than inventing their own key". This one did not, and
# tested the exact date only.
#
# ESPN dates a card by its UTC start, so a Saturday-night US event lands on
# Sunday in the payload and the same bout we already hold reads as new. On a
# 27-fighter sample that was 22 of 36 candidate bouts -- ~60% duplicates. With
# CI running this file with --apply, fixing the NameError above WITHOUT this
# would have re-created the 231 double-written bouts that the spine cleanup on
# 2026-08-31 removed, and src/elo.py replays raw names, so each duplicate is
# scored twice against a phantom node.
def _same_day_bout(by_fighter, anchor, d):
    """
    Does this fighter ALREADY have a bout on (or beside) this date?

    A SECOND DUPLICATE CLASS, which the +-1 day window above does not reach.
    The spine holds opponent names that an older import truncated -- "Sylvain
    Sommerei", "Alexandre Guille", "Franck Lebouyon" -- where ESPN sends
    "Sommereisen", "Guillemant", "LeBouyonnec". The pair key cannot match
    those, so each one re-appends as a new bout. It put Michael Aljarouj on 24
    spine rows against a 13-3 record, history_coverage 1.375, and
    tests/test_spine_integrity.py is what caught it.

    NOT SOLVED BY FUZZIER NAMES. "Leno Rodrigo"/"Lennon Rodrigo" and
    "Kanguichev"/"Kanguichiev" are not prefixes of each other, so the rule
    that covers them is a similarity threshold -- and CLAUDE.md is explicit
    that this project does not do that, because it conflates real people.

    So this declines instead of deciding. A fighter with a bout already on
    that date is ambiguous, and an ambiguous bout is REPORTED, NEVER
    APPENDED. It costs the occasional genuine same-night tournament bout,
    which is a far cheaper error than a phantom Elo node scored twice.
    """
    try:
        day = dt.date.fromisoformat(str(d)[:10])
    except (ValueError, TypeError):
        return False
    held = by_fighter.get(fold(anchor), set())
    return any(abs((day - h).days) <= 1 for h in held)


def _seen(have, a, b, d):
    try:
        day = dt.date.fromisoformat(str(d)[:10])
    except (ValueError, TypeError):
        return key(a, b, d) in have
    for off in (-1, 0, 1):
        if key(a, b, (day + dt.timedelta(days=off)).isoformat()) in have:
            return True
    return False


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

    # THE SCOREBOARD IS NOT THE ONLY PLACE AN ID LIVES. data/espn_athlete_ids.csv
    # holds 2,785 name-to-id pairs and this script never read it, so a fighter
    # the scoreboard does not list -- a late replacement, added after his card
    # stopped being a future card -- was unreachable even when we had his id.
    # Pavel Andrusca went onto 2026-09-05 with 0 of his 8 bouts in the spine
    # for exactly this reason. Scoreboard ids still win: they came from the
    # card being backfilled.
    try:
        _disk = pd.read_csv("data/espn_athlete_ids.csv")
        _added = 0
        for _, _r in _disk.iterrows():
            _k = fold(_r["name"])
            if _k not in id_map and str(_r.get("espn_id") or "").strip():
                id_map[_k] = str(_r["espn_id"]).strip()
                _added += 1
        print(f"  + {_added} id(s) from data/espn_athlete_ids.csv not on the scoreboard")
    except (OSError, pd.errors.EmptyDataError, KeyError) as _e:
        print(f"  (no on-disk id map: {_e})")

    # Every date each fighter already has a bout on, for the ambiguity guard.
    by_fighter = {}
    for _, _r in hist.iterrows():
        try:
            _d = dt.date.fromisoformat(str(_r["date"])[:10])
        except (ValueError, TypeError):
            continue
        for _side in ("fighter_a", "fighter_b"):
            by_fighter.setdefault(fold(_r[_side]), set()).add(_d)

    new_rows, no_id, ambiguous = [], [], []
    for n in sorted(names):
        aid = id_map.get(fold(n))
        if not aid:
            no_id.append(n)
            continue
        for row in fetch_espn_fight_history(aid, n):
            if _seen(have, row["fighter_a"], row["fighter_b"], row["date"]):
                continue
            if _same_day_bout(by_fighter, n, row["date"]):
                ambiguous.append((n, row["date"], row["fighter_a"], row["fighter_b"]))
                continue
            have.add(key(row["fighter_a"], row["fighter_b"], row["date"]))
            new_rows.append(row)
        time.sleep(0.2)

    if ambiguous:
        print(f"\nHELD BACK -- {len(ambiguous)} bout(s) whose fighter already has one that day.")
        print("  Almost always the same bout under a differently-spelled opponent; "
              "appending would double-score it in elo. Not appended, listed so a "
              "human can reconcile the spelling:")
        for who, d, a, b in sorted(ambiguous)[:10]:
            print(f"    {d}  {who}: {a} vs {b}")
        if len(ambiguous) > 10:
            print(f"    ... and {len(ambiguous) - 10} more")
        print()
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
    # kind="stable": pandas defaults to quicksort, which is NOT stable, so a
    # plain sort_values("date") silently reshuffles rows WITHIN a date. Elo
    # replays row by row, so that alone moved 271 fighters (Don Frye +23.8)
    # with zero data change. Measured 2026-08-31.
    out = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True).sort_values("date", kind="stable")
    out.to_csv(HISTORY, index=False)
    print(f"\nWritten: {len(hist)} -> {len(out)} rows.")
    print("Re-run generate_site.py -- ratings, streaks and facts all read this.")


if __name__ == "__main__":
    main()
