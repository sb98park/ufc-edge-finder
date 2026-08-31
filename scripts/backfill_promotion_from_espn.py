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

SCOPE, AND WHY IT IS NOT THE WHOLE FILE. ESPN athlete ids are only reachable
through an event scoreboard -- there is no name search -- so a fighter is
resolvable only if they appear on a card we track. That covers everyone whose
numbers are published this week, which is the point. Rows for fighters we
cannot resolve keep their blank promotion, which is exactly the state they are
in today: this can only improve the file, never degrade it.

RE-RUNNABLE. Only ever writes rows it can positively classify as non-UFC.
A row already labelled is left alone.

Usage:
    python3 scripts/backfill_promotion_from_espn.py            # dry run
    python3 scripts/backfill_promotion_from_espn.py --apply
"""

import datetime as dt
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


def main() -> int:
    apply = "--apply" in sys.argv
    hist = pd.read_csv(HISTORY)
    if "promotion" not in hist.columns:
        hist["promotion"] = ""

    ids = athlete_ids()
    print(f"resolved {len(ids)} athlete id(s) from tracked card scoreboards")

    # (folded pair, date) -> row index, with a one-day tolerance on lookup.
    index = {}
    dates = pd.to_datetime(hist["date"], errors="coerce")
    for i in hist.index:
        if pd.isna(dates.loc[i]):
            continue
        index[fight_key(hist.at[i, "fighter_a"], hist.at[i, "fighter_b"], dates.loc[i])] = i

    # NEVER SKIP SILENTLY. The first version of this script reported "0 rows
    # to stamp" because every athlete fetch was failing and being swallowed by
    # a bare `continue` -- the same defect the audit found in results_fetcher.
    stamped, unresolved, failed, empty = {}, 0, [], 0
    for name, aid in sorted(ids.items()):
        try:
            r = requests.get(SITE_ATHLETE_URL.format(aid), headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                failed.append(f"{name} HTTP {r.status_code}")
                time.sleep(2.0 if r.status_code in (429, 403) else 0.4)
                continue
            events = r.json().get("eventsMap") or {}
            if not events:
                empty += 1
        except requests.RequestException as exc:
            failed.append(f"{name} {type(exc).__name__}")
            continue
        time.sleep(0.4)
        for rec in events.values():
            if not isinstance(rec, dict):
                continue
            promo = _promotion_of(rec)
            if not promo:
                continue                      # UFC, or unclassifiable: leave blank
            when = pd.to_datetime(str(rec.get("gameDate") or "")[:10], errors="coerce")
            opp = rec.get("opponent")
            if isinstance(opp, dict):
                opp = opp.get("displayName") or opp.get("shortDisplayName")
            if pd.isna(when) or not opp:
                continue
            hit = None
            for off in (-1, 0, 1):
                k = fight_key(name, opp, (when + dt.timedelta(days=off)).date().isoformat())
                if k in index:
                    hit = index[k]
                    break
            if hit is None:
                unresolved += 1
                continue
            # NOT `x or ""` -- an unlabelled promotion cell is NaN, and NaN is
            # TRUTHY, so that idiom yields the string "nan" and every row looks
            # already-labelled. This reported "0 rows to stamp" over 8 regional
            # bouts it had correctly matched (CLAUDE.md s4).
            existing = hist.at[hit, "promotion"]
            if pd.notna(existing) and str(existing).strip():
                continue
            stamped[hit] = promo

    if failed:
        print(f"athlete fetch FAILED for {len(failed)}: {failed[:6]}"
              f"{' ...' if len(failed) > 6 else ''}")
    print(f"athletes with an empty eventsMap: {empty}")
    print(f"rows to stamp non-UFC : {len(stamped)}")
    print(f"ESPN bouts not in the spine (nothing to stamp): {unresolved}")
    by_promo = {}
    for v in stamped.values():
        by_promo[v] = by_promo.get(v, 0) + 1
    for k, v in sorted(by_promo.items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {v:4d}  {k}")

    if not stamped:
        print("\nNothing to do.")
        return 0
    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return 0
    for i, promo in stamped.items():
        hist.at[i, "promotion"] = promo
    hist.to_csv(HISTORY, index=False)
    print(f"\nWritten: {len(stamped)} row(s) labelled non-UFC in {HISTORY}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
