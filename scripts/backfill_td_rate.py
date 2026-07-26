"""
Backfill career takedown RATE (takedowns landed per 15 minutes) into
data/fighters.csv.

WHY. head_to_head_adjustment.py showed, on a frozen 2019+ holdout, that
production's wrestling term is the weakest link. Holding everything else
fixed and swapping ONLY the wrestling signal:

    production's shape, clipped accuracy-vs-defense : 56.0%  Brier 0.2424
    production's shape, takedown-RATE differential  : 57.2%  Brier 0.2389

(Elo alone is 55.9%/0.2431 -- so the current wrestling term is contributing
almost nothing.) The rate version needs a field fighters.csv doesn't have:
td_accuracy_pct and td_defense_pct are percentages, not volume. A fighter
who goes 1-for-1 on takedowns has 100% accuracy and almost no wrestling
output; one who goes 6-for-12 has half the accuracy and far more control of
where the fight happens. Rate captures that; accuracy cannot.

WHERE THE DATA COMES FROM. ufc_fight_stats.csv has takedowns landed per
round; ufc_fight_results.csv has each fight's end round and time, which
gives real fight duration. Both already power the backtests. Neither is
committed (they're large and local-only), which is exactly why this is a
manual backfill that WRITES ITS RESULT INTO fighters.csv -- that file IS
committed, so scheduled Actions builds get the field without ever needing
the big CSVs.

NAME MATCHING. The stats CSVs use UFCStats spellings; fighters.csv uses
ours. Matching is by normalised name, and anything unmatched is REPORTED
rather than silently dropped -- a quiet 30% miss rate would degrade the
model invisibly.

Usage:
    python3 scripts/backfill_td_rate.py            # dry run, prints coverage
    python3 scripts/backfill_td_rate.py --apply    # writes fighters.csv
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.card_matcher import _normalize_name  # noqa: E402

FIGHTERS = "data/fighters.csv"
STATS = next((p for p in ("data/ufc_fight_stats.csv",
                          "/mnt/user-data/uploads/ufc_fight_stats.csv") if os.path.exists(p)), None)
RESULTS = next((p for p in ("data/ufc_fight_results.csv",
                            "/mnt/user-data/uploads/ufc_fight_results.csv") if os.path.exists(p)), None)
COLUMN = "td_per_15"
MIN_MINUTES = 15.0  # under one full fight's worth of cage time, a rate is noise


def _of_pair(s):
    try:
        landed, _ = str(s).split(" of ")
        return int(landed)
    except (ValueError, AttributeError):
        return 0


def _duration_seconds(round_num, time_str):
    try:
        m, sec = str(time_str).split(":")
        return (int(round_num) - 1) * 300 + int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return None


def main():
    apply = "--apply" in sys.argv
    if not STATS or not RESULTS:
        print("Missing ufc_fight_stats.csv / ufc_fight_results.csv.")
        print("Put both in data/ (they're local-only, not committed) and re-run.")
        sys.exit(1)

    stats = pd.read_csv(STATS)
    stats.columns = [c.strip() for c in stats.columns]
    results = pd.read_csv(RESULTS)
    results.columns = [c.strip() for c in results.columns]

    # Fight duration, keyed by (event, bout)
    durations = {}
    for r in results.itertuples(index=False):
        d = _duration_seconds(r.ROUND, r.TIME)
        if d and d > 0:
            durations[(str(r.EVENT).strip(), str(r.BOUT).strip())] = d

    # Per fighter: total takedowns landed and total cage time
    totals = {}
    unmatched_fights = 0
    for r in stats.itertuples(index=False):
        key = (str(r.EVENT).strip(), str(r.BOUT).strip())
        dur = durations.get(key)
        if dur is None:
            unmatched_fights += 1
            continue
        name = str(r.FIGHTER).strip()
        t = totals.setdefault(name, {"td": 0, "seconds": 0.0, "rounds": 0})
        t["td"] += _of_pair(r.TD)
        t["rounds"] += 1

    # Cage time is per FIGHT, not per round row -- add it once per fighter/fight.
    seen = set()
    for r in stats.itertuples(index=False):
        key = (str(r.EVENT).strip(), str(r.BOUT).strip())
        dur = durations.get(key)
        if dur is None:
            continue
        name = str(r.FIGHTER).strip()
        if (key, name) in seen:
            continue
        seen.add((key, name))
        totals[name]["seconds"] += dur

    rates = {}
    for name, t in totals.items():
        minutes = t["seconds"] / 60.0
        if minutes >= MIN_MINUTES:
            rates[_normalize_name(name)] = t["td"] / minutes * 15.0

    fighters = pd.read_csv(FIGHTERS)
    matched, missing = 0, []
    values = []
    for name in fighters["name"]:
        v = rates.get(_normalize_name(str(name)))
        if v is None:
            missing.append(str(name))
            values.append("")
        else:
            matched += 1
            values.append(round(v, 3))

    print(f"stats rows: {len(stats)} | fights with a usable duration: {len(durations)}")
    if unmatched_fights:
        print(f"  ({unmatched_fights} stat rows had no matching fight duration -- skipped)")
    print(f"fighters with >= {MIN_MINUTES:.0f} min of tracked cage time: {len(rates)}")
    print(f"\nfighters.csv rows: {len(fighters)} | matched: {matched} ({matched/len(fighters):.0%}) "
          f"| unmatched: {len(missing)}")
    if missing:
        print("  unmatched (will have a blank rate, production falls back for these):")
        for n in missing[:15]:
            print(f"    {n}")
        if len(missing) > 15:
            print(f"    ... and {len(missing)-15} more")

    if matched:
        got = [v for v in values if v != ""]
        got.sort()
        print(f"\ntd_per_15 distribution: min {got[0]:.2f} | median {got[len(got)//2]:.2f} "
              f"| max {got[-1]:.2f}")

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to write the column.")
        return

    fighters[COLUMN] = values
    fighters.to_csv(FIGHTERS, index=False)
    print(f"\nDone -- wrote {COLUMN} for {matched} fighters into {FIGHTERS}.")
    print("Commit fighters.csv so scheduled builds get the field too.")


if __name__ == "__main__":
    main()
