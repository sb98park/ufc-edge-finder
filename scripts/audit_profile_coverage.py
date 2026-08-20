"""
Audit what the fighter profile actually has data for, per fighter on the card.

WHY THIS EXISTS. src/fighter_profile.py is now the single source for three
things a reader sees: the six scout rails under the fighter buttons, the five
Tale of the Tape categories and their tap-through, and the tier words on the
scouting drawer's rate strip. All three go silent for the same reason -- a
fighter short of MIN_UFC_BOUTS gets no profile at all -- so one audit covers
them.

This replaces an audit of the old six-axis radar, which plotted
compute_radar_metrics from fighters.csv. That chart no longer exists: the
radar draws categories from pit_stats instead, and the old script was
reporting coverage of columns that feed nothing. It ran clean while auditing
the wrong thing, which is the failure mode worth avoiding here.

WHAT ABSENCE MEANS, and it is not a data gap. Measured across the booked
roster: every fighter without a profile is simply short of UFC bouts, and
their pit_stats bout count matches their ufc_fight_results count exactly. No
scrape failure, no name-match failure. A blank here is a career length, and
the UI says so in words rather than leaving a hole.

Usage:  python3 scripts/audit_profile_coverage.py
        python3 scripts/audit_profile_coverage.py --all   # whole roster
"""

import collections
import csv
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fighter_history import fold_name
from src.fighter_profile import (ATTRIBUTES, CATEGORIES, DRAWER_RANKS,
                                 MIN_UFC_BOUTS, RAIL_LABELS, build_profiles,
                                 _aggregate)


def main():
    everyone = "--all" in sys.argv
    if everyone:
        names = sorted(set(pd.read_csv("data/fighters.csv")["name"].dropna()))
        scope = "whole roster"
    else:
        cards = pd.concat([pd.read_csv("data/fight_cards.csv"),
                           pd.read_csv("data/future_cards.csv")])
        names = sorted(set(cards["fighter_a"]) | set(cards["fighter_b"]))
        scope = "fighters booked on fight_cards + future_cards"

    profiles = build_profiles(names)
    agg = _aggregate()
    have = [n for n in names if (profiles.get(fold_name(n)) or {}).get("pct")]
    missing = [n for n in names if n not in have]

    print(f"Profile coverage for {len(names)} {scope}")
    print(f"  profiled ({MIN_UFC_BOUTS}+ UFC bouts): {len(have)}  ({len(have)/len(names):.1%})")
    print(f"  no profile:                          {len(missing)}\n")

    # WHY each one is missing. If any of these is NOT simply short of bouts,
    # that is a real defect -- a scrape or name-match failure -- and it should
    # be loud, because it looks identical to a debutant in the UI.
    print("WHY THERE IS NO PROFILE (bout count, not missing data)")
    dist = collections.Counter()
    suspicious = []
    truth = _ufc_bout_counts()
    for n in missing:
        k = fold_name(n)
        pit = (agg.get(k) or {}).get("bouts", 0)
        real = truth.get(k, 0)
        dist[pit] += 1
        if real != pit:
            suspicious.append((n, pit, real))
    for bouts in sorted(dist):
        print(f"  {bouts} UFC bout(s): {dist[bouts]} fighter(s)")
    if suspicious:
        print("\n  DEFECT -- pit_stats disagrees with ufc_fight_results:")
        for n, pit, real in suspicious:
            print(f"    {n:<26} pit_stats {pit}, results {real}")
    else:
        print("  every one matches ufc_fight_results exactly -- no data is missing")
    print()

    print(f"ATTRIBUTE COVERAGE among the {len(have)} profiled")
    for label, _fn, _hb, cat in ATTRIBUTES:
        n = sum(1 for f in have if label in (profiles[fold_name(f)]["pct"] or {}))
        flag = ""
        if label in RAIL_LABELS:
            flag += " [rail]"
        if label in DRAWER_RANKS.values():
            flag += " [drawer]"
        print(f"  {label:<14} {cat:<11} {n}/{len(have)}{flag}")
    print()

    print("CATEGORY COVERAGE (radar axes)")
    for cat in CATEGORIES:
        n = sum(1 for f in have if cat in (profiles[fold_name(f)].get("cats") or {}))
        print(f"  {cat:<12} {n}/{len(have)}")
    print()

    # A fight needs BOTH corners to draw a radar, so pair coverage is the
    # number that actually decides how many charts appear on the page.
    if not everyone:
        cards = pd.concat([pd.read_csv("data/fight_cards.csv"),
                           pd.read_csv("data/future_cards.csv")])
        both = one = none = 0
        for r in cards.itertuples():
            a = (profiles.get(fold_name(r.fighter_a)) or {}).get("pct")
            b = (profiles.get(fold_name(r.fighter_b)) or {}).get("pct")
            both += bool(a and b); one += bool(bool(a) ^ bool(b)); none += bool(not a and not b)
        tot = both + one + none
        print("PAIR COVERAGE (a radar needs both corners)")
        print(f"  both profiled: {both}/{tot} ({both/tot:.0%})   one only: {one}   neither: {none}")


def _ufc_bout_counts() -> dict:
    """Ground truth from the ufcstats spine, to catch a real data gap."""
    out = collections.Counter()
    for row in csv.DictReader(open("data/ufc_fight_results.csv", encoding="utf-8")):
        for who in str(row.get("BOUT", "")).split(" vs. "):
            who = who.strip()
            if who:
                out[fold_name(who)] += 1
    return out


if __name__ == "__main__":
    main()
