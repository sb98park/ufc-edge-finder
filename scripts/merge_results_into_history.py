"""
Append confirmed results from fight_results.csv into fight_history.csv.

WHY THIS IS NEEDED. fight_history.csv is built by etl_fight_history.py from
the raw UFC dataset, which lags by weeks or months. Results the site records
live -- via ESPN, every card it tracks -- land in fight_results.csv and stop
there. Nothing carries them across.

The consequence is not cosmetic. Everything downstream reads history:

  Elo ratings          a fighter's recent wins are invisible
  the streak bonus     shipped today, keyed on tracked wins
  recency weighting    weights fights it doesn't know happened
  fun facts            Salkilld showed 4 wins against a real 5, one short
                       of the streak threshold, so his card had no facts

So the site watches a card, records who won, and then predicts the next one
as though it never happened. This closes that loop.

SAFE BY CONSTRUCTION:
  - append only, never edits or removes an existing row
  - skips any fight already present, matched on an unordered name pair plus
    date, so a differing fighter order can't create a duplicate
  - keeps a draw/no-contest (empty winner, NC or Draw as the method) and
    skips anything else with no winner
  - writes nothing when there is nothing to add

Usage:
    python3 scripts/merge_results_into_history.py            # dry run
    python3 scripts/merge_results_into_history.py --apply
"""

import os
import sys
import unicodedata

import pandas as pd

HISTORY = "data/fight_history.csv"
RESULTS = "data/fight_results.csv"


def fold(name: str) -> str:
    """Accent-insensitive key. Same folding used at every other join."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def key(a, b, date) -> tuple:
    # Unordered pair: the two sources disagree about which fighter is "a",
    # and an ordered key would let the same fight in twice.
    return (frozenset({fold(a), fold(b)}), str(date)[:10])


def main():
    apply = "--apply" in sys.argv
    if not os.path.exists(HISTORY) or not os.path.exists(RESULTS):
        print("Need both data/fight_history.csv and data/fight_results.csv.")
        return

    hist = pd.read_csv(HISTORY)
    res = pd.read_csv(RESULTS)
    have = {key(r["fighter_a"], r["fighter_b"], r.get("date"))
            for _, r in hist.iterrows()}

    date_col = next((c for c in ("event_date", "date", "date_added") if c in res.columns), None)
    if not date_col:
        print(f"No date column in {RESULTS}. Columns: {list(res.columns)}")
        return

    new_rows, skipped_known, skipped_nowinner = [], 0, 0
    for _, r in res.iterrows():
        a, b, w = r.get("fighter_a"), r.get("fighter_b"), r.get("winner")
        method = str(r.get("method") or "").strip()
        decisive = isinstance(w, str) and w.strip()
        # A NO CONTEST OR DRAW IS A RESULT, AND IT BELONGS IN THE SPINE. It
        # used to be dropped here with everything else that had no winner, so
        # the bout was invisible to layoff and to "last fight" -- Michael
        # Aljarouj's 2025-04-12 no contest is why his site card read a
        # four-year layoff. Carried through with an empty winner, which every
        # reader of fight_history.csv now handles: elo skips it, pit_roster
        # counts it toward neither record, matchup_model's recent-form term
        # ignores it, fun_facts treats it as not-a-win.
        no_contest = not decisive and method.lower() in ("nc", "no contest", "draw")
        if not a or not b or not (decisive or no_contest):
            skipped_nowinner += 1
            continue
        d = str(r.get(date_col))[:10]
        if key(a, b, d) in have:
            skipped_known += 1
            continue
        new_rows.append({
            "date": d, "fighter_a": a, "fighter_b": b,
            "winner": w if decisive else "",
            "method": method,
        })
        have.add(key(a, b, d))

    print(f"history rows      : {len(hist)}")
    print(f"recorded results  : {len(res)}")
    print(f"  already present : {skipped_known}")
    print(f"  no clear winner : {skipped_nowinner}")
    print(f"  NEW to add      : {len(new_rows)}")
    if new_rows:
        print()
        for r in new_rows[:12]:
            print(f"    {r['date']}  {r['fighter_a']} vs {r['fighter_b']}"
                  f"  -> {r['winner'] or r['method'] or 'no winner'}")
        if len(new_rows) > 12:
            print(f"    ... and {len(new_rows) - 12} more")

    if not new_rows:
        print("\nNothing to merge.")
        return
    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return

    out = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
    if "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)
    out.to_csv(HISTORY, index=False)
    print(f"\nWritten: {len(hist)} -> {len(out)} rows.")
    print("Re-run generate_site.py -- ratings, streaks and facts all read this.")


if __name__ == "__main__":
    main()
