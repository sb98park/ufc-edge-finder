"""
Clear last-fight data that is dated in the FUTURE (or holds a status word).

TWO BUGS LEFT RESIDUE IN fighters.csv, and neither self-heals:

1. FUTURE DATES. ESPN's eventLog marks some SCHEDULED bouts as played=true,
   so an upcoming card was accepted as a fighter's most recent fight --
   producing a "last fight" dated after today, with a result attached to a
   fight that hasn't happened. Real case: two fighters showing 2026-08-08
   on 2026-08-01.

2. STATUS WORDS AS METHODS. ESPN's status.type.detail returns "Final" -- a
   completion state, not a method -- which rendered as "W by Final against
   X". The parser now rejects these, but rows written before that fix keep
   the bad value.

WHY A SCRIPT IS NEEDED AT ALL. The backfill only refreshes a fighter when a
field is NULL. A wrong-but-present value is never re-checked, so fixing the
source only helps fighters added afterwards. Blanking the bad fields makes
them eligible for gap-fill, and the next generate_site run refetches them
correctly.

Usage:
    python3 scripts/fix_future_last_fights.py            # dry run
    python3 scripts/fix_future_last_fights.py --apply
"""

import datetime as dt
import sys

import pandas as pd

FIGHTERS = "data/fighters.csv"
COLS = ["last_fight_date", "last_fight_result", "last_fight_method", "last_fight_opponent"]
STATUS_WORDS = {"final", "final/ot", "ft", "completed", "complete", "status_final", "end", "ended"}


def main():
    apply = "--apply" in sys.argv
    f = pd.read_csv(FIGHTERS)
    today = dt.date.today().isoformat()

    future, status = [], []
    for i, row in f.iterrows():
        d = str(row.get("last_fight_date") or "").strip()
        m = str(row.get("last_fight_method") or "").strip()
        bad_date = len(d) >= 10 and d[:10] > today
        bad_method = m.lower() in STATUS_WORDS
        if not bad_date and not bad_method:
            continue
        name = row.get("name")
        if bad_date:
            future.append((name, d[:10], m))
        elif bad_method:
            status.append((name, d[:10] if d else "?", m))
        if apply:
            for c in COLS:
                if c in f.columns:
                    # Blank rather than delete the row: the next backfill sees
                    # a NULL and refetches, which is exactly the path that
                    # already works for a brand-new fighter.
                    f.at[i, c] = pd.NA

    print(f"today is {today}\n")
    if future:
        print(f"FUTURE-DATED last fights ({len(future)}):")
        for n, d, m in future:
            print(f"   {str(n)[:26]:28} {d}  method={m or '(none)'}")
    if status:
        print(f"\nSTATUS WORD stored as method ({len(status)}):")
        for n, d, m in status:
            print(f"   {str(n)[:26]:28} {d}  method={m}")
    if not future and not status:
        print("No bad last-fight data found.")
        return

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to clear these,")
        print("then run generate_site.py so the backfill refetches them.")
        return
    f.to_csv(FIGHTERS, index=False)
    print(f"\nCleared {len(future) + len(status)} fighter(s). Run generate_site.py to refetch,")
    print("then commit data/fighters.csv.")


if __name__ == "__main__":
    main()
