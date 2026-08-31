"""
Remove rows that record the same fight twice.

WHY THIS EXISTS. Cards get re-scraped, and a fighter whose name folds
differently between passes lands in the spine under both spellings -- so the
same bout is appended a second time. Elo replays the file row by row, so a
duplicated fight is scored twice: the winner is paid for one win and gets
credited for two.

Found 2026-08-31 while adding a hand-entered bout that the dedupe key in
add_history_manually.py failed to catch. All four affected fights involve one
fighter, Liu Ce, who ESPN has also called "Ce Liu" (CLAUDE.md s4). He was on
an upcoming card at the time, so this was live: removing the duplicates moves
14 fighters, Ivan Gnizditskiy by -38.2 and Liu Ce himself by -12.3.

THE KEY INCLUDES THE DATE, deliberately. frozenset({a, b}) alone carries no
event, so a rematch collides with the first meeting. Adding the date makes a
rematch safe (different night) while still collapsing a genuine double-write.
Two distinct bouts between the same pair on the same date do not exist.

Usage:
    python3 scripts/dedupe_fight_history.py            # dry run
    python3 scripts/dedupe_fight_history.py --apply
"""

import sys
import unicodedata

import pandas as pd

HISTORY = "data/fight_history.csv"


def fold(n) -> str:
    s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def main() -> int:
    apply = "--apply" in sys.argv
    h = pd.read_csv(HISTORY)
    keys = [(frozenset({fold(a), fold(b)}), str(d)[:10])
            for a, b, d in zip(h["fighter_a"], h["fighter_b"], h["date"])]
    dup = pd.Series(keys).duplicated(keep="first").to_numpy()

    if not dup.any():
        print(f"{len(h)} rows, no duplicated fights.")
        return 0

    print(f"{len(h)} rows; {int(dup.sum())} duplicate row(s) to drop:")
    for _, r in h[dup].iterrows():
        print(f"   {r['date']}  {r['fighter_a']} / {r['fighter_b']}  [{r.get('method') or ''}]")

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return 0
    h[~dup].to_csv(HISTORY, index=False)
    print(f"\nWritten: {len(h)} -> {int((~dup).sum())} rows.")
    print("Re-run generate_site.py -- Elo is replayed from this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
