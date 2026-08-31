"""
Remove rows that record the same fight twice.

WHY THIS EXISTS. Cards get re-scraped and four different paths append to
data/fight_history.csv. Elo replays the file row by row, so a bout written
twice is scored twice: the winner is paid for one win and credited with two.

WHY THE FIRST VERSION OF THIS SCRIPT MISSED 231 OF THEM. It folded names with
NFKD + lowercase + whitespace collapse, which does not touch punctuation, and
it demanded an exact date match. Both are wrong for this file:

  PUNCTUATION. ESPN writes "Benoit Saint Denis"; the card writes
  "Benoit Saint-Denis". src/card_matcher._normalize_name -- the fold the rest
  of the repo matches names with -- substitutes punctuation with a space, so
  those are one fighter to the card path and were two to this one. Five bouts
  sat in the file twice, invisibly. One of them charged Dan Hooker two losses
  for a single 2026-01-31 fight and moved the published probability on the
  2026-09-05 main event by 3.3 points.

  THE DATE. ESPN dates an event in UTC, so a card starting 10pm ET is stamped
  the following day; and merge_results_into_history wrote `date_added` -- the
  CI scrape timestamp -- as the event date on cards that straddle midnight.
  226 bouts are in the file twice, one day apart.

Both now go through card_matcher.fight_key, which is the single definition of
"which bout is this" for the spine. Do not add a thirteenth fold (CLAUDE.md
s4); if the key is wrong, fix it there and every path moves together.

WHICH ROW SURVIVES IS NOT ARBITRARY, and picking the earlier one blindly was
wrong twice over:

  NAMES. src/elo.py replays RAW name strings, so the surviving spelling
  becomes the node in the rating graph. Keep "Benoit Saint Denis" over
  "Benoit Saint-Denis" and the rating is built under a name fighters.csv
  never uses, so build_effective_ratings looks it up, misses, and falls back
  to the neutral 1500. Roster spelling therefore wins first.

  METHOD. In 17 of 231 groups the LATER row is the one carrying the method;
  in 214 it is the earlier. Dropping on date alone loses real data either
  way, so the keeper inherits a method from a discarded sibling when it has
  none of its own. Nothing else is merged -- these rows describe one bout and
  agree on everything but spelling, spacing and a day.

Order: roster-canonical names, then has-a-method, then earliest date. The
winner string is rewritten to the keeper's own spelling so it cannot become a
phantom node either.

A REMATCH IS SAFE. The window is one day, and no pair has ever fought twice
inside 24 hours. A fighter with two bouts on one night (a regional tournament
-- real, e.g. Mario Pinto on 2023-03-11) has a different opponent each time,
so the pair differs and nothing collapses.

Usage:
    python3 scripts/dedupe_fight_history.py            # dry run
    python3 scripts/dedupe_fight_history.py --apply
"""

import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_matcher import _normalize_name, fight_key   # noqa: E402

HISTORY = "data/fight_history.csv"
# ESPN's UTC roll-over and a scrape timestamp are both at most one day late.
DATE_WINDOW_DAYS = 1


def _groups(history: pd.DataFrame) -> list:
    """Row indices grouped by bout, tolerating a one-day date difference."""
    dates = pd.to_datetime(history["date"], errors="coerce")
    seen: dict = {}
    out: list = []
    for i in dates.sort_values(kind="stable").index:
        when = dates.loc[i]
        if pd.isna(when):
            continue
        pair = fight_key(history.at[i, "fighter_a"], history.at[i, "fighter_b"], when)[0]
        hit = None
        for off in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1):
            k = (pair, (when + dt.timedelta(days=off)).date().isoformat())
            if k in seen:
                hit = k
                break
        if hit is not None:
            out[seen[hit]].append(i)
        else:
            seen[(pair, when.date().isoformat())] = len(out)
            out.append([i])
    return out


def _has_method(history, i) -> bool:
    m = history.at[i, "method"]
    return pd.notna(m) and str(m).strip() != ""


def resolve(history: pd.DataFrame, roster: set) -> tuple:
    """(kept dataframe, indices dropped, methods recovered from dropped rows)."""
    dates = pd.to_datetime(history["date"], errors="coerce")
    out = history.copy()
    dropped, recovered = [], 0
    for idx in _groups(history):
        if len(idx) == 1:
            continue
        def rank(i):
            names = {history.at[i, "fighter_a"], history.at[i, "fighter_b"]}
            canonical = len(names & roster)          # 0, 1 or 2 exact roster hits
            stamp = dates.loc[i]
            return (canonical, _has_method(history, i),
                    -(stamp.toordinal() if pd.notna(stamp) else 0))
        keep = max(idx, key=rank)
        for other in idx:
            if other == keep:
                continue
            if not _has_method(history, keep) and _has_method(history, other):
                out.at[keep, "method"] = history.at[other, "method"]
                recovered += 1
            dropped.append(other)
        # The winner must be spelled the way the surviving row spells it, or
        # elo.py builds the rating under a name nothing else uses.
        w = _normalize_name(out.at[keep, "winner"])
        if w:
            for col in ("fighter_a", "fighter_b"):
                if _normalize_name(out.at[keep, col]) == w:
                    out.at[keep, "winner"] = out.at[keep, col]
                    break
    return out.drop(index=dropped), dropped, recovered


def main() -> int:
    apply = "--apply" in sys.argv
    h = pd.read_csv(HISTORY)
    roster = set(pd.read_csv("data/fighters.csv")["name"].dropna().astype(str))
    kept, dropped, recovered = resolve(h, roster)

    if not dropped:
        print(f"{len(h)} rows, no duplicated fights.")
        return 0

    print(f"{len(h)} rows; {len(dropped)} duplicate row(s) to drop, "
          f"{recovered} method(s) recovered from a dropped row:")
    for i in dropped[:20]:
        print(f"   {str(h.at[i, 'date'])[:10]}  {h.at[i, 'fighter_a']} / {h.at[i, 'fighter_b']}")
    if len(dropped) > 20:
        print(f"   ... and {len(dropped) - 20} more")

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return 0
    kept.to_csv(HISTORY, index=False)
    print(f"\nWritten: {len(h)} -> {len(kept)} rows.")
    print("Re-run generate_site.py -- Elo is replayed from this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
