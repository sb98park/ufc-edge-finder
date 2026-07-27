"""
Repair a wrongly-promoted "This Weekend" card.

WHY THIS IS NEEDED SEPARATELY FROM THE BUG FIX. promote_card_if_stale used
to hand off to future_cards.csv's FIRST ROW rather than the soonest event
by date, and that file is in discovery/append order (the dedupe and resync
helpers rewrite it by concatenating groups, never sorting). In production
that promoted a card 19 days out ahead of one happening the same weekend.

Fixing the logic does not undo an already-bad promotion: the wrong card now
sits in fight_cards.csv with a date in the FUTURE, so days_since is
negative and the corrected promotion never fires -- the site would stay
stuck on it for weeks. This script puts things back.

WHAT IT DOES: finds the soonest event across the current card and every
tracked future card. If the current card is already the soonest, it changes
nothing. Otherwise it moves the current card back into future_cards.csv and
promotes the soonest one, preserving every column in both files.

Usage:
    python3 scripts/fix_card_promotion.py            # dry run
    python3 scripts/fix_card_promotion.py --apply    # write both files
"""

import sys

import pandas as pd

CURRENT = "data/fight_cards.csv"
FUTURE = "data/future_cards.csv"


def _first_date(df, name):
    rows = df[df["event_name"] == name]
    return str(rows["event_date"].iloc[0]) if not rows.empty else None


def main():
    apply = "--apply" in sys.argv
    current = pd.read_csv(CURRENT)
    future = pd.read_csv(FUTURE)

    if current.empty:
        print("fight_cards.csv is empty -- nothing to repair.")
        return

    current_name = str(current["event_name"].iloc[0])
    current_date = str(current["event_date"].iloc[0])

    # Every candidate: the current card plus each distinct future event.
    candidates = [(current_name, current_date, "current")]
    for name in future["event_name"].dropna().unique():
        d = _first_date(future, str(name))
        if d:
            candidates.append((str(name), d, "future"))
    candidates.sort(key=lambda c: c[1])

    print("tracked events by date:")
    for name, date, where in candidates:
        mark = "  <-- currently shown as This Weekend" if where == "current" else ""
        print(f"  {date}  {name}{mark}")

    soonest_name, soonest_date, soonest_where = candidates[0]
    if soonest_where == "current":
        print(f"\nThe current card ({current_name}) IS the soonest. Nothing to repair.")
        return

    print(f"\nMISMATCH: showing {current_name!r} ({current_date}) but "
          f"{soonest_name!r} ({soonest_date}) comes first.")
    print(f"  -> move {current_name!r} back into future_cards.csv")
    print(f"  -> promote {soonest_name!r} to fight_cards.csv")

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to repair.")
        return

    new_current = future[future["event_name"] == soonest_name].reset_index(drop=True)
    remaining_future = future[future["event_name"] != soonest_name]
    # Columns can differ slightly between the two files (e.g. a `cancelled`
    # flag added to the current card); concat keeps the union rather than
    # silently dropping anything.
    new_future = pd.concat([remaining_future, current], ignore_index=True)

    new_current.to_csv(CURRENT, index=False)
    new_future.to_csv(FUTURE, index=False)
    print(f"\nDone -- This Weekend is now {soonest_name!r} ({soonest_date}).")
    print("Run generate_site.py, then commit data/fight_cards.csv and data/future_cards.csv.")
    print("\nNote: while the wrong card was current, the pipeline may have logged")
    print("predictions for its fights early. Those rows keep refreshing until the")
    print("fights resolve, but their recorded pick odds were captured sooner than")
    print("usual, which slightly affects CLV for that event only.")


if __name__ == "__main__":
    main()
