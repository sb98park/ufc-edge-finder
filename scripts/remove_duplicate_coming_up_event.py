"""
One-time direct fix for a duplicate event tracked in "Coming Up" (July
2026) - "UFC Fight Night: Ankalaev vs. Guskov" is the SAME real event as
the active "This Weekend" card, just independently discovered and
tracked in future_cards.csv before the event-name rename fix existed
(at that point the active card was still misnamed "...vs. Rountree
Jr.", so the automated pipeline's own new-event discovery didn't
recognize "...vs. Guskov" as already-tracked and added it as if it were
a separate, later event).

Run this once against the real, live data file:
    python scripts/remove_duplicate_coming_up_event.py

Removes every row in future_cards.csv whose event_name is the Guskov
card, since that event is now correctly tracked as the active card in
fight_cards.csv and has no business also appearing in Coming Up.
"""
import pandas as pd

DUPLICATE_EVENT_NAME = "UFC Fight Night: Ankalaev vs. Guskov"

FUTURE_CARDS_PATH = "data/future_cards.csv"


def main():
    try:
        df = pd.read_csv(FUTURE_CARDS_PATH)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"[remove_duplicate_coming_up_event] {FUTURE_CARDS_PATH} not found or empty -- skipping")
        return

    mask = df["event_name"] == DUPLICATE_EVENT_NAME
    count = int(mask.sum())
    if count == 0:
        print(f"[remove_duplicate_coming_up_event] no rows for {DUPLICATE_EVENT_NAME!r} in "
              f"{FUTURE_CARDS_PATH} -- nothing to remove")
        return

    df[~mask].to_csv(FUTURE_CARDS_PATH, index=False)
    print(f"[remove_duplicate_coming_up_event] {FUTURE_CARDS_PATH}: removed {count} row(s) for "
          f"{DUPLICATE_EVENT_NAME!r} (duplicate of the active card)")


if __name__ == "__main__":
    main()
