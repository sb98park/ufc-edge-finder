"""
One-time direct fix for the stale event name (July 2026) - the event is
still titled "UFC Fight Night: Ankalaev vs. Rountree Jr." throughout
fight_cards.csv/future_cards.csv, left over from before Rountree Jr. was
replaced by Bogdan Guskov as Ankalaev's opponent. The real, current name
is "UFC Fight Night: Ankalaev vs. Guskov".

Run this once against the real, live data files:
    python scripts/fix_event_name.py

Renames the event_name field on every row that currently has the stale
name, in both fight_cards.csv and future_cards.csv. Does not touch any
other column or any row with a different event_name.
"""
import pandas as pd

OLD_NAME = "UFC Fight Night: Ankalaev vs. Rountree Jr."
NEW_NAME = "UFC Fight Night: Ankalaev vs. Guskov"

FILES_TO_CHECK = ["data/fight_cards.csv", "data/future_cards.csv"]


def main():
    for path in FILES_TO_CHECK:
        try:
            df = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"[fix_event_name] {path} not found or empty -- skipping")
            continue

        mask = df["event_name"] == OLD_NAME
        count = int(mask.sum())
        if count == 0:
            print(f"[fix_event_name] no rows with the stale name in {path} -- nothing to fix there")
            continue

        df.loc[mask, "event_name"] = NEW_NAME
        df.to_csv(path, index=False)
        print(f"[fix_event_name] {path}: renamed {count} row(s) from {OLD_NAME!r} to {NEW_NAME!r}")


if __name__ == "__main__":
    main()
