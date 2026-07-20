"""
One-time direct removal of a specific stale fight row (July 2026) -
Umar Nurmagomedov vs David Martinez, a pre-replacement pairing still
lingering in the tracked data (Nurmagomedov's real, current opponent is
Song Yadong, confirmed via the site's own displayed data elsewhere).

Run this once against the real, live data files:
    python scripts/remove_stale_fight.py

Checks both fight_cards.csv (the active card) and future_cards.csv,
since either could be holding a lingering copy. Matches by fighter
pair, order-independent (fighter_a/fighter_b assignment isn't always
identical between sources), and only removes an exact match - never
touches anything else.
"""
import pandas as pd

TARGET_PAIR = {"umar nurmagomedov", "david martinez"}

FILES_TO_CHECK = ["data/fight_cards.csv", "data/future_cards.csv"]


def _pair(row) -> set:
    return {str(row["fighter_a"]).strip().lower(), str(row["fighter_b"]).strip().lower()}


def main():
    for path in FILES_TO_CHECK:
        try:
            df = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"[remove_stale_fight] {path} not found or empty -- skipping")
            continue

        mask = df.apply(lambda r: _pair(r) == TARGET_PAIR, axis=1)
        matched = df[mask]
        if matched.empty:
            print(f"[remove_stale_fight] no match in {path} -- nothing to remove there")
            continue

        for _, row in matched.iterrows():
            print(f"[remove_stale_fight] removing from {path}: {row['fighter_a']} vs {row['fighter_b']} "
                  f"({row['event_name']}, {row['card_position']})")

        df[~mask].to_csv(path, index=False)
        print(f"[remove_stale_fight] {path} updated -- {len(matched)} row(s) removed")


if __name__ == "__main__":
    main()
