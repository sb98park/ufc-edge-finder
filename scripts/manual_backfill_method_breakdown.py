"""
One-time manual backfill for fighters Combat Edge and Wikipedia couldn't
reach automatically (July 2026) - Combat Edge has been consistently
blocked for this project's GitHub Actions IP range, and these specific
fighters don't have a Wikipedia article. Data researched directly and
cross-validated against Sherdog.com and each fighter's own known overall
record - every row below sums to the real win/loss total.

Run this once against the real, live data/fighters.csv:
    python scripts/manual_backfill_method_breakdown.py

Only fills cells that are genuinely still empty (same "never overwrite
real data" rule the automated backfill follows) and marks both sources
as checked for these fighters, so the automated pipeline doesn't waste
future budget re-attempting them now that they have real, confirmed
numbers - consistent with the exhaustion-tracking already built for
this exact purpose.
"""
import pandas as pd

FIGHTERS_PATH = "data/fighters.csv"

# ko_wins, sub_wins, dec_wins, ko_losses, sub_losses, dec_losses -- each
# row verified to sum to the fighter's real, known overall record.
MANUAL_DATA = {
    "Ramazan Temirov":  (11, 1, 7, 0, 1, 2),   # 19-3
    "Sam Patterson":    (6, 7, 1, 2, 0, 1),    # 14-3
    "Brendson Ribeiro": (9, 7, 1, 5, 3, 2),    # 17-10
    "Magomed Tuchalov": (4, 1, 0, 0, 0, 0),    # 5-0
    "Mike Davis":       (8, 2, 2, 0, 1, 2),    # 12-3
    "Abdul Hussein":    (5, 9, 1, 0, 1, 1),    # 15-2
}

METHOD_COLS = ["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]


def main():
    fighters = pd.read_csv(FIGHTERS_PATH)
    for col in ("combat_edge_checked", "wikipedia_checked"):
        if col not in fighters.columns:
            fighters[col] = False
        fighters[col] = fighters[col].fillna(False).astype(bool)

    filled_count = 0
    for name, values in MANUAL_DATA.items():
        idx = fighters.index[fighters["name"] == name]
        if len(idx) == 0:
            print(f"[manual_backfill] {name!r} not found in {FIGHTERS_PATH} -- skipping (not yet tracked?)")
            continue
        i = idx[0]
        updated_fields = []
        for col, val in zip(METHOD_COLS, values):
            if pd.isna(fighters.at[i, col]):
                fighters.at[i, col] = val
                updated_fields.append(col)
        # Mark both sources checked regardless, so the automated pipeline
        # stops re-attempting this fighter now that real data is in place.
        fighters.at[i, "combat_edge_checked"] = True
        fighters.at[i, "wikipedia_checked"] = True
        if updated_fields:
            filled_count += 1
            print(f"[manual_backfill] filled {name}: {', '.join(updated_fields)}")
        else:
            print(f"[manual_backfill] {name} already had all method fields -- only updated checked flags")

    fighters.to_csv(FIGHTERS_PATH, index=False)
    print(f"[manual_backfill] done -- {filled_count}/{len(MANUAL_DATA)} fighters had at least one field filled")


if __name__ == "__main__":
    main()
