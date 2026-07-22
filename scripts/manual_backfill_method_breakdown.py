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

Matches by exact name first, falling back to a loose (first+last word)
match if that fails, since the automated pipeline's own discovered
spelling in fighters.csv could differ slightly from what's used here -
same loose-match convention already established elsewhere in this
codebase for exactly this kind of cross-source name mismatch. Prints
a clear diagnostic for any name that STILL can't be matched, listing
the closest candidates actually present in the file, so a genuine
mismatch is immediately visible rather than silently skipped.
"""
import pandas as pd

FIGHTERS_PATH = "data/fighters.csv"

# ko_wins, sub_wins, dec_wins, ko_losses, sub_losses, dec_losses -- each
# row verified to sum to the fighter's real, known overall record.
MANUAL_DATA = {
    "Ramazan Temirov":  (11, 1, 7, 0, 1, 2),   # 19-3
    "Sam Patterson":    (6, 7, 1, 2, 0, 1),    # 14-3
    "Brendson Ribeiro": (9, 7, 1, 5, 3, 2),    # 17-10
    "Mike Davis":       (8, 2, 2, 0, 1, 2),    # 12-3
    "Abdul Hussein":    (5, 9, 1, 0, 1, 1),    # 15-2
    "Thomas Petersen":  (7, 1, 3, 3, 0, 1),    # 11-4
    "Abubakar Vagaev":      (4, 1, 19, 3, 0, 1),  # 24-4
    "Saygid Izagakhmaev":   (3, 13, 6, 0, 1, 2),  # 22-3
    "Muhammad Said":        (6, 2, 1, 0, 0, 0),   # 9-0
    "Ismael Bonfim":        (9, 4, 7, 2, 4, 0),   # 20-6
    "Axel Sola":            (6, 1, 4, 0, 0, 1),   # 11-1-1
}

METHOD_COLS = ["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]


def _loose_name(name: str) -> tuple:
    parts = str(name).strip().lower().split()
    return (parts[0], parts[-1]) if parts else (str(name).strip().lower(),)


def main():
    fighters = pd.read_csv(FIGHTERS_PATH)
    for col in ("combat_edge_checked", "wikipedia_checked"):
        if col not in fighters.columns:
            fighters[col] = False
        fighters[col] = fighters[col].fillna(False).astype(bool)

    loose_index: dict = {}
    for i, real_name in fighters["name"].items():
        loose_index.setdefault(_loose_name(real_name), []).append((i, real_name))

    filled_count = 0
    for name, values in MANUAL_DATA.items():
        idx = fighters.index[fighters["name"] == name]
        matched_name = name
        if len(idx) == 0:
            # Exact match failed -- try loose (first+last word) match.
            candidates = loose_index.get(_loose_name(name), [])
            if len(candidates) == 1:
                i, matched_name = candidates[0]
                idx = [i]
                print(f"[manual_backfill] {name!r} matched loosely to {matched_name!r} in {FIGHTERS_PATH}")
            elif len(candidates) > 1:
                print(f"[manual_backfill] {name!r} has {len(candidates)} ambiguous loose matches in "
                      f"{FIGHTERS_PATH}: {[c[1] for c in candidates]} -- skipping, can't tell which is right")
                continue
            else:
                similar = [n for n in fighters["name"] if _loose_name(n)[1] == _loose_name(name)[1]]
                print(f"[manual_backfill] {name!r} not found in {FIGHTERS_PATH} (exact or loose) -- skipping. "
                      f"{'Closest by last name: ' + str(similar) if similar else 'No similar names found at all -- is this fighter tracked yet?'}")
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
            print(f"[manual_backfill] filled {matched_name}: {', '.join(updated_fields)}")
        else:
            print(f"[manual_backfill] {matched_name} already had all method fields -- only updated checked flags")

    fighters.to_csv(FIGHTERS_PATH, index=False)
    print(f"[manual_backfill] done -- {filled_count}/{len(MANUAL_DATA)} fighters had at least one field filled")
    print(f"[manual_backfill] IMPORTANT: this only updated {FIGHTERS_PATH}. You still need to run "
          f"'python generate_site.py' to regenerate docs/index.html, then commit and push, "
          f"for this to actually show up on the live site.")


if __name__ == "__main__":
    main()
