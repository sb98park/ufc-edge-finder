"""
Set reach / height / age on roster rows that no automated source carries.

WHY THIS EXISTS. check_card_data_coverage flags carded fighters missing
physicals; for genuine debutants ESPN has no reach at all and UFC.com's bio
carries only Status/Age/Octagon Debut. Someone has to read it off a fighter
database and type it in. This is where that goes, so the provenance lives in
version control instead of in a shell history.

WHY DOB AND NOT AGE. fighters.csv stores `age` as a frozen number, and
fighter_backfill only refills a gap column when it is NaN -- so a hand-typed
age is never corrected and is wrong the day after the fighter's next
birthday. Recording the birthdate instead makes this script idempotent:
re-run it any time and every age is recomputed against today.

REACH MATTERS MORE THAN IT LOOKS. power_rating adds 4.0 * (reach_in - 70),
so a missing reach is not neutral-ish -- it lands exactly on the term's
centring constant and contributes precisely zero. Four rating points an inch.

Usage:
    python3 scripts/add_physicals_manually.py            # dry run
    python3 scripts/add_physicals_manually.py --apply
"""

import datetime as dt
import sys

import pandas as pd

FIGHTERS = "data/fighters.csv"

# name -> {reach_in, height_in, dob}. Omit a key that is genuinely unknown;
# leaving it NaN keeps the fighter on the coverage alarm, which is correct --
# we should keep being told, not quietly record a guess.
PHYSICALS = {
    # Read off Tapology by the owner, 2026-08-31.
    "Michael Aljarouj":   {"reach_in": 68.0, "height_in": 67.0, "dob": "1997-08-21"},
    "Rodrigo Vera":       {"reach_in": 68.0},
    # Not on Tapology or Sherdog; birthdate from a general web search, so the
    # provenance is weaker than the rest of this table. Reach still unknown.
    "Delphine Benouaich": {"dob": "1995-03-21"},
    # Checked and genuinely unavailable anywhere as of 2026-08-31 -- listed so
    # the next person does not repeat the search:
    #   Fabia Sintes          reach
    #   Ilimbek Akylbek Uulu  reach
    #   Mehemmedeli Osmanli   reach
}

# Matches fighter_backfill's own convention (integer years, //365) so a
# hand-set age and a scraped one are computed the same way.
def _age_from(dob: str, today: dt.date) -> int:
    return (today - dt.date.fromisoformat(dob)).days // 365


def main() -> int:
    apply = "--apply" in sys.argv
    today = dt.date.today()
    f = pd.read_csv(FIGHTERS)

    changes = []
    for name, vals in PHYSICALS.items():
        hit = f.index[f["name"] == name]
        if len(hit) != 1:
            print(f"  SKIP {name!r}: matched {len(hit)} roster rows")
            continue
        i = hit[0]
        wanted = {k: v for k, v in vals.items() if k != "dob"}
        if "dob" in vals:
            wanted["age"] = float(_age_from(vals["dob"], today))
        for col, new in wanted.items():
            old = f.at[i, col]
            if pd.notna(old) and float(old) == float(new):
                continue
            changes.append((name, col, old, new, i))

    if not changes:
        print("Nothing to change -- every value already matches.")
        return 0
    for name, col, old, new, _ in changes:
        was = "blank" if pd.isna(old) else f"{old:g}"
        print(f"  {name:22s} {col:10s} {was:>7s} -> {new:g}")

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return 0
    for _, col, _, new, i in changes:
        f.at[i, col] = new
    f.to_csv(FIGHTERS, index=False)
    print(f"\nWritten: {len(changes)} value(s) into {FIGHTERS}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
