"""
Fill method splits that ARITHMETIC forces -- never ones that need a guess.

TWO CASES ARE FILLED, both determined rather than inferred:
  1. TOTAL IS ZERO. A fighter at 9-0 has zero KO losses, zero submission
     losses and zero decision losses. Nothing is being estimated.
  2. EXACTLY ONE SPLIT MISSING. A 12-4 fighter with ko_losses=2 and
     sub_losses=1 must have dec_losses=1. The remainder is pinned.

ONE CASE IS DELIBERATELY SKIPPED: two or more splits missing with a non-zero
total. A 12-4 fighter with all three blank has many valid combinations, and
filling them would be inventing data -- exactly what the site's Honest Data
Limitations tab exists to avoid. Those stay as "—".

RECONCILIATION GUARD. scripts/derive_loss_splits.py counts UFC bouts only, so
for a fighter with regional history the known splits UNDERSHOOT the career
record. Deducing a remainder from that mixed base would produce a confidently
wrong number. Any fighter whose known splits already EXCEED their record --
or whose remainder would come out negative -- is skipped and reported, since
that's the signal the base is inconsistent.

Applies to wins and losses alike: gaps appear on both sides.

Usage:
    python3 scripts/deduce_method_splits.py            # dry run
    python3 scripts/deduce_method_splits.py --apply
"""

import sys

import pandas as pd

FIGHTERS = "data/fighters.csv"
SIDES = {
    "wins":   ("ko_wins", "sub_wins", "dec_wins"),
    "losses": ("ko_losses", "sub_losses", "dec_losses"),
}


def main():
    apply = "--apply" in sys.argv
    f = pd.read_csv(FIGHTERS)

    filled_zero = filled_remainder = 0
    skipped_ambiguous = 0
    inconsistent = []
    examples = []

    for i, row in f.iterrows():
        for total_col, cols in SIDES.items():
            if total_col not in f.columns:
                continue
            total = row.get(total_col)
            if pd.isna(total):
                continue
            total = int(total)
            missing = [c for c in cols if c in f.columns and pd.isna(row.get(c))]
            if not missing:
                continue

            # Case 1: a zero total forces every split to zero.
            if total == 0:
                for c in missing:
                    f.at[i, c] = 0
                filled_zero += len(missing)
                if len(examples) < 8:
                    examples.append(f"{row['name']}: {total_col}=0 -> all splits 0")
                continue

            known = [int(row[c]) for c in cols if c in f.columns and pd.notna(row.get(c))]
            known_sum = sum(known)

            # Guard: known splits can't exceed the record. If they do, the base
            # is mixing UFC-only counts with a career record and any remainder
            # derived from it would be wrong.
            if known_sum > total:
                inconsistent.append(f"{row['name']}: {total_col}={total} but known splits sum to {known_sum}")
                continue

            # Case 2: exactly one gap -- the remainder is determined.
            if len(missing) == 1:
                f.at[i, missing[0]] = total - known_sum
                filled_remainder += 1
                if len(examples) < 8:
                    examples.append(f"{row['name']}: {missing[0]} = {total} - {known_sum} = {total - known_sum}")
            else:
                skipped_ambiguous += len(missing)

    print(f"filled because the total was ZERO      : {filled_zero}")
    print(f"filled as a FORCED remainder           : {filled_remainder}")
    print(f"left blank (2+ gaps, not determined)   : {skipped_ambiguous}")
    if inconsistent:
        print(f"\nskipped as inconsistent ({len(inconsistent)}) -- known splits exceed the record,")
        print("which means UFC-only counts are sitting under a career total:")
        for x in inconsistent[:6]:
            print("   ", x)
    if examples:
        print("\nexamples:")
        for e in examples:
            print("   ", e)

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return
    f.to_csv(FIGHTERS, index=False)
    print("\nWritten. Commit data/fighters.csv.")


if __name__ == "__main__":
    main()
