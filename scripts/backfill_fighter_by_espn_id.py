"""
Fill in a fighter's roster row from ESPN, addressed by ATHLETE ID.

WHY THIS EXISTS, and why it isn't a duplicate of ensure_roster_rows().
Both end up calling the same ESPN athlete endpoints. The difference is how
the athlete is FOUND. ensure_roster_rows resolves ids by fetching the
scoreboard for the event date and reading that event's competitors, so it can
only ever see fighters ESPN already lists on the card. When ESPN is late
publishing a bout -- which on one real card meant a four-days-out addition, a
one-day replacement, and an outright wrong opponent -- its fighters are
invisible to that route even though ESPN has complete profiles for them.
Addressing the athlete directly sidesteps the card entirely.

That gap is not academic. A roster row carrying only name and record leaves
slpm, sapm and takedown rate empty, and the ENTIRE adjustment layer of the
model reads those columns. With them missing the pick falls back to Elo
alone, with no striking, wrestling, style or form input -- while the site
still presents the result with a normal confidence label. Pulling the real
numbers is the fix. (Discounting the confidence instead is NOT: signals based
on how complete a fighter's data is were tested on this project and failed,
because completeness also measures how established a fighter is, and
matchmaking dominates the result.)

WHAT THIS DOES NOT FIX: Elo. Ratings are replayed from fight_history.csv, so
a fighter whose bouts aren't in that file sits at the default 1500 no matter
how complete this row is. For a genuine debutant that default is correct and
validated. For someone with UFC bouts that history simply hasn't caught up
on, it is not -- so check fight_history.csv separately rather than assuming
this script settled it.

FINDING THE ID: open the fighter on ESPN; the URL is
  espn.com/mma/fighter/_/id/<ID>/<name>
and the number is what this wants.

Usage:
  python3 scripts/backfill_fighter_by_espn_id.py --name "Gianni Vazquez" --espn-id 5158441
  python3 scripts/backfill_fighter_by_espn_id.py --name "Gianni Vazquez" --espn-id 5158441 --apply
"""

import argparse
import os
import sys
import unicodedata

import pandas as pd

# Python puts the SCRIPT's directory on sys.path, not the directory you ran
# from, so `from src...` fails when this is invoked as
# `python3 scripts/backfill_fighter_by_espn_id.py` from the project root --
# which is how every other script here is run. Adding the project root
# explicitly keeps that invocation working without a PYTHONPATH prefix.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fighter_backfill import (
    _fetch_espn_athlete_detail,
    _fetch_espn_method_records,
    _fetch_last_fight_from_events_map,
)

FIGHTERS = "data/fighters.csv"


def _fold(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--name", required=True, help="name EXACTLY as it appears in fighters.csv")
    ap.add_argument("--espn-id", required=True)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace values already present; default fills only blanks")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    fighters = pd.read_csv(FIGHTERS)
    mask = fighters["name"].map(_fold) == _fold(args.name)
    if not mask.any():
        print(f"No row in {FIGHTERS} for {args.name!r}. Create it first "
              f"(scripts/add_fight_manually.py --create-roster-row).")
        sys.exit(1)

    print(f"Fetching ESPN athlete {args.espn_id} ...")
    physical, _ = _fetch_espn_athlete_detail(args.espn_id)
    recs = _fetch_espn_method_records(args.espn_id)
    last = _fetch_last_fight_from_events_map(args.espn_id, args.name)

    fetched = {}
    fetched.update(physical or {})
    career_w, career_l = recs.pop("_career_w", None), recs.pop("_career_l", None)
    if career_w is not None:
        fetched["wins"], fetched["losses"] = career_w, career_l
    fetched.update(recs or {})
    fetched.update(last or {})
    fetched.pop("name", None)

    if not fetched:
        print("ESPN returned nothing usable for that id. Check the id is right "
              "and that it's an MMA athlete.")
        sys.exit(1)

    idx = fighters.index[mask][0]
    changes = []
    for col, new in fetched.items():
        if new is None or (isinstance(new, str) and not new.strip()):
            continue
        current = fighters.at[idx, col] if col in fighters.columns else None
        blank = current is None or (isinstance(current, float) and pd.isna(current)) or str(current).strip() == ""
        if not blank and not args.overwrite:
            if str(current) != str(new):
                # Surfaced rather than silently skipped: a disagreement between
                # a hand-entered value and ESPN is exactly the thing worth
                # seeing, and it's how the record entered from press reports
                # gets caught if ESPN counts it differently.
                print(f"    (kept) {col}: {current}   [ESPN says {new} -- rerun with --overwrite to take it]")
            continue
        changes.append((col, current if not blank else "", new))

    if not changes:
        print("Nothing to fill -- every field ESPN returned is already populated.")
        return

    print(f"\nWould update {fighters.at[idx, 'name']}:")
    for col, old, new in changes:
        print(f"    {col}: {old!r} -> {new!r}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to write.")
        return

    for col, _, new in changes:
        if col not in fighters.columns:
            fighters[col] = None
        fighters[col] = fighters[col].astype("object")
        fighters.at[idx, col] = new
    fighters.to_csv(FIGHTERS, index=False)
    print(f"\nUpdated {FIGHTERS}. Now run generate_site.py.")


if __name__ == "__main__":
    main()
