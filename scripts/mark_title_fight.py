"""
Flag a bout as a TITLE FIGHT.

NOT COSMETIC. is_title_fight also drives is_five_round in card_matcher,
because a championship bout is scheduled for five rounds WHEREVER it sits on
the card. Before this flag existed that was derived from card_position alone,
so a title fight in the CO-MAIN slot -- which happens on any card carrying two
belts -- was modelled as three rounds. That silently corrupts its round
distribution, its finish probability and every Over/Under line built on them,
with nothing anywhere to indicate a problem. Setting this flag fixes the
model, and the badge is the visible side effect.

WHY IT'S MANUAL. Nothing in the pipeline carries a title indicator: ESPN's
scoreboard rows give fighters, weight class, card position and times, and
none of them say "championship". Title fights are also rare (one or two per
card) and known weeks ahead, so a hand flag costs almost nothing and is
right by construction, where an inferred one could quietly be wrong on the
most important fight of the night.

Usage (dry run first):
    python3 scripts/mark_title_fight.py "Makhachev" "Garry"
    python3 scripts/mark_title_fight.py "Makhachev" "Garry" --apply
    python3 scripts/mark_title_fight.py "Makhachev" "Garry" --unset --apply
"""

import argparse
import sys
import unicodedata

import pandas as pd

CARD_FILES = ["data/fight_cards.csv", "data/future_cards.csv"]


def _fold(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fighter_a")
    ap.add_argument("fighter_b")
    ap.add_argument("--unset", action="store_true", help="clear the flag instead of setting it")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    want = args.unset is False
    a, b = _fold(args.fighter_a), _fold(args.fighter_b)
    touched = 0

    for path in CARD_FILES:
        try:
            df = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        # Substring match on either corner, so a surname is enough -- but it
        # must match BOTH fighters, which makes a wrong bout very unlikely.
        mask = df.apply(
            lambda r: (a in _fold(r["fighter_a"]) or a in _fold(r["fighter_b"]))
            and (b in _fold(r["fighter_a"]) or b in _fold(r["fighter_b"])), axis=1)
        if not mask.any():
            continue
        for _, r in df[mask].iterrows():
            pos = r.get("card_position", "?")
            print(f"  {path}: {r['fighter_a']} vs {r['fighter_b']} ({pos})")
            if want and str(pos).strip() != "Main Event":
                print(f"     -> will also become FIVE ROUNDS (currently modelled as three, "
                      f"since it is not the Main Event)")
        touched += int(mask.sum())

        if args.apply:
            if "is_title_fight" not in df.columns:
                df["is_title_fight"] = None
            # Flag columns read back from CSV as float64 when empty, and
            # pandas raises assigning a bool into that.
            df["is_title_fight"] = df["is_title_fight"].astype("object")
            df.loc[mask, "is_title_fight"] = want
            df.to_csv(path, index=False)

    if not touched:
        print(f"No bout matching {args.fighter_a!r} vs {args.fighter_b!r} on any card.")
        sys.exit(1)
    if args.apply:
        print(f"\nSet is_title_fight={want} on {touched} row(s). Now run generate_site.py.")
    else:
        print(f"\nDRY RUN -- {touched} row(s) would change. Re-run with --apply.")


if __name__ == "__main__":
    main()
