"""
Record a Double Chance / goes-the-distance price seen in a sportsbook app.

WHY THIS IS MANUAL. The open question is narrow and specific: a book quoting
Double Chance as its OWN two-way market should price it near a two-way margin
(5-8%) rather than the 21.8% the six-cell method grid charges, and at 5-8% the
measured 4-point finish-overpricing bias is not obviously dead. Nothing
automated can answer it:

  - data/external_odds.csv has no Double Chance quote anywhere, across 7,177
    bouts. The archive cannot be made to answer it.
  - TheRundown catalogues method_of_victory_double_chance (1371) and
    fight_to_start_round (1369) for sport 7 and returns NO DATA for either;
    that was checked against a live card, not assumed.
  - Polymarket lists the markets but they sit untraded at 0.5/0.5 with no
    volume, so live_props drops them before they reach anything.

So the only source of a real two-way book price on these markets is a person
looking at the app. Twenty seconds a week, and in a few months the question
that has blocked every other line of enquiry becomes answerable.

Prices land in the same ledger as the automatic ones, tagged with their book
and is_vig_free=False, so scripts/grade_prop_prices.py settles and reports
them alongside everything else -- and reports the vig-free and book-priced
cohorts SEPARATELY, because pooling them is what this project keeps having to
undo.

USAGE

  python3 scripts/record_book_price.py \\
      --fighters "Anthony Hernandez" "Gregory Rodrigues" \\
      --market double_chance --selection "Anthony Hernandez by KO/TKO or SUB" \\
      --price -155 --book DraftKings

  Markets:  double_chance | goes_distance | round_start | method | total_rounds
  --price is American odds. --book is free text; use the app's real name.
  --event and --date default to the tracked card in data/fight_cards.csv.

Recording BOTH SIDES of a two-way market when the app shows them is worth the
extra ten seconds: the pair is what measures the actual margin, which is the
entire point of the exercise.
"""

import argparse
import sys

import pandas as pd

sys.path.insert(0, ".")

from src.prop_ledger import LEDGER_PATH, record_prop_prices

# The ledger keys on `market`, and grade_prop_prices settles on those exact
# strings -- so these map to the vocabulary the grader already understands
# rather than inventing a parallel one.
MARKET_NAMES = {
    "double_chance": "Method: DoubleChance",
    "goes_distance": "GoesTheDistance",
    "round_start": "RoundStart",
    "method": "Method",
    "total_rounds": "TotalRounds",
}


def tracked_card():
    try:
        c = pd.read_csv("data/fight_cards.csv")
        if not c.empty:
            return str(c.event_name.iloc[0]), str(c.event_date.iloc[0])
    except Exception:
        pass
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fighters", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--market", required=True, choices=sorted(MARKET_NAMES))
    ap.add_argument("--selection", required=True,
                    help='Exactly what the bet is, e.g. "Anthony Hernandez by KO/TKO or SUB"')
    ap.add_argument("--price", required=True, type=float, help="American odds")
    ap.add_argument("--book", required=True, help="DraftKings, FanDuel, ...")
    ap.add_argument("--selection-method", default="", help="KO/TKO, SUB, 2.5, ...")
    ap.add_argument("--event", default=None)
    ap.add_argument("--date", default=None)
    a = ap.parse_args()

    event, date = tracked_card()
    event, date = a.event or event, a.date or date
    fa, fb = a.fighters

    row = {
        "fight_id": f"{fa}|{fb}",
        "fighter_a": fa,
        "fighter_b": fb,
        "market": MARKET_NAMES[a.market],
        "selection": a.selection,
        "selection_method": a.selection_method,
        "odds_american": a.price,
        "source": a.book,
        # THE WHOLE REASON THIS SCRIPT EXISTS. A book quote carries margin;
        # the automatic Polymarket rows do not. The grader splits on this and
        # the open question is only about the False side.
        "source_is_vig_free": False,
    }
    total = record_prop_prices([row], event, date)
    if total:
        print(f"recorded: {a.book} {a.price:+.0f} on {a.selection}")
        print(f"          {fa} vs {fb}  ({event or 'unknown event'}, {date or 'unknown date'})")
        print(f"          {LEDGER_PATH} now holds {total} quote(s)")
        print("\nSettle and report with:  python3 scripts/grade_prop_prices.py")
    else:
        print("nothing recorded -- see the error above")


if __name__ == "__main__":
    main()
