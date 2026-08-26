"""
Write a play into the ledger at the price it was actually taken at.

WHY THIS EXISTS. The ledger is already set-once: record_plays keeps every
field at the value it was first written with and only advances last_seen and
closing_odds, so a play published on Tuesday at -400 still reads -400 on
Saturday when the board says -700. That part works and was verified.

What it cannot do is reach backwards. The price a row freezes at is the price
the pipeline happened to see on the render that first created it, and if a
row is removed and rebuilt -- which is what happened to the two moneylines
for Nurmagomedov vs. Song -- it re-freezes at whatever the board says now.
Both were rebuilt after the market had moved several hundred points, and
neither DraftKings opener survives anywhere in our own data: odds_snapshot is
a thirty-point ring buffer with no venue attribution, and prop_price_log
excludes Moneyline by design.

So the opening price is not recoverable by measurement, and this script does
not pretend otherwise. It takes the price as an argument, from whoever placed
the bet, and writes it as the published price with published_at set to when
the pick was actually made rather than to now.

PROVENANCE LIVES IN GIT, not in a column. Adding one would mean classifying
it in src/tiering.py and teaching every reader about it, for a field that is
blank on all but a handful of rows. Anything written by this script should be
committed on its own with the source of the number in the message.

IT WILL NOT OVERWRITE. A row that already exists is already frozen and this
refuses to touch it, which is the whole point of the file. Delete the row
first if it genuinely needs rewriting, and say why in the commit.
"""

import argparse
import csv
import sys

sys.path.insert(0, ".")

from src.plays import HURDLE_MONEYLINE, decimal_odds, ev_per_unit, required_prob
from src.plays_ledger import FIELDNAMES, LEDGER_PATH, load, play_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", required=True)
    ap.add_argument("--date", required=True, help="event date, YYYY-MM-DD")
    ap.add_argument("--fighter-a", required=True)
    ap.add_argument("--fighter-b", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--market", default="Moneyline")
    ap.add_argument("--price", type=float, required=True, help="American, as taken")
    ap.add_argument("--venue", required=True)
    ap.add_argument("--units", type=float, required=True)
    ap.add_argument("--at", required=True, help="when the pick was made")
    ap.add_argument("--model", type=float, required=True)
    ap.add_argument("--fair", type=float, required=True)
    ap.add_argument("--blend", type=float, required=True)
    ap.add_argument("--tier", default="")
    ap.add_argument("--card-position", default="")
    ap.add_argument("--weight-class", default="")
    ap.add_argument("--path", default=LEDGER_PATH)
    a = ap.parse_args()

    # THE SAME INVARIANT card_plays STAKES ON. A hand-written row is exactly
    # as capable of carrying numbers from two different passes as a generated
    # one, and this is the file most likely to be run in a hurry.
    if not (min(a.model, a.fair) - 1e-6 <= a.blend <= max(a.model, a.fair) + 1e-6):
        print(f"refusing: blend {a.blend} is outside [{min(a.model, a.fair)}, "
              f"{max(a.model, a.fair)}] -- these did not come from one pass")
        return 1

    pid = play_id(a.event, a.fighter_a, a.fighter_b, a.market, a.selection)
    rows = load(a.path)
    if any(r.get("play_id") == pid for r in rows):
        print(f"refusing: {pid} is already in the ledger and already frozen")
        return 1

    row = {
        "play_id": pid, "event_name": a.event, "event_date": a.date,
        "fighter_a": a.fighter_a, "fighter_b": a.fighter_b,
        "card_position": a.card_position, "weight_class": a.weight_class,
        "axis": "outcome", "market": a.market, "selection": a.selection,
        "label": f"{a.selection} {a.market}", "tier": a.tier,
        "is_lock": "0", "is_prop": "0",
        "odds_american": round(a.price), "venue": a.venue, "units": a.units,
        "to_win": round(a.units * (decimal_odds(a.price) - 1.0), 2),
        "model_prob": a.model, "fair_prob": a.fair, "blended_prob": a.blend,
        "ev_per_unit": round(ev_per_unit(a.blend, a.price), 4),
        "required_prob": round(required_prob(a.price, HURDLE_MONEYLINE), 4),
        "published_at": a.at, "last_seen": a.at, "closing_odds": "",
        "result": "", "units_result": "", "graded_at": "", "void_reason": "",
    }

    with open(a.path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow(row)

    print(f"froze {a.selection} {a.market} at {row['odds_american']:+d} ({a.venue}), "
          f"{a.units}U to win {row['to_win']}, published {a.at}")
    print(f"  ev/unit {row['ev_per_unit']}  required {row['required_prob']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
