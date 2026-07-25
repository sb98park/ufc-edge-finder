"""
One-time batch backfill: all 12 results from UFC Fight Night: Ankalaev vs.
Guskov (July 25, 2026), for the card whose results didn't auto-confirm
(see scripts/record_fight_result.py's docstring for why).

Writes to data/fight_results.csv exactly like the automated fetcher would
have -- each fight becomes indistinguishable from a real confirmation:
drops out of the live schedule, appears in Track Record on next generate.

Skips any fight that already has a result recorded (safe to re-run).

Usage:
  python3 scripts/backfill_ankalaev_guskov_card.py           # dry run
  python3 scripts/backfill_ankalaev_guskov_card.py --apply   # write
"""

import sys
import datetime as dt

import pandas as pd

FIGHT_RESULTS = "data/fight_results.csv"
EVENT = "UFC Fight Night: Ankalaev vs. Guskov"

# (fighter_a, fighter_b, winner, method, round, time)
RESULTS = [
    ("Magomed Ankalaev", "Bogdan Guskov", "Magomed Ankalaev", "KO/TKO", 5, "2:41"),
    ("Ramazan Temirov", "Steve Erceg", "Ramazan Temirov", "KO/TKO", 1, "4:21"),
    ("Magomed Zaynukov", "Damian Rzepecki", "Magomed Zaynukov", "DEC", 3, "5:00"),
    ("Rizvan Kuniev", "Tyrell Fortune", "Rizvan Kuniev", "KO/TKO", 3, "1:12"),
    ("Abubakar Vagaev", "Saygid Izagakhmaev", "Abubakar Vagaev", "DEC", 3, "5:00"),
    ("Valter Walker", "Thomas Petersen", "Valter Walker", "SUB", 1, "1:32"),
    ("Muhammad Said", "Dustin Jacoby", "Muhammad Said", "KO/TKO", 2, "4:49"),
    ("Sam Patterson", "Santiago Ponzinibbio", "Sam Patterson", "KO/TKO", 2, "3:06"),
    ("Axel Sola", "Ismael Bonfim", "Axel Sola", "SUB", 1, "4:44"),
    ("Magomed Tuchalov", "Brendson Ribeiro", "Magomed Tuchalov", "DEC", 3, "5:00"),
    ("Nurullo Aliev", "Mike Davis", "Nurullo Aliev", "DEC", 3, "5:00"),
    ("Abdul Hussein", "Cody Gibson", "Abdul Hussein", "SUB", 3, "3:07"),
]


def _norm(s) -> str:
    return str(s).strip().lower()


def main():
    apply = "--apply" in sys.argv
    results = pd.read_csv(FIGHT_RESULTS)

    existing_keys = {
        frozenset({_norm(r["fighter_a"]), _norm(r["fighter_b"])})
        for _, r in results.iterrows() if pd.notna(r.get("winner"))
    }

    new_rows = []
    for a, b, winner, method, rnd, time in RESULTS:
        key = frozenset({_norm(a), _norm(b)})
        if key in existing_keys:
            print(f"SKIP (already recorded): {a} vs {b}")
            continue
        row = {c: "" for c in results.columns}
        row.update({
            "event_name": EVENT, "fighter_a": a, "fighter_b": b,
            "winner": winner, "method": method, "end_round": rnd, "end_time": time,
            "date_added": dt.date.today().isoformat(),
        })
        new_rows.append(row)
        loser = b if winner == a else a
        print(f"{'WILL RECORD' if not apply else 'RECORDED'}: {winner} def. {loser} by {method}, R{rnd} {time}")

    if not new_rows:
        print("\nNothing to do -- every fight already has a result.")
        return

    if not apply:
        print(f"\nDRY RUN -- {len(new_rows)} result(s) above, nothing written. Re-run with --apply to write.")
        return

    results = pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)
    results.to_csv(FIGHT_RESULTS, index=False)
    print(f"\nDone -- {len(new_rows)} result(s) written. Now run generate_site.py and push.")


if __name__ == "__main__":
    main()
