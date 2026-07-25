"""
Manually record a fight result the automated fetcher hasn't picked up.

results_fetcher.py deliberately refuses to log a partial result (a winner
with no confidently-parsed method) so it can keep retrying rather than
silently going stale with a blank method -- see _fetch_from_espn's
docstring. That's the right default, but it means a fight whose method
text just never parses cleanly (an unusual finish description, an API
quirk for that one bout) can stay "unconfirmed" indefinitely: missing
from Track Record and, per apply_live_corrections' stuck-fight handling,
silently excluded from the live/next countdown too.

This writes a real row to fight_results.csv directly -- the same file
the automated fetcher writes to -- so once used, the fight is confirmed
exactly as if the fetcher had succeeded on its own: it drops out of the
live schedule and appears in Track Record on the next generate.

Only the fields Track Record and the schedule actually use are required;
the rest (strike/takedown breakdowns) are optional and left blank if
not supplied -- Track Record doesn't need them, only the fight-detail
stat panels do, and those simply show less detail for a manually-entered
result rather than showing something wrong.

Usage:
  python3 scripts/record_fight_result.py \
      --event "UFC Fight Night: Ankalaev vs. Rountree Jr." \
      --a "Cody Gibson" --b "Abdul Hussein" \
      --winner "Cody Gibson" --method "KO/TKO" \
      [--round 2] [--time "3:14"]              # dry run
  # add --apply to write
"""

import argparse
import datetime as dt

import pandas as pd

FIGHT_RESULTS = "data/fight_results.csv"


def _norm(s) -> str:
    return str(s).strip().lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", required=True)
    p.add_argument("--a", required=True, help="fighter_a exactly as it appears in fight_cards.csv")
    p.add_argument("--b", required=True, help="fighter_b exactly as it appears in fight_cards.csv")
    p.add_argument("--winner", required=True)
    p.add_argument("--method", required=True, help="e.g. KO/TKO, SUB, DEC, DQ")
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--time", default=None, help="e.g. 3:14")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    if args.winner.strip().lower() not in (args.a.strip().lower(), args.b.strip().lower()):
        print(f"--winner {args.winner!r} doesn't match --a or --b exactly -- check spelling.")
        return

    results = pd.read_csv(FIGHT_RESULTS)
    already = results.apply(
        lambda r: {_norm(r["fighter_a"]), _norm(r["fighter_b"])} == {_norm(args.a), _norm(args.b)}, axis=1
    )
    if already.any():
        print(f"A result for {args.a} vs {args.b} already exists -- not touching it. "
              f"If the automated fetcher already caught this, no action needed.")
        return

    row = {c: "" for c in results.columns}
    row.update({
        "event_name": args.event, "fighter_a": args.a, "fighter_b": args.b,
        "winner": args.winner, "method": args.method,
        "end_round": args.round if args.round is not None else "",
        "end_time": args.time or "",
        "date_added": dt.date.today().isoformat(),
    })
    print(f"Will record: {args.winner} def. {args.a if args.winner != args.a else args.b} "
          f"by {args.method}" + (f", R{args.round} {args.time}" if args.round else ""))

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to write.")
        return

    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    results.to_csv(FIGHT_RESULTS, index=False)
    print("\nDone -- now run generate_site.py and push.")


if __name__ == "__main__":
    main()
