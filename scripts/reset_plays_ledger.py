"""
Delete the plays ledger for one event, on purpose, before anything settles.

WHY THIS IS A SCRIPT AND NOT A `rm`.

data/plays_ledger.csv is set-once by design and guarded by
scripts/check_plays_ledger.py, which fails the build if a published row is
rewritten or removed. That is exactly what it should do -- a record assembled
with hindsight is not a record.

But the guarantee has a legitimate escape: a STAKING RULE CHANGE, made before
a single play on that card has settled. When the rule that produced a row no
longer exists, republishing the card under the new rule is honest and leaving
the old rows is not -- they claim we would have made bets that the current
system would never make.

The line is settlement, and it is absolute. Any row with a result is a bet
that resolved; this refuses to touch those, and refuses the whole event if any
of them has. Past that point the only honest move is to leave the record alone
and let the new rule apply to the next card.

ALSO WHY IT EXISTS: .gitattributes points data/* at an "ours" merge driver, so
a hand-deleted ledger is silently restored from CI's copy on the next rebase.
It happened. A reset has to be a deliberate, repeatable operation rather than
a file edit that the next pull quietly undoes.

Run:  python3 scripts/reset_plays_ledger.py "UFC Fight Night: ..." --yes
      python3 scripts/reset_plays_ledger.py --all --yes
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plays_ledger import LEDGER_PATH, FIELDNAMES  # noqa: E402


def main(argv):
    args = [a for a in argv[1:] if a != "--yes"]
    confirmed = "--yes" in argv
    everything = "--all" in args
    event = next((a for a in args if not a.startswith("--")), None)

    if not everything and not event:
        print(__doc__)
        return 2
    if not os.path.exists(LEDGER_PATH):
        print(f"[reset] no {LEDGER_PATH} -- nothing to do")
        return 0

    with open(LEDGER_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    doomed = [r for r in rows if everything or r.get("event_name") == event]
    if not doomed:
        print(f"[reset] no rows for {event!r} -- nothing to do")
        return 0

    settled = [r for r in doomed if (r.get("result") or "").strip()]
    if settled:
        print(f"[reset] REFUSED: {len(settled)} of {len(doomed)} row(s) have already "
              f"settled.\n"
              f"        A resolved bet is a fact, not a draft. If the rule has "
              f"changed,\n        it applies to the next card.")
        for r in settled[:5]:
            print(f"          {r['result']:5s} {r['units']}U  {r['label']}")
        return 1

    keep = [r for r in rows if r not in doomed]
    print(f"[reset] {len(doomed)} unsettled row(s) would be removed, "
          f"{len(keep)} kept:")
    for r in doomed:
        print(f"          {r['units']}U  {r['label']}  {r['odds_american']}")
    if not confirmed:
        print("\n        Nothing written. Re-run with --yes to do it.")
        return 0

    with open(LEDGER_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in keep:
            w.writerow(r)
    print(f"\n[reset] done. The next build republishes the card under the "
          f"current rules.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
