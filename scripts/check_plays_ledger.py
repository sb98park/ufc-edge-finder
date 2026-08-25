"""
Fail the build if a published play has been restated.

WHAT THIS PROTECTS. data/plays_ledger.csv records what the model said it was
betting AT THE MOMENT IT SAID IT -- the price, the stake Kelly wanted at that
price, and the probabilities behind both. Every one of those is set once and
never rewritten, because a ledger that moves with the line is not a record of
bets, it is a record of hindsight. src/plays_ledger enforces that in the merge;
this enforces it against the merge, by comparing the working tree to the last
committed copy.

The same argument as scripts/check_stake_schedule.py, one layer along: that
one refuses to let a stake change rewrite the published moneyline record, this
one refuses to let a price change rewrite a published bet.

WHAT IS ALLOWED TO MOVE: last_seen (the play is still on the board),
closing_odds (the line at the bell, which is the input to CLV and never to a
stake), and result / units_result / graded_at, which are written once by
grading and are empty until then.

ROWS MAY BE ADDED, NEVER REMOVED. A play that stops qualifying is still a
play that was made.

Exits 0 clean, 1 on any violation. Run: python3 scripts/check_plays_ledger.py
"""

import csv
import io
import subprocess
import sys

LEDGER = "data/plays_ledger.csv"

MUTABLE = {"last_seen", "closing_odds", "result", "units_result", "graded_at"}


def _committed_copy():
    """The ledger as of HEAD, or None when it is not tracked yet."""
    try:
        out = subprocess.run(["git", "show", f"HEAD:{LEDGER}"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[plays-ledger] cannot read HEAD copy ({exc}) -- skipping")
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _rows(text):
    return {r["play_id"]: r for r in csv.DictReader(io.StringIO(text))}


def main():
    try:
        with open(LEDGER, encoding="utf-8") as fh:
            current = _rows(fh.read())
    except FileNotFoundError:
        print("[plays-ledger] no ledger yet -- nothing to check")
        return 0

    before_text = _committed_copy()
    if before_text is None:
        print(f"[plays-ledger] {LEDGER} is not tracked at HEAD yet -- "
              f"{len(current)} row(s) will be the baseline")
        return 0
    before = _rows(before_text)

    violations = []

    for pid, was in before.items():
        now = current.get(pid)
        if now is None:
            violations.append(f"REMOVED: {pid}")
            continue
        for field, old in was.items():
            if field in MUTABLE:
                continue
            new = now.get(field, "")
            if (old or "") != (new or ""):
                violations.append(f"REWRITTEN: {pid}\n    {field}: {old!r} -> {new!r}")

    # Grading is one-way. A settled play going back to ungraded, or changing
    # its answer, means something is re-deciding fights that already happened.
    for pid, was in before.items():
        now = current.get(pid)
        if not now or not (was.get("result") or ""):
            continue
        if (now.get("result") or "") != was["result"]:
            violations.append(f"RE-GRADED: {pid}\n    result: "
                              f"{was['result']!r} -> {now.get('result')!r}")

    added = len(current) - len([p for p in before if p in current])
    if violations:
        print(f"[plays-ledger] {len(violations)} VIOLATION(S)\n")
        for v in violations:
            print("  " + v)
        print("\nA published play is a claim about a moment that has passed. "
              "If the rule changed, the new rule applies to the NEXT card.")
        return 1

    print(f"[plays-ledger] clean -- {len(before)} published row(s) unchanged, "
          f"{added} added")
    return 0


if __name__ == "__main__":
    sys.exit(main())
