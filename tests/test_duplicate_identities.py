"""
One fighter under two names must be caught before the card, not after it.

It has happened twice and both times it was found by accident:

  Sean King / "Sean King III"    after the card. card_discovery read the
                                 variant's row as a REPLACEMENT, cancelled the
                                 real bout and voided its prediction -- a
                                 CORRECT pick left the published record.
  Valesca Machado / "Tina Black" three days before the card, while looking at
                                 something unrelated. Every book priced her
                                 under the canonical name, so the prices
                                 attached to the row marked cancelled.

THE SIGNAL: a bout keyed (date, opponent) belongs to one fighter. A's fight
with X on D is (D, X); X's side of it is (D, A). Two different names holding
the SAME key are the same person -- with one honest exception, a tournament
night where X fought both A and B on D, which MIN_SHARED exists to swallow.

Synthetic bouts only; nothing here reads data/ or the network.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_duplicate_identities import MIN_SHARED, find_duplicates  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def career(name, opps, start=2020):
    """One fighter's bouts as the index sees them: (date, opponent)."""
    return {name: {(f"{start + i // 3}-0{i % 3 + 1}-15", o) for i, o in enumerate(opps)}}


# --- the two real incidents, in shape --------------------------------------
OPPS = ["nayara cascardo", "cleudilene costa", "elaine leal", "maria oliveira"]
b = {}
b.update(career("valesca machado", OPPS))
b.update(career("tina black", OPPS))          # same career, second name
found = find_duplicates(b)
check("a duplicated career is flagged", len(found) == 1)
check("  ...naming both sides",
      found and set(found[0][:2]) == {"valesca machado", "tina black"})
check("  ...and counting every shared bout", found and found[0][2] == len(OPPS))

# --- a genuine pair of fighters is NOT flagged -----------------------------
b2 = {}
b2.update(career("fighter one", ["opp a", "opp b", "opp c", "opp d"]))
b2.update(career("fighter two", ["opp e", "opp f", "opp g", "opp h"]))
check("two real careers are left alone", find_duplicates(b2) == [])

# Sharing OPPONENTS is not sharing BOUTS. Two fighters who both beat the same
# three men on different nights are not one person, and a detector that could
# not tell those apart would flag half a division.
b3 = {
    "alpha": {("2021-01-15", "x"), ("2021-02-15", "y"), ("2021-03-15", "z")},
    "beta":  {("2022-01-15", "x"), ("2022-02-15", "y"), ("2022-03-15", "z")},
}
check("same opponents on different dates is not a duplicate", find_duplicates(b3) == [])

# --- the tournament exception ----------------------------------------------
# X fights A and B on the same night, so A and B each legitimately hold
# (D, X). One shared key is a bracket, not an identity.
b4 = {
    "a": {("2021-06-05", "x"), ("2021-06-05", "c")},
    "b": {("2021-06-05", "x"), ("2021-07-05", "d")},
}
check("a single shared bout is not enough to flag", find_duplicates(b4) == [])
check("  ...and MIN_SHARED is what makes that true", MIN_SHARED >= 2)

# --- the threshold behaves at its boundary ---------------------------------
shared = {(f"2021-0{i}-15", f"opp{i}") for i in range(1, MIN_SHARED + 1)}
just_under = {"p": set(list(shared)[:MIN_SHARED - 1]), "q": set(list(shared)[:MIN_SHARED - 1])}
just_over = {"p": shared, "q": shared}
check(f"{MIN_SHARED - 1} shared bouts stays quiet", find_duplicates(just_under) == [])
check(f"{MIN_SHARED} shared bouts flags", len(find_duplicates(just_over)) == 1)

# --- three names for one fighter -------------------------------------------
b5 = {}
for n in ("liu ce", "ce liu", "liu, ce"):
    b5.update(career(n, OPPS))
trip = find_duplicates(b5)
check("three spellings produce all three pairings", len(trip) == 3)

# --- ordering puts the worst case first ------------------------------------
b6 = {}
b6.update(career("big a", OPPS * 2))
b6.update(career("big b", OPPS * 2))
b6.update(career("small a", OPPS[:MIN_SHARED]))
b6.update(career("small b", OPPS[:MIN_SHARED]))
ranked = find_duplicates(b6)
check("the most-shared pair is reported first",
      len(ranked) >= 2 and ranked[0][2] >= ranked[-1][2])

# --- degenerate input ------------------------------------------------------
for label, payload in (("no fighters", {}),
                       ("one fighter", career("solo", OPPS)),
                       ("a fighter with no bouts", {"ghost": set()})):
    try:
        check(f"{label} -> no crash", find_duplicates(payload) == [])
    except Exception as exc:                      # noqa: BLE001
        check(f"{label} -> no crash ({exc})", False)

# --- the pipeline actually runs it -----------------------------------------
wf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       ".github", "workflows", "refresh.yml")).read()
check("the workflow runs the check", "check_duplicate_identities.py" in wf)
# Non-blocking on purpose: a name-hygiene check must never be able to freeze a
# site real money is staked against.
_blk = wf[wf.index("Look for one fighter recorded under two names"):]
check("  ...and cannot fail the build",
      "continue-on-error: true" in _blk[:_blk.index("run:")])

print(f"{'PASS' if not fail else 'FAIL'}: test_duplicate_identities -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
