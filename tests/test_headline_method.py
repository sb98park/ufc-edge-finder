"""
The named method must not contradict the grid printed underneath it.

The headline used to be a three-way argmax over the favourite's KO/SUB/DEC
cells. Decision is the largest SINGLE cell far more often than a finish is
unlikely, so on 2026-09-06 the site labelled 69 of 93 fights "by decision" --
and on 23 of those the same page also printed P(finish) > P(decision).
Salahdine Parnasse: KO 28.8, SUB 13.7, DEC 39.9, headline "by decision",
table "Fight ends by Decision 45.6%" against 54.5% for a finish.

The rule decides the exhaustive pair first (finish vs decision, which is a
real traded market), then which finish. The invariant below is the point:
never name decision when the grid says a finish is likelier.

Synthetic fixtures only; nothing here reads data/.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.method_model import headline_method  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


# The fight that motivated this. Decision is the biggest cell; a finish is
# still likelier, so a finish is what gets named.
check("Parnasse: finish beats decision, so KO/TKO is named",
      headline_method(28.8, 13.7, 39.9) == ("KO/TKO", 28.8))
check("a genuine decision read is still named decision",
      headline_method(11.9, 8.1, 31.1) == ("Decision", 31.1))
check("a clear KO read is unchanged", headline_method(42.8, 10.6, 28.9) == ("KO/TKO", 42.8))
check("submission is named when it leads the finish split",
      headline_method(9.0, 21.0, 25.0) == ("Submission", 21.0))
check("narrow case: 22.3 dec vs 28.2 finish -> finish",
      headline_method(19.8, 8.4, 22.3)[0] == "KO/TKO")

# Ties resolve toward decision, and toward KO within a finish -- arbitrary but
# fixed, so the label cannot flicker between builds on identical inputs.
check("exact tie decision vs finish goes to decision",
      headline_method(20.0, 20.0, 40.0) == ("Decision", 40.0))
check("exact tie KO vs SUB goes to KO", headline_method(20.0, 20.0, 10.0) == ("KO/TKO", 20.0))

# The rate returned must belong to the method named, or the headline would
# print one method's name beside another's number.
for trio in [(28.8, 13.7, 39.9), (11.9, 8.1, 31.1), (9.0, 21.0, 25.0), (42.8, 10.6, 28.9)]:
    name, rate = headline_method(*trio)
    want = {"KO/TKO": trio[0], "Submission": trio[1], "Decision": trio[2]}[name]
    check(f"rate matches the named method for {trio}", rate == want)

# THE INVARIANT, over random grids: decision is never named while a finish
# is strictly likelier. This is the whole point of the change.
rng = random.Random(20260906)
bad = 0
for _ in range(20000):
    ko, su, de = rng.uniform(0, 60), rng.uniform(0, 40), rng.uniform(0, 70)
    name, _ = headline_method(ko, su, de)
    if name == "Decision" and ko + su > de:
        bad += 1
check("no decision label ever contradicts the grid (20k random grids)", bad == 0)

# And the converse: a finish is never named when decision is strictly ahead
# of the pair.
bad2 = 0
for _ in range(20000):
    ko, su, de = rng.uniform(0, 60), rng.uniform(0, 40), rng.uniform(0, 70)
    name, _ = headline_method(ko, su, de)
    if name != "Decision" and de > ko + su:
        bad2 += 1
check("no finish label when decision leads the pair", bad2 == 0)

# Scale invariance: the grid rows sum to the fighter's WIN probability, not to
# 1, so the rule must not depend on the total.
for scale in (0.25, 0.5, 2.0):
    check(f"same call when the grid is scaled by {scale}",
          headline_method(28.8 * scale, 13.7 * scale, 39.9 * scale)[0]
          == headline_method(28.8, 13.7, 39.9)[0])

print(f"{'PASS' if not fail else 'FAIL'}: test_headline_method -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
