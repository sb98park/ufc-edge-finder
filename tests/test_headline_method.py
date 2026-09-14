"""
The named method must never be less likely than the alternative.

This rule has been wrong twice, in opposite directions, and both times the
symptom was the headline disagreeing with the table printed under it.

  three-way argmax   named the largest single cell, so it said "by decision"
                     on 69 of 93 fights -- and on 23 of those the same page
                     showed P(finish) > P(decision)
  larger-finish      named the bigger of KO and SUB, so Arman Tsarukyan read
                     "by KO/TKO" at 30.7% while decision sat at 37.3% on the
                     row below it

Now the exhaustive pair decides: finish (KO + SUB) against decision, and the
finish branch is named "Finish" rather than guessing which one. 43.0% beats
37.3%, and a reader can verify it by adding two rows they can see.

"Finish" is graded as a match for KO/TKO or Submission (see
track_record._method_matches). That is deliberately easier to hit than
naming KO/TKO outright.

Synthetic fixtures only; nothing here reads data/.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.method_model import headline_method  # noqa: E402
from src.track_record import _method_matches  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


# --- naming the specific method, when it is genuinely specific -----------
# The complaint that produced this branch: every headline read "by finish" or
# "by decision" and KO never appeared, even on a fighter who wins by KO and
# almost never by submission.
check("a lopsided KO read names KO/TKO", headline_method(47.0, 5.0, 30.0) == ("KO/TKO", 47.0))
check("a lopsided submission read names Submission",
      headline_method(5.0, 40.0, 30.0) == ("Submission", 40.0))
# ...but a near-even KO/SUB split is a coin flip dressed as a read.
check("a 32/31 split stays Finish", headline_method(32.0, 31.0, 30.0)[0] == "Finish")
# ...and a mode that does NOT outrank decision can never be named, because
# the table directly below would show decision as the larger number.
check("KO below decision stays Finish", headline_method(40.0, 8.0, 45.0)[0] == "Finish")

# --- the two fights that drove each change -------------------------------
check("Arman: finish 43.0 beats decision 37.3, so Finish",
      headline_method(30.7, 12.3, 37.3) == ("Finish", 43.0))
check("Parnasse: finish 42.5 beats decision 39.9, so Finish",
      headline_method(28.8, 13.7, 39.9)[0] == "Finish")
check("a genuine decision read is still Decision",
      headline_method(11.9, 8.1, 31.1) == ("Decision", 31.1))

# --- it never names the less likely side ---------------------------------
check("a dominant KO read is never a Decision",
      headline_method(42.8, 10.6, 28.9)[0] == "KO/TKO")
check("a submission that trails decision stays Finish",
      headline_method(9.0, 21.0, 25.0)[0] == "Finish")

# Ties resolve to Decision -- arbitrary but fixed, so the label cannot
# flicker between builds on identical inputs.
check("an exact tie goes to Decision", headline_method(20.0, 20.0, 40.0) == ("Decision", 40.0))

# --- the rate belongs to the thing named ---------------------------------
for trio in [(30.7, 12.3, 37.3), (11.9, 8.1, 31.1), (9.0, 21.0, 25.0),
             (42.8, 10.6, 28.9), (47.0, 5.0, 30.0), (5.0, 40.0, 30.0)]:
    name, rate = headline_method(*trio)
    want = {"KO/TKO": trio[0], "Submission": trio[1], "Decision": trio[2],
            "Finish": trio[0] + trio[1]}[name]
    check(f"rate matches the named thing for {trio}", abs(rate - want) < 1e-9)

# --- THE INVARIANT, over random grids ------------------------------------
rng = random.Random(20260914)
bad = contradicts_table = 0
for _ in range(20000):
    ko, sub, dec = rng.uniform(0, 60), rng.uniform(0, 40), rng.uniform(0, 70)
    name, rate = headline_method(ko, sub, dec)
    # The claim always beats the alternative claim it displaced.
    if name == "Decision":
        bad += dec < ko + sub
    else:
        bad += (ko + sub) <= dec
    # And a SPECIFIC method must be the largest single cell on the grid, or
    # the reader sees a bigger number on the row below the headline. That is
    # the complaint this rule has now been rewritten twice to answer.
    if name in ("KO/TKO", "Submission"):
        contradicts_table += rate < max(ko, sub, dec) - 1e-9
check("the named side is never the less likely one (20k grids)", bad == 0)
check("a named KO/Submission is always the largest cell (20k grids)",
      contradicts_table == 0)

check("only the four labels are ever produced",
      {headline_method(rng.uniform(0, 60), rng.uniform(0, 40), rng.uniform(0, 70))[0]
       for _ in range(5000)} <= {"KO/TKO", "Submission", "Finish", "Decision"})

# Scale invariance: the grid sums to the fighter's WIN probability, not to 1.
for scale in (0.25, 0.5, 2.0):
    check(f"same call when scaled by {scale}",
          headline_method(30.7 * scale, 12.3 * scale, 37.3 * scale)[0] == "Finish")
    check(f"  ...and for a KO read scaled by {scale}",
          headline_method(47.0 * scale, 5.0 * scale, 30.0 * scale)[0] == "KO/TKO")

# --- and it has to grade ------------------------------------------------
check("Finish counts as a hit on a KO", _method_matches("Finish", "KO/TKO") is True)
check("Finish counts as a hit on a submission", _method_matches("Finish", "Submission") is True)
check("Finish is a miss on a decision", _method_matches("Finish", "Decision - Unanimous") is False)
check("Decision still grades as before", _method_matches("Decision", "Decision - Split") is True)
check("a named KO grades strictly", _method_matches("KO/TKO", "KO/TKO") is True)
check("  ...and misses on a submission", _method_matches("KO/TKO", "Submission") is False)
check("a named submission grades strictly", _method_matches("Submission", "Submission") is True)
check("a missing side is still None", _method_matches("Finish", None) is None)

print(f"{'PASS' if not fail else 'FAIL'}: test_headline_method -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
