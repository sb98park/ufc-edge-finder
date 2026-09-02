"""
The orphan cleaner must not touch a rematch.

Fight identity in this project is frozenset({a, b}) and carries NO event
(CLAUDE.md s4), so "the same pair under two event names" describes a rename
artefact AND a legitimate rematch equally well. Telling them apart is the only
hard part of this cleaner, and getting it wrong deletes a real published
prediction -- so that is what these tests are mostly about.

The discriminator is whether the EVENT exists: a rematch's two events both
produced results or sit on a tracked card, while a renamed card leaves behind
a name that appears nowhere at all.

Synthetic fixtures only; nothing here reads data/.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from clean_orphaned_event_rows import analyse  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def L(rows):
    return pd.DataFrame([{"event_name": e, "fighter_a": a, "fighter_b": b,
                          "favorite": a, "favorite_prob": 0.6, "voided": v}
                         for e, a, b, v in rows])


def R(rows):
    return pd.DataFrame([{"event_name": e, "fighter_a": a, "fighter_b": b}
                         for e, a, b in rows])


# 1. A REMATCH IS NOT AN ORPHAN. Same pair, two events, BOTH real.
log = L([("UFC 1", "Ann Lee", "Bo Ross", ""), ("UFC 2", "Ann Lee", "Bo Ross", "")])
res = R([("UFC 1", "Ann Lee", "Bo Ross"), ("UFC 2", "Ann Lee", "Bo Ross")])
drop, void = analyse(log, res, set())
check("a graded rematch is left alone", drop == [] and void == [])

# 2. ...including when the second meeting has not happened yet.
log = L([("UFC 1", "Ann Lee", "Bo Ross", ""), ("UFC 9", "Ann Lee", "Bo Ross", "")])
res = R([("UFC 1", "Ann Lee", "Bo Ross")])
drop, void = analyse(log, res, {"UFC 9"})           # UFC 9 is on a tracked card
check("an upcoming rematch is left alone", drop == [] and void == [])

# 3. THE RENAME. Old name exists nowhere; the pair has a row under the real one.
log = L([("Old Name", "Ann Lee", "Bo Ross", ""), ("Real Card", "Ann Lee", "Bo Ross", "")])
res = R([("Real Card", "Ann Lee", "Bo Ross")])
drop, void = analyse(log, res, set())
check("the stranded copy is dropped", len(drop) == 1 and drop[0]["event"] == "Old Name")
check("...and names its keeper", drop[0]["kept_under"] == "Real Card")
check("...and the real row is never a candidate", all(d["event"] != "Real Card" for d in drop))

# 4. A BOOKING THAT NEVER HAPPENED is voided, not deleted.
log = L([("Old Name", "Cal Vega", "Dee Winn", ""), ("Real Card", "Ann Lee", "Bo Ross", "")])
res = R([("Real Card", "Ann Lee", "Bo Ross")])
drop, void = analyse(log, res, set())
check("a cancelled booking is voided, not dropped",
      drop == [] and len(void) == 1 and void[0]["a"] == "Cal Vega")

# 5. THE DULATOV CASE. The pair never graded, but it HAS a row under the real
#    name carrying voided=true. Treating it as a cancelled booking would
#    void-and-re-point it into a SECOND row on the real card.
log = L([("Old Name", "Cal Vega", "Dee Winn", ""), ("Real Card", "Cal Vega", "Dee Winn", "true")])
res = R([("Real Card", "Ann Lee", "Bo Ross")])
drop, void = analyse(log, res, set())
check("a cancelled fight already recorded on the real card is a duplicate, not a re-point",
      len(drop) == 1 and void == [])

# 6. Quiet when every event is real.
log = L([("Real Card", "Ann Lee", "Bo Ross", "")])
res = R([("Real Card", "Ann Lee", "Bo Ross")])
check("no orphans means no findings", analyse(log, res, set()) == ([], []))

# 7. An event known only from a tracked card still counts as real.
log = L([("Upcoming", "Ann Lee", "Bo Ross", "")])
res = R([])
drop, void = analyse(log, res, {"Upcoming"})
check("a future card is a real event", (drop, void) == ([], []))
check("...and without it the row would be flagged",
      analyse(log, res, set()) != ([], []))

print(f"test_orphaned_event_rows: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
