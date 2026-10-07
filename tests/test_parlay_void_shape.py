"""
A voided slip and a settled slip are not the same row, and the page has to
survive both.

On 2026-10-07 the first slip ever to void -- 74ff7228c89e -- froze the whole
site for seven hours. grade_slips writes result, leg_states, legs_won,
units_result, settled_decimal and graded_at together. void_stale_slips writes
result, units_result, graded_at and void_reason, and nothing else. The
template read settled_decimal unconditionally, american_from_decimal called
float() on the Undefined, and `Generate site` -- which has no
continue-on-error by design, because there is no site without it -- failed
every run for seven hours.

Nothing was wrong with the grader. The void path is RIGHT not to invent a
settled price: money that was never at risk has no price, which is the same
reason units_result is 0.0 and not -1.0.

So this pins the asymmetry rather than papering over it, and asserts the
template copes with it.

Synthetic rows only; nothing here reads data/.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parlay_grader import void_stale_slips  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def slip(**kw):
    base = {"slip_id": "abc123", "pinned_at": "2026-09-01 10:00 AM ET",
            "last_seen": "2026-09-05", "first_seen": "2026-09-01",
            "legs": [{"label": "A ML", "decimal_odds": 1.5},
                     {"label": "B ML", "decimal_odds": 2.0}]}
    base.update(kw)
    return base


# --- what a void actually writes ------------------------------------------
rows = [slip()]
changed = void_stale_slips(rows, "2026-10-07 09:00 AM ET")
check("a stale committed slip voids", len(changed) == 1)
r = rows[0]
check("  ...with result void", r.get("result") == "void")
check("  ...and zero units, because it never lost", r.get("units_result") == 0.0)
check("  ...and a reason", bool(r.get("void_reason")))

# The absence is the point. If a later change starts writing these, this test
# should be revisited deliberately -- not silently satisfied.
for f in ("settled_decimal", "legs_won", "leg_states"):
    check(f"a void carries NO {f}", f not in r)

# --- the template must not require what a void lacks ----------------------
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "templates", "site.html"), encoding="utf-8").read()
start = tpl.index('<div class="pr-slip">')
block = tpl[start:start + 2600]

check("the slip block branches on a void", "s.result == 'void'" in block)
check("  ...and guards settled_decimal", "s.settled_decimal is defined" in block)

# settled_decimal must never be piped straight into the filter without a guard
# ahead of it -- that exact line is what raised.
for m in re.finditer(r"\{\{\s*s\.settled_decimal\s*\|", block):
    before = block[:m.start()]
    guarded = ("s.settled_decimal is defined" in before.rsplit("{% if", 1)[-1]
               or "s.result == 'void'" in before)
    check("every settled_decimal render sits behind a guard", guarded)

# A void must not print a won-count it never computed.
# Slice from the void branch to ITS else -- the pr-state span above has an
# earlier "{% else %}Void{% endif %}", and anchoring on the first one sliced
# backwards into an empty string that passed the "no legs_won" check for the
# wrong reason.
_v = block.index("s.result == 'void'")
void_branch = block[_v:block.index("{% else %}", _v)]
check("a void prints no won-count", "legs_won" not in void_branch)
check("  ...but still says how many legs it had", "s.legs|length" in void_branch)

# --- a settled slip is untouched by any of this ---------------------------
settled = slip(result="cashed", settled_decimal=3.0, legs_won=2,
               leg_states=["won", "won"], units_result=2.0)
rows2 = [settled]
check("a slip that already settled is not re-voided",
      void_stale_slips(rows2, "2026-10-07 09:00 AM ET") == [])
check("  ...and keeps its price", rows2[0]["settled_decimal"] == 3.0)

# An unpinned slip was never committed to and must not collect a result.
unpinned = slip(pinned_at=None)
check("an unpinned slip is never voided",
      void_stale_slips([unpinned], "2026-10-07 09:00 AM ET") == [])

print(f"{'PASS' if not fail else 'FAIL'}: test_parlay_void_shape -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
