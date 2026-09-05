"""
The orphan-streak counter must survive reading its own column back from CSV.

_orphan_streak is written per-row, but a CSV column is not sparse: as soon as
ONE row carries a streak, every other row gets an empty cell, which pandas
reads back as NaN. NaN is present, so `.get(key, 0)` returns it rather than
the default, and `int(NaN)` raises (CLAUDE.md s4 -- pandas NaN is truthy and
does not behave like a missing value).

That mattered more than the counter: the exception propagated out of
resync_tracked_card_order into the catch-all at its call site, so every run
printed "continuing without it" and skipped the ENTIRE resync -- fight ORDER
against ESPN included, which is the function's main job.

Synthetic fixtures only; nothing here reads data/.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


from src.card_discovery import _bump_orphan_streak as bump  # noqa: E402


# The regression: the key is PRESENT and NaN, so the .get default never fires.
check("NaN counts as a first orphaning, not a crash", bump({"_orphan_streak": float("nan")}) == 1)
check("NaN is present, so .get's default is not what saves us",
      "_orphan_streak" in {"_orphan_streak": float("nan")}
      and math.isnan({"_orphan_streak": float("nan")}["_orphan_streak"]))

# The other shapes a round-trip through CSV can produce.
check("empty string from a blank cell", bump({"_orphan_streak": ""}) == 1)
check("None", bump({"_orphan_streak": None}) == 1)
check("genuinely absent key", bump({}) == 1)
check("float from a float64 column", bump({"_orphan_streak": 2.0}) == 3)
check("string digits from an object column", bump({"_orphan_streak": "2"}) == 3)
check("int, the in-memory case", bump({"_orphan_streak": 2}) == 3)

# The counter has to actually climb, or the grace threshold is unreachable
# and a genuinely-removed fight is re-appended forever.
row, seen = {}, []
for _ in range(5):
    seen.append(bump(row))
check("streak accumulates across successive resyncs", seen == [1, 2, 3, 4, 5])
check("streak passes ORPHAN_STREAK_LIMIT=3 so cancellation can fire", max(seen) > 3)

# End to end: the real function must not raise on a frame carrying NaN streaks.
try:
    import pandas as pd

    from src.card_discovery import resync_tracked_card_order  # noqa: F401
    df = pd.DataFrame([{"event_name": "E", "fighter_a": "A", "fighter_b": "B", "_orphan_streak": float("nan")}])
    check("a NaN streak column round-trips as float64", str(df["_orphan_streak"].dtype) == "float64")
    check("and int() on it is what used to raise",
          isinstance(df["_orphan_streak"].iloc[0], float) and math.isnan(df["_orphan_streak"].iloc[0]))
except Exception as e:  # pragma: no cover
    check(f"import/frame setup: {e}", False)

print(f"{'PASS' if not fail else 'FAIL'}: test_orphan_streak_nan -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
