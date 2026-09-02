"""
The point-in-time method rates must be the SERVING path with a clock on it.

A validation harness that re-derives what production serves measures its own
re-derivation. The first attempt at this used rates_or_prior directly, which
reads today's table and therefore the scored fight's own result; it posted AUC
0.804 against the production model's measured 0.720. A backtest beating the
live model is a receipt for leakage, not a finding.

So the guarantee these tests exist for is narrow and total: wind the clock past
every bout and ufc_method_rates_as_of must equal ufc_method_rates, fighter for
fighter, to the last decimal. Any divergence means the windowed path has
started answering a different question from the one that ships.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ufc_method_rates import (  # noqa: E402
    MIN_UFC_FIGHTS, load_ufc_records, load_dated_ufc_bouts,
    ufc_method_rates, ufc_method_rates_as_of, rates_or_prior,
    rates_or_prior_as_of,
)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


FUTURE = "2999-01-01"
dated = load_dated_ufc_bouts()
undated = load_ufc_records()

check("the dated index found fighters", len(dated) > 1000)
check("...roughly as many as the undated table",
      abs(len(dated) - len(undated)) <= max(40, 0.02 * len(undated)))

# WHO CANNOT MATCH, AND WHY, established before asserting that everyone else
# does. 27 of 8,859 bout rows (0.30%) sit on events missing from
# ufc_event_details.csv, so they carry no date and the windowed index drops
# them -- a bout that cannot be placed in time cannot be windowed by one.
# Their fighters legitimately run a denominator short. Naming them is the
# difference between a known 0.3% and a test loosened until it passes.
import csv as _csv                                                    # noqa: E402
from src.ufc_method_rates import _fold, RESULTS, EVENTS               # noqa: E402

_dated_events = {str(r["EVENT"]).strip() for r in _csv.DictReader(open(EVENTS, encoding="utf-8"))}
undateable = set()
for _r in _csv.DictReader(open(RESULTS, encoding="utf-8")):
    if str(_r.get("EVENT", "")).strip() in _dated_events:
        continue
    for _p in str(_r.get("BOUT", "")).split(" vs. "):
        undateable.add(_fold(_p.strip()))
check("the undated events touch only a handful of fighters",
      len(undateable) <= 0.03 * len(undated))

mismatch, measured = [], 0
for name in undated:
    a = ufc_method_rates(name)
    b = ufc_method_rates_as_of(name, FUTURE)
    if a is not None:
        measured += 1
    if name in undateable:
        continue                    # denominator is short by construction
    if a is None and b is None:
        continue
    if a is None or b is None or max(abs(x - y) for x, y in zip(a, b)) > 1e-9:
        mismatch.append((name, a, b))
check(f"every dateable fighter matches production with the clock wound "
      f"forward ({measured} measured, {len(undateable)} excluded)", not mismatch)
if mismatch:
    for m in mismatch[:5]:
        print(f"      {m[0]}: production {m[1]} vs as_of {m[2]}")

# THE WINDOW HAS TO ACTUALLY BITE, or the equivalence above is vacuous -- a
# function returning today's answer regardless of `when` would pass it.
before_any = sum(1 for n in list(undated)[:400]
                 if ufc_method_rates_as_of(n, "1990-01-01") is not None)
check("nothing is measurable before the UFC existed", before_any == 0)

moved = 0
for n in list(undated)[:400]:
    late, early = ufc_method_rates_as_of(n, FUTURE), ufc_method_rates_as_of(n, "2016-01-01")
    if late is not None and (early is None or max(abs(x - y) for x, y in zip(late, early)) > 1e-9):
        moved += 1
check("and a mid-career cutoff changes most fighters", moved > 100)

# The min_fights gate and the fallback have to survive the rewind.
thin = [n for n, r in undated.items() if 0 < r["fights"] < MIN_UFC_FIGHTS]
check("a fighter under the fights floor is still None",
      not thin or ufc_method_rates_as_of(thin[0], FUTURE) is None)

priors = {"Lightweight": {"KO/TKO": 0.30, "SUB": 0.20, "DEC": 0.50}}
fb_now = rates_or_prior("Nobody At All Xyz", priors, "Lightweight")
fb_pit = rates_or_prior_as_of("Nobody At All Xyz", FUTURE, priors, "Lightweight")
check("an unknown fighter falls back identically", fb_now == fb_pit)
check("...to half the divisional rate", abs(fb_pit[0] - 0.15) < 1e-9)

print(f"test_pit_method_rates: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
