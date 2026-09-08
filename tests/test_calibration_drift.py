"""
The drift warning must be quiet unless it means something.

The old rule fired on a fixed 15 points at n>=10, and its own comment
conceded that 15pp at n=10 sits inside plausible binomial noise. It
oscillated -- the 50-60% bin tripped it at 15.2pp on n=46 and fell silent
at 13.6pp on n=50, having learned nothing in between -- and it ignored that
four or five bins are tested every run. With the worst bin at |z|=1.93, the
chance of SOME bin looking that extreme by luck is about 0.20.

One warning in five being real is not a warning. The week this was written
produced four failures whose only symptom was a misleading line in a build
log; a detector that is wrong four times in five teaches the reader to skip
the line that finally matters.

Captures stdout, because "says nothing" is the property under test.

Synthetic fixtures only; nothing here reads data/.
"""

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.track_record import (DRIFT_ALPHA, MIN_BIN_FOR_DRIFT,  # noqa: E402
                              _warn_calibration_drift)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def bins(*specs):
    return [{"predicted": p, "actual": a, "n": n, "_lo": lo, "_hi": lo + 0.1}
            for p, a, n, lo in specs]


def said(points):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _warn_calibration_drift(points)
    return buf.getvalue().strip()


# --- THE REGRESSION: the actual record, which must be silent --------------
# 116 graded picks, worst bin |z|=1.93, family-wise p about 0.20.
REAL = bins((0.544, 0.680, 50, 0.5), (0.648, 0.576, 33, 0.6),
            (0.752, 0.818, 22, 0.7), (0.841, 0.900, 10, 0.8))
check("the real record produces no warning", said(REAL) == "")

# The old rule's exact trigger: 15pp at n=46. It must no longer fire alone.
check("15pp on one bin of several is not enough",
      said(bins((0.544, 0.696, 46, 0.5), (0.648, 0.640, 33, 0.6),
                (0.752, 0.760, 22, 0.7))) == "")

# --- IT MUST STILL FIRE ON REAL DRIFT ------------------------------------
loud = said(bins((0.75, 0.45, 60, 0.7), (0.55, 0.56, 40, 0.5)))
check("a badly miscalibrated bin is reported", "CALIBRATION DRIFT" in loud)
check("...named as overconfident", "OVERconfident" in loud)
check("...with the bin", "70%-80%" in loud)
check("...and the correction is stated", "correcting for 2 bins" in loud)

under = said(bins((0.55, 0.85, 60, 0.5), (0.75, 0.74, 40, 0.7)))
check("underconfidence is reported too", "UNDERconfident" in under)

# --- THE CORRECTION ACTUALLY BITES ---------------------------------------
# One bin alone at p just under .05 speaks; the same bin among many does not,
# because more bins mean more chances to look extreme.
# Chosen to be MARGINAL on purpose: z=-2.19, per-bin p=0.029. Strong enough
# to speak alone, not strong enough to survive five chances at looking this
# extreme. A bin at p=0.003 would survive the correction and prove nothing
# about it.
solo = bins((0.75, 0.60, 40, 0.7))
many = solo + bins((0.55, 0.55, 40, 0.5), (0.65, 0.65, 40, 0.6),
                   (0.85, 0.85, 40, 0.8), (0.60, 0.60, 40, 0.55))
check("a lone bin at that gap is reported", said(solo) != "")
check("the same bin among five is not", said(many) == "")

# --- THIN BINS ARE NOT TESTED AT ALL -------------------------------------
check(f"a bin under n={MIN_BIN_FOR_DRIFT} says nothing however wild",
      said(bins((0.75, 0.05, MIN_BIN_FOR_DRIFT - 1, 0.7))) == "")
check("...and does not count toward the correction either",
      said(bins((0.75, 0.45, 60, 0.7), (0.9, 0.1, 3, 0.9))) ==
      said(bins((0.75, 0.45, 60, 0.7))))

# --- degenerate input -----------------------------------------------------
check("no bins is a no-op", said([]) == "")
check("a perfectly calibrated record is silent",
      said(bins((0.55, 0.55, 50, 0.5), (0.75, 0.75, 50, 0.7))) == "")
check("alpha is a real threshold", 0 < DRIFT_ALPHA < 1)

print(f"{'PASS' if not fail else 'FAIL'}: test_calibration_drift -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
