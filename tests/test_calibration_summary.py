"""
The sentence under the chart must not outrun the evidence in it.

_warn_calibration_drift corrects for the four or five bands tested every run
and, on the record as of 2026-09-15, stays silent: nothing survives. The
reader-facing summary took a raw look at the same bins and called one of them
"the one place the model has oversold itself" -- a band at z=-1.10, p~0.27
corrected, while the WIDEST swing on the same chart was 12 points the other
way and went unmentioned. Two components, one dataset, opposite conclusions,
and the page showed the uncorrected one.

The band is still named when the chart shows a dip. That part was a real fix:
on 2026-08-23 the average read "if anything, we've been modest" directly under
a chart whose middle point visibly dipped the other way, and a sentence is
what people read. What changed is the claim attached to the name.

Synthetic bins only; nothing here reads data/.
"""

import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.track_record import (DRIFT_ALPHA, MIN_BIN_FOR_DRIFT,  # noqa: E402
                              _calibration_family_p, _warn_calibration_drift,
                              _compute_calibration)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def picks(pred, actual_rate, n):
    """n graded picks in one band, `actual_rate` of them correct."""
    hits = round(actual_rate * n)
    return [{"favorite_prob": pred, "correct": i < hits} for i in range(n)]


# --- the live shape: a visible dip that is not a finding ------------------
# 54% band running high, 65% band running low -- the real 2026-09-15 curve.
matched = picks(0.545, 0.661, 56) + picks(0.648, 0.564, 39) + picks(0.75, 0.826, 23)
with redirect_stdout(io.StringIO()):
    cal = _compute_calibration(matched)
check("calibration is ready at this sample", cal["ready"] is True)
check("the low band is NOT called significant", cal["worst_bin_significant"] is False)
text = cal["summary"]
check("the dip is still named", "65%" in text or "64%" in text)
check("  ...without claiming it is the one place we oversold",
      "one place the model has oversold" not in text)
check("  ...and says plainly it is inside chance", "inside what chance produces" in text)
check("  ...and names the bigger swing the other way", "OTHER way" in text)
check("  ...and does not say the average HIDES it", "average hides one band" not in text)

# --- the same numbers must not make the log warning speak ----------------
buf = io.StringIO()
with redirect_stdout(buf):
    _warn_calibration_drift([dict(p, _lo=0.0, _hi=1.0) for p in cal["points"]])
check("THE LOG WARNING AGREES -- it stays silent too", buf.getvalue().strip() == "")

# --- a real one still gets called a real one -----------------------------
# A band far enough off, at enough picks, that correcting for every band
# tested still leaves it below alpha.
loud = picks(0.70, 0.30, 120) + picks(0.55, 0.56, 60)
# _compute_calibration warns on the way through; captured so a passing run
# stays quiet and the warning is asserted on deliberately, below.
with redirect_stdout(io.StringIO()):
    cal2 = _compute_calibration(loud)
check("a genuine miss IS significant", cal2["worst_bin_significant"] is True)
check("  ...and gets the strong sentence",
      "one place the model has oversold" in cal2["summary"])
check("  ...and the average hides it", "average hides one band" in cal2["summary"])
buf2 = io.StringIO()
with redirect_stdout(buf2):
    _warn_calibration_drift([dict(p, _lo=0.0, _hi=1.0) for p in cal2["points"]])
check("  ...and the log warning speaks for the same bin",
      "CALIBRATION DRIFT" in buf2.getvalue())

# --- the two readers share one test --------------------------------------
# This is the property that stops them diverging again. Every testable bin
# must get the same verdict from the helper as the warning acts on.
for cal_x in (cal, cal2):
    for pt in cal_x["points"]:
        fam = _calibration_family_p(cal_x["points"], pt)
        if pt["n"] < MIN_BIN_FOR_DRIFT:
            check(f"a bin of {pt['n']} is untestable, not 'fine'", fam is None)
        else:
            check(f"bin n={pt['n']} gets a real p", isinstance(fam, float) and 0 <= fam <= 1)

check("a tiny bin can never be called significant",
      _calibration_family_p([{"predicted": .5, "actual": 1.0, "n": 3}],
                            {"predicted": .5, "actual": 1.0, "n": 3}) is None)

# Correcting for more bins can only make a given gap LESS significant --
# that is the whole point of the correction.
one = [{"predicted": 0.7, "actual": 0.5, "n": 40}]
many = one + [{"predicted": 0.5, "actual": 0.5, "n": 40} for _ in range(4)]
check("more bins tested -> a weaker claim, never a stronger one",
      _calibration_family_p(many, one[0]) > _calibration_family_p(one, one[0]))

print(f"{'PASS' if not fail else 'FAIL'}: test_calibration_summary -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
