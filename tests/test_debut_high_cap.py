"""
A fighter the model has never seen in the UFC cannot carry the flagship pick.

_confidence_label had two caps at the High band and consulted only one.
thinner_record counts PROFESSIONAL bouts, so a 7-0 regional fighter clears
its floor of 4 while having never fought in the UFC -- the module's own
comment names exactly that case. debut_corner is the quantity that catches
him, and it was checked only in the Medium branch.

Because DEBUT_MEDIUM_CEILING equals high_bar, the result was a
discontinuity rather than a lenient rule: the same debutant read LOW at
0.74 and HIGH at 0.76, skipping Medium entirely, at exactly the probability
where a five-unit stake appears.

Shipped for consistency, not for a measured gain: across all 147 logged
picks, three have a corner with zero UFC bouts and none reached 0.75.

Synthetic fixtures only; nothing here reads data/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_preview import (DEBUT_MEDIUM_CEILING,  # noqa: E402
                               MIN_RECORD_FOR_HIGH_CONFIDENCE,
                               _confidence_capped, _confidence_label)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def L(p, record=8, debut=False, prev=None):
    return _confidence_label(p, thinner_record=record, debut_corner=debut,
                             previous_label=prev)


# --- THE REGRESSION: no debut corner reaches a staked tier ---------------
check("a debutant at 0.80 is capped to Medium", L(0.80, debut=True) == "Medium Confidence")
check("...at 0.90 too", L(0.90, debut=True) == "Medium Confidence")
check("...and at 0.99", L(0.99, debut=True) == "Medium Confidence")

# The discontinuity itself: Low below the ceiling, and it must not become
# HIGH the moment it crosses.
check("debutant just under the ceiling is Low",
      L(DEBUT_MEDIUM_CEILING - 0.01, debut=True) == "Low Confidence")
check("debutant just over it is Medium, not High",
      L(DEBUT_MEDIUM_CEILING + 0.01, debut=True) == "Medium Confidence")

# --- what must NOT change ------------------------------------------------
check("a non-debutant at 0.80 is still High", L(0.80, debut=False) == "High Confidence")
check("a non-debutant at 0.76 is still High", L(0.76, debut=False) == "High Confidence")
check("the thin-record cap still works",
      L(0.80, record=MIN_RECORD_FOR_HIGH_CONFIDENCE - 1) == "Medium Confidence")
check("a deep record with no debut is untouched", L(0.85, record=25) == "High Confidence")
check("Medium band unaffected for a non-debutant", L(0.65, debut=False) == "Medium Confidence")
check("below the medium bar is still Low", L(0.55, debut=False) == "Low Confidence")

# A professional record clearing the floor is NOT enough on its own -- that
# is the whole point, and the case the module's comment describes.
check("7-0 as a pro but never in the UFC does not get High",
      L(0.83, record=7, debut=True) == "Medium Confidence")

# --- the cap must be explainable -----------------------------------------
# _confidence_cap_reason renders off _confidence_capped; if that disagrees
# with the label, the UI shows a capped pick with no reason.
check("a debut cap reports as capped",
      _confidence_capped(0.80, 8, True) is True)
check("a non-debut High is not reported as capped",
      _confidence_capped(0.80, 8, False) is False)
check("the thin-record cap still reports as capped",
      _confidence_capped(0.80, MIN_RECORD_FOR_HIGH_CONFIDENCE - 1, False) is True)

# --- hysteresis must not reopen the hole ---------------------------------
# The caps are deliberately not subject to hysteresis; holding a stale High
# through one is the failure they exist to prevent.
check("a previously-High debutant is still capped",
      L(0.80, debut=True, prev="High Confidence") == "Medium Confidence")

# --- None means unknown, not zero ----------------------------------------
check("unknown debut status does not cap",
      L(0.80, record=8, debut=None) == "High Confidence")
check("unknown record does not cap", L(0.80, record=None, debut=False) == "High Confidence")

print(f"{'PASS' if not fail else 'FAIL'}: test_debut_high_cap -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
