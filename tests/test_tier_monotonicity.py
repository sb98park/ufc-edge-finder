"""
A higher probability must never carry a lower confidence tier.

Hysteresis lowers the bar for a fight that already held a tier -- entering
High costs 0.75, holding costs 0.74 -- which is correct on its own and made a
card contradict itself. On 2026-09-05 Donchenko at 0.7440 read High (held)
while Pinto at 0.7490 read Medium (entering). The tiers were right and the
ladder was unreadable, which from the outside is the same thing.
"""
import sys

sys.path.insert(0, ".")
from src.card_matcher import _enforce_tier_monotonicity            # noqa: E402
from src.model_preview import (CONFIDENCE_HYSTERESIS,              # noqa: E402
                               _confidence_capped, _confidence_label)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def card(*rows):
    """rows of (prob, label) or (prob, label, capped)."""
    return [{"preview": {"favorite_prob": p, "confidence_label": l,
                         "confidence_capped": (r[2] if len(r) > 2 else False)}}
            for r in rows for p, l in [(r[0], r[1])]]


def labels(fights):
    return [f["preview"]["confidence_label"] for f in fights]


# The real card that exposed it.
f = card((0.7440, "High Confidence"), (0.7490, "Medium Confidence"),
         (0.7260, "Medium Confidence"), (0.5020, "Low Confidence"))
_enforce_tier_monotonicity(f)
check("the 0.7490 fight is lifted to match the held 0.7440 one",
      labels(f)[1] == "High Confidence")
check("the held fight is untouched", labels(f)[0] == "High Confidence")
check("a fight BELOW the lowered bar is not lifted", labels(f)[2] == "Medium Confidence")
check("a Low fight far below is not lifted", labels(f)[3] == "Low Confidence")
check("the lift is recorded, not silent",
      f[1]["preview"].get("confidence_lifted_from") == "Medium Confidence")

# Monotonicity itself, stated directly.
f = card((0.80, "High Confidence"), (0.76, "Medium Confidence"),
         (0.62, "Medium Confidence"), (0.61, "Low Confidence"))
_enforce_tier_monotonicity(f)
order = ["Low Confidence", "Medium Confidence", "High Confidence"]
probs = [x["preview"]["favorite_prob"] for x in f]
ranks = [order.index(l) for l in labels(f)]
pairs = list(zip(probs, ranks))
check("no fight outranks a higher-probability fight",
      all(not (p1 > p2 and r1 < r2) for p1, r1 in pairs for p2, r2 in pairs))

# A CAPPED fight must not drag the bar down for everyone above it.
f = card((0.82, "Medium Confidence", True),      # thin record, deliberately demoted
         (0.79, "Medium Confidence", False))
_enforce_tier_monotonicity(f)
check("a capped fight is not treated as evidence the bar is lower",
      labels(f)[1] == "Medium Confidence")
check("and the capped fight is left where the cap put it",
      labels(f)[0] == "Medium Confidence")

# Nothing to do cases.
f = card((0.80, "High Confidence"))
_enforce_tier_monotonicity(f)
check("a one-fight card is untouched", labels(f) == ["High Confidence"])
f = []
_enforce_tier_monotonicity(f)
check("an empty card does not raise", f == [])
f = [{"preview": None}, {"preview": {"favorite_prob": 0.9,
                                     "confidence_label": "High Confidence"}}]
_enforce_tier_monotonicity(f)
check("a missing preview is skipped", f[1]["preview"]["confidence_label"] == "High Confidence")

# The lift can never reach below what hysteresis itself allows.
f = card((0.7400, "High Confidence"), (0.7401, "Medium Confidence"),
         (0.7399, "Medium Confidence"))
_enforce_tier_monotonicity(f)
check("the effective bar cannot fall below the hysteresis floor",
      labels(f)[2] == "Medium Confidence"
      and 0.75 - CONFIDENCE_HYSTERESIS == 0.74)

# The cap flag itself must agree with the label it explains.
check("a thin-record High is reported as capped",
      _confidence_label(0.80, 2, False) == "Medium Confidence"
      and _confidence_capped(0.80, 2, False))
check("a clean High is not reported as capped",
      _confidence_label(0.80, 10, False) == "High Confidence"
      and not _confidence_capped(0.80, 10, False))
check("a fight simply below the bar is not 'capped'",
      not _confidence_capped(0.749, 10, False))

print(f"test_tier_monotonicity: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
