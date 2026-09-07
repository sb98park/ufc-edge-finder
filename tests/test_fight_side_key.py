"""
One fight must key the same however its fighters are spelled or ordered.

A fight-level market row carries its side as "A vs B", and dedupe was
normalising that WHOLE string. _normalize_name's middle-name rule only fires
on a bare name, so the priced feed's "Jean Silva vs Jose Miguel Delgado" and
the projection's "Jean Silva vs Jose Delgado" produced two keys for one
fight. Both rows survived, and the card printed every rounds line twice at
identical probabilities:

    FAIL [round-monotonic] Under 3.5 and Under 3.5 are both 53.9%
    FAIL [round-monotonic] Under 4.5 and Under 4.5 are both 59.4%

which reads like two lines disagreeing and is really one line duplicated.

Synthetic fixtures only; nothing here reads data/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_matcher import _dedupe_market_rows, _fight_side_key  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


# The regression: a middle name present in one source and not the other.
check("middle-name variant keys the same",
      _fight_side_key("Jean Silva vs Jose Miguel Delgado")
      == _fight_side_key("Jean Silva vs Jose Delgado"))

# Corners get swapped between re-scrapes (CLAUDE.md s4), so order must not matter.
check("corner order does not matter",
      _fight_side_key("Jose Delgado vs Jean Silva")
      == _fight_side_key("Jean Silva vs Jose Miguel Delgado"))

# The stroked-l case that motivated the original dedupe.
check("stroked letters fold",
      _fight_side_key("Klaudia Syguła vs Nora Cornolle")
      == _fight_side_key("Klaudia Sygula vs Nora Cornolle"))

check("'vs.' with a period is the same separator",
      _fight_side_key("Jean Silva vs. Jose Delgado")
      == _fight_side_key("Jean Silva vs Jose Delgado"))

# It must NOT collapse genuinely different fights.
check("different fights stay distinct",
      _fight_side_key("Jean Silva vs Jose Delgado")
      != _fight_side_key("Jean Silva vs Mario Pinto"))

# A bare name still folds exactly as before -- this is _normalize_name at a
# different granularity, not a new fold.
check("a bare name still folds", _fight_side_key("Jose Miguel Delgado") == "jose delgado")
check("a non-fight label is passed through the same helper",
      _fight_side_key("Under 2.5") == _fight_side_key("Under 2.5"))

# End to end through the dedupe, which is where the duplicate rows survived.
rows = [
    {"market": "Total Rounds Under 3.5", "fighter": "Jean Silva vs Jose Miguel Delgado",
     "model_prob": 0.5393, "has_line": True},
    {"market": "Total Rounds Under 3.5", "fighter": "Jean Silva vs Jose Delgado",
     "model_prob": 0.5393, "has_line": False},
    {"market": "Total Rounds Under 4.5", "fighter": "Jean Silva vs Jose Miguel Delgado",
     "model_prob": 0.5942, "has_line": True},
    {"market": "Total Rounds Under 4.5", "fighter": "Jose Delgado vs Jean Silva",
     "model_prob": 0.5942, "has_line": False},
]
out = _dedupe_market_rows(rows)
check("the duplicate pair collapses to one row per line", len(out) == 2)
# Priced beats unpriced -- a model-only row carries strictly less information.
check("the priced row is the survivor", all(r.get("has_line") for r in out))
check("both distinct lines are still present",
      {r["market"] for r in out} == {"Total Rounds Under 3.5", "Total Rounds Under 4.5"})

# Two genuinely different fights must both survive.
two = _dedupe_market_rows([
    {"market": "Total Rounds Under 2.5", "fighter": "A One vs B Two", "has_line": True},
    {"market": "Total Rounds Under 2.5", "fighter": "C Three vs D Four", "has_line": True},
])
check("two different fights are not merged", len(two) == 2)

print(f"{'PASS' if not fail else 'FAIL'}: test_fight_side_key -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
