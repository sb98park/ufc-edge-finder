"""
The paths that only run on card night: draw/no-contest grading, the live
method cache, and matching an ESPN result back to our card.

Every check here corresponds to a defect the 2026-08-31 audit reproduced.
"""
import pathlib
import sys

sys.path.insert(0, ".")
from src.card_matcher import _normalize_name                       # noqa: E402
from src.parlay_grader import _went_distance, grade_condition      # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


# ---------------------------------------------- draws and no contests
check("a decision went the distance", _went_distance("decision", "A") is True)
check("a draw went the distance", _went_distance("draw", "") is True)
check("case does not matter", _went_distance("Draw", "") is True)
check("a no contest is UNKNOWN, not False", _went_distance("nc", "") is None)
check("'no contest' spelled out is also unknown", _went_distance("no contest", "") is None)
check("a KO did not go the distance", _went_distance("kotko", "A") is False)
check("a submission did not go the distance", _went_distance("submission", "A") is False)

# The inversion itself: a draw IS a judges' decision, so a distance leg on one
# must settle the same way it would on any other decision.
_draw = {"went_distance": _went_distance("draw", ""), "no_winner": True,
         "method_slug": "draw", "end_round": 3, "end_time": "5:00"}
check("'Goes The Distance' WINS on a draw",
      grade_condition({"kind": "distance", "value": True}, _draw) is True)
check("'Ends In Finish' LOSES on a draw",
      grade_condition({"kind": "distance", "value": False}, _draw) is False)

_nc = {"went_distance": _went_distance("nc", ""), "no_winner": True,
       "method_slug": "nc", "end_round": 1, "end_time": "1:12"}
check("a no contest leaves a distance leg unresolved rather than guessing",
      grade_condition({"kind": "distance", "value": True}, _nc) is None)

# A finish must still lose a distance leg -- guard against over-correcting.
_ko = {"went_distance": _went_distance("kotko", "A"), "no_winner": False,
       "method_slug": "kotko", "end_round": 2, "end_time": "3:01"}
check("'Goes The Distance' still LOSES on a knockout",
      grade_condition({"kind": "distance", "value": True}, _ko) is False)

# ------------------------------------- matching an ESPN result to our card
# Exact-lowercase silently discarded completed bouts. Replaying UFC 329 lost
# three: an accent, a hyphen and a token-order swap.
_card = ["Cong Wang", "Adrian Yanez", "Benoit Saint-Denis", "Morgan Charrière"]
_folded = {_normalize_name(n) for n in _card}
_sorted = {" ".join(sorted(_normalize_name(n).split())) for n in _card}


def _on_card(name):
    return (_normalize_name(name) in _folded
            or " ".join(sorted(_normalize_name(name).split())) in _sorted)


check("token order folds (Ce Liu / Liu Ce)", _on_card("Wang Cong"))
check("accents fold", _on_card("Adrian Yañez"))
check("hyphenation folds", _on_card("Benoît Saint Denis"))
check("Charriere folds -- he is on the Sept 5 card", _on_card("Morgan Charriere"))
check("a fighter genuinely not on the card is still rejected",
      not _on_card("Some Other Guy"))

_rf = pathlib.Path("src/results_fetcher.py").read_text(encoding="utf-8")
check("an unmatched ESPN result is reported, never silently dropped",
      "could not be" in _rf and "skipped_off_card" in _rf)

# ------------------------------------------------ the live method cache
_site = pathlib.Path("templates/site.html").read_text(encoding="utf-8")
_fn = _site[_site.index("async function fetchLiveMethods"):_site.index("function paintMethod")]
check("a failed status response clears `complete`",
      "if (!sr.ok) { complete = false; continue; }" in _fn)
# Four sites must clear it: no $ref, a failed status fetch, a bout with no
# result yet, and a thrown request. Before the fix only two did, and the two
# missing ones are the two that fire on a card that has not started.
check("all four failure paths clear `complete`", _fn.count("complete = false;") == 4)
check("the no-result branch exists", "} else {" in _fn)
check("the map is still only cached when complete",
      "if (complete) methodCache[eventId] = byComp;" in _fn)
check("the live distance rule mirrors the server's",
      "/draw/i.test(_m.slug" in _site and "no ?contest" in _site)

print(f"test_card_night: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
