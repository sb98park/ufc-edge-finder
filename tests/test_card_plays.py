"""
The selector, pinned to a real card.

tests/test_plays.py pins the ARITHMETIC against worked examples. This pins the
POLICY against five fights lifted verbatim out of the build for UFC Fight
Night: Nurmagomedov vs. Song -- real prices, real model probabilities, real
edge rows, including the cancelled bout and the two High Confidence favorites
whose prices are too short to play.

The fixture is frozen on purpose. When a rule changes, these numbers move, and
having to update them is the point: it is the only place where "we changed the
staking rule" and "we changed what we would have bet" become the same edit.
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_plays import (  # noqa: E402
    build_card_plays, axis_for_market, label_for, candidates_for_fight,
)
from src.plays import AXIS_OUTCOME, AXIS_METHOD, AXIS_DURATION, MAX_UNITS_PER_CARD  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "card_nurmagomedov_song.json")
FAILURES = []


def check(label, got, want, tol=1e-6):
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:56s} got {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


card = json.load(open(FIXTURE))
by_name = {f["fighter_a"]: f for f in card["fights"]}
built = build_card_plays(card)
plays = {p["label"]: p for p in built["plays"]}

print("\nwhat kind of risk is this")
check("a moneyline is an outcome", axis_for_market("Moneyline"), AXIS_OUTCOME)
# A fighter-specific method bet cannot win unless that fighter wins, so it is
# the same risk as the moneyline and must not buy a second slot on the fight.
check("'Method: KO/TKO' is an outcome, not a method",
      axis_for_market("Method: KO/TKO"), AXIS_OUTCOME)
check("'Fight Method: SUB' IS a method", axis_for_market("Fight Method: SUB"), AXIS_METHOD)
check("round totals are duration", axis_for_market("Total Rounds Over 2.5"), AXIS_DURATION)
check("distance-vs-finish is duration too",
      axis_for_market("Fight Outcome: Goes The Distance"), AXIS_DURATION)
check("an unknown market is not staked", axis_for_market("Fighter Props: Sig Strikes"), None)

print("\nplays are named the way someone would say them out loud")
check("moneyline", label_for("Moneyline", "Denise Gomes", ""), "Denise Gomes to win")
check("negated method", label_for("Fight Method: Not SUB", None, ""), "Does not end by SUB")
check("round total", label_for("Total Rounds Over 2.5", None, ""), "Over 2.5 rounds")

print("\nTHE CASE THAT STARTED THIS: a pick too short to bet")
# Umar at -388 is a High Confidence pick and NOT a play. The model has him at
# 75.1%, the blend at 78.2%, and the price demands 83.5%. This is the whole
# argument for the plays layer existing: conviction and value are different
# questions, and the tiers only ever answered the first.
passed = {p["selection"]: p for p in built["passed"]}
check("Umar is picked", by_name["Umar Nurmagomedov"]["preview"]["favorite"], "Umar Nurmagomedov")
check("  ...at High Confidence",
      by_name["Umar Nurmagomedov"]["preview"]["confidence_label"], "High Confidence")
check("  ...and is NOT played", "Umar Nurmagomedov to win" in plays, False)
check("  ...but is listed as passed, not silently dropped",
      "Umar Nurmagomedov" in passed, True)
check("  ...with the price named as the reason",
      "below the" in passed["Umar Nurmagomedov"]["reason"], True)
check("the other short favorite is passed too", "Rei Tsuruya" in passed, True)

print("\nand a pick the market disagrees with IS bet")
check("Gomes plays", "Denise Gomes to win" in plays, True)
check("  ...at +153", plays["Denise Gomes to win"]["odds_american"], 153)
check("  ...on the blend, not the model's 62.4%",
      plays["Denise Gomes to win"]["blended_prob"] < 0.5, True)

print("\ncancelled fights are not bet")
ce = by_name["Ce Liu"]
check("the fixture really does carry a cancelled bout", bool(ce["cancelled"]), True)
check("  ...and it offered prices", len(ce["edges"]) > 0, True)
taken, refused = candidates_for_fight(ce)
check("  ...yet produces no candidates", len(taken) + len(refused), 0)
check("  ...and no play", any("Ce Liu" in p for p in plays), False)

print("\nwe never stake against our own pick")
# Kai Asakura is the market favorite and the model's underdog. His moneyline
# is the better-priced side of that fight by a distance, and it must never
# appear -- tipping one fighter and betting the other is the one thing this
# section cannot do.
check("the model picks the dog here",
      by_name["Aoriqileng"]["preview"]["favorite"], "Aoriqileng")
check("  ...so Asakura is never staked", any("Asakura" in p for p in plays), False)
check("  ...and our own dog is", "Aoriqileng to win" in plays, True)

print("\nthe tier describes the pick, not every bet on the fight")
check("a moneyline carries its tier",
      plays["Denise Gomes to win"]["tier"], "Medium Confidence")
check("a duration bet carries none",
      plays["Fight goes the distance"]["tier"], None)

print("\ntotals")
check("the card stays inside its ceiling", built["total_units"] <= MAX_UNITS_PER_CARD, True)
check("every dropped candidate says why",
      all(d.get("dropped") for d in built["dropped"]), True)
check("no play is staked at zero", all(p["units"] > 0 for p in built["plays"]), True)
check("one play per fight per axis",
      len({(p["fight_key"], p["axis"]) for p in built["plays"]}), len(built["plays"]))

print("\n" + ("-" * 68))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print(f"the selector holds -- {len(built['plays'])} plays, "
      f"{built['total_units']}U on this fixture")
