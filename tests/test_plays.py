"""
The staking rule, pinned to worked examples.

These are the specification. Every number here was computed by hand from the
formulas in src/plays.py before being written down, so a change that "fixes"
one of these is changing the rule, not the code.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plays import (  # noqa: E402
    decimal_odds, implied_prob, ev_per_unit, required_prob, kelly_fraction,
    size_play, select_card, HURDLE_MONEYLINE, HURDLE_PROP,
    PROP_CAP_UNITS, MAX_UNITS_PER_FIGHT, MAX_UNITS_PER_CARD, MAX_UNITS_PER_AXIS,
    TIER_CAP_UNITS,
    AXIS_OUTCOME, AXIS_METHOD, AXIS_DURATION,
)

FAILURES = []


def check(label, got, want, tol=1e-3):
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:52s} got {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


print("\nodds conversion")
check("decimal(-150)", decimal_odds(-150), 1.6667)
check("decimal(+150)", decimal_odds(150), 2.5)
check("implied(-1000)", implied_prob(-1000), 0.9091)
check("implied(+300)", implied_prob(300), 0.25)
check("implied(+100) == implied(-100)", implied_prob(100), implied_prob(-100))

print("\nthe hurdle demands MORE absolute edge as the price shortens")
for price, want_pts in [(-1000, 4.55), (-450, 4.09), (-190, 3.27), (160, 1.92), (300, 1.25)]:
    edge = (required_prob(price, HURDLE_MONEYLINE) - implied_prob(price)) * 100
    check(f"edge required at {price:+}", edge, want_pts, tol=0.02)

print("\nexpected value per unit risked")
check("EV: 75% at -190", ev_per_unit(0.75, -190), 0.1447)
check("EV: 45% at +160", ev_per_unit(0.45, 160), 0.17)
check("EV: a fair coin at +100 is zero", ev_per_unit(0.5, 100), 0.0)
check("EV: 95% at -1000 is thin", ev_per_unit(0.95, -1000), 0.045)

print("\nraw Kelly is violent at short prices -- this is why it gets clamped")
check("Kelly: 97% at -1000 wants 67% of bankroll", kelly_fraction(0.97, -1000), 0.67)
check("Kelly: 50% at +150 wants 16.7%", kelly_fraction(0.50, 150), 0.1667)

print("\nTHE CASE THAT STARTED THIS: Over 0.5 rounds at -1000")
# At the 10% prop hurdle a -1000 price needs 0.909 * 1.10 = 100%, which no
# honest model ever prints. So the short end of the props board is closed
# outright -- not by a banned-price list but by arithmetic. That is the
# designed behaviour and it is worth a test of its own, because it is the
# single decision most likely to be second-guessed later.
r = size_play(0.95, -1000, "Lock of the Week", is_prop=True)
check("95% read is rejected", r["play"], False)
check("  ...and says why", "below the" in (r["reason"] or ""), True)
r = size_play(0.97, -1000, "Lock of the Week", is_prop=True)
check("97% is ALSO rejected -- a -1000 prop needs 100%", r["play"], False)
check("required prob at -1000 on the prop hurdle", required_prob(-1000, HURDLE_PROP), 1.0)
check("  ...so no prop shorter than -1000 can ever play",
      required_prob(-1100, HURDLE_PROP) > 1.0, True)

print("\nthe same price as a MONEYLINE, where the hurdle is half as stiff")
r = size_play(0.97, -1000, "Lock of the Week", is_prop=False)
check("97% at -1000 qualifies as a moneyline", r["play"], True)
check("  ...and the tier cap sizes it, not Kelly", r["units"], 10.0)
check("  ...flagged as capped", r["capped"], True)

print("\nmoneylines size below their ceiling, which is the point of quarter Kelly")
r = size_play(0.75, -190, "High Confidence")
check("75% at -190 plays", r["play"], True)
check("  ...at 5U, its tier ceiling", r["units"], 5.0)
r = size_play(0.45, 160, "Lock of the Week")
check("45% at +160 plays", r["play"], True)
check("  ...at 2.5U, well under the 10U cap", r["units"], 2.5)
check("  ...so the cap is not binding", r.get("capped"), False)

print("\nfloors and rejections")
# +160 needs 40.4%; 41% clears it by 0.6pt and Kelly sizes it at 1.03U, so it
# plays at the floor. 27% at +300 clears its hurdle too but Kelly only wants
# 0.67U -- that is what the floor is for.
r = size_play(0.41, 160, "High Confidence")
check("a thin but qualifying edge plays at the floor", r["units"], 1.0)
r = size_play(0.27, 300, "High Confidence")
check("qualifying but under 1U of Kelly is rejected", r["play"], False)
check("  ...on the raw size, not the rounded one", "0.67U" in (r["reason"] or ""), True)
r = size_play(0.50, 100, "Low Confidence")
check("a coin flip at evens never plays", r["play"], False)
r = size_play(0.75, -190, "No Such Tier")
check("an unknown tier plays nothing", r["play"], False)

print("\nprops face a stiffer hurdle than moneylines")
p, price = 0.70, -190
check("70% at -190 clears the moneyline hurdle",
      size_play(p, price, "High Confidence", is_prop=False)["play"], True)
check("  ...and fails the prop hurdle",
      size_play(p, price, "High Confidence", is_prop=True)["play"], False)
check("prop hurdle is double the moneyline one", HURDLE_PROP, 2 * HURDLE_MONEYLINE)

print("\ncorrelation: one play per fight per axis")
card = select_card([
    {"fight_id": "f1", "axis": AXIS_DURATION, "units": 3.0, "ev_per_unit": 0.20, "sel": "Over 2.5"},
    {"fight_id": "f1", "axis": AXIS_DURATION, "units": 3.0, "ev_per_unit": 0.09, "sel": "Decision"},
    {"fight_id": "f1", "axis": AXIS_OUTCOME,  "units": 5.0, "ev_per_unit": 0.15, "sel": "ML"},
])
check("the duplicate duration play is dropped", len(card["plays"]), 2)
check("  ...and the better EV kept the slot", card["plays"][0]["sel"], "Over 2.5")
check("  ...with a reason recorded", "already have a duration play" in card["dropped"][0]["dropped"], True)

print("\nexposure caps")
card = select_card([
    {"fight_id": "f1", "axis": AXIS_OUTCOME,  "units": 10.0, "ev_per_unit": 0.30},
    {"fight_id": "f1", "axis": AXIS_METHOD,   "units": 3.0,  "ev_per_unit": 0.20},
])
check(f"one fight cannot exceed {MAX_UNITS_PER_FIGHT:.0f}U", card["total_units"], 10.0)

print("\nthe axis cap, and the thing it must never do")
# A maximum-stake Lock has to survive every rule. An axis ceiling below the
# largest tier cap would silently forbid the product's headline bet, and only
# on the card where it mattered most.
check("no axis ceiling below the biggest single stake",
      MAX_UNITS_PER_AXIS >= max(TIER_CAP_UNITS.values()), True)
card = select_card([
    {"fight_id": "f1", "axis": AXIS_OUTCOME, "units": 10.0, "ev_per_unit": 0.30},
])
check("  ...so a 10U lock still places on its own", card["total_units"], 10.0)
# Six fights leaning the same way on the finish curve is one assumption, not six.
spread = [{"fight_id": f"g{i}", "axis": AXIS_DURATION, "units": 3.0, "ev_per_unit": 0.3 - i * 0.01}
          for i in range(6)]
card = select_card(spread)
check(f"one shared assumption cannot exceed {MAX_UNITS_PER_AXIS:.0f}U",
      card["total_units"] <= MAX_UNITS_PER_AXIS, True)
check("  ...and the ones it refused say so",
      "across the card" in card["dropped"][0]["dropped"], True)
big = [{"fight_id": f"f{i}", "axis": AXIS_OUTCOME, "units": 10.0, "ev_per_unit": 0.3 - i * 0.01}
       for i in range(8)]
card = select_card(big)
check(f"one card cannot exceed {MAX_UNITS_PER_CARD:.0f}U", card["total_units"] <= MAX_UNITS_PER_CARD, True)
check("  ...and the rest are recorded, not silently lost",
      len(card["plays"]) + len(card["dropped"]), 8)

print("\n" + ("-" * 62))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print("all staking rules hold")
