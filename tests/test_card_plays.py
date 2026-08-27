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

import sys, os, json, copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_plays import (  # noqa: E402
    build_card_plays, axis_for_market, label_for, candidates_for_fight,
)
from src.plays import AXIS_OUTCOME, AXIS_MANNER, MAX_UNITS_PER_CARD  # noqa: E402

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

def at_book(raw, book="DraftKings"):
    """
    The same card with every price attributed to a real sportsbook.

    THE FIXTURE IS PRICED AT POLYMARKET, and since the bettable-venue rule
    landed that means nothing on it can be staked -- a reference line produces
    a PICK but never a STAKE. That rule is the thing being protected, so the
    fixture keeps its Polymarket provenance and the ladder assertions below
    run against this copy instead. Testing the ladder on a card that cannot
    be staked at all would prove nothing about the ladder.
    """
    out = copy.deepcopy(raw)
    for fight in out.get("fights") or []:
        for edge in fight.get("edges") or []:
            # BOTH FIELDS. _venue prefers best_book and only falls back to
            # source, so rewriting source alone leaves the row still claiming
            # Polymarket and this helper silently does nothing.
            edge["source"] = book
            edge["best_book"] = book
    return out


# EVERY SIZING ASSERTION READS THE BOOK-PRICED COPY. On the Polymarket
# original the venue rule refuses the whole card, so "nothing outside the
# ladder is staked" and friends would pass vacuously on a card where nothing
# is staked at all -- a green test proving the opposite of what it claims.
# `built` is kept for the venue rule's own assertions, which need the
# reference-priced card to have anything to say.
_book = build_card_plays(at_book(card))
_bplays = {p["label"]: p for p in _book["plays"]}
# With the discretionary layer paused, everything outside the ladder is sized
# and then held back rather than never computed -- so the assertions about
# SIZING still have something to look at. See DISCRETIONARY_PLAYS.
_bshelved = {p["label"]: p for p in _book["shelved"]}
shelved = {p["label"]: p for p in built["shelved"]}
sizeable = {**_bshelved, **_bplays}

print("\nwhat kind of risk is this")
check("a moneyline is an outcome", axis_for_market("Moneyline"), AXIS_OUTCOME)
# A fighter-specific method bet cannot win unless that fighter wins, so it is
# the same risk as the moneyline and must not buy a second slot on the fight.
check("'Method: KO/TKO' is an outcome, not a method",
      axis_for_market("Method: KO/TKO"), AXIS_OUTCOME)
# ONE AXIS FOR EVERY PROP. Method and duration were separate, and the card
# published both "Fight goes the distance" and "Does not end by KO/TKO" on the
# same bout -- two rows, 4 units, one thing happening.
check("'Fight Method: SUB' is about the manner",
      axis_for_market("Fight Method: SUB"), AXIS_MANNER)
check("so is a round total", axis_for_market("Total Rounds Over 2.5"), AXIS_MANNER)
check("so is distance-vs-finish",
      axis_for_market("Fight Outcome: Goes The Distance"), AXIS_MANNER)
check("  ...so one bout can only carry one of them",
      len({axis_for_market("Fight Method: SUB"), axis_for_market("Total Rounds Over 2.5"),
           axis_for_market("Fight Outcome: Goes The Distance")}), 1)

print("\na negated method prop is never staked")
# "Does not end by SUB" at -102 is a near-certainty wearing a coinflip's
# price: it wins on a decision, a KO, a DQ or a doctor stoppage. Being right
# every week adds variance and nothing else.
check("'Fight Method: Not SUB' has no axis", axis_for_market("Fight Method: Not SUB"), None)
check("nor does 'Not KO/TKO'", axis_for_market("Fight Method: Not KO/TKO"), None)
check("  ...and none reaches the card",
      any(p["market"].startswith("Fight Method: Not") for p in _book["plays"]), False)
check("an unknown market is not staked", axis_for_market("Fighter Props: Sig Strikes"), None)

print("\nplays are named the way someone would say them out loud")
check("moneyline is named the way the book names it",
      label_for("Moneyline", "Denise Gomes", ""), "Denise Gomes Moneyline")
check("round total", label_for("Total Rounds Under 1.5", None, ""), "Under 1.5 rounds")
check("round total", label_for("Total Rounds Over 2.5", None, ""), "Over 2.5 rounds")

print("\nTHE LADDER TIERS ARE PLAYED ON THE RECORD, NOT ON THE HURDLE")
# This test used to assert the opposite, and the record says the old rule was
# wrong. Measured across every graded pick: the locks and high-confidence
# picks the EV hurdle REFUSED went 12-0 for +21.10U, and -- the part that
# matters, since 12-0 is above expectation -- they averaged +1.02% EV per unit
# on our own blended probability. They were thin positive-expectation bets
# under a threshold built for the props board, not negative ones that got
# lucky. And the published record IS every lock at 10U and every high pick at
# 5U, so a plays section that skips half of them is not the system the
# landing page draws a curve of.
check("Umar is picked", by_name["Umar Nurmagomedov"]["preview"]["favorite"], "Umar Nurmagomedov")
check("  ...at High Confidence",
      by_name["Umar Nurmagomedov"]["preview"]["confidence_label"], "High Confidence")
check("  ...and IS played, at a book price", "Umar Nurmagomedov Moneyline" in _bplays, True)
check("  ...at the ladder stake, not a Kelly size",
      _bplays["Umar Nurmagomedov Moneyline"]["units"], 5.0)
check("  ...flagged as riding the ladder",
      _bplays["Umar Nurmagomedov Moneyline"]["on_ladder"], True)
check("the other short favorite plays too", "Rei Tsuruya Moneyline" in _bplays, True)
# The price is still short enough that the hurdle would refuse it, and the row
# says so honestly: 5 units to win less than one.
check("  ...and the row is honest about what that returns",
      _bplays["Rei Tsuruya Moneyline"]["to_win"] < 1.0, True)
check("the HURDLE never benches a ladder pick",
      any(p["tier"] in ("Lock of the Week", "High Confidence")
          and "reference line" not in (p.get("reason") or "")
          for p in _book["passed"]), False)

print("\nA REFERENCE LINE MAKES A PICK, NEVER A STAKE")
# The rule the fixture's own Polymarket provenance exists to protect: staking
# a vig-free midpoint would compute units, ROI and the bankroll at prices no
# book offers, so the record would run flatteringly high forever while
# describing bets nobody could place.
check("nothing on a Polymarket-priced card is staked", len(built["plays"]), 0)
check("  ...and the ladder picks are not silently dropped",
      len([p for p in built["passed"] if p["tier"] in ("Lock of the Week", "High Confidence")]) > 0,
      True)
check("  ...each saying the price is not one you can bet",
      all("reference line" in (p.get("reason") or "")
          for p in built["passed"] if p["market"] == "Moneyline"), True)

print("\nTHE DISCRETIONARY LAYER IS SIZED, THEN HELD BACK")
# Backtested over every graded pick: ladder-only returned +60.30U on 150U
# staked, ladder-plus-discretionary +55.94U on 184U. The layer cost money, so
# it is paused -- but the arithmetic still runs, so the pause can be lifted
# without rebuilding it and the page can say what it is not betting.
# Against the book-priced copy for the same reason as the ladder above: on
# the Polymarket original nothing is staked at all, so "nothing outside the
# ladder is staked" would pass on a card where nothing is staked full stop.
check("nothing outside the ladder is staked",
      all(p["on_ladder"] for p in _book["plays"]), True)
check("  ...and there is something on the card to say that about",
      len(_book["plays"]) > 0, True)
check("  ...and the rest is shelved, not discarded", len(_book["shelved"]) > 0, True)
check("  ...with a flag saying which mode this is", _book["discretionary_on"], False)
check("Gomes is still sized", "Denise Gomes Moneyline" in _bshelved, True)
check("  ...at +153", _bshelved["Denise Gomes Moneyline"]["odds_american"], 153)
check("  ...on the blend, not the model's 62.4%",
      _bshelved["Denise Gomes Moneyline"]["blended_prob"] < 0.5, True)
check("  ...and says what it returns if it lands",
      _bshelved["Denise Gomes Moneyline"]["to_win"],
      round(_bshelved["Denise Gomes Moneyline"]["units"] * 1.53, 2))
check("  ...but is not on the card", "Denise Gomes Moneyline" in _bplays, False)

print("\ncancelled fights are not bet")
ce = by_name["Ce Liu"]
check("the fixture really does carry a cancelled bout", bool(ce["cancelled"]), True)
check("  ...and it offered prices", len(ce["edges"]) > 0, True)
taken, refused = candidates_for_fight(ce)
check("  ...yet produces no candidates", len(taken) + len(refused), 0)
check("  ...and no play", any("Ce Liu" in p for p in plays), False)

print("\nand a cap invented later cannot cancel a bet the record is made of")
# Three locks is 30U and the card ceiling is 20U. The ladder is placed
# regardless; what it leaves behind is the discretionary budget, which on a
# card like that is nothing -- and that is the right answer.
from src.plays import select_card, MAX_UNITS_PER_CARD  # noqa: E402
_locks = [{"fight_id": f"L{i}", "axis": AXIS_OUTCOME, "units": 10.0,
           "ev_per_unit": 0.01, "priority": 2, "caps_exempt": True} for i in range(3)]
_extra = [{"fight_id": "X", "axis": AXIS_MANNER, "units": 3.0, "ev_per_unit": 0.9}]
_card = select_card(_locks + _extra)
check(f"all three locks place, past the {MAX_UNITS_PER_CARD:.0f}U ceiling",
      sum(1 for p in _card["plays"] if p.get("caps_exempt")), 3)
check("  ...and the discretionary play is what yields",
      any(d["fight_id"] == "X" for d in _card["dropped"]), True)

print("\nwe never stake against our own pick")
# Kai Asakura is the market favorite and the model's underdog. His moneyline
# is the better-priced side of that fight by a distance, and it must never
# appear -- tipping one fighter and betting the other is the one thing this
# section cannot do.
check("the model picks the dog here",
      by_name["Aoriqileng"]["preview"]["favorite"], "Aoriqileng")
check("  ...so Asakura is never staked", any("Asakura" in p for p in plays), False)
# Aoriqileng is ALSO the card's worst disagreement -- market 20.5%, model
# 53.5% -- so the sanity cap refuses him even though the arithmetic likes him.
check("  ...and a 33-point disagreement is refused, not bet",
      "Aoriqileng Moneyline" in sizeable, False)

print("\nthe tier describes the pick, not every bet on the fight")
check("a moneyline carries its tier",
      sizeable["Denise Gomes Moneyline"]["tier"], "Medium Confidence")
check("  ...and Medium is NOT on the ladder -- it still has to earn its place",
      sizeable["Denise Gomes Moneyline"]["on_ladder"], False)
check("a prop carries none",
      [p for p in _book["shelved"] if p["is_prop"]][0]["tier"], None)

print("\nA COMMITTED PLAY IS NEVER LISTED AS UNPLAYED")
# Denise Gomes appeared as a live 2U play and as a "Picked, not played" pick
# on the same screen. passed was built from THIS render's plays alone, so once
# a moneyline was on the board, any later render where its price no longer
# cleared the hurdle reported it as unbet.
_gomes = [f for f in card["fights"] if f["fighter_b"] == "Denise Gomes"][0]
_moved = json.loads(json.dumps(card))
for _f in _moved["fights"]:
    if _f["fighter_b"] == "Denise Gomes":
        for _e in _f["edges"]:
            if _e["market"] == "Moneyline" and _e["fighter"] == "Denise Gomes":
                # The line comes all the way in; the edge is gone.
                _e.update(odds_american=-140, book_fair_prob=0.583, blended_prob=0.595)
_committed = [{"fight_id": f"{_gomes['fighter_a']}|{_gomes['fighter_b']}",
               "axis": AXIS_OUTCOME, "units": 2.0}]
_later = build_card_plays(_moved, committed=_committed)
check("her price no longer qualifies", "Denise Gomes Moneyline" in
      {p["label"] for p in _later["plays"]}, False)
check("  ...and she is STILL not listed as unplayed",
      any(p["selection"] == "Denise Gomes" for p in _later["passed"]), False)
# The other direction, so this is testing the fix and not just its absence:
# the SAME moved price with nothing committed does surface as unplayed.
_uncommitted = build_card_plays(_moved)
check("  ...while the same pick, never bet, does show as unplayed",
      any(p["selection"] == "Denise Gomes" for p in _uncommitted["passed"]), True)
check("  ...with the price given as the reason",
      "below the" in next(p["reason"] for p in _uncommitted["passed"]
                          if p["selection"] == "Denise Gomes"), True)

print("\ntotals")
check("the card stays inside its ceiling", _book["total_units"] <= MAX_UNITS_PER_CARD, True)
check("every dropped candidate says why",
      all(d.get("dropped") for d in _book["dropped"]), True)
check("no play is staked at zero", all(p["units"] > 0 for p in _book["plays"]), True)
check("one play per fight per axis",
      len({(p["fight_key"], p["axis"]) for p in _book["plays"]}), len(_book["plays"]))

print("\n" + ("-" * 68))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print(f"the selector holds -- {len(_book['plays'])} plays, "
      f"{_book['total_units']}U on this fixture")
