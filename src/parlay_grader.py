"""
Settle published parlay slips against recorded results.

WHY THIS EXISTS. Nine slips a week went out for months and nothing wrote
down whether any of them landed. The rules to settle one were never the
missing piece -- templates/site.html has carried gradeCondition(),
gradeLeg() and slipState() since live tracking shipped, and they run
correctly during a card. What was missing is that they run in the browser
and nothing persists the answer: at the final bell the page reloads and
the result is gone. This is those three functions in Python, run after the
fact against data/fight_results.csv instead of a live scoreboard.

THE CLOCK RUNS THE OTHER WAY HERE, and it is the one thing in this file
that would silently invert results rather than fail loudly. ESPN's live
scoreboard counts DOWN inside a round, so the JS computes elapsed time as
300 minus what it shows. fight_results.csv stores ELAPSED time -- every
decision in it reads 5:00, which a countdown would render as 0:00. Both
representations are five characters of mm:ss and neither announces which
it is. Applying the JS conversion to this file would flip every Over/Under
rounds leg, and rounds legs are currently the entire content of both
pinned slips.

WHAT GETS GRADED. The pinned slip for a card, not everything the ledger
holds for it. Before src/parlay_pin.py the builder re-picked on every
render, so one card accumulated 51 distinct bankroll slips and 93 lotto
ones, most alive for a single five-minute build. Grading all of them would
answer "what if you had bet every slip the site ever displayed", which
nobody did. The pinned slip is the recommendation; the rest is churn.

A SLIP IS SCORED AT 1U FLAT AND IS NEVER STAKED. The plays ledger is the
record of money at risk; this is a record of whether reads that were
published would have paid. Every figure it produces is a hypothetical at a
named unit size, the same footing as "at $100 a unit, that's..." on the
Units Tracker -- and the site says so wherever it is shown.

Per the decision recorded in decisions/2026-08-26-parlay-record.md, the
graded record is FREE. The slips themselves stay member-only; whether a
published read paid is evidence, and evidence on this site is public.
"""

import csv

from src.parlay_ledger import LEDGER_PATH, load, write_graded

RESULTS_PATH = "data/fight_results.csv"

# A round is five minutes. Named because the arithmetic below is meaningless
# without it and a bare 300 reads like a timeout.
ROUND_SECONDS = 300.0

# Three-valued, matching the JS: True, False, or None for "no answer yet".
# "void" is a fourth state and deliberately not a boolean -- a voided leg is
# removed from the slip rather than won or lost by it.
VOID = "void"


def _canon(name) -> str:
    return " ".join(str(name or "").strip().lower().split())


def load_results(path: str = RESULTS_PATH) -> dict:
    """
    Fights keyed on frozenset({a, b}) -- the convention the rest of the
    pipeline uses, and order-insensitive, which matters because a card can be
    re-scraped with the corners the other way round.
    """
    fights = {}
    try:
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
    except (FileNotFoundError, OSError):
        return fights
    for r in rows:
        method = (r.get("method") or "").strip().lower()
        slug = ("ko/tko" if ("ko" in method or "tko" in method) else
                "submission" if "sub" in method else
                "decision" if "dec" in method else method)
        end_round = (r.get("end_round") or "").strip()
        winner = (r.get("winner") or "").strip()
        fights[frozenset({_canon(r.get("fighter_a")), _canon(r.get("fighter_b"))})] = {
            "winner": winner,
            # A draw or no-contest has a result but no winner, which voids a
            # side or method leg while leaving a rounds leg perfectly settlable.
            "no_winner": (not winner) or slug in ("draw", "nc", "no contest"),
            "method_slug": slug,
            "end_round": int(end_round) if end_round.isdigit() else None,
            "end_time": (r.get("end_time") or "").strip(),
            # A DRAW WENT THE DISTANCE. `slug == "decision"` alone made this
            # False for a draw, which does not merely fail to settle -- it
            # INVERTS: "Goes The Distance" graded LOST and "Ends In Finish"
            # graded WON on a fight the judges had just scored. A draw is a
            # judges' decision; that is what a draw is.
            #
            # A NO CONTEST IS GENUINELY UNKNOWN and stays None. An NC can be
            # called at any point -- an eye poke in round one or an overturned
            # decision months later -- so the slug says nothing about length,
            # and grade_condition returns unresolved on None rather than
            # guessing. Roughly 1 card in 5 carries a draw or NC (158 of
            # 8,859 bouts), so this is rare, not hypothetical.
            "went_distance": _went_distance(slug, winner),
            "cancelled": False,
        }
    return fights


def _went_distance(slug: str, winner: str):
    """True / False / None -- None meaning we genuinely cannot tell."""
    s = (slug or "").lower()
    if "decision" in s or s == "dec":
        return True
    if "draw" in s:
        return True                      # a draw IS a judges' decision
    if s in ("nc", "no contest") or (not winner and not s):
        return None                      # an NC can stop at any moment
    return False


def _elapsed_rounds(fight) -> float | None:
    """
    How much of the fight was actually fought, in rounds.

    None when the clock is unusable rather than a guess: a round-two finish at
    0:30 and one at 4:30 fall on opposite sides of an Over 1.5 line, so a
    midpoint assumption would be a coin flip dressed as a settlement.
    """
    if fight.get("end_round") is None:
        return None
    parts = str(fight.get("end_time") or "").split(":")
    if len(parts) != 2 or not parts[0].strip().lstrip("-").isdigit():
        return None
    try:
        elapsed = int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return None
    # ELAPSED, not remaining. See the module docstring.
    return (fight["end_round"] - 1) + elapsed / ROUND_SECONDS


def grade_condition(cond: dict, fight: dict | None):
    """True / False / None (no answer yet) / VOID."""
    if not fight:
        return None
    if fight.get("cancelled"):
        return VOID
    kind = cond.get("kind")

    if kind == "winner":
        if fight["no_winner"]:
            return VOID
        if not fight["winner"]:
            return None
        return _canon(fight["winner"]) == _canon(cond.get("fighter"))

    if kind == "method":
        if fight["no_winner"]:
            return VOID
        if not fight["method_slug"]:
            return None
        return any(fight["method_slug"].startswith(s) for s in (cond.get("any_of") or []))

    if kind == "rounds":
        # A DRAW STILL WENT A CERTAIN LENGTH. Unlike winner and method this
        # does not void on one: how long the fight lasted is a fact whoever
        # was awarded it.
        #
        # A DECISION IS OVER EVERY OFFERED LINE. Total-rounds lines sit
        # strictly below the scheduled distance (1.5/2.5 on a three-rounder,
        # 3.5/4.5 on a five), so "went the distance" settles all of them
        # without needing to know which distance it was -- fortunate, because
        # the results file does not say.
        if fight["went_distance"]:
            return False if cond.get("op") == "under" else True
        total = _elapsed_rounds(fight)
        if total is None:
            return None
        line = cond.get("line")
        if line is None:
            return None
        return total < line if cond.get("op") == "under" else total > line

    if kind == "distance":
        if fight.get("went_distance") is None:
            return None
        return bool(fight["went_distance"]) is bool(cond.get("value"))

    # An unrecognised predicate is permanently unknown rather than assumed.
    return None


def grade_leg(leg: dict, fights: dict) -> str:
    """'won' | 'lost' | 'void' | 'unresolved'"""
    conds = leg.get("conditions") or []
    # No stored predicate means the market was never gradeable -- see
    # _leg_conditions in parlay_builder. Honest answer is permanently unknown.
    if not conds:
        return "unresolved"
    key = leg.get("fight_key")
    parts = [p for p in str(key or "").split("|") if p.strip()]
    fight = fights.get(frozenset(_canon(p) for p in parts)) if len(parts) == 2 else None
    verdicts = [grade_condition(c, fight) for c in conds]
    if any(v == VOID for v in verdicts):
        return "void"
    # AND across conditions, with unknown absorbing: one false settles the leg
    # immediately, but a missing answer holds everything else open.
    if any(v is False for v in verdicts):
        return "lost"
    if any(v is None for v in verdicts):
        return "unresolved"
    return "won"


def slip_state(states: list[str]) -> str:
    """'cashed' | 'dead' | 'void' | 'open'"""
    if any(s == "lost" for s in states):
        return "dead"
    live = [s for s in states if s != "void"]
    if not live:
        return "void"
    if any(s == "unresolved" for s in live):
        return "open"
    return "cashed" if all(s == "won" for s in live) else "open"


def effective_decimal(row: dict, states: list[str]) -> float:
    """
    What the slip actually pays, with voided legs divided out the way a book
    would settle it -- the slip shortens rather than dying.

    Priced at FIRST PUBLICATION. decimal_odds moves with the market on a
    pinned slip, so grading at the current number would quietly let the record
    shop for a better one after the fact.

    NOTHING VOIDED IS THE COMMON CASE, and it uses combined_decimal_first --
    the whole-slip price, recorded since the ledger was written. Rebuilding it
    from the legs would be worse rather than equivalent: decimal_odds_first is
    a newer field, so for any slip already published when it shipped the
    per-leg figures are stamped at their next render rather than at their
    actual first publication. The combined number has no such gap.

    Only a void forces the per-leg path, because dividing one leg back out
    needs that leg's own number and there is nowhere else to get it.
    """
    legs = row.get("legs") or []
    if not any(st == "void" for st in states):
        try:
            d = float(row.get("combined_decimal_first")
                      or row.get("combined_decimal") or 0.0)
            if d > 0:
                return d
        except (TypeError, ValueError):
            pass
    dec = 1.0
    for leg, st in zip(legs, states):
        if st == "void":
            continue
        try:
            d = float(leg.get("decimal_odds_first") or leg.get("decimal_odds") or 1.0)
        except (TypeError, ValueError):
            d = 1.0
        dec *= d if d > 0 else 1.0
    return dec


def grade_slips(rows: list[dict], fights: dict, now: str,
                only_slip_ids: set | None = None) -> int:
    """
    Settle every ungraded slip we now have results for. Returns how many
    changed. A slip this cannot settle stays ungraded rather than guessed at.
    """
    changed = 0
    for r in rows:
        if r.get("result"):
            continue
        if only_slip_ids is not None and r.get("slip_id") not in only_slip_ids:
            continue
        legs = r.get("legs") or []
        if not legs:
            continue
        states = [grade_leg(l, fights) for l in legs]
        state = slip_state(states)
        if state == "open":
            continue
        dec = effective_decimal(r, states)
        if state == "cashed":
            units = round(dec - 1.0, 2)
        elif state == "void":
            # Money that was never at risk did not lose.
            units = 0.0
        else:
            units = -1.0
        r["result"] = state
        r["leg_states"] = states
        r["legs_won"] = sum(1 for s in states if s == "won")
        r["units_result"] = units          # at 1U flat, never staked
        r["settled_decimal"] = round(dec, 6)
        r["graded_at"] = now
        changed += 1
    return changed


def grade_pinned(pins: dict, now: str, path: str = LEDGER_PATH,
                 results_path: str = RESULTS_PATH) -> int:
    """
    Grade only the slips a card is committed to, and write the ledger back.

    `pins` is data/pinned_parlays.json as loaded by src.parlay_pin.
    """
    wanted = set()
    for tiers in (pins or {}).values():
        for entry in (tiers or {}).values():
            sid = ((entry or {}).get("snapshot") or {}).get("slip_id")
            if sid:
                wanted.add(sid)
    if not wanted:
        return 0
    rows = load(path)
    if not rows:
        return 0
    changed = grade_slips(rows, load_results(results_path), now, only_slip_ids=wanted)
    if changed:
        write_graded(rows, path)
        print(f"[parlay_grader] settled {changed} pinned slip(s)")
    return changed


def summarise(rows: list[dict]) -> dict:
    """
    The published-not-staked record, for the block under All calls.

    Deliberately no staked column: nothing here was staked, and a column of
    zeros reads as a loss rather than as an abstention. `units_flat` is the
    only money figure and it is a hypothetical.
    """
    graded = [r for r in rows if r.get("result") in ("cashed", "dead", "void")]
    by_tier = {}
    for r in graded:
        t = by_tier.setdefault(r.get("tier") or "other",
                               {"n": 0, "cashed": 0, "units_flat": 0.0})
        t["n"] += 1
        t["cashed"] += 1 if r["result"] == "cashed" else 0
        t["units_flat"] += float(r.get("units_result") or 0.0)
    for t in by_tier.values():
        t["units_flat"] = round(t["units_flat"], 2)
    total = round(sum(float(r.get("units_result") or 0.0) for r in graded), 2)
    cashed = sum(1 for r in graded if r["result"] == "cashed")
    return {
        "n": len(graded),
        "cashed": cashed,
        "hit_pct": round(100.0 * cashed / len(graded), 1) if graded else None,
        "units_flat": total,
        "by_tier": by_tier,
        "events": len({r.get("event") for r in graded if r.get("event")}),
    }


def summarise_by_event(rows: list[dict]) -> dict:
    """Graded slips keyed by event name, newest-agnostic -- the caller orders."""
    out = {}
    for r in rows:
        if r.get("result") not in ("cashed", "dead", "void"):
            continue
        out.setdefault(r.get("event") or "", []).append(r)
    return out
