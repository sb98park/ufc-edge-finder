"""
ESPN re-serving a bout we already cancelled must not kill a live bout.

PRODUCTION CASE. On the 2026-11-07 card, Lucas Brennan withdrew and Carlos
Calderon replaced him, so `Austin Bashi vs Lucas Brennan` was correctly
cancelled and `Austin Bashi vs Carlos Calderon` became the live bout. At
2026-11-04 23:54Z ESPN's card served Bashi vs BRENNAN again -- the dead
pairing. The resync's replacement rule has no notion of direction: it treats
any fresh bout sharing a fighter as superseding the tracked one. So it ran
the substitution BACKWARDS, cancelled the live Calderon bout as "replaced"
by the corpse, and moved the replacement badge onto the Brennan row.

ESPN flipped forward again on a later fetch, but `cancelled` is sticky, so
the damage stuck. Austin Bashi beat Carlos Calderon by submission while our
card said the fight was off, and the published Low Confidence pick on it was
voided instead of graded -- the record read 144/225 where it should have
read 145/226.

The guard: a pairing we have already marked cancelled is the one case where
the direction is known for certain. Re-serving it changes nothing on its own
(such rows are pinned), and it must not be allowed to kill anything else.

Drives the REAL resync against a stubbed ESPN fetch and a temp CSV, rather
than reimplementing the rule. Nothing here reads data/ or hits the network.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv  # noqa: E402

import src.card_discovery as cd  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


EVENT = "UFC Fight Night: Test vs. Case"
DATE = "2099-11-07"          # far future, so the post-card shortcut never fires
COLS = ["event_name", "event_date", "card_position", "weight_class",
        "fighter_a", "fighter_b", "event_start_time_et", "is_womens_division",
        "event_location", "event_main_card_time_et", "replacement",
        "replaced_fighter", "cancelled", "manually_added", "bout_order",
        "_orphan_since", "_orphan_streak"]


def row(a, b, cancelled="", replacement="", replaced="", bout="", pos="Prelims"):
    r = {c: "" for c in COLS}
    r.update({"event_name": EVENT, "event_date": DATE, "card_position": pos,
              "weight_class": "Featherweight", "fighter_a": a, "fighter_b": b,
              "event_start_time_et": "17:00", "is_womens_division": "False",
              "event_location": "Test Arena", "event_main_card_time_et": "20:00",
              "replacement": replacement, "replaced_fighter": replaced,
              "cancelled": cancelled, "bout_order": bout})
    return r


def fresh(a, b, pos="Prelims"):
    return {"event_name": EVENT, "event_date": DATE, "card_position": pos,
            "weight_class": "Featherweight", "fighter_a": a, "fighter_b": b,
            "event_start_time_et": "17:00", "event_main_card_time_et": "20:00",
            "is_womens_division": False, "event_location": "Test Arena",
            "bout_order": ""}


def run_resync(tracked, espn_card):
    """Write tracked rows, stub the fetch, run the real resync, read back."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            for r in tracked:
                w.writerow(r)
        original = cd._fetch_espn_full_card
        cd._fetch_espn_full_card = lambda name, date: list(espn_card)
        try:
            cd.resync_tracked_card_order(path)
        finally:
            cd._fetch_espn_full_card = original
        with open(path, newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    finally:
        os.unlink(path)


def state(rows, a, b):
    """The (cancelled, replacement) state of one pairing, or None if absent."""
    want = frozenset({a.lower(), b.lower()})
    for r in rows:
        if frozenset({r["fighter_a"].lower(), r["fighter_b"].lower()}) == want:
            return (str(r.get("cancelled", "")).strip().lower() == "true",
                    str(r.get("replacement", "")).strip().lower() == "true")
    return None


# --- THE REGRESSION -------------------------------------------------------
# Tracked: the swap already happened. ESPN serves the DEAD pairing.
tracked = [
    row("Austin Bashi", "Lucas Brennan", cancelled="True", bout="9.0"),
    row("Austin Bashi", "Carlos Calderon", replacement="True",
        replaced="Lucas Brennan"),
    row("Other Guy", "Someone Else", bout="1.0"),
]
espn = [fresh("Austin Bashi", "Lucas Brennan"), fresh("Other Guy", "Someone Else")]
out = run_resync(tracked, espn)

live = state(out, "Austin Bashi", "Carlos Calderon")
check("the live bout still exists after a stale listing", live is not None)
check("the live bout is NOT cancelled by a bout we already cancelled",
      live is not None and live[0] is False)
dead = state(out, "Austin Bashi", "Lucas Brennan")
check("the already-cancelled bout stays cancelled", dead is not None and dead[0] is True)
check("the unrelated bout is untouched",
      state(out, "Other Guy", "Someone Else") == (False, False))

# --- THE CASE THE RULE EXISTS FOR MUST STILL WORK -------------------------
# A genuine withdrawal: ESPN swaps the opponent for one we have NOT cancelled.
tracked2 = [
    row("Austin Bashi", "Lucas Brennan", bout="9.0"),
    row("Other Guy", "Someone Else", bout="1.0"),
]
espn2 = [fresh("Austin Bashi", "Carlos Calderon"), fresh("Other Guy", "Someone Else")]
out2 = run_resync(tracked2, espn2)
old = state(out2, "Austin Bashi", "Lucas Brennan")
new = state(out2, "Austin Bashi", "Carlos Calderon")
check("a genuine withdrawal still cancels the old pairing",
      old is not None and old[0] is True)
check("a genuine withdrawal still adds the new pairing", new is not None)
check("the new pairing is flagged as a replacement",
      new is not None and new[1] is True)
check("the new pairing is not itself cancelled",
      new is not None and new[0] is False)

# --- NEITHER SIDE MAY END UP WITH NO LIVE BOUT ---------------------------
# The invariant the production bug violated, stated directly: if ESPN lists a
# fighter at all, he must not be left with every one of his bouts cancelled.
for label, rows_out in (("after a stale listing", out),
                        ("after a genuine withdrawal", out2)):
    bashi = [r for r in rows_out if "Austin Bashi" in (r["fighter_a"], r["fighter_b"])]
    alive = [r for r in bashi
             if str(r.get("cancelled", "")).strip().lower() != "true"]
    check(f"Bashi has exactly one live bout {label}", len(alive) == 1)

print(f"{'PASS' if not fail else 'FAIL'}: test_backwards_substitution -- "
      f"{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
