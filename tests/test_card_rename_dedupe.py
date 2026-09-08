"""
A renamed card must collapse; two real cards must not.

deduplicate_tracked_fights grouped by event_name, so it only ever saw
duplicates WITHIN one event. A rename produces the other shape entirely --
every fight present exactly once under each of two names -- which a
per-event pass cannot see.

ESPN published UFC 332 as "UFC 332" and later as "UFC 332: Silva vs. Wang".
Both survived in future_cards.csv and all twelve fights were tracked twice.
backfill_fighters matches by event name and appends per event, so it staged
Khaos Williams and Roberto Soldic once per name in one pass; the two
duplicate roster rows tripped the one-row-per-fighter test and froze the
refresh for eleven hours on 2026-09-07.

The merging half is easy. These tests are mostly about the NOT-merging half,
because collapsing two real cards would be far worse than tracking one
twice.

Synthetic fixtures in a temp dir; nothing here reads data/.
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_discovery import deduplicate_tracked_fights  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


COLS = ["event_name", "event_date", "card_position", "weight_class",
        "fighter_a", "fighter_b", "cancelled", "manually_added"]


def run(rows):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "future_cards.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    removed = deduplicate_tracked_fights(p)
    with open(p, newline="") as fh:
        out = list(csv.DictReader(fh))
    return removed, out


def bout(event, date, a, b):
    return {"event_name": event, "event_date": date, "fighter_a": a, "fighter_b": b,
            "card_position": "Prelims", "weight_class": "Lightweight"}


# --- the regression: a provisional name extended into a full one ----------
rows = ([bout("UFC 332", "2026-10-24", a, b) for a, b in
         (("Khaos Williams", "Roberto Soldic"), ("Natalia Silva", "Wang Cong"))] +
        [bout("UFC 332: Silva vs. Wang", "2026-10-24", a, b) for a, b in
         (("Khaos Williams", "Roberto Soldic"), ("Natalia Silva", "Wang Cong"))])
removed, out = run(rows)
check("the provisional rows are dropped", removed == 2)
check("only the full name survives",
      {r["event_name"] for r in out} == {"UFC 332: Silva vs. Wang"})
check("every fight is kept exactly once", len(out) == 2)

# --- what must NOT merge --------------------------------------------------
# Two real cards on one date. They share no fighters, and neither name
# extends the other.
removed, out = run([bout("UFC 300", "2026-10-24", "A One", "B Two"),
                    bout("UFC Fight Night: X vs. Y", "2026-10-24", "C Three", "D Four")])
check("two unrelated cards on one date are untouched", removed == 0 and len(out) == 2)

# A name that extends another but describes DIFFERENT fights is not a rename.
removed, out = run([bout("UFC 332", "2026-10-24", "A One", "B Two"),
                    bout("UFC 332: Silva vs. Wang", "2026-10-24", "C Three", "D Four")])
check("an extending name with different fights is left alone",
      removed == 0 and len(out) == 2)

# The provisional card carrying a fight the full one does not: dropping it
# would LOSE that bout, so nothing is dropped.
removed, out = run([bout("UFC 332", "2026-10-24", "A One", "B Two"),
                    bout("UFC 332", "2026-10-24", "E Five", "F Six"),
                    bout("UFC 332: Silva vs. Wang", "2026-10-24", "A One", "B Two")])
check("a provisional card with an extra fight is not dropped", removed == 0)
check("...and that extra fight survives",
      any(r["fighter_a"] == "E Five" for r in out))

# Same names, DIFFERENT dates -- a rematch or a moved card, not a duplicate.
removed, out = run([bout("UFC 332", "2026-10-24", "A One", "B Two"),
                    bout("UFC 332: Silva vs. Wang", "2026-11-07", "A One", "B Two")])
check("the same pair on two dates is not merged", removed == 0 and len(out) == 2)

# The boundary must be a real separator, not a prefix collision: "UFC 33"
# is not the provisional name of "UFC 332".
removed, out = run([bout("UFC 33", "2026-10-24", "A One", "B Two"),
                    bout("UFC 332", "2026-10-24", "A One", "B Two")])
check("a bare prefix with no separator is not treated as a rename", removed == 0)

# --- the within-event pass still works ------------------------------------
removed, out = run([bout("UFC 332: Silva vs. Wang", "2026-10-24", "Jose Delgado", "Jean Silva"),
                    bout("UFC 332: Silva vs. Wang", "2026-10-24", "Jose Miguel Delgado", "Jean Silva")])
check("a name variant inside one event still merges", removed == 1 and len(out) == 1)

# --- idempotence: this runs every build -----------------------------------
rows2 = ([bout("UFC 332", "2026-10-24", "A One", "B Two")] +
         [bout("UFC 332: Silva vs. Wang", "2026-10-24", "A One", "B Two")])
r1, out1 = run(rows2)
r2, out2 = run(out1)
check("a second pass removes nothing", r1 == 1 and r2 == 0)

# --- an empty or single-event file is a clean no-op ------------------------
check("empty file is a no-op", run([])[0] == 0)
check("one event is a no-op", run([bout("UFC 332", "2026-10-24", "A One", "B Two")])[0] == 0)

print(f"{'PASS' if not fail else 'FAIL'}: test_card_rename_dedupe -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
