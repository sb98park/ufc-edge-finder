"""
One card listed under three names is still one card.

On 2026-10-31 the same eleven bouts were tracked three times:

  UFC Fight Night: Las Vegas            the venue placeholder, used before
                                        the main event was announced
  UFC Fight Night: Moicano vs. Nolan    the real billing
  UFC Fight Night: Nolan vs. Moicano    the same billing, reversed

The existing merge only handled a provisional name being EXTENDED
("UFC 332" -> "UFC 332: Silva vs. Wang"), and none of these three extends
another, so none merged and the upcoming-fights list showed the card three
times.

WHAT MAKES THE MERGE SAFE is the rule the function already states: two
genuinely different events on one date do not share fighters. These share
ALL of them. Subset is deliberately not enough -- a card that lost a bout to
a cancellation is a subset of its own earlier self -- so only an IDENTICAL
bout set triggers this.

Synthetic cards only; nothing here reads data/.
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


COLS = ["event_name", "event_date", "fighter_a", "fighter_b", "card_position", "bout_order"]


def write(rows):
    p = os.path.join(tempfile.mkdtemp(), "future_cards.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    return p


def card(name, date, bouts, main=0):
    return [{"event_name": name, "event_date": date, "fighter_a": a, "fighter_b": b,
             "card_position": "Main Event" if i == main else "Prelims", "bout_order": i + 1}
            for i, (a, b) in enumerate(bouts)]


BOUTS = [("Renato Moicano", "Tom Nolan"), ("Ann Doe", "Bea Roe"), ("Cal Fox", "Dan Gray")]

# --- the real case ---------------------------------------------------------
p = write(card("UFC Fight Night: Las Vegas", "2026-10-31", BOUTS)
          + card("UFC Fight Night: Nolan vs. Moicano", "2026-10-31", BOUTS)
          + card("UFC Fight Night: Moicano vs. Nolan", "2026-10-31", BOUTS))
removed = deduplicate_tracked_fights(p)
left = {r["event_name"] for r in csv.DictReader(open(p, encoding="utf-8"))}
check("two of the three groups are dropped", removed == len(BOUTS) * 2)
check("exactly one name survives", len(left) == 1)
# Read off the card, not guessed: the main event row is Moicano vs Nolan.
check("  ...the one the main event bills", left == {"UFC Fight Night: Moicano vs. Nolan"})

# --- two REAL cards on one date must survive -------------------------------
# The function's own stated reason this is safe: different events on one date
# do not share fighters. Merging these would be the serious failure.
other = [("Ed Hall", "Fay Ives"), ("Gus Jones", "Hal King")]
p2 = write(card("UFC Fight Night: Moicano vs. Nolan", "2026-10-31", BOUTS)
           + card("Contender Series 2026-10-31", "2026-10-31", other))
check("two genuinely different cards on one date are left alone",
      deduplicate_tracked_fights(p2) == 0)
check("  ...and both names survive",
      len({r["event_name"] for r in csv.DictReader(open(p2, encoding="utf-8"))}) == 2)

# --- a SUBSET is not a duplicate -------------------------------------------
# A card that lost a bout to a cancellation is a subset of its earlier self.
# Only the prefix rule may act on that; this one must not.
p3 = write(card("UFC Fight Night: Alpha vs. Beta", "2026-11-07", BOUTS)
           + card("UFC Fight Night: Gamma vs. Delta", "2026-11-07", BOUTS[:2]))
check("a subset does not trigger the identical-bouts merge",
      deduplicate_tracked_fights(p3) == 0)

# --- the prefix rule still works -------------------------------------------
p4 = write(card("UFC 332", "2026-12-05", BOUTS)
           + card("UFC 332: Silva vs. Wang", "2026-12-05", BOUTS))
check("a provisional name still merges into its extension",
      deduplicate_tracked_fights(p4) == len(BOUTS))
check("  ...keeping the full name",
      {r["event_name"] for r in csv.DictReader(open(p4, encoding="utf-8"))}
      == {"UFC 332: Silva vs. Wang"})

# --- no main event row: fall back to the longest name ----------------------
p5 = write([dict(r, card_position="Prelims") for r in
            card("UFC Fight Night: Vegas", "2027-01-09", BOUTS)]
           + [dict(r, card_position="Prelims") for r in
              card("UFC Fight Night: Someone vs. Other", "2027-01-09", BOUTS)])
deduplicate_tracked_fights(p5)
check("with no main event the longest name wins",
      {r["event_name"] for r in csv.DictReader(open(p5, encoding="utf-8"))}
      == {"UFC Fight Night: Someone vs. Other"})

# --- a single card is untouched -------------------------------------------
p6 = write(card("UFC Fight Night: Solo vs. Lone", "2027-02-06", BOUTS))
check("one card on its own is left alone", deduplicate_tracked_fights(p6) == 0)

print(f"{'PASS' if not fail else 'FAIL'}: test_duplicate_card_names -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
