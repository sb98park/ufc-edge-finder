"""
A scratched bout must not outlive its own card.

PRODUCTION CASE THIS PINS. Gabriel Bonfim vs Sean Brady was the original
main event of the 2026-11-07 Fight Night. It was scratched; the UFC re-billed
the card around Cortes Acosta vs Walker, who fought it. Two days later the
Bonfim row was STILL on the active card, still carrying card_position
"Main Event", at _orphan_streak 571 -- so the site named the card after it
and counted it in "13 of 14 results confirmed", where it could never be
confirmed.

TWO INDEPENDENT FAULTS, both pinned here:

1. The stamp was never written. _orphan_since reads back from pandas as NaN
   for an unset cell, and the guard was `not str(v or "").strip()`. NaN is
   truthy, so `v or ""` yields the NaN, and str(NaN) is "nan" -- a non-empty
   string the guard read as "already stamped". So it never stamped, and
   _hours_since("nan") returns its safe 0.0, so the hours half of the rule
   could never be satisfied. CLAUDE.md s4. _bump_orphan_streak documents
   this exact trap for the sibling column; the fix missed this one.

2. Even stamped, it would have restarted the clock the day AFTER the card.
   The 24-hour grace exists to survive ESPN serving a partial prelim card
   before an event -- a pre-card hazard only. Post-card it just preserves a
   phantom on the one day anyone reads the results count.

Synthetic fixtures only; nothing here reads data/ or hits the network.
"""

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_discovery import (ORPHAN_GRACE_HOURS, ORPHAN_STREAK_LIMIT,  # noqa: E402
                                _days_since_event, _is_blank, _orphan_is_ripe)

NOW = dt.datetime(2026, 11, 9, 12, 0, tzinfo=dt.timezone.utc)
now_iso = NOW.isoformat()
ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def ago(**kw):
    return (NOW - dt.timedelta(**kw)).isoformat()


PAST = "2026-11-07"     # two days before NOW
FUTURE = "2026-11-21"

# --- fault 1: the NaN stamp -------------------------------------------------
check("NaN reads as blank, so it gets stamped", _is_blank(float("nan")))
check("an empty cell reads as blank", _is_blank(""))
check("the round-tripped string 'nan' also reads as blank", _is_blank("nan"))
check("None reads as blank", _is_blank(None))
check("a real stamp does NOT read as blank", not _is_blank("2026-11-07T12:00:00+00:00"))
check("a weight class does not read as blank", not _is_blank("Welterweight"))

# The precise expression that shipped. Kept as a literal so this fails if
# anyone reintroduces it.
_nan = float("nan")
check("the old guard really did misread NaN as already-stamped",
      bool(str(_nan or "").strip()))

# --- fault 2: the exact production row -------------------------------------
check("the production row (streak 571, NaN stamp, 2 days post-card) is ripe",
      _orphan_is_ripe(571, _nan, PAST, now_iso))
check("same row BEFORE its card is still protected",
      not _orphan_is_ripe(571, _nan, FUTURE, now_iso))

# --- post-card ripens on the streak alone, pre-card does not ---------------
check("post-card, freshly stamped, above the streak limit -> ripe",
      _orphan_is_ripe(ORPHAN_STREAK_LIMIT + 1, now_iso, PAST, now_iso))
check("pre-card, freshly stamped, above the streak limit -> NOT ripe",
      not _orphan_is_ripe(ORPHAN_STREAK_LIMIT + 1, now_iso, FUTURE, now_iso))
check("pre-card still ripens once the hours are served",
      _orphan_is_ripe(ORPHAN_STREAK_LIMIT + 1, ago(hours=ORPHAN_GRACE_HOURS + 1),
                      FUTURE, now_iso))

# --- the streak limit is NOT bypassed by the post-card branch --------------
# This is the half that keeps a single bad fetch on the morning after a card
# from cancelling a bout that did happen.
check("post-card but at the streak limit -> NOT ripe",
      not _orphan_is_ripe(ORPHAN_STREAK_LIMIT, now_iso, PAST, now_iso))
check("post-card with streak 1 -> NOT ripe",
      not _orphan_is_ripe(1, now_iso, PAST, now_iso))

# --- the day boundary ------------------------------------------------------
check("the card's own day does not count as post-card",
      not _orphan_is_ripe(9, now_iso, NOW.date().isoformat(), now_iso))
check("the day after does",
      _orphan_is_ripe(9, now_iso, (NOW - dt.timedelta(days=1)).date().isoformat(),
                      now_iso))

# --- junk must fail CLOSED, never cancel -----------------------------------
check("an unreadable event_date keeps the conservative pre-card grace",
      not _orphan_is_ripe(9, now_iso, "not-a-date", now_iso))
check("an unreadable event_date reads as -1 days, i.e. still ahead",
      _days_since_event("not-a-date") == -1.0)
check("a missing event_date keeps the grace",
      not _orphan_is_ripe(9, now_iso, None, now_iso))
check("a NaN streak never ripens", not _orphan_is_ripe(_nan, now_iso, PAST, now_iso))
check("a junk streak never ripens", not _orphan_is_ripe("", now_iso, PAST, now_iso))
check("a None streak never ripens", not _orphan_is_ripe(None, now_iso, PAST, now_iso))

# --- a future-dated event is never post-card ------------------------------
check("a card a fortnight out is not post-card", _days_since_event(FUTURE) < 1)

print(f"{'PASS' if not fail else 'FAIL'}: test_orphan_post_card -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
