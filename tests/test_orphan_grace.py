"""
A bout must not be cancelled because ESPN was briefly missing it.

The orphan grace was a COUNT of consecutive resyncs, and at this repo's
cadence that is close to meaningless: refresh.yml is cron "*/5" and the
resync runs every build, so "more than 3 consecutive resyncs" is about
twenty minutes. ESPN routinely serves a partial prelim card for longer than
that while an event is still being built out.

The threshold had also never actually executed -- _bump_orphan_streak raised
on NaN every run until 2026-09-05, so the counter never advanced past 1 and
the branch was dead. Its first live action was to cancel Ramiro Jimenez vs
Rodrigo Vera on the 2026-09-12 Noche card, a bout Polymarket had active and
accepting orders with $130k of liquidity behind it.

The costs are not symmetric. A false cancellation pulls a real fight off the
card and strands a published pick; waiting a day on a genuine scratch costs
nothing, because a scratched bout stays absent for days and is still caught
well before the card.

Synthetic fixtures only; nothing here reads data/ or hits the network.
"""

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_discovery import (ORPHAN_GRACE_HOURS, ORPHAN_STREAK_LIMIT,  # noqa: E402
                                _bump_orphan_streak, _hours_since)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


NOW = dt.datetime(2026, 9, 7, 12, 0, tzinfo=dt.timezone.utc)
now_iso = NOW.isoformat(timespec="seconds")


def ago(**kw):
    return (NOW - dt.timedelta(**kw)).isoformat(timespec="seconds")


def ripe(streak, since):
    """The shipped rule: both conditions must hold."""
    return streak > ORPHAN_STREAK_LIMIT and _hours_since(since, now_iso) >= ORPHAN_GRACE_HOURS


# --- the regression --------------------------------------------------------
# Twenty minutes is what the old count amounted to. It must not be enough,
# however high the streak has climbed.
check("20 minutes missing is NOT cancellable", not ripe(4, ago(minutes=20)))
check("...nor with a streak of 50", not ripe(50, ago(minutes=20)))
check("2 hours missing is NOT cancellable", not ripe(10, ago(hours=2)))
check("23 hours is still inside the grace", not ripe(10, ago(hours=23)))

# --- a genuine scratch is still caught -------------------------------------
check("24 hours with a real streak IS cancellable", ripe(4, ago(hours=24)))
check("3 days missing is cancellable", ripe(20, ago(days=3)))

# --- the streak still has to be real ---------------------------------------
# Duration alone is not enough: the streak is what proves the fetches
# succeeded rather than the resync having been broken for a day.
check("long absence but streak 1 is NOT cancellable", not ripe(1, ago(days=3)))
check("streak exactly at the limit is NOT cancellable (strictly greater)",
      not ripe(ORPHAN_STREAK_LIMIT, ago(days=3)))
check("one above the limit is", ripe(ORPHAN_STREAK_LIMIT + 1, ago(days=3)))

# --- the clock helper itself -----------------------------------------------
check("hours are computed", abs(_hours_since(ago(hours=30), now_iso) - 30.0) < 0.01)
check("an unreadable stamp reads as 0, the SAFE direction",
      _hours_since("not-a-date", now_iso) == 0.0)
check("None reads as 0", _hours_since(None, now_iso) == 0.0)
check("a naive stamp is treated as UTC rather than raising",
      abs(_hours_since("2026-09-07T06:00:00", now_iso) - 6.0) < 0.01)
check("a future stamp never goes negative", _hours_since(ago(hours=-5), now_iso) == 0.0)

# --- a row with no stamp yet must not be instantly ripe --------------------
# Rows written before this field existed carry a streak but no _orphan_since.
# They get stamped on first sight, which restarts the clock -- the safe
# direction, since the alternative is cancelling on a streak accumulated
# under the old rule.
legacy = {"_orphan_streak": 99}
_bump_orphan_streak(legacy)
stamped = legacy.setdefault("_orphan_since", now_iso)
check("a legacy row with a huge streak is not immediately cancellable",
      not ripe(legacy["_orphan_streak"], stamped))

# --- the constants must stay sane against the CI cadence -------------------
# 5-minute cron: the streak limit alone buys ~20 minutes, which is why the
# hours exist. If someone drops the hours back to single digits, this fails.
check("the grace is at least half a day", ORPHAN_GRACE_HOURS >= 12)
check("the streak limit is still meaningful", ORPHAN_STREAK_LIMIT >= 3)

print(f"{'PASS' if not fail else 'FAIL'}: test_orphan_grace -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
