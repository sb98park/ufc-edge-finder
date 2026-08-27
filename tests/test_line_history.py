"""
The price-history buffer, and the point it must never throw away.

hist[-30:] discarded the OPEN, which is the one quote nothing else can
reconstruct. On 2026-08-26 the DraftKings opening price on two staked fights
had to be recovered from a phone screenshot because this buffer had already
rolled past it. Everything below is about that not happening again.
"""

import sys, os, datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.line_movement import (  # noqa: E402
    _thin_history, MAX_HISTORY_POINTS, RECENT_HISTORY_POINTS, _snapshot_points,
)

FAILURES = []


def check(label, got, want=True):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:58s} got {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def hist(n, minutes=15):
    """
    A plausible line that drifts inside a normal band.

    ODDS THAT WALK OFF TO -2099 ARE NOT TEST DATA, they are a settled market:
    _snapshot_points runs _drop_settled, which strips anything outside
    _SETTLED_CHART_PROB, so a synthetic ramp silently loses two thirds of its
    points before the chart sees them and the assertion below measures the
    wrong thing. Oscillating between roughly -140 and -260 keeps every point a
    real quote.
    """
    start = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
    return [{"odds": -200 + (i % 13 - 6) * 10,
             "timestamp": (start + dt.timedelta(minutes=minutes * i)).isoformat()}
            for i in range(n)]


def span_hours(h):
    a = dt.datetime.fromisoformat(h[0]["timestamp"])
    b = dt.datetime.fromisoformat(h[-1]["timestamp"])
    return (b - a).total_seconds() / 3600


print("\nthe open survives, whatever else goes")
for n in (31, 60, 500, 2000):
    check(f"{n:>5d} points -> the first is still first",
          _thin_history(hist(n))[0], hist(n)[0])

print("\nand so does the most recent price")
for n in (31, 500, 2000):
    check(f"{n:>5d} points -> the last is still last",
          _thin_history(hist(n))[-1], hist(n)[-1])

print("\nthe buffer never grows")
check("never over the cap, at any input length",
      all(len(_thin_history(hist(n))) <= MAX_HISTORY_POINTS for n in range(1, 300)))
check("short histories are untouched", _thin_history(hist(5)), hist(5))
check("a full-but-not-over history is untouched",
      _thin_history(hist(MAX_HISTORY_POINTS)), hist(MAX_HISTORY_POINTS))

print("\nthe series stays well formed")
for n in (31, 200, 2000):
    out = _thin_history(hist(n))
    ts = [p["timestamp"] for p in out]
    check(f"{n:>5d} points -> still in order", ts == sorted(ts))
    check(f"{n:>5d} points -> no duplicates", len(set(ts)), len(ts))

print("\nrecent detail is kept at full resolution")
_out = _thin_history(hist(2000))
_tail = hist(2000)[-RECENT_HISTORY_POINTS:]
check("the last stretch is verbatim, not sampled",
      [p["timestamp"] for p in _out[-RECENT_HISTORY_POINTS:]] ==
      [p["timestamp"] for p in _tail])

print("\nTHE WHOLE POINT: the buffer covers the life of the line, not the last hour")
for n in (100, 500, 2000):
    old_span = span_hours(hist(n)[-MAX_HISTORY_POINTS:])
    new_span = span_hours(_thin_history(hist(n)))
    check(f"{n:>5d} points -> {old_span:.0f}h under the old tail, {new_span:.0f}h now",
          new_span > old_span)

print("\nthe charts can still read it")
# _snapshot_points parses timestamps rather than assuming even spacing, which
# is what lets a thinned series render at its true dates instead of implying
# the gaps are uniform.
_pts = _snapshot_points(_thin_history(hist(2000)))
check("every thinned point converts to (time, probability)",
      len(_pts), len(_thin_history(hist(2000))))
check("and they come out time-ordered", [p[0] for p in _pts] == sorted(p[0] for p in _pts))

print("\n" + ("-" * 70))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print("the open is safe")
