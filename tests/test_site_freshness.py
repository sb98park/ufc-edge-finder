#!/usr/bin/env python3
"""The freshness watchdog's thresholds, with no network and no wall clock.

Run as a plain script, not under pytest -- that is how CI runs it.
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_site_freshness import evaluate, parse_build_iso

NOW = dt.datetime(2026, 11, 6, 18, 0, 0, tzinfo=dt.timezone.utc)


def page(iso):
    # The shape templates/site.html emits, with enough around it that a bare
    # substring match would not be what makes this pass.
    return (
        '<div class="meta-row" data-live-token="x"\n'
        f'       data-build-iso="{iso}"\n'
        '       data-card="UFC 999">Updated Nov 6</div>'
    )


def check(label, cond):
    if not cond:
        print(f"FAIL: {label}")
        sys.exit(1)
    print(f"  ok: {label}")


def test_parses_the_attribute():
    when = parse_build_iso(page("2026-11-06T17:30:00+00:00"))
    check("parses an offset-bearing stamp",
          when == dt.datetime(2026, 11, 6, 17, 30, tzinfo=dt.timezone.utc))


def test_naive_stamp_is_read_as_utc():
    when = parse_build_iso(page("2026-11-06T17:30:00"))
    check("a stamp with no offset is read as UTC, not left naive",
          when == dt.datetime(2026, 11, 6, 17, 30, tzinfo=dt.timezone.utc))


def test_fresh_page_passes():
    ok, msg = evaluate(page("2026-11-06T17:30:00+00:00"), NOW)
    check(f"30m old page passes ({msg})", ok)


def test_the_legitimate_worst_case_passes():
    # should_refresh.py caps its interval at 30m, and a lost deploy still
    # records a build time, so ~60m can happen with nothing broken. If this
    # test ever fails, the threshold has been lowered into the range of
    # normal cadence -- which CLAUDE.md section 2 forbids.
    ok, msg = evaluate(page("2026-11-06T17:00:00+00:00"), NOW)
    check(f"60m old page still passes ({msg})", ok)


def test_stale_page_alarms():
    ok, msg = evaluate(page("2026-11-06T16:30:00+00:00"), NOW)
    check(f"90m old page alarms ({msg})", not ok)
    check("the message says how old it is", "90m ago" in msg)


def test_threshold_boundary():
    at = evaluate(page("2026-11-06T16:45:00+00:00"), NOW)[0]   # exactly 75m
    over = evaluate(page("2026-11-06T16:44:00+00:00"), NOW)[0]  # 76m
    check("exactly at the threshold passes", at)
    check("one minute past the threshold alarms", not over)


def test_missing_stamp_alarms():
    ok, msg = evaluate("<html><body>404 Not Found</body></html>", NOW)
    check(f"a page with no stamp alarms ({msg})", not ok)
    check("the message names the attribute", "data-build-iso" in msg)


def test_unparseable_stamp_alarms():
    ok, _ = evaluate(page("not-a-date"), NOW)
    check("an unparseable stamp alarms rather than raising", not ok)


def test_future_stamp_alarms():
    ok, msg = evaluate(page("2026-11-06T19:00:00+00:00"), NOW)
    check(f"a stamp an hour in the future alarms ({msg})", not ok)
    check("the message says FUTURE", "FUTURE" in msg)


def test_small_clock_skew_is_tolerated():
    # A couple of minutes of skew between the builder and the checker is not
    # worth an email.
    ok, _ = evaluate(page("2026-11-06T18:02:00+00:00"), NOW)
    check("2m of forward skew is tolerated", ok)


if __name__ == "__main__":
    print("test_site_freshness")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("PASS")
