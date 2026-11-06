#!/usr/bin/env python3
"""Alarm when the PUBLISHED page goes stale.

WHY THIS EXISTS AS ITS OWN WORKFLOW, and not as a step in refresh.yml or a
fourth entry in the gate loop: this check CAN fail for a reason outside the
repo (Pages is down, the CDN is slow, the domain is unreachable), and
CLAUDE.md section 2 is explicit that a gate exiting non-zero freezes the
whole site on stale data. A watchdog that can fail must never sit in the
path that commits data. It lives in .github/workflows/site-freshness.yml,
which touches nothing.

WHAT IT IS WATCHING FOR, precisely: `deploy` in refresh.yml is
continue-on-error, because a failed Pages deploy is normally won back by the
next tick (it builds a new sha, which gets a new Pages deployment ID -- the
old ID is the sha and cannot be redeployed, which is why the in-run retry
that used to live there went 0 for 5). That trade gives up the per-incident
email. This is what replaces it: one alarm when the page a reader actually
loads has stopped moving, rather than five an hour when an attempt loses a
race it was always going to win on the next pass.

THE THRESHOLD IS NOT ARBITRARY. scripts/should_refresh.py caps its own
interval at 30 minutes ("no card close"), so the published page is
legitimately up to 30m + build + deploy old with nothing wrong. Worse: a
failed deploy still records a build time, so the next build waits out the
full interval anyway -- a single lost deploy can legitimately show ~60m.
75 minutes sits above that and below anything a reader would call current.
Do not lower it to "catch things sooner"; that is how this becomes a gate
that fails for a number that legitimately moved.
"""
import argparse
import datetime as dt
import re
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://octanealpha.com/"
DEFAULT_MAX_AGE_MIN = 75.0
FETCH_ATTEMPTS = 3

# The attribute is emitted by templates/site.html:
#     data-build-iso="{{ generated_at_iso }}"
# Anchored on the attribute name rather than on any surrounding markup so a
# layout change to the meta row does not silently stop the watchdog. If this
# ever stops matching, that is itself an alarm -- see _missing_stamp below.
_BUILD_ISO = re.compile(r'data-build-iso="([^"]+)"')


def parse_build_iso(html):
    """Return the page's build time as an aware datetime, or None."""
    m = _BUILD_ISO.search(html or "")
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        when = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    # A stamp without an offset is read as UTC: generated_at_iso has always
    # carried one, but a naive value here would otherwise make the
    # subtraction below raise rather than alarm.
    return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)


def age_minutes(build_time, now):
    return (now - build_time).total_seconds() / 60.0


def evaluate(html, now, max_age_min=DEFAULT_MAX_AGE_MIN):
    """(ok, message). Pure: no network, no clock, so it is testable."""
    build_time = parse_build_iso(html)
    if build_time is None:
        return False, (
            "could not find data-build-iso in the served page -- either the "
            "page is not the built site (a 404 or an error page would also "
            "land here) or templates/site.html stopped emitting the stamp"
        )
    age = age_minutes(build_time, now)
    if age > max_age_min:
        return False, (
            f"published page was built {age:.0f}m ago "
            f"({build_time.isoformat()}), over the {max_age_min:.0f}m "
            "threshold -- the Pages deploy is not landing"
        )
    if age < -5:
        return False, (
            f"published page claims a build {abs(age):.0f}m in the FUTURE "
            f"({build_time.isoformat()}) -- clock or stamp is wrong"
        )
    return True, (
        f"published page was built {age:.0f}m ago "
        f"({build_time.isoformat()}), under the {max_age_min:.0f}m threshold"
    )


def fetch(url, attempts=FETCH_ATTEMPTS):
    """Return (html, error). Retried: one refused connection is not news."""
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    # Cache-Control, because a CDN hit would report the age of
                    # whatever the edge is holding rather than what was last
                    # deployed, and this check exists to catch a deploy that
                    # never landed.
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                    "User-Agent": "octanealpha-freshness-check",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", "replace"), None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            if i + 1 < attempts:
                import time

                time.sleep(5)
    return None, last


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--max-age-min", type=float, default=DEFAULT_MAX_AGE_MIN)
    args = ap.parse_args(argv)

    html, err = fetch(args.url)
    if html is None:
        print(f"[freshness] FAIL {args.url} unreachable after "
              f"{FETCH_ATTEMPTS} attempts: {err}")
        return 1

    ok, message = evaluate(html, dt.datetime.now(dt.timezone.utc),
                           args.max_age_min)
    print(f"[freshness] {'OK' if ok else 'FAIL'} {args.url} -- {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
