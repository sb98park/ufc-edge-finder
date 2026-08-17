"""
Decide whether this scheduled run needs to do the expensive work.

WHY. The cron runs every 5 minutes -- up to 288 builds a day, each committing
a regenerated site and triggering its own Pages deployment. That kept the
deployment queue near saturation: a manual push spent an hour stuck behind
automated ones, and three consecutive deployments failed before anyone looked
at the cause.

Odds move meaningfully over hours, not minutes -- except close to a card,
where they move fast and late money matters. So the cadence should follow the
schedule rather than being uniform:

    within 12h of an event      every 5 minutes
    within 72h                  every 15 minutes
    otherwise                   every 30 minutes

MEASURED AS TIME SINCE THE LAST BUILD, not as a slot of the wall clock.
The first version computed `slot = now.minute // 5` and built when
`slot % every == 0` -- i.e. only at :00, :15, :30, :45. That assumes cron
fires exactly on the five-minute boundary, and GitHub Actions gives no such
guarantee: */5 schedules are routinely delayed or dropped under load, so runs
arrive at :07, :23, :41. Those are slots 1, 4 and 8, none divisible by 3, so
every single run skipped. The gate starved and the site went 12 hours without
updating two days before a card, while the workflow reported success on every
run -- it was doing exactly what it was told.
Elapsed time is indifferent to when the scheduler actually fires.

Exits 0 to proceed, 1 to skip. A skipped run still starts, but stops before
generating -- costing seconds instead of a build and a deployment.

Fails OPEN: any error proceeds. A guard that silently stops the site updating
is worse than one that occasionally lets a redundant build through.

STDLIB ONLY, DELIBERATELY -- see the CSV read in main(). This file must stay
importable without installing anything, so the workflow can run it before
`pip install`. Adding a third-party import here silently costs ~90s on every
skipped run.
"""

import csv
import datetime as dt
import os
import subprocess
import sys

# THE BUILD CLOCK, written by this file and by nothing else. Lives in data/
# because refresh.yml's commit step is `git add -A -- data/ docs/`, so it is
# carried to the next run; the runner's checkout is fresh every time, so an
# on-disk-only marker would always read as "never built".
STAMP = "data/last_build"


def _stamp_time():
    """The recorded build time from STAMP, or None if it is missing/unreadable."""
    try:
        with open(STAMP, encoding="utf-8") as fh:
            value = fh.read().strip()
        if not value:
            return None
        last = dt.datetime.fromisoformat(value)
        # A stamp written without a zone is UTC by construction (below); read
        # it that way rather than letting it default to the runner's locale.
        return last if last.tzinfo else last.replace(tzinfo=dt.timezone.utc)
    except (OSError, ValueError):
        return None


def _commit_time():
    """The last commit's time, or None if git cannot answer."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%ct"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return dt.datetime.fromtimestamp(int(out.stdout.strip()), dt.timezone.utc)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def record_build_time():
    """
    Stamp "a build starts now". Called only when this gate answers BUILD.

    Best-effort on purpose: if the write fails the gate must still let the
    build through, and the next run simply falls back to the commit time.
    """
    try:
        os.makedirs(os.path.dirname(STAMP) or ".", exist_ok=True)
        now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        with open(STAMP, "w", encoding="utf-8") as fh:
            fh.write(now.isoformat() + "\n")
    except OSError:
        pass


def _minutes_since_last_build():
    """
    Minutes since the most recent build, or None if nothing can answer.

    THE LATER OF TWO CLOCKS, because neither is complete on its own.

    STAMP alone misses the builds that bypass this gate: workflow_dispatch and
    push always build without consulting it (refresh.yml's cadence step), so a
    manual refresh would leave the stamp stale and the very next tick would
    rebuild on top of it.

    The COMMIT alone was the original implementation and it silently stopped
    working. It relied on "the refresh commits on every build", which was true
    while docs/ was tracked. It is not any more: refresh.yml stages `data/ docs/`
    and .gitignore excludes docs/index.html and docs/movements/, so a build that
    changes no file in data/ -- no priced market to move odds_snapshot.json, no
    settled fight to log -- makes no commit at all. HEAD's timestamp then
    freezes, `age + 1.5 >= interval` is true forever, and the throttle inverts
    into 288 builds a day at exactly the moment there is least to publish.

    Taking the max keeps the old behaviour wherever it was right and stops the
    clock freezing where it was wrong.
    """
    times = [t for t in (_stamp_time(), _commit_time()) if t is not None]
    if not times:
        return None
    return (dt.datetime.now(dt.timezone.utc) - max(times)).total_seconds() / 60.0


def main():
    try:
        now = dt.datetime.now(dt.timezone.utc)
        dates = []
        for path in ("data/fight_cards.csv", "data/future_cards.csv"):
            # csv.DictReader rather than pd.read_csv: pulling one column out
            # of two files of a few dozen rows never needed a DataFrame, and
            # the pandas import is what forced this gate to run below `pip
            # install` in the workflow. Blank cells are skipped here, which
            # is what .dropna() did; any other unparseable value (a literal
            # "nan", a malformed date) falls through to the strptime below
            # and is discarded there, exactly as before.
            try:
                with open(path, newline="", encoding="utf-8-sig") as fh:
                    reader = csv.DictReader(fh)
                    if not reader.fieldnames or "event_date" not in reader.fieldnames:
                        continue
                    for row in reader:
                        value = (row.get("event_date") or "").strip()
                        if value:
                            dates.append(value[:10])
            except (FileNotFoundError, OSError, csv.Error):
                continue
        if not dates:
            print("[cadence] no card dates found -- proceeding")
            return 0

        upcoming = []
        for s in set(dates):
            try:
                d = dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            except ValueError:
                continue
            # Cards run in the evening US time; treat the event as ~23:00 UTC
            # on its date so "hours away" isn't off by most of a day.
            d += dt.timedelta(hours=23)
            if d > now - dt.timedelta(hours=6):
                upcoming.append(d)
        if not upcoming:
            print("[cadence] no upcoming card -- proceeding at reduced rate")
            hours = 9999
        else:
            hours = min((d - now).total_seconds() / 3600 for d in upcoming)

        if hours <= 12:
            interval, why = 5, "within 12h of a card"
        elif hours <= 72:
            interval, why = 15, "within 72h of a card"
        else:
            interval, why = 30, "no card close"

        age = _minutes_since_last_build()
        if age is None:
            print(f"[cadence] {why} ({hours:.0f}h) -- cannot read last build time, proceeding")
            return 0

        # Tolerance, because a run arriving at 14m50s into a 15m interval
        # would otherwise wait a whole further cycle. Without it, irregular
        # scheduling pushes the effective cadence out to nearly double.
        if age + 1.5 >= interval:
            print(f"[cadence] {why} ({hours:.0f}h) -- last build {age:.0f}m ago, "
                  f"interval {interval}m -- BUILDING")
            return 0
        print(f"[cadence] {why} ({hours:.0f}h) -- last build {age:.0f}m ago, "
              f"interval {interval}m -- skipping")
        return 1
    except Exception as exc:
        print(f"[cadence] check failed, proceeding anyway: {exc}")
        return 0


if __name__ == "__main__":
    code = main()
    if code == 0:
        # Stamped HERE rather than at the end of generate_site.py so that a
        # build which crashes half way still moves the clock. The alternative
        # hammers: a build that fails leaves no commit and no stamp, so every
        # tick behind it rebuilds and fails again at full cost. Bounding a
        # broken build to one attempt per interval is the point of the gate.
        record_build_time()
    sys.exit(code)
