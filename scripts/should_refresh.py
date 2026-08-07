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

    within 12h of an event      every run          (5 min)
    within 72h                  every 3rd run      (~15 min)
    otherwise                   every 6th run      (~30 min)

Exits 0 to proceed, 1 to skip. A skipped run still starts, but stops before
generating -- costing seconds instead of a build and a deployment.

Fails OPEN: any error proceeds. A guard that silently stops the site updating
is worse than one that occasionally lets a redundant build through.
"""

import datetime as dt
import sys

import pandas as pd


def main():
    try:
        now = dt.datetime.now(dt.timezone.utc)
        dates = []
        for path in ("data/fight_cards.csv", "data/future_cards.csv"):
            try:
                d = pd.read_csv(path)
            except FileNotFoundError:
                continue
            if "event_date" in d.columns:
                dates += [str(x)[:10] for x in d["event_date"].dropna()]
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

        # Which 5-minute slot of the hour this is. Deterministic, so the
        # spacing is even rather than random.
        slot = now.minute // 5

        if hours <= 12:
            every, why = 1, "within 12h of a card"
        elif hours <= 72:
            every, why = 3, "within 72h of a card"
        else:
            every, why = 6, "no card close"

        if slot % every == 0:
            print(f"[cadence] {why} ({hours:.0f}h) -- building (every {every*5} min)")
            return 0
        print(f"[cadence] {why} ({hours:.0f}h) -- skipping this slot "
              f"(builds every {every*5} min)")
        return 1
    except Exception as exc:
        print(f"[cadence] check failed, proceeding anyway: {exc}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
