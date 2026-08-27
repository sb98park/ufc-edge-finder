"""
Check what TheRundown is actually serving, and what it costs to ask.

WHY THIS EXISTS, and it is not "because every client should have a test".
src/rundown_source.py is the only source on this site whose failures are
SILENT and INVISIBLE at the same time:

  - RUNDOWN_API_KEY is a CI secret and is unset on every developer machine, so
    a local build is Polymarket-only and looks perfectly healthy while
    carrying none of the book prices the product is about. A whole afternoon
    went into "why are there no DraftKings moneylines" before the answer
    turned out to be that there was no key in the shell asking.
  - fetch_rundown_ufc_odds returns [] on ANY failure, on purpose -- a metered
    second source must not be able to take a build down. That is the right
    call and it means an expired key, a 403, a quota wall and a card the feed
    has not loaded yet are all the same empty list.
  - _rows_from_event drops any affiliate not in AFFILIATES without a word. The
    day TheRundown renumbers a book or adds one worth having, the prices stop
    arriving and nothing says so.

So this exists to make those four cases distinguishable, out loud.

IT RUNS OFFLINE. --fixture reads a saved payload instead of calling the API,
which means the parsing and every check below can be exercised with no key and
no quota. Capture one with --save on a machine that has the key. A harness
that can only run where the secret lives is a harness nobody runs.

WHAT IT CHECKS, each earned rather than invented:

  coverage    per market and per book. The module docstring warns that a
              totals block came back carrying DraftKings alone while the
              moneyline beside it had both. Measured, not assumed.
  overround   the two-sided vig per book. On 2026-08-26 the opening DraftKings
              price on two fights had to be de-vigged BY HAND off a phone
              screenshot to recover a fair line, because nothing in the repo
              measured it live. A sum below 1.0 is not a thin market, it is
              bad data or a genuine arbitrage, and either one is worth
              stopping for.
  quota       the free tier allows 20,000 points a day, a point being one
              participant x one line x one book. This counts the real payload
              and replays the client's actual ramping schedule against it, day
              by day -- the cadence is no longer one number, so multiplying one
              interval by 1440 would describe nothing the client does.
  dropped     raw prices whose affiliate id is not in AFFILIATES, counted
              rather than discarded in silence.
  staleness   how old the quoted prices are, which is the difference between
              a quiet market and a feed that has stopped moving.

Exit 0 clean or skipped, 1 on something that should stop a person.
"""

import argparse
import collections
import datetime as dt
import json
import os
import sys

sys.path.insert(0, ".")

from src.odds_utils import american_to_implied_prob
from src.rundown_source import (AFFILIATES, BOOK_AFFILIATES, BUDGET_SAFETY,
                                DAILY_POINT_CAP as SOURCE_CAP, MARKET_IDS,
                                SPORT_MMA, _rows_from_event, plan_pull)

# Mirrored from the client so the harness cannot drift from what it checks.
DAILY_POINT_CAP = SOURCE_CAP

# A two-sided book price sums above 1.0 by construction -- that is the margin.
# Below 1.0 means the two sides can be backed for a guaranteed profit, which no
# book leaves standing, so it is a data fault until proven otherwise. The upper
# bound is loose on purpose: 15% is punitive but real on a thin prelim.
OVERROUND_MIN, OVERROUND_MAX = 1.0, 1.15


def _raw_points(payload) -> tuple[int, collections.Counter]:
    """Data points in a payload, and how the affiliate ids break down."""
    n, affs = 0, collections.Counter()
    for ev in payload.get("events") or []:
        for m in ev.get("markets") or []:
            for p in m.get("participants") or []:
                for line in p.get("lines") or []:
                    for aff_id in (line.get("prices") or {}):
                        n += 1
                        affs[str(aff_id)] += 1
    return n, affs


def _fetch(date: str, save: str | None):
    from src.rundown_source import BASE, _get  # noqa: F401  (key check is in _get)
    mkts = ",".join(str(m) for m in MARKET_IDS)
    # BOOK_AFFILIATES, not AFFILIATES. The client requests only the two
    # books a reader bets; asking for the third here would measure a call
    # nobody makes and overstate the quota bill it is trying to report.
    affs = ",".join(str(a) for a in BOOK_AFFILIATES)
    payload = _get(f"/sports/{SPORT_MMA}/events/{date}"
                   f"?market_ids={mkts}&affiliate_ids={affs}&main_line=true")
    if save:
        with open(save, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print(f"  saved the raw payload to {save}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--fixture", help="read a saved payload instead of calling the API")
    ap.add_argument("--save", help="write the raw payload here (live calls only)")
    a = ap.parse_args()

    if a.fixture:
        with open(a.fixture, encoding="utf-8") as fh:
            payload = json.load(fh)
        print(f"[rundown-verify] fixture {a.fixture} (no API call, no quota spent)")
    elif not os.environ.get("RUNDOWN_API_KEY"):
        # NOT A FAILURE. It is the normal state of a developer machine, and
        # saying so plainly is most of the point of this script.
        print("[rundown-verify] RUNDOWN_API_KEY is not set in this shell.\n"
              "  This is expected locally -- the key is a CI secret, so local\n"
              "  builds are Polymarket-only and carry no book prices. That is\n"
              "  not a bug and it is not a broken client.\n"
              "  To check the live feed:  export RUNDOWN_API_KEY=... \n"
              "  To check the parsing without a key or quota:  --fixture PATH")
        return 0
    else:
        print(f"[rundown-verify] live call for {a.date}")
        payload = _fetch(a.date, a.save)

    events = payload.get("events") or []
    rows = [r for ev in events for r in _rows_from_event(ev)]
    points, affs = _raw_points(payload)
    problems: list[str] = []

    print(f"\n  events {len(events)}   parsed rows {len(rows)}   raw data points {points}")

    if events and not rows:
        problems.append("the payload has events but nothing parsed out of them -- "
                        "either every market is unserved or _rows_from_event has "
                        "stopped matching the feed's shape")

    # ---- affiliates, including the ones the client throws away ----
    known = {str(k) for k in AFFILIATES}
    dropped = {k: v for k, v in affs.items() if k not in known}
    _known_bits = ", ".join(f"{AFFILIATES[int(k)]}({k}) {v} price(s)"
                            for k, v in sorted(affs.items()) if k in known)
    print(f"\n  AFFILIATES  {_known_bits or 'none'}")
    if dropped:
        # Loud but not fatal: an unknown book is a feed change to look at, not
        # a reason to fail a build.
        print(f"  DROPPED     {dropped} -- affiliate id(s) not in AFFILIATES, "
              f"silently discarded by _rows_from_event")

    # ---- coverage, per market and per book ----
    fights = {r["fight_id"] for r in rows}
    print(f"\n  COVERAGE    {len(fights)} fight(s)")
    for market in sorted({r["market"] for r in rows}):
        mrows = [r for r in rows if r["market"] == market]
        mfights = {r["fight_id"] for r in mrows}
        by_book = collections.Counter(r["source"] for r in mrows)
        both = sum(1 for f in mfights
                   if len({r["source"] for r in mrows if r["fight_id"] == f}) > 1)
        print(f"    {market:12s} {len(mfights):2d}/{len(fights)} fight(s), "
              f"{both} with more than one book   {dict(by_book)}")

    # ---- the two-sided vig, per book, per fight ----
    print(f"\n  OVERROUND   (two-sided moneyline, per book)")
    ml = [r for r in rows if r["market"] == "Moneyline"]
    seen = 0
    for fight in sorted({r["fight_id"] for r in ml}):
        for book in sorted({r["source"] for r in ml if r["fight_id"] == fight}):
            sides = [r for r in ml if r["fight_id"] == fight and r["source"] == book]
            if len(sides) != 2:
                continue
            seen += 1
            total = sum(american_to_implied_prob(r["odds_american"]) for r in sides)
            flag = "" if OVERROUND_MIN <= total <= OVERROUND_MAX else "   <-- OUT OF RANGE"
            print(f"    {fight[:38]:38s} {book:11s} {total:.4f}  "
                  f"({(total - 1) * 100:+.2f}% vig){flag}")
            if total < OVERROUND_MIN:
                problems.append(f"{book} prices {fight} at an implied sum of "
                                f"{total:.4f} -- below 1.0 is an arbitrage or bad data")
            elif total > OVERROUND_MAX:
                problems.append(f"{book} prices {fight} at {(total - 1) * 100:.1f}% vig, "
                                f"above the {(OVERROUND_MAX - 1) * 100:.0f}% this expects")
    if not seen:
        print("    no fight had both sides from one book -- nothing to measure")

    # ---- how old are these prices ----
    stamps = [r["price_updated_at"] for r in rows if r.get("price_updated_at")]
    print(f"\n  FRESHNESS   {len(stamps)}/{len(rows)} row(s) carry price_updated_at")
    if stamps:
        print(f"    oldest {min(stamps)}   newest {max(stamps)}")

    # ---- the quota bill, walked forward day by day ----
    #
    # NOT A SINGLE CADENCE ANY MORE. The client ramps toward the card and
    # paces fight day off whatever budget is left, so one interval times 1440
    # describes nothing it actually does. This replays the real schedule
    # against the measured cost of this payload.
    print(f"\n  QUOTA       {points} point(s) per pull, measured from this payload")
    print(f"              allowance {int(DAILY_POINT_CAP * BUDGET_SAFETY):,} "
          f"of {DAILY_POINT_CAP:,} ({BUDGET_SAFETY:.0%} safety margin)")
    cost = points or None
    if not cost:
        print("    nothing came back, so there is no cost to project")
    else:
        card = dt.date.fromisoformat(a.date)
        print(f"    {'day':12s} {'pulls':>6s} {'points':>8s} {'% cap':>7s}  spacing")
        worst = 0
        for offset in range(6, -1, -1):
            day = card - dt.timedelta(days=offset)
            budget = {"points": 0, "pulls": 0, "last_cost": cost}
            last, n, gaps = None, 0, []
            t = dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc)
            stop = t + dt.timedelta(days=1)
            while t < stop:
                plan = plan_pull([a.date], budget, t)
                if plan["affordable"] and (
                        last is None or (t - last).total_seconds() >= plan["interval"]):
                    budget["points"] += cost
                    budget["pulls"] += 1
                    last, n = t, n + 1
                    gaps.append(plan["interval"])
                t += dt.timedelta(minutes=1)
            pct = budget["points"] / DAILY_POINT_CAP * 100
            worst = max(worst, budget["points"])
            gap = (sorted(gaps)[len(gaps) // 2] / 60) if gaps else 0
            tag = "  <- fight day" if offset == 0 else ""
            print(f"    {day.isoformat():12s} {n:6d} {budget['points']:8,d} "
                  f"{pct:6.1f}% {gap:8.0f}m{tag}")
        if worst > DAILY_POINT_CAP:
            problems.append(f"the schedule peaks at {worst:,} points in a day against "
                            f"a {DAILY_POINT_CAP:,} cap -- widen CADENCE_BY_DAYS_OUT "
                            f"or lower BUDGET_SAFETY")

    print()
    if problems:
        print(f"FAIL  {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASS  feed shape, coverage, vig and quota all within expectations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
