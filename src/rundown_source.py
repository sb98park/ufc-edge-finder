"""
DraftKings and FanDuel prices, via TheRundown.

WHY THIS EXISTS. Every edge on this site was computed against a Polymarket
midpoint -- a peer-to-peer, vig-free number that no sportsbook will honour.
The premise of the product is "what the book offers versus what the model
thinks", and the book side of that comparison was a price nobody could bet.
This supplies the real one.

WHAT IT ACTUALLY DELIVERS, verified against a live card rather than assumed:

    market_id 1  moneyline   present on 7 of 8 fights
    market_id 3  totals      present on 7 of 8 fights

and nothing else. TheRundown's catalogue lists eighteen markets for sport 7
including `method_of_victory_double_chance` (1371) and `fight_to_start_round`
(1369) -- exactly the two markets worth having -- but requesting them for a
real card returns no data. They are DEFINED, not SERVED, at least on the free
tier. Do not re-add them to MARKET_IDS on the strength of the catalogue; that
was checked and it comes back empty.

A useful accident during that check: the one fight with no markets at all was
Kody Steele vs Gauge Young, which this repo had independently flagged as
cancelled. The feed agreed without being asked.

THREE OPERATIONAL FACTS THAT ARE NOT OPTIONAL.

1. CLOUDFLARE BLOCKS THE DEFAULT PYTHON USER AGENT. A request without a
   browser-like UA returns 403, while the identical request from curl
   succeeds. This is not documented anywhere; it cost an hour.

2. THE QUOTA IS THE BINDING CONSTRAINT, not the rate limit. The free tier
   allows 20,000 "data points" a day, where a point is one participant x one
   line x one book. A 13-fight card is roughly:

       moneyline   2 fighters x 2 books                  =  52
       totals      2 sides x ~3 lines x 2 books          = 156

   about 208 points -- though that totals figure assumes ~3 lines, and the
   request sets main_line=true, which collapses them to one and brings a
   13-fight card nearer 104. BOTH NUMBERS ARE ARITHMETIC, NOT MEASUREMENT;
   nobody has counted a live payload. At the site's 5-minute rebuild cadence
   either one is over the cap, so this source runs on its own clock.

   That clock RAMPS rather than sitting at one interval: six-hourly a week
   out, hourly a few days out, twenty minutes the day before, and on fight
   day paced off whatever budget is left over however much of the day is
   left. Prices barely move on a Tuesday and move constantly on fight night,
   so the allowance belongs where the movement is. See CADENCE_BY_DAYS_OUT.

   The ramp is a preference. The GUARANTEE is a measured daily ledger and a
   hard stop: every pull's real cost is counted from the rows it returned,
   and one that cannot be paid for is refused however long it has been. That
   is what makes staying inside the free tier true even though the cost per
   pull above is still an estimate. Simulated minute by minute across a
   week, the worst day lands at 85% of the cap at 104 points a pull and at
   85% at 400 -- the schedule absorbs the error rather than the cap doing
   it.

3. COVERAGE IS PATCHY PER MARKET. A totals block came back carrying only
   affiliate 19 (DraftKings) while the moneyline on the same fight carried
   both. Never assume both books quoted a market; a row that silently shows
   one book where the reader expects two is worse than one that says so.

WHAT THIS SOURCE IS FOR, AND WHAT POLYMARKET BECOMES. These prices carry vig
and are bettable, so they are the BOOK. Polymarket keeps its place as the
FAIR line -- the cleanest available estimate of true probability, precisely
because it has no margin in it. Two different jobs, and the provenance column
on every row is what keeps them from being confused for one another.
"""

import datetime as dt
import json
import os
import time

import requests

from src.odds_utils import american_to_implied_prob

BASE = "https://therundown.io/api/v2"
SPORT_MMA = 7

# Cloudflare 403s the default python UA. See the module docstring.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Verified against the live /affiliates endpoint, which needs no auth.
AFFILIATES = {19: "DraftKings", 23: "FanDuel", 22: "BetMGM"}
BOOK_AFFILIATES = (19, 23)          # the two the reader actually bets

# Only what is actually served. See the docstring before adding to this.
MARKET_IDS = (1, 3)
_MARKET_NAME = {1: "Moneyline", 3: "TotalRounds"}

# Own clock, slower than the site's 5-minute rebuild, because the daily point
# budget rather than the per-second limit is what binds. This is the fallback
# spacing used when no card date is known; the real cadence ramps -- see
# CADENCE_BY_DAYS_OUT.
MIN_SECONDS_BETWEEN_PULLS = 15 * 60
_last_pull = {"at": 0.0, "rows": None}

# ---------------------------------------------------------------------------
# THE DAILY BUDGET, AND WHY IT IS MEASURED RATHER THAN ASSUMED
#
# The free tier allows 20,000 data points a day, a point being one participant
# x one line x one book. Everything about staying inside that turns on the
# cost of a single pull, and this repo has carried TWO different numbers for
# it: the module docstring works out ~208 for a 13-fight card, and that figure
# was computed for totals at roughly three lines each. The client requests
# main_line=true, which collapses totals to the headline line, so the real
# figure should be nearer 104. Nobody has run a live pull and counted.
#
# The gap matters: at 208 a ten-minute cadence is 150% of the cap, and at 104
# it is 75% of it. A hand-tuned interval table would be picking one of those
# and hoping. So the cost is MEASURED off each pull -- len(rows) is exactly
# one row per participant x line x book, which is the same thing a point is --
# and the spacing is derived from what the budget has left.
#
# THE HARD STOP IS THE GUARANTEE. The ramp below is a preference; the check
# against the day's remaining points is what makes "never exceed the free
# tier" true even if every estimate here is wrong by a factor of two.
BUDGET_PATH = "data/rundown_budget.json"
DAILY_POINT_CAP = 20_000

# Never spend the last 15%. Headroom for a bigger card than usual, a second
# date, and the retries a flaky morning produces.
BUDGET_SAFETY = 0.85

# Until a real pull measures it, assume the EXPENSIVE reading. Guessing high
# costs a few slow pulls on the first day; guessing low is how a cap gets
# breached before anything notices.
BOOTSTRAP_COST = 250

# Spacing by how far the next card is. Prices barely move on a Tuesday and
# move constantly on fight night, so the budget belongs where the movement is.
# Fight day is deliberately absent: it is paced by whatever is left, which is
# both faster and safer than any number that could be written here.
CADENCE_BY_DAYS_OUT = (
    (7, 6 * 3600),      # a week or more out: four pulls a day
    (4, 3 * 3600),
    (2, 60 * 60),
    (1, 20 * 60),       # the day before
)

# However much budget is left, do not pull faster than this. The site rebuilds
# every five minutes and the front end already updates displayed prices live,
# so anything under this buys nothing a reader can see.
FIGHT_DAY_FLOOR_SECONDS = 5 * 60


def _utc_day(now: dt.datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _budget_load(now: dt.datetime, path: str = BUDGET_PATH) -> dict:
    """
    The day's spend so far. Resets when the UTC day rolls over.

    UTC IS AN ASSUMPTION, and a deliberate one. TheRundown documents a daily
    allowance without saying which midnight it turns on, so this picks the
    earliest plausible one: if their day actually starts later, this resets
    early and underspends, which is the harmless direction.
    """
    today = _utc_day(now)
    try:
        with open(path, encoding="utf-8") as fh:
            b = json.load(fh)
        if b.get("utc_day") == today:
            return {"utc_day": today,
                    "points": int(b.get("points") or 0),
                    "pulls": int(b.get("pulls") or 0),
                    "last_cost": int(b.get("last_cost") or 0)}
        # A new day inherits the cost estimate and nothing else -- the card is
        # the same size this morning as it was last night.
        return {"utc_day": today, "points": 0, "pulls": 0,
                "last_cost": int(b.get("last_cost") or 0)}
    except (OSError, ValueError, TypeError):
        return {"utc_day": today, "points": 0, "pulls": 0, "last_cost": 0}


def _budget_save(b: dict, path: str = BUDGET_PATH) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(b, fh, indent=1, sort_keys=True)
    except OSError as exc:
        # Losing the counter costs accuracy, not safety: the next load starts
        # from zero and the ramp still applies.
        print(f"[rundown] budget not written ({exc}) -- continuing")


def _days_out(dates, now: dt.datetime):
    """Days until the SOONEST requested date, or None if none parse."""
    best = None
    for d in dates or []:
        try:
            when = dt.datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        delta = (when - now.date()).days
        best = delta if best is None else min(best, delta)
    return best


def plan_pull(dates, budget: dict, now: dt.datetime) -> dict:
    """
    How long to wait before the next pull, and whether one is affordable.

    Split out and pure so scripts/verify_rundown.py can print a whole week's
    schedule without calling anything, and so the arithmetic is testable
    without a key. Returns {interval, cost, remaining, affordable, why}.
    """
    cost = budget.get("last_cost") or BOOTSTRAP_COST
    allowance = int(DAILY_POINT_CAP * BUDGET_SAFETY)
    remaining = max(0, allowance - int(budget.get("points") or 0))
    affordable = remaining >= cost

    days = _days_out(dates, now)
    if days is None:
        return {"interval": MIN_SECONDS_BETWEEN_PULLS, "cost": cost,
                "remaining": remaining, "affordable": affordable,
                "why": "no card date -- fallback spacing"}
    if days < 0:
        # The card has been and gone and nothing has replaced it yet.
        return {"interval": 6 * 3600, "cost": cost, "remaining": remaining,
                "affordable": affordable, "why": "card is in the past"}

    for threshold, interval in CADENCE_BY_DAYS_OUT:
        if days >= threshold:
            return {"interval": interval, "cost": cost, "remaining": remaining,
                    "affordable": affordable,
                    "why": f"{days} day(s) out"}

    # FIGHT DAY. Spread whatever is left over whatever is left of the day, so
    # the cadence tightens as the card approaches and slackens by itself if
    # the morning was expensive. The UTC rollover lands mid-card for a US
    # evening event, which hands the late rounds a fresh allowance.
    end = dt.datetime.combine(now.date() + dt.timedelta(days=1),
                              dt.time.min, tzinfo=now.tzinfo)
    left = max(60.0, (end - now).total_seconds())
    pulls_affordable = max(1.0, remaining / float(cost or BOOTSTRAP_COST))
    interval = max(FIGHT_DAY_FLOOR_SECONDS, left / pulls_affordable)
    return {"interval": interval, "cost": cost, "remaining": remaining,
            "affordable": affordable,
            "why": f"fight day -- {remaining} point(s) over {left / 3600:.1f}h"}


def _get(path: str, timeout: int = 20):
    key = os.environ.get("RUNDOWN_API_KEY")
    if not key:
        raise RuntimeError("RUNDOWN_API_KEY is not set")
    r = requests.get(f"{BASE}{path}",
                     headers={"X-TheRundown-Key": key, "User-Agent": _UA,
                              "Accept": "application/json"},
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


def _rows_from_event(ev: dict) -> list[dict]:
    """Flatten one event's markets into the row shape live_props expects."""
    teams = ev.get("teams") or []
    if len(teams) < 2:
        return []
    # is_away first, to match how the rest of the repo orders a pairing.
    away = next((t for t in teams if t.get("is_away")), teams[0])
    home = next((t for t in teams if t.get("is_home")), teams[-1])
    fa, fb = away.get("name"), home.get("name")
    if not fa or not fb:
        return []
    fight_id = f"{fa}|{fb}"

    out = []
    for m in ev.get("markets") or []:
        market = _MARKET_NAME.get(m.get("market_id"))
        if not market:
            continue
        for p in m.get("participants") or []:
            pname = p.get("name")
            for line in p.get("lines") or []:
                value = line.get("value")
                for aff_id, price in (line.get("prices") or {}).items():
                    book = AFFILIATES.get(int(aff_id))
                    if not book:
                        continue
                    am = price.get("price")
                    if am is None:
                        continue
                    if market == "Moneyline":
                        selection, sel_method = pname, ""
                    else:
                        # "Over"/"Under" plus the line value, matching the
                        # selection strings the rest of the pipeline parses.
                        selection = f"{pname} {value}" if value else pname
                        sel_method = str(value or "")
                    out.append({
                        "fight_id": fight_id, "fighter_a": fa, "fighter_b": fb,
                        "market": market, "selection": selection,
                        "selection_method": sel_method,
                        "odds_american": float(am),
                        # PROVENANCE. These are vigged, bettable prices -- the
                        # opposite of the Polymarket rows they sit beside, and
                        # the whole reason the flag travels per row.
                        "source": book,
                        "source_is_vig_free": False,
                        "is_main_line": bool(price.get("is_main_line")),
                        "price_updated_at": price.get("updated_at"),
                    })
    return out


def fetch_rundown_ufc_odds(dates: list[str], force: bool = False) -> list[dict]:
    """
    Moneyline and totals for the given YYYY-MM-DD dates, from DraftKings and
    FanDuel. Returns [] on any failure -- a metered second source must never
    be able to take a site build down with it.

    Cached on its own clock: repeated calls inside MIN_SECONDS_BETWEEN_PULLS
    return the previous result rather than spending quota.
    """
    now = time.time()
    now_utc = dt.datetime.now(dt.timezone.utc)
    budget = _budget_load(now_utc)
    plan = plan_pull(dates, budget, now_utc)

    # THE HARD STOP, CHECKED BEFORE THE CLOCK. A pull that cannot be paid for
    # is refused however long it has been, which is what makes the free tier a
    # guarantee rather than an intention. Serving the last rows is the right
    # failure: prices that are an hour old beat a page with no book on it.
    if not force and not plan["affordable"]:
        print(f"[rundown] budget spent: {budget['points']} point(s) of "
              f"{int(DAILY_POINT_CAP * BUDGET_SAFETY)} today over {budget['pulls']} "
              f"pull(s) -- serving the last prices until the UTC day rolls over")
        return _last_pull["rows"] or []

    if not force and _last_pull["rows"] is not None \
            and now - _last_pull["at"] < plan["interval"]:
        return _last_pull["rows"]

    mkts = ",".join(str(m) for m in MARKET_IDS)
    affs = ",".join(str(a) for a in BOOK_AFFILIATES)
    rows: list[dict] = []
    for i, d in enumerate(dates):
        if i:
            time.sleep(1.1)          # documented limit is 1 req/sec
        try:
            # main_line=true is a QUOTA decision, not a display one: alternate
            # totals lines multiply the point cost by roughly three for a
            # market the site shows one line of anyway.
            data = _get(f"/sports/{SPORT_MMA}/events/{d}"
                        f"?market_ids={mkts}&affiliate_ids={affs}&main_line=true")
        except Exception as exc:
            print(f"[rundown] {d} failed ({exc}) -- continuing without it")
            continue
        for ev in data.get("events") or []:
            rows.extend(_rows_from_event(ev))

    # ONE ROW IS ONE DATA POINT. _rows_from_event emits exactly one row per
    # participant x line x book, which is the unit TheRundown meters, so the
    # spend needs no estimating -- it is counted from what came back. A pull
    # that returned nothing still cost a request but no points.
    budget["points"] += len(rows)
    budget["pulls"] += 1
    if rows:
        budget["last_cost"] = len(rows)
    _budget_save(budget)

    books = {r["source"] for r in rows}
    spent_pct = budget["points"] / (DAILY_POINT_CAP * BUDGET_SAFETY) * 100
    print(f"[rundown] {len(rows)} price(s) across {len(dates)} date(s) "
          f"from {', '.join(sorted(books)) if books else 'no book'} "
          f"-- {budget['points']} point(s) today ({spent_pct:.0f}% of allowance), "
          f"next in {plan['interval'] / 60:.0f}m ({plan['why']})")
    _last_pull.update(at=now, rows=rows)
    return rows


def best_book_price(rows: list[dict], fight_id: str, market: str,
                    selection: str) -> dict | None:
    """
    The best available price for one bet across the books, and which book has
    it. "Best" is the highest implied payout, i.e. the LOWEST implied
    probability -- the same bet for less money.

    This is what line shopping is, and it is the one edge on this page that
    needs no model to be real: taking +116 over +114 on the identical
    selection is free, and the only way to see it is to hold two books
    side by side.
    """
    cands = [r for r in rows
             if r["fight_id"] == fight_id and r["market"] == market
             and r["selection"] == selection]
    if not cands:
        return None
    best = min(cands, key=lambda r: american_to_implied_prob(r["odds_american"]))
    return {
        "book": best["source"],
        "odds_american": best["odds_american"],
        "implied": american_to_implied_prob(best["odds_american"]),
        "all_books": {r["source"]: r["odds_american"] for r in cands},
        # A single-book quote is not a shopped price, and the reader should be
        # told rather than left to assume both were checked.
        "books_quoting": len(cands),
    }
