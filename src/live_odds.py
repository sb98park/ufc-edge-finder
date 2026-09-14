"""
Fetch live UFC/MMA odds from The Odds API (https://the-odds-api.com).

Confirmed via their own docs: fight winner (h2h) is covered broadly, and
"limited coverage of total rounds odds are also available from some
bookmakers" on the free tier too -- so both markets are worth requesting,
not just moneyline. Method-of-victory isn't offered here at all; that
still needs Polymarket/DraftKings or the manual data/upcoming_props.csv path.
"""

import os
import statistics
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import json
import time

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"

# THE QUOTA MATH THAT FORCES THIS CACHE. The free tier allows 500 requests a
# month. Actions rebuilds roughly every 5 minutes, which is ~8,600 calls a
# month -- the quota is gone in about two days, every month, and the site then
# runs blind until it rolls over. That's exactly what happened: a live run
# returned OUT_OF_USAGE_CREDITS with used=499.
#
# Upgrading the plan only buys a longer runway before the same thing recurs.
# The real fix is call frequency: moneyline odds do not move meaningfully in
# five minutes, so one fetch an hour is no less accurate in practice and takes
# usage from thousands a month to a few hundred -- permanently inside the free
# tier.
ODDS_CACHE_PATH = "data/odds_api_cache.json"

# TWO HOURS, AND THE QUOTA IS THE WHOLE ARGUMENT. At one hour this is
# 24 x 30 = 720 requests a month against a 500-request free tier -- over the
# cap by 44%, which is the same overrun that spent the quota in two days
# before the cache existed. Two hours is ~360 a month and fits with room to
# spare. Moneylines do not move meaningfully in two hours; the quota running
# out moves them to "no price at all", which is worse than a slightly old one.
ODDS_CACHE_TTL_SEC = 2 * 60 * 60

# HOW OLD A FALLBACK MAY GET BEFORE IT IS REFUSED.
#
# When the key is rejected this module serves the cache instead, on the
# reasoning that stale odds beat no odds. That was true and unbounded, and
# unbounded is how it served a cache from 2026-08-30 until 2026-09-14 --
# FIFTEEN DAYS -- with nothing but a print to say so. Nobody reads a print.
#
# These prices are staked against. A price a book offered two weeks ago is
# not a price, it is a number; src/card_plays carries a dropped book quote
# for twelve hours for exactly this reason and refuses it after that. This is
# the same bound one notch looser, because this source is the coarser one.
ODDS_CACHE_MAX_SERVE_SEC = 24 * 60 * 60

# The books whose prices can actually be staked. Keys are The Odds API's own
# bookmaker keys; values must match src.card_plays.BETTABLE_VENUES exactly --
# a venue spelled differently there is a price that is fetched, displayed and
# then silently refused at the staking gate.
BETTABLE_BOOK_KEYS = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "betmgm": "BetMGM",
}


def _read_cache() -> tuple[list[dict] | None, float]:
    try:
        with open(ODDS_CACHE_PATH, encoding="utf-8") as fh:
            blob = json.load(fh)
        return blob.get("data"), float(blob.get("fetched_at", 0))
    except (OSError, ValueError, TypeError):
        return None, 0.0


def _write_cache(data: list[dict]) -> None:
    try:
        os.makedirs(os.path.dirname(ODDS_CACHE_PATH), exist_ok=True)
        with open(ODDS_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump({"fetched_at": time.time(), "data": data}, fh)
    except OSError as exc:
        print(f"[odds_api] could not write cache ({exc}) -- continuing uncached")


def fetch_mma_odds(api_key: str | None = None, regions: str = "us") -> list[dict]:
    cached, fetched_at = _read_cache()
    age = time.time() - fetched_at
    if cached is not None and age < ODDS_CACHE_TTL_SEC:
        _record_health(ok=True, age_sec=age, detail="served from cache, quota untouched")
        print(f"[odds_api] serving cached odds ({age/60:.0f} min old, {len(cached)} events) "
              f"-- no request made, quota untouched")
        return cached

    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set the ODDS_API_KEY environment variable "
            "(or pass one in), get a free key at https://the-odds-api.com"
        )

    resp = requests.get(
        ODDS_API_BASE,
        params={
            "regions": regions,
            "markets": "h2h,totals",
            "oddsFormat": "american",
            "apiKey": api_key,
        },
        timeout=15,
    )
    # A bare raise_for_status() turns every failure into an opaque "401
    # Unauthorized", which is the one message that DOESN'T distinguish the two
    # things it can mean: a key that's wrong/revoked, versus a valid key whose
    # monthly quota is spent. The Odds API reports usage in response headers
    # and puts a human-readable reason in the body, so surface both -- the
    # difference decides whether you regenerate a key or just wait for the
    # quota to roll over.
    if resp.status_code in (401, 403, 429) and cached is not None:
        if age > ODDS_CACHE_MAX_SERVE_SEC:
            # PAST THE BOUND THE CACHE STOPS BEING A PRICE. See
            # ODDS_CACHE_MAX_SERVE_SEC: serving this is how fifteen days went by.
            _record_health(ok=False, age_sec=age, detail=(
                f"key rejected (HTTP {resp.status_code}) and the cache is "
                f"{age/86400:.1f} days old -- past the {ODDS_CACHE_MAX_SERVE_SEC/3600:.0f}h "
                f"limit, so no book price was served"))
            print(f"[odds_api] request rejected (HTTP {resp.status_code}) and the cache is "
                  f"{age/86400:.1f} DAYS old -- refusing to serve it. Book prices are "
                  f"staked against; regenerate ODDS_API_KEY or wait for the quota.")
            return []
        _record_health(ok=False, age_sec=age, detail=(
            f"key rejected (HTTP {resp.status_code}) -- serving cache from "
            f"{age/3600:.1f}h ago"))
        print(f"[odds_api] request rejected (HTTP {resp.status_code}) -- falling back to "
              f"cached odds from {age/3600:.1f}h ago. Stale odds beat no odds, and this "
              f"keeps the site running through a quota outage instead of going dark.")
        return cached
    if resp.status_code in (401, 403, 429):
        used = resp.headers.get("x-requests-used")
        left = resp.headers.get("x-requests-remaining")
        body = (resp.text or "")[:200].strip()
        quota = f" | quota used={used} remaining={left}" if used or left else ""
        raise RuntimeError(
            f"The Odds API rejected the key (HTTP {resp.status_code}){quota}. "
            f"Response: {body or '(empty)'}. "
            f"If remaining=0 the key is fine and the quota is spent; otherwise the key "
            f"itself is invalid or revoked -- regenerate at the-odds-api.com and update "
            f"the ODDS_API_KEY secret in GitHub Actions."
        )
    resp.raise_for_status()
    data = resp.json()
    # Persist so the next ~12 scheduled runs in this hour cost nothing.
    _write_cache(data)
    remaining = resp.headers.get("x-requests-remaining")
    _record_health(ok=True, age_sec=0.0,
                   detail=f"fetched {len(data)} events"
                          + (f", quota remaining {remaining}" if remaining else ""))
    print(f"[odds_api] fetched {len(data)} events"
          + (f" (quota remaining: {remaining})" if remaining else "")
          + f" -- cached for {ODDS_CACHE_TTL_SEC // 60} min")
    return data


def _record_health(ok: bool, age_sec: float, detail: str) -> None:
    """
    Publish this source's freshness so a dead key is visible on the site.

    Never raises: a health write that breaks the build would be a strictly
    worse failure than the one it reports.
    """
    try:
        from src.source_health import record
        record("odds_api", {
            "ok": bool(ok),
            "cache_age_hours": round(max(age_sec, 0.0) / 3600.0, 2),
            "max_serve_hours": ODDS_CACHE_MAX_SERVE_SEC / 3600.0,
            "detail": detail,
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    except Exception as exc:                      # noqa: BLE001 -- see docstring
        print(f"[odds_api] health not recorded ({exc}) -- continuing")


def _event_dates(commence_time) -> set:
    """
    The dates this start time could plausibly belong to, as YYYY-MM-DD.

    A UFC card is dated by its US Eastern calendar day, so a main event at
    2026-09-20T03:30Z is the 09-19 card. Matching on the UTC date alone would
    file half the schedule a day late and miss every card it should match.
    The UTC date is kept alongside it for cards that run in Europe or Asia,
    where the ET conversion is the one that misfiles.
    """
    try:
        parsed = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return set()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    out = {parsed.astimezone(timezone.utc).date().isoformat()}
    try:
        out.add(parsed.astimezone(ZoneInfo("America/New_York")).date().isoformat())
    except Exception:                             # noqa: BLE001 -- no tzdata
        pass
    return out


def to_book_moneyline_rows(events: list[dict], dates=None) -> list[dict]:
    """
    One row per (fight, fighter, BOOK) -- the bettable moneylines.

    This exists because to_upcoming_rows takes the MEDIAN across bookmakers,
    and a median is not a price anybody offers. It is a fine reference number
    and a useless bet: src.card_plays will only stake a venue in
    BETTABLE_VENUES, so a median row stamped "The Odds API" is fetched,
    displayed, and then refused at the gate.

    Moneyline only. The ladder is moneyline-only, so this is the market where
    a missing book price costs a real play; totals stay on the median path
    rather than quietly repricing every parlay leg in the same change.

    `dates` filters to the cards we actually track. The feed carries duplicate
    and far-future listings of the same fight -- Arman Tsarukyan appeared
    three times in one response, at 09-10, 09-20 and 12-31 -- so an unfiltered
    pass can price a card off the wrong event entirely.
    """
    wanted = {str(d)[:10] for d in (dates or [])} or None
    rows: list[dict] = []
    for fight_id, event in enumerate(events or [], start=1):
        fighter_a = event.get("home_team")
        fighter_b = event.get("away_team")
        if not fighter_a or not fighter_b:
            continue
        start_date = event.get("commence_time")
        if wanted is not None and not (_event_dates(start_date) & wanted):
            continue

        for book in event.get("bookmakers", []) or []:
            venue = BETTABLE_BOOK_KEYS.get(str(book.get("key") or "").lower())
            if venue is None:
                continue
            for market in book.get("markets", []) or []:
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []) or []:
                    name = outcome.get("name")
                    if name not in (fighter_a, fighter_b):
                        continue
                    try:
                        price = float(outcome["price"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    rows.append({
                        "fight_id": f"oa_{fight_id}",
                        "fighter_a": fighter_a, "fighter_b": fighter_b,
                        "event_name": "", "start_date": start_date,
                        "weight_class": "", "card_position": "",
                        "market": "Moneyline", "selection": name,
                        "selection_method": "",
                        "odds_american": price,
                        # Vigged and bettable -- the opposite of the Polymarket
                        # rows these sit beside. The flag travels per row.
                        "source": venue,
                        "source_is_vig_free": False,
                        "price_updated_at": book.get("last_update"),
                    })
    return rows


def to_upcoming_rows(events: list[dict]) -> list[dict]:
    """
    Converts The Odds API's per-bookmaker response into the same row shape
    edge_finder expects, by taking the MEDIAN price across all returned
    bookmakers for each fighter/line (reduces noise from any single book
    being an outlier).
    """
    rows = []
    for fight_id, event in enumerate(events, start=1):
        fighter_a = event.get("home_team")
        fighter_b = event.get("away_team")
        start_date = event.get("commence_time")

        prices_a, prices_b = [], []
        totals_prices: dict[tuple[str, float], list[float]] = {}  # (Over/Under, point) -> prices

        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == fighter_a:
                            prices_a.append(outcome["price"])
                        elif outcome["name"] == fighter_b:
                            prices_b.append(outcome["price"])
                elif market.get("key") == "totals":
                    for outcome in market.get("outcomes", []):
                        point = outcome.get("point")
                        name = outcome.get("name")  # "Over" or "Under"
                        if point is None or name not in ("Over", "Under"):
                            continue
                        totals_prices.setdefault((name, point), []).append(outcome["price"])

        if prices_a and prices_b:
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "event_name": "", "start_date": start_date, "weight_class": "", "card_position": "",
                "market": "Moneyline", "selection": fighter_a, "selection_method": "",
                "odds_american": statistics.median(prices_a),
            })
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "event_name": "", "start_date": start_date, "weight_class": "", "card_position": "",
                "market": "Moneyline", "selection": fighter_b, "selection_method": "",
                "odds_american": statistics.median(prices_b),
            })

        for (side, point), prices in totals_prices.items():
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "event_name": "", "start_date": start_date, "weight_class": "", "card_position": "",
                "market": "TotalRounds", "selection": f"{side} {point}", "selection_method": str(point),
                "odds_american": statistics.median(prices),
            })

    return rows
