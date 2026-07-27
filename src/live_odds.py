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
ODDS_CACHE_TTL_SEC = 60 * 60


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
    print(f"[odds_api] fetched {len(data)} events"
          + (f" (quota remaining: {remaining})" if remaining else "")
          + f" -- cached for {ODDS_CACHE_TTL_SEC // 60} min")
    return data


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
