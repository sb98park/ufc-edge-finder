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

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"


def fetch_mma_odds(api_key: str | None = None, regions: str = "us") -> list[dict]:
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
    resp.raise_for_status()
    return resp.json()


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
