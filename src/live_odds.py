"""
Fetch live UFC/MMA moneyline odds from The Odds API (https://the-odds-api.com).

Free tier note: MMA currently only has the h2h (moneyline) market available
via this API -- method-of-victory and round totals props aren't offered for
MMA by mainstream odds APIs yet, so this only covers moneylines. Props still
work through the manual data/upcoming_props.csv path in edge_finder.py.
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
            "markets": "h2h",
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
    bookmakers for each fighter (reduces noise from any single book being
    an outlier).
    """
    rows = []
    for fight_id, event in enumerate(events, start=1):
        fighter_a = event.get("home_team")
        fighter_b = event.get("away_team")

        prices_a, prices_b = [], []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == fighter_a:
                        prices_a.append(outcome["price"])
                    elif outcome["name"] == fighter_b:
                        prices_b.append(outcome["price"])

        if not prices_a or not prices_b:
            continue  # no usable odds yet for this fight

        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "Moneyline", "selection": fighter_a, "selection_method": "",
            "odds_american": statistics.median(prices_a),
        })
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "Moneyline", "selection": fighter_b, "selection_method": "",
            "odds_american": statistics.median(prices_b),
        })

    return rows
