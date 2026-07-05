"""Shared live-odds fetching logic: try DraftKings (full props), fall back to The Odds API (moneyline only)."""

import pandas as pd

from src.draftkings_scraper import fetch_draftkings_mma_props
from src.live_odds import fetch_mma_odds, to_upcoming_rows


def get_live_props() -> tuple[pd.DataFrame, str]:
    try:
        rows = fetch_draftkings_mma_props()
        if rows:
            return pd.DataFrame(rows), "DraftKings (moneyline + props)"
    except Exception as exc:
        print(f"[warn] DraftKings scrape failed ({exc}), falling back to The Odds API (moneyline only)")

    try:
        events = fetch_mma_odds()
        rows = to_upcoming_rows(events)
        return pd.DataFrame(rows), "The Odds API (moneyline only)"
    except Exception as exc:
        raise RuntimeError(f"Both DraftKings and The Odds API failed: {exc}")
