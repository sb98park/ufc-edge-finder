"""Shared live-odds fetching logic: Polymarket first (public, documented, stable),
falling back to DraftKings (undocumented, can break/get blocked), then The Odds API
(moneyline only for MMA) as a last resort."""

import pandas as pd

from src.polymarket_source import fetch_polymarket_ufc_props
from src.draftkings_scraper import fetch_draftkings_mma_props
from src.live_odds import fetch_mma_odds, to_upcoming_rows


def get_live_props() -> tuple[pd.DataFrame, str]:
    try:
        rows = fetch_polymarket_ufc_props()
        if rows:
            return pd.DataFrame(rows), "Polymarket"
    except Exception as exc:
        print(f"[warn] Polymarket fetch failed ({exc}), falling back to DraftKings")

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
        raise RuntimeError(f"Polymarket, DraftKings, and The Odds API all failed: {exc}")
