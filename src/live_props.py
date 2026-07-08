"""
Shared live-odds fetching logic. Both Polymarket and DraftKings are tried
and MERGED (not first-success-wins) -- if Polymarket doesn't have a
method-of-victory market for a given fight but DraftKings does, that
DraftKings row still makes it into the final data instead of being
discarded just because Polymarket answered first. The Odds API is a
last resort only if both of the above return nothing at all.
"""

import pandas as pd

from src.polymarket_source import fetch_polymarket_ufc_props
from src.draftkings_scraper import fetch_draftkings_mma_props
from src.live_odds import fetch_mma_odds, to_upcoming_rows


def _pair_key(row: dict) -> frozenset | None:
    """Normalized fighter-pair key used to match the same fight across different sources."""
    from src.card_matcher import _normalize_name  # local import avoids any load-order issues

    fighter_a, fighter_b = row.get("fighter_a"), row.get("fighter_b")
    if not fighter_a or not fighter_b:
        return None
    return frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)})


def _bet_key(row: dict) -> tuple:
    """Full identity of a specific bet -- fighter pair + market + exact selection, not just the fight."""
    pair = _pair_key(row) or frozenset({row.get("fighter_a"), row.get("fighter_b")})
    return (pair, row.get("market"), row.get("selection"), row.get("selection_method"))


def get_live_props() -> tuple[pd.DataFrame, str]:
    sources_used = []
    pm_rows, dk_rows = [], []

    try:
        pm_rows = fetch_polymarket_ufc_props()
        if pm_rows:
            sources_used.append("Polymarket")
    except Exception as exc:
        print(f"[warn] Polymarket fetch failed ({exc})")

    try:
        dk_rows = fetch_draftkings_mma_props()
        if dk_rows:
            sources_used.append("DraftKings")
    except Exception as exc:
        print(f"[warn] DraftKings scrape failed ({exc})")

    if not pm_rows and not dk_rows:
        try:
            events = fetch_mma_odds()
            rows = to_upcoming_rows(events)
            if rows:
                return pd.DataFrame(rows), "The Odds API (moneyline only)"
        except Exception as exc:
            raise RuntimeError(f"Polymarket, DraftKings, and The Odds API all failed: {exc}")
        return pd.DataFrame(), "no source returned data"

    # Merge: Polymarket rows are kept as-is (no-vig, more trustworthy pricing).
    # DraftKings rows only get ADDED for (fighter-pair, market) combos
    # Polymarket didn't already cover -- filling gaps, not overriding.
    covered = {(_pair_key(r), r["market"]) for r in pm_rows if _pair_key(r)}
    supplemental = [r for r in dk_rows if (_pair_key(r), r["market"]) not in covered]

    combined_rows = pm_rows + supplemental

    # Final safety net: the same specific bet can show up twice at two
    # different prices (confirmed live) -- most likely from Polymarket
    # having two separate market listings covering the same fight. Keep
    # only the first occurrence of each exact bet.
    seen: dict[tuple, dict] = {}
    dupes_removed = 0
    upgrades = 0
    for row in combined_rows:
        key = _bet_key(row)
        if key not in seen:
            seen[key] = row
            continue
        dupes_removed += 1
        existing = seen[key]
        # Prefer whichever duplicate actually has a usable clob_token_id --
        # blindly keeping "whichever came first" was silently discarding
        # rows with real chart data in favor of rows without it, for no
        # reason other than list order (confirmed live: this is exactly
        # why McGregor vs Holloway's chart fell back to sparse tracking
        # data while every other fight got full CLOB history).
        if not existing.get("clob_token_id") and row.get("clob_token_id"):
            seen[key] = row
            upgrades += 1
    deduped = list(seen.values())
    if dupes_removed:
        msg = f"[live_props] removed {dupes_removed} duplicate bet(s) (same fighter/market/selection, different price)"
        if upgrades:
            msg += f", upgraded {upgrades} to keep the copy with a working clob_token_id"
        print(msg)

    source_label = " + ".join(sources_used) if len(sources_used) > 1 else sources_used[0]
    return pd.DataFrame(deduped), source_label
