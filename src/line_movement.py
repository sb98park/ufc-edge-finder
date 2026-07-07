"""
Line movement tracking: each run compares current live odds against the
last saved snapshot (committed to the repo by the previous Action run),
so real price movement is visible instead of a clean slate every time.

Honest scope note: this tracks PRICE movement only. True "sharp money"
detection needs bet-volume/handle data (what % of bets vs. what % of
dollars are on each side) that no free source provides -- without that,
there's no way to distinguish a line moving because of one large bet
from a genuine public-money swing. What this DOES give you: real,
verifiable price movement, which is useful signal on its own even without
knowing exactly who's behind it.
"""

import json
import os
from datetime import datetime, timezone

from src.odds_utils import american_to_decimal

SNAPSHOT_PATH = "data/odds_snapshot.json"
NOTABLE_MOVEMENT_THRESHOLD_PCT = 15.0


def _bet_key_str(row: dict) -> str:
    """String key for JSON serialization (JSON dict keys must be strings)."""
    return f"{row.get('fighter', '')}|{row.get('market', '')}"


def load_snapshot() -> dict:
    if not os.path.exists(SNAPSHOT_PATH):
        return {}
    try:
        with open(SNAPSHOT_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_snapshot(edges: list[dict]) -> None:
    snapshot = {}
    now = datetime.now(timezone.utc).isoformat()
    for row in edges:
        if row.get("odds_american") is None:
            continue
        snapshot[_bet_key_str(row)] = {"odds": row["odds_american"], "timestamp": now}
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)


def annotate_movement(edges: list[dict], previous_snapshot: dict) -> None:
    """Mutates each edge dict in place, adding a 'movement' field when prior data exists for it."""
    for row in edges:
        if row.get("odds_american") is None:
            row["movement"] = None
            continue
        prev = previous_snapshot.get(_bet_key_str(row))
        if not prev:
            row["movement"] = None
            continue

        prev_odds, curr_odds = prev["odds"], row["odds_american"]
        if prev_odds == curr_odds:
            row["movement"] = {"direction": "flat", "from": prev_odds, "to": curr_odds, "notable": False}
            continue

        # Compare via implied probability so direction is consistent across
        # the +/- sign flip at even money, not just raw number comparison
        prev_prob = 1 / american_to_decimal(prev_odds)
        curr_prob = 1 / american_to_decimal(curr_odds)
        pct_change = abs(curr_prob - prev_prob) / prev_prob * 100 if prev_prob else 0

        row["movement"] = {
            "direction": "shortening" if curr_prob > prev_prob else "drifting",
            "from": prev_odds, "to": curr_odds,
            "pct_change": round(pct_change, 1),
            "notable": pct_change >= NOTABLE_MOVEMENT_THRESHOLD_PCT,
        }
