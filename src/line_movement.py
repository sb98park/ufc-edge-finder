"""
Line movement tracking: each run appends the current odds onto a per-bet
history (committed to the repo by the previous Action run), so real price
movement over time is visible -- both as a quick before/after delta and as
an actual chart, similar to what Polymarket itself shows.

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
MAX_HISTORY_POINTS = 30  # keeps the snapshot file from growing unbounded


def _bet_key_str(row: dict) -> str:
    """String key for JSON serialization (JSON dict keys must be strings)."""
    return f"{row.get('fighter', '')}|{row.get('market', '')}"


def load_snapshot() -> dict:
    """
    Returns {bet_key: {"history": [{"odds": X, "timestamp": Y}, ...]}}.
    Transparently upgrades the older single-value format (from before
    history tracking existed) into a one-point history list.
    """
    if not os.path.exists(SNAPSHOT_PATH):
        return {}
    try:
        with open(SNAPSHOT_PATH) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    normalized = {}
    for key, val in raw.items():
        if isinstance(val, dict) and "history" in val:
            normalized[key] = val
        elif isinstance(val, dict) and "odds" in val:
            normalized[key] = {"history": [val]}
    return normalized


def save_snapshot(edges: list[dict], previous_snapshot: dict) -> dict:
    """Appends current odds onto each bet's history and writes the result to disk."""
    now = datetime.now(timezone.utc).isoformat()
    new_snapshot = {k: {"history": list(v.get("history", []))} for k, v in previous_snapshot.items()}

    for row in edges:
        if row.get("odds_american") is None:
            continue
        key = _bet_key_str(row)
        entry = new_snapshot.setdefault(key, {"history": []})
        entry["history"].append({"odds": row["odds_american"], "timestamp": now})
        entry["history"] = entry["history"][-MAX_HISTORY_POINTS:]

    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(new_snapshot, f, indent=2)
    return new_snapshot


def annotate_movement(edges: list[dict], previous_snapshot: dict) -> None:
    """Mutates each edge dict in place, adding a 'movement' field when prior data exists for it."""
    for row in edges:
        if row.get("odds_american") is None:
            row["movement"] = None
            continue
        prev_entry = previous_snapshot.get(_bet_key_str(row))
        history = prev_entry.get("history", []) if prev_entry else []
        if not history:
            row["movement"] = None
            continue

        prev_odds, curr_odds = history[-1]["odds"], row["odds_american"]
        if prev_odds == curr_odds:
            row["movement"] = {"direction": "flat", "from": prev_odds, "to": curr_odds, "notable": False}
            continue

        prev_prob = 1 / american_to_decimal(prev_odds)
        curr_prob = 1 / american_to_decimal(curr_odds)
        pct_change = abs(curr_prob - prev_prob) / prev_prob * 100 if prev_prob else 0

        row["movement"] = {
            "direction": "shortening" if curr_prob > prev_prob else "drifting",
            "from": prev_odds, "to": curr_odds,
            "pct_change": round(pct_change, 1),
            "notable": pct_change >= NOTABLE_MOVEMENT_THRESHOLD_PCT,
        }


def attach_charts_to_fight(fight: dict, full_snapshot: dict) -> None:
    """
    Attaches a moneyline chart (always shown) and a list of other-market
    charts (shown behind a toggle) to a fight dict, using whatever history
    exists in the snapshot -- independent of whether that market has a live
    price THIS run, since charting is about trend, not just current status.
    """
    ml_key = f"{fight['fighter_a']}|Moneyline"
    ml_entry = full_snapshot.get(ml_key)
    fight["moneyline_chart"] = build_sparkline_svg(ml_entry["history"]) if ml_entry else None

    other_charts = []
    seen_keys = {ml_key}
    for edge in fight.get("edges", []):
        key = f"{edge['fighter']}|{edge['market']}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        entry = full_snapshot.get(key)
        if entry:
            svg = build_sparkline_svg(entry["history"])
            if svg:
                other_charts.append({"label": f"{edge['fighter']} — {edge['market']}", "svg": svg})
    fight["other_charts"] = other_charts


def build_sparkline_svg(history: list[dict], width: int = 220, height: int = 48) -> str | None:
    """
    Renders a small SVG line chart of implied PROBABILITY over time (not raw
    American odds, which have a non-linear scale around even money) --
    similar in spirit to Polymarket's own price charts. Returns None if
    there isn't enough history yet for a meaningful chart.
    """
    if len(history) < 2:
        return None

    probs = []
    for point in history:
        try:
            probs.append(1 / american_to_decimal(point["odds"]))
        except (ZeroDivisionError, ValueError):
            continue
    if len(probs) < 2:
        return None

    min_p, max_p = min(probs), max(probs)
    range_p = (max_p - min_p) or 1.0
    padding = 4
    plot_w, plot_h = width - 2 * padding, height - 2 * padding

    points = []
    for i, p in enumerate(probs):
        x = padding + (i / (len(probs) - 1)) * plot_w
        y = padding + (1 - (p - min_p) / range_p) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    line_color = "#3ddc84" if probs[-1] >= probs[0] else "#ff5c5c"
    polyline_points = " ".join(points)
    area_points = f"{padding:.1f},{height - padding:.1f} " + polyline_points + f" {width - padding:.1f},{height - padding:.1f}"

    start_pct = round(probs[0] * 100)
    end_pct = round(probs[-1] * 100)

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="sparkline" role="img" '
        f'aria-label="Probability moved from {start_pct}% to {end_pct}%">'
        f'<polygon points="{area_points}" fill="{line_color}" opacity="0.12"/>'
        f'<polyline points="{polyline_points}" fill="none" stroke="{line_color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )
