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

    # If this fight HAS live odds this run but doesn't have a chart yet
    # (only 1 history point logged so far), say so explicitly rather than
    # just silently showing nothing -- makes it clear more refreshes will
    # fill this in, instead of looking like charts only exist for one fight.
    has_live_ml = any(e.get("market") == "Moneyline" for e in fight.get("edges", []))
    fight["chart_building"] = has_live_ml and not fight["moneyline_chart"]

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


def build_sparkline_svg(history: list[dict], width: int = 280, height: int = 100) -> str | None:
    """
    Renders an SVG line chart of implied PROBABILITY over time (not raw
    American odds, which have a non-linear scale around even money) --
    styled similarly to Polymarket's own price charts: percentage gridlines,
    a start/end value callout, and a highlighted current-price dot. Returns
    None if there isn't enough history yet for a meaningful chart.
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
    raw_range = max_p - min_p

    # Genuinely flat (or near-flat) price history is real, honest
    # information -- draw a clean flat line at the ACTUAL value with a
    # clear "stable" label, instead of the old behavior of padding to a
    # fake range and drawing the line through an arbitrary midpoint, which
    # looked like a rendering bug rather than "the price hasn't moved."
    is_stable = raw_range < 0.005  # less than half a percentage point of movement

    if is_stable:
        pct = round(probs[-1] * 100)
        left_pad, right_pad, top_pad, bottom_pad = 34, 12, 14, 20
        plot_w = width - left_pad - right_pad
        y = height / 2
        return (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="sparkline" role="img" '
            f'aria-label="Price has been stable at {pct}%">'
            f'<line x1="{left_pad}" y1="{y:.1f}" x2="{left_pad + plot_w:.1f}" y2="{y:.1f}" '
            f'stroke="#8a8f9a" stroke-width="2" stroke-dasharray="4,3"/>'
            f'<text x="{left_pad}" y="{y - 10:.1f}" font-size="11" font-weight="700" fill="#8a8f9a">{pct}% · stable, no significant movement yet</text>'
            f'</svg>'
        )

    range_p = raw_range
    pad_p = range_p * 0.15
    min_p, max_p = max(0.0, min_p - pad_p), min(1.0, max_p + pad_p)
    range_p = max_p - min_p

    left_pad, right_pad, top_pad, bottom_pad = 34, 12, 14, 20
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    def x_at(i):
        return left_pad + (i / (len(probs) - 1)) * plot_w

    def y_at(p):
        return top_pad + (1 - (p - min_p) / range_p) * plot_h

    points = [f"{x_at(i):.1f},{y_at(p):.1f}" for i, p in enumerate(probs)]
    polyline_points = " ".join(points)
    area_points = f"{left_pad},{top_pad + plot_h} " + polyline_points + f" {left_pad + plot_w:.1f},{top_pad + plot_h}"

    line_color = "#3ddc84" if probs[-1] >= probs[0] else "#ff5c5c"

    # Horizontal gridlines at 25/50/75% (only the ones that actually fall
    # within the visible probability range, so a tightly-clustered chart
    # doesn't show meaningless gridlines miles off in unused space)
    gridlines = []
    for pct in (0.25, 0.50, 0.75):
        if min_p <= pct <= max_p:
            y = y_at(pct)
            gridlines.append(
                f'<line x1="{left_pad}" y1="{y:.1f}" x2="{left_pad + plot_w}" y2="{y:.1f}" '
                f'stroke="#262b36" stroke-width="1" stroke-dasharray="2,3"/>'
                f'<text x="{left_pad - 6}" y="{y + 3:.1f}" font-size="9" fill="#8a8f9a" text-anchor="end">{round(pct*100)}%</text>'
            )

    end_x, end_y = x_at(len(probs) - 1), y_at(probs[-1])
    end_pct = round(probs[-1] * 100)
    start_pct = round(probs[0] * 100)

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="sparkline" role="img" '
        f'aria-label="Probability moved from {start_pct}% to {end_pct}%">'
        + "".join(gridlines) +
        f'<polygon points="{area_points}" fill="{line_color}" opacity="0.12"/>'
        f'<polyline points="{polyline_points}" fill="none" stroke="{line_color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3.5" fill="{line_color}"/>'
        f'<text x="{end_x:.1f}" y="{max(10, end_y - 8):.1f}" font-size="11" font-weight="700" fill="{line_color}" '
        f'text-anchor="{"end" if end_x > width - 30 else "middle"}">{end_pct}%</text>'
        f'</svg>'
    )
