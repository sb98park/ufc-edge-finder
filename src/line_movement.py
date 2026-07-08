"""
Line movement tracking, two layers:

1. Our own accumulated snapshot history (odds_snapshot.json, committed to
   the repo each run) -- used for the quick "shortening/drifting X%" badge
   next to odds throughout the site.

2. REAL historical price data pulled directly from Polymarket's CLOB API
   (prices-history endpoint) for the main chart -- this is the same data
   backing Polymarket's own charts, going back to when the market opened,
   not just what we've accumulated since this site started tracking. Public,
   no auth required.

Honest scope note: this tracks PRICE movement only. True "sharp money"
detection needs bet-volume/handle data (what % of bets vs. what % of
dollars are on each side) that no free source provides.
"""

import json
import os
from datetime import datetime, timezone

from src.odds_utils import american_to_decimal
from src.polymarket_source import fetch_price_history

SNAPSHOT_PATH = "data/odds_snapshot.json"
NOTABLE_MOVEMENT_THRESHOLD_PCT = 15.0
MAX_HISTORY_POINTS = 30

LINE_COLOR_A = "#d4af37"
LINE_COLOR_B = "#8a8f9a"


def _bet_key_str(row: dict) -> str:
    """String key for JSON serialization (JSON dict keys must be strings)."""
    return f"{row.get('fighter', '')}|{row.get('market', '')}"


def load_snapshot() -> dict:
    """Returns {bet_key: {"history": [{"odds": X, "timestamp": Y}, ...]}}."""
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


def _clob_points(history: list[dict]) -> list[tuple[float, float]]:
    """CLOB history [{"t": unix_ts, "p": price}] -> [(timestamp, probability)]."""
    points = []
    for pt in history:
        try:
            points.append((float(pt["t"]), float(pt["p"])))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _snapshot_points(history: list[dict]) -> list[tuple[float, float]]:
    """Our snapshot history [{"odds": X, "timestamp": iso}] -> [(timestamp, probability)]."""
    points = []
    for pt in history:
        try:
            ts = datetime.fromisoformat(pt["timestamp"]).timestamp()
            prob = 1 / american_to_decimal(pt["odds"])
            points.append((ts, prob))
        except (KeyError, ValueError, ZeroDivisionError):
            continue
    return points


def build_dual_line_chart_svg(
    points_a: list[tuple[float, float]], points_b: list[tuple[float, float]],
    name_a: str, name_b: str, width: int = 300, height: int = 170,
) -> str | None:
    """
    Renders both fighters' probability history on one chart with a real
    date axis and percentage gridlines -- styled after Polymarket's own
    chart (two colored lines, endpoint % callouts, axis labels).
    """
    if len(points_a) < 2 and len(points_b) < 2:
        return None

    all_points = points_a + points_b
    all_ts = [p[0] for p in all_points]
    all_probs = [p[1] for p in all_points]
    min_ts, max_ts = min(all_ts), max(all_ts)
    ts_range = (max_ts - min_ts) or 1.0

    min_p, max_p = min(all_probs), max(all_probs)
    range_p = max_p - min_p
    pad = max(range_p * 0.15, 0.015)
    min_p, max_p = max(0.0, min_p - pad), min(1.0, max_p + pad)
    range_p = (max_p - min_p) or 0.1

    left_pad, right_pad, top_pad, bottom_pad = 36, 10, 30, 22
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    def x_at(ts):
        return left_pad + ((ts - min_ts) / ts_range) * plot_w

    def y_at(p):
        return top_pad + (1 - (p - min_p) / range_p) * plot_h

    gridline_candidates = [0.10, 0.20, 0.25, 0.35, 0.40, 0.50, 0.60, 0.65, 0.75, 0.80, 0.90]
    shown_gridlines = sorted({c for c in gridline_candidates if min_p <= c <= max_p})[:5]
    grid_svg = ""
    for pct in shown_gridlines:
        y = y_at(pct)
        grid_svg += (
            f'<line x1="{left_pad}" y1="{y:.1f}" x2="{left_pad + plot_w}" y2="{y:.1f}" '
            f'stroke="#262b36" stroke-width="1" stroke-dasharray="2,3"/>'
            f'<text x="{left_pad - 6}" y="{y + 3:.1f}" font-size="9" fill="#8a8f9a" text-anchor="end">{round(pct*100)}%</text>'
        )

    axis_svg = (
        f'<line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{top_pad + plot_h}" stroke="#3a3f4a" stroke-width="1"/>'
        f'<line x1="{left_pad}" y1="{top_pad + plot_h}" x2="{left_pad + plot_w}" y2="{top_pad + plot_h}" stroke="#3a3f4a" stroke-width="1"/>'
    )

    start_label = datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime("%b %-d")
    end_label = datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime("%b %-d")
    x_labels_svg = (
        f'<text x="{left_pad}" y="{height - 4}" font-size="9" fill="#8a8f9a" text-anchor="start">{start_label}</text>'
        f'<text x="{left_pad + plot_w}" y="{height - 4}" font-size="9" fill="#8a8f9a" text-anchor="end">{end_label}</text>'
    )

    def render_line(points, color):
        if len(points) < 2:
            return "", None
        pts_sorted = sorted(points, key=lambda p: p[0])
        coords = " ".join(f"{x_at(t):.1f},{y_at(p):.1f}" for t, p in pts_sorted)
        last_t, last_p = pts_sorted[-1]
        end_x, end_y = x_at(last_t), y_at(last_p)
        svg = (
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3.5" fill="{color}"/>'
        )
        return svg, round(last_p * 100)

    line_a_svg, pct_a = render_line(points_a, LINE_COLOR_A)
    line_b_svg, pct_b = render_line(points_b, LINE_COLOR_B)

    legend_svg = ""
    ly = 10
    if pct_a is not None:
        short_name_a = name_a.split()[-1]
        legend_svg += (
            f'<circle cx="{width - 8}" cy="{ly}" r="3" fill="{LINE_COLOR_A}"/>'
            f'<text x="{width - 14}" y="{ly + 3}" font-size="9" font-weight="700" fill="{LINE_COLOR_A}" text-anchor="end">{short_name_a} {pct_a}%</text>'
        )
        ly += 13
    if pct_b is not None:
        short_name_b = name_b.split()[-1]
        legend_svg += (
            f'<circle cx="{width - 8}" cy="{ly}" r="3" fill="{LINE_COLOR_B}"/>'
            f'<text x="{width - 14}" y="{ly + 3}" font-size="9" font-weight="700" fill="{LINE_COLOR_B}" text-anchor="end">{short_name_b} {pct_b}%</text>'
        )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="dual-chart" role="img" '
        f'aria-label="{name_a} vs {name_b} probability over time">'
        + grid_svg + axis_svg + line_a_svg + line_b_svg + legend_svg + x_labels_svg +
        '</svg>'
    )


def attach_charts_to_fight(fight: dict, full_snapshot: dict) -> None:
    """
    Attaches a dual-line moneyline chart (always shown, using REAL CLOB
    history when a token ID is available on the edge itself, falling back
    to our own accumulated snapshot otherwise) and a list of other-market
    charts (method/rounds/distance, shown behind a toggle) -- same
    real-data-first approach applies uniformly to every market type now,
    since every edge carries its own clob_token_id when Polymarket provided one.
    """
    fighter_a, fighter_b = fight["fighter_a"], fight["fighter_b"]

    ml_edges = [e for e in fight.get("edges", []) if e.get("market") == "Moneyline"]
    token_a = next((e.get("clob_token_id") for e in ml_edges if e.get("fighter") == fighter_a), None)
    token_b = next((e.get("clob_token_id") for e in ml_edges if e.get("fighter") == fighter_b), None)

    if ml_edges and not (token_a and token_b):
        # Pinpoint diagnostic: shows exactly what's in ml_edges (fighter name
        # as it actually appears, and whether a token is present on each row)
        # so a name-matching mismatch is distinguishable from a genuinely
        # missing token, instead of guessing again.
        debug_rows = [(e.get("fighter"), bool(e.get("clob_token_id"))) for e in ml_edges]
        print(f"[charts] token lookup failed for {fighter_a!r} vs {fighter_b!r} -- "
              f"ml_edges fighter/has_token pairs: {debug_rows}")

    points_a = _clob_points(fetch_price_history(token_a)) if token_a else []
    points_b = _clob_points(fetch_price_history(token_b)) if token_b else []

    if len(points_a) < 2 and len(points_b) < 2:
        # no real CLOB history available -- fall back to our own accumulated snapshot
        entry_a = full_snapshot.get(f"{fighter_a}|Moneyline")
        points_a = _snapshot_points(entry_a["history"]) if entry_a else []
        entry_b = full_snapshot.get(f"{fighter_b}|Moneyline")
        points_b = _snapshot_points(entry_b["history"]) if entry_b else []
        print(f"[charts] {fighter_a} vs {fighter_b}: using OWN SNAPSHOT data "
              f"({len(points_a)} + {len(points_b)} points) -- no usable CLOB history")
    else:
        print(f"[charts] {fighter_a} vs {fighter_b}: using REAL CLOB data "
              f"({len(points_a)} + {len(points_b)} points)")

    fight["moneyline_chart"] = build_dual_line_chart_svg(points_a, points_b, fighter_a, fighter_b)

    has_live_ml = bool(ml_edges)
    fight["chart_building"] = has_live_ml and not fight["moneyline_chart"]

    other_charts = []
    seen_keys = {f"{fighter_a}|Moneyline", f"{fighter_b}|Moneyline"}
    for edge in fight.get("edges", []):
        key = f"{edge['fighter']}|{edge['market']}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        points = []
        token_id = edge.get("clob_token_id")
        if token_id:
            points = _clob_points(fetch_price_history(token_id))
        if len(points) < 2:
            entry = full_snapshot.get(key)
            points = _snapshot_points(entry["history"]) if entry else []

        if len(points) >= 2:
            svg = build_dual_line_chart_svg(points, [], edge["fighter"], "", width=260, height=90)
            if svg:
                other_charts.append({"label": f"{edge['fighter']} — {edge['market']}", "svg": svg})
    fight["other_charts"] = other_charts
