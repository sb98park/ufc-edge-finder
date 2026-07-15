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
from src.card_matcher import _normalize_name

SNAPSHOT_PATH = "data/odds_snapshot.json"
TOKEN_CACHE_PATH = "data/clob_token_cache.json"
NOTABLE_MOVEMENT_THRESHOLD_PCT = 15.0
MAX_HISTORY_POINTS = 30

LINE_COLOR_A = "#d4af37"
LINE_COLOR_B = "#8a8f9a"


def load_token_cache() -> dict:
    """{normalized_fighter_name: clob_token_id}, persisted across runs so a
    fight's chart doesn't lose its token just because THIS run's Polymarket
    discovery didn't happen to surface that market again."""
    if not os.path.exists(TOKEN_CACHE_PATH):
        return {}
    try:
        with open(TOKEN_CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_token_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(TOKEN_CACHE_PATH), exist_ok=True)
    with open(TOKEN_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def update_token_cache(edges: list[dict], cache: dict) -> dict:
    """Merges any freshly-discovered tokens from this run's edges into the persisted cache."""
    updated = dict(cache)
    for row in edges:
        token_id = row.get("clob_token_id")
        fighter = row.get("fighter")
        if token_id and fighter and row.get("market") == "Moneyline":
            updated[_normalize_name(fighter)] = token_id
    return updated


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
    implied_a: bool = False, implied_b: bool = False,
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

    # Pick gridlines with a MINIMUM PIXEL GAP between adjacent labels, not
    # just "falls within the value range" -- the old approach could select
    # candidates that were numerically valid but rendered only a few pixels
    # apart on a short chart, causing labels to visually overlap (confirmed
    # live on the smaller single-fighter charts: "60% 50% 40% 35%" crammed
    # together unreadably). Working in pixel space instead of percentage
    # space means this scales correctly regardless of chart height.
    MIN_GRIDLINE_GAP_PX = 16
    MAX_GRIDLINES = 4
    gridline_candidates = [0.10, 0.20, 0.25, 0.35, 0.40, 0.50, 0.60, 0.65, 0.75, 0.80, 0.90]
    in_range = sorted(c for c in gridline_candidates if min_p <= c <= max_p)

    shown_gridlines = []
    last_y = None
    for pct in in_range:
        y = y_at(pct)
        if last_y is None or abs(y - last_y) >= MIN_GRIDLINE_GAP_PX:
            shown_gridlines.append(pct)
            last_y = y
        if len(shown_gridlines) >= MAX_GRIDLINES:
            break

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
            f'stroke-linejoin="round" stroke-linecap="round" class="chart-draw-line"/>'
            f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3.5" fill="{color}" '
            f'class="chart-endpoint-halo" style="transform-box: fill-box; transform-origin: center;"/>'
            f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3.5" fill="{color}" class="chart-draw-endpoint"/>'
        )
        return svg, round(last_p * 100)

    line_a_svg, pct_a = render_line(points_a, LINE_COLOR_A)
    line_b_svg, pct_b = render_line(points_b, LINE_COLOR_B)

    legend_svg = ""
    ly = 10
    if pct_a is not None:
        short_name_a = name_a.split()[-1] + (" ~" if implied_a else "")
        legend_svg += (
            f'<circle cx="{width - 8}" cy="{ly}" r="3" fill="{LINE_COLOR_A}"/>'
            f'<text x="{width - 14}" y="{ly + 3}" font-size="9" font-weight="700" fill="{LINE_COLOR_A}" text-anchor="end">{short_name_a} {pct_a}%</text>'
        )
        ly += 13
    if pct_b is not None:
        short_name_b = name_b.split()[-1] + (" ~" if implied_b else "")
        legend_svg += (
            f'<circle cx="{width - 8}" cy="{ly}" r="3" fill="{LINE_COLOR_B}"/>'
            f'<text x="{width - 14}" y="{ly + 3}" font-size="9" font-weight="700" fill="{LINE_COLOR_B}" text-anchor="end">{short_name_b} {pct_b}%</text>'
        )

    # Reveal mask: a rect covering the plot area that shrinks away via
    # transform:scaleX (anchored to the right edge, so it uncovers left to
    # right) instead of animating the lines' own stroke properties
    # directly. This deliberately reuses the same transform-based
    # technique already proven reliable for the radar chart's reveal --
    # stroke-dasharray/dashoffset animation (both CSS-transitioned and
    # later JS-rAF-driven) proved unreliable specifically on iOS Safari
    # across multiple rounds of testing, which lines up with a known,
    # documented gap in WebKit: transform animations get real hardware
    # compositing, direct SVG stroke-property animation often doesn't.
    mask_svg = (
        f'<rect x="{left_pad}" y="{top_pad}" width="{plot_w}" height="{plot_h}" fill="#1a1e28" '
        f'class="chart-reveal-mask" style="transform-box: fill-box; transform-origin: right center;"/>'
    )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="dual-chart" role="img" '
        f'aria-label="{name_a} vs {name_b} probability over time{" (one side implied)" if (implied_a or implied_b) else ""}">'
        + grid_svg + axis_svg + line_a_svg + line_b_svg + legend_svg + x_labels_svg + mask_svg +
        '</svg>'
    )


def attach_charts_to_fight(fight: dict, full_snapshot: dict, token_cache: dict | None = None) -> None:
    """
    Attaches a dual-line moneyline chart (always shown, using REAL CLOB
    history when a token ID is available, falling back to our own
    accumulated snapshot otherwise) and a list of other-market charts
    (method/rounds/distance, shown behind a toggle).

    token_cache: a persisted {normalized_fighter_name: clob_token_id} map
    from PAST runs, used when this run's live discovery didn't happen to
    surface a fight -- Polymarket's volume-based discovery doesn't find
    every fight every run (confirmed live: even a card's main event can
    miss the cut against the whole platform's volume ranking), so without
    this, a fight's chart would silently regress to sparse data any time
    discovery has an off run, even after previously having full history.
    """
    fighter_a, fighter_b = fight["fighter_a"], fight["fighter_b"]
    token_cache = token_cache if token_cache is not None else {}

    ml_edges = [e for e in fight.get("edges", []) if e.get("market") == "Moneyline"]
    # Normalized matching, not exact string equality -- Polymarket's raw
    # fighter name can differ from our canonical name in accents/hyphenation
    # (confirmed live: "Benoît Saint Denis" vs our "Benoit Saint-Denis"),
    # which silently broke token lookup even though the token was right there.
    norm_a, norm_b = _normalize_name(fighter_a), _normalize_name(fighter_b)
    token_a = next((e.get("clob_token_id") for e in ml_edges if _normalize_name(e.get("fighter", "")) == norm_a), None)
    token_b = next((e.get("clob_token_id") for e in ml_edges if _normalize_name(e.get("fighter", "")) == norm_b), None)

    # Fall back to the persisted cache if this run's live discovery didn't
    # find a token for one or both sides.
    if not token_a:
        token_a = token_cache.get(norm_a)
    if not token_b:
        token_b = token_cache.get(norm_b)

    if ml_edges and not (token_a and token_b):
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

    # If exactly one side has real history, derive the other as its
    # complement (1 - p at each timestamp) instead of leaving it blank --
    # a two-way moneyline's two probabilities are genuinely complementary
    # (ignoring vig), so this is a legitimate derived line, not a guess.
    # Confirmed live: fights like Cortez/Wang were only showing one side's
    # movement even though the other side's line is fully implied by it.
    implied_a, implied_b = False, False
    if len(points_a) >= 2 and len(points_b) < 2:
        points_b = [(t, 1 - p) for t, p in points_a]
        implied_b = True
        print(f"[charts] {fighter_b}: derived as inverse of {fighter_a}'s real line (no independent data)")
    elif len(points_b) >= 2 and len(points_a) < 2:
        points_a = [(t, 1 - p) for t, p in points_b]
        implied_a = True
        print(f"[charts] {fighter_a}: derived as inverse of {fighter_b}'s real line (no independent data)")
    elif len(points_a) >= 2 and len(points_b) >= 2:
        # Both sides have independently-sourced real data. A genuine
        # two-way market's two prices are complementary (ignoring vig),
        # but independently-scraped sides are rarely sampled at the exact
        # same timestamps, so their raw latest points can drift apart by
        # a few percent even when nothing is actually wrong -- and
        # unlike a sportsbook's displayed odds, this chart shows no vig
        # figure to explain that gap, so ANY visible gap reads as a bug
        # to someone looking at it, not just a large one. Always trust
        # whichever side's most recent point is actually more recent
        # (not whichever has more total accumulated points -- snapshot
        # data is captured opportunistically per-run, unlike CLOB history
        # which covers both sides over the identical window, so a side
        # with more total points can still have a staler latest reading
        # than a side with fewer but fresher ones. Confirmed live:
        # McGregor had 15 points but a stale latest one; Holloway had
        # fewer but more current data) and derive the other side as its
        # exact complement across the whole line, not just the latest
        # point, so the two displayed lines always sum to 100% everywhere
        # on the chart, not only at one end of it.
        sorted_a = sorted(points_a, key=lambda p: p[0])
        sorted_b = sorted(points_b, key=lambda p: p[0])
        latest_ts_a, latest_a = sorted_a[-1]
        latest_ts_b, latest_b = sorted_b[-1]
        if abs((latest_a + latest_b) - 1.0) > 0.005:
            if latest_ts_a >= latest_ts_b:
                points_b = [(t, 1 - p) for t, p in points_a]
                implied_b = True
                print(f"[charts] {fighter_a} vs {fighter_b}: independent sides didn't sum to 100% "
                      f"({latest_a*100:.0f}% + {latest_b*100:.0f}%) -- trusting {fighter_a}'s more "
                      f"recent point ({len(points_a)} pts, latest at {latest_ts_a:.0f}) over "
                      f"{fighter_b}'s ({len(points_b)} pts, latest at {latest_ts_b:.0f}), "
                      f"deriving {fighter_b} as its complement")
            else:
                points_a = [(t, 1 - p) for t, p in points_b]
                implied_a = True
                print(f"[charts] {fighter_a} vs {fighter_b}: independent sides didn't sum to 100% "
                      f"({latest_a*100:.0f}% + {latest_b*100:.0f}%) -- trusting {fighter_b}'s more "
                      f"recent point ({len(points_b)} pts, latest at {latest_ts_b:.0f}) over "
                      f"{fighter_a}'s ({len(points_a)} pts, latest at {latest_ts_a:.0f}), "
                      f"deriving {fighter_a} as its complement")

    fight["moneyline_chart"] = build_dual_line_chart_svg(
        points_a, points_b, fighter_a, fighter_b, implied_a=implied_a, implied_b=implied_b
    )
    fight["moneyline_chart_has_implied"] = implied_a or implied_b
    if points_a and points_b:
        final_a = sorted(points_a, key=lambda p: p[0])[-1][1]
        final_b = sorted(points_b, key=lambda p: p[0])[-1][1]
        print(f"[charts] {fighter_a} vs {fighter_b}: final displayed values -- "
              f"{fighter_a}={final_a*100:.1f}% {fighter_b}={final_b*100:.1f}% (sum={round((final_a+final_b)*100)}%)")

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
