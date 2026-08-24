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
from datetime import datetime, timedelta, timezone

from src.odds_utils import american_to_decimal, add_estimated_vig, implied_prob_to_american, format_american_odds
from src.polymarket_source import fetch_price_history
from src.card_matcher import _normalize_name

SNAPSHOT_PATH = "data/odds_snapshot.json"
TOKEN_CACHE_PATH = "data/clob_token_cache.json"
NOTABLE_MOVEMENT_THRESHOLD_PCT = 15.0
MAX_HISTORY_POINTS = 30
# How long an untouched bet stays in the snapshot. Long enough to survive a
# quiet stretch between cards, short enough that orphaned keys -- settled
# fights, and every entry stranded by a key-format change -- do not accumulate.
STALE_ENTRY_DAYS = 45

# CORNER COLOURS, matching the waterfall. Fighter A was gold and fighter B a
# neutral grey -- so the chart said "model" in the site's colour language
# while actually meaning "the fighter on the left", and gave the second
# fighter no identity at all.
# Red/blue reads as two corners, and it returns gold to meaning MODEL
# everywhere. Safe here specifically because this chart has no direction
# colouring: it draws in greys and one accent, so corner-red can't be
# confused with a falling line.
# Scrub samples kept per chart. Above this a fingertip cannot address
# individual points anyway, and the payload is shipped inline per fight.
SCRUB_MAX_POINTS = 240


def _html_escape(text: str) -> str:
    """For a single-quoted SVG attribute: quotes and & only."""
    return (text.replace('&', '&amp;').replace("'", '&#39;')
                .replace('<', '&lt;').replace('>', '&gt;'))

LINE_COLOR_A = "#e53935"
LINE_COLOR_B = "#3b82f6"


def _clean_movement_label(label: str, fighter_a: str, fighter_b: str) -> str:
    """
    Strip the matchup and match the Fight props table's wording.

    Two problems with the raw label. It prefixed every fight-level chart with
    "A vs B", which the surrounding card already states. And it used the
    market's internal name ("Fight Method: KO/TKO") where the table two
    inches below says "Fight ends by KO/TKO" -- the same market under two
    names on one screen.
    """
    for pair in (f"{fighter_a} vs {fighter_b}", f"{fighter_b} vs {fighter_a}"):
        label = label.replace(f"{pair} — ", "").replace(f"{pair} - ", "").replace(pair, "")
    label = label.strip(" -\u2014")

    low = label.lower()
    for prefix in ("fight method:", "method:", "fight outcome:"):
        if low.startswith(prefix):
            method = label[len(prefix):].strip()
            mapped = {"sub": "Submission", "ko/tko": "KO/TKO",
                      "dec": "Decision", "goes the distance": "Decision"}.get(
                          method.lower(), method)
            return f"Fight ends by {mapped}"
    if low.startswith("goesthedistance") or low == "goes the distance":
        return "Fight ends by Decision"
    return label


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


def _price_basis(row: dict) -> str:
    """
    Which KIND of price this row carries: a bettable book quote, or the
    vig-free reference line shown when no book quoted.

    These are not comparable and must never share a history series. See
    _bet_key_str.
    """
    flag = row.get("source_is_vig_free")
    # Unknown provenance is treated as a reference line: the conservative
    # branch, since mistaking a book price for a fair one merely splits a
    # series while the reverse silently fabricates a move.
    if flag is None:
        return "fair"
    try:
        return "fair" if bool(flag) else "book"
    except (TypeError, ValueError):
        return "fair"


def _bet_key_str(row: dict) -> str:
    """
    String key for JSON serialization (JSON dict keys must be strings).

    THE PRICE BASIS IS PART OF THE KEY, and leaving it out fabricated line
    moves. `odds_american` used to be one feed's number; it is now the best
    BETTABLE price where a book quoted and the vig-free fair line where none
    did. Books post lines progressively through the week, so a fight flips
    from the second to the first mid-week -- and keyed only on fighter|market,
    that flip appended a book price onto a series of fair ones and the site
    reported a move that never happened. Measured at 2.2-5.2% depending on the
    price, which is simply the vig being mistaken for market action.

    Splitting the series means a fight that gains a book quote shows no
    movement until it has been quoted twice. That is correct: there is no
    prior bettable price to have moved from.
    """
    return f"{row.get('fighter', '')}|{row.get('market', '')}|{_price_basis(row)}"


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
    # PRUNE WHAT WILL NEVER MATCH AGAIN. This carried every key it had ever
    # seen forward unconditionally, which was survivable while keys were
    # stable. Adding the price basis to the key orphaned every pre-existing
    # entry at a stroke, and a settled card's bets never appear again either,
    # so without this the file only grows.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=STALE_ENTRY_DAYS)).isoformat()
    new_snapshot = {}
    for k, v in previous_snapshot.items():
        hist = list(v.get("history", []))
        last = hist[-1].get("timestamp") if hist else None
        if last is not None and str(last) < cutoff:
            continue
        new_snapshot[k] = {"history": hist}

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


# A RESOLVED MARKET IS NOT A PRICE. Polymarket does not delist the instant
# the horn sounds -- the contract prints 0.9995 / 0.0005 while it settles,
# and those ticks were being charted as though they were quotes. Every
# concluded fight's chart then terminated at the same impossible pair
# (-5000 / +2112) with the same gridlines: twelve fights on one card, one
# ending price. The polyline itself is real; only the tail is not.
#
# 0.95 is an APPROXIMATION of "stop at the last pre-fight price", and worth
# naming as one. The exact fix is to truncate at the fight's scheduled start,
# which needs a start time threaded into the chart builder; this instead uses
# the fact that no real MMA market sits this high beforehand. The heaviest
# favourites on record price around -1500 (0.938) -- Makhachev opened near
# -600 -- so 0.95 clears every genuine pre-fight quote while cutting both the
# resolution ticks and the in-fight drift toward them.
#
# At 0.99 one chart still ended at -5000: the line had drifted to ~0.98
# DURING the fight, which is not a resolution artifact but is not a closing
# line either. Anything past here also collides with the min(0.995, vig_p)
# guard below, which maps every such value onto the same -5000 label.
_SETTLED_CHART_PROB = 0.95


def _drop_settled(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Strip resolution ticks so a chart ends at the last real quote."""
    return [(t, p) for t, p in points
            if _SETTLED_CHART_PROB > p > (1.0 - _SETTLED_CHART_PROB)]


def _clob_points(history: list[dict]) -> list[tuple[float, float]]:
    """CLOB history [{"t": unix_ts, "p": price}] -> [(timestamp, probability)]."""
    points = []
    for pt in history:
        try:
            points.append((float(pt["t"]), float(pt["p"])))
        except (KeyError, TypeError, ValueError):
            continue
    return _drop_settled(points)


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
    return _drop_settled(points)


def build_dual_line_chart_svg(
    points_a: list[tuple[float, float]], points_b: list[tuple[float, float]],
    name_a: str, name_b: str, width: int = 300, height: int = 170,
    implied_a: bool = False, implied_b: bool = False,
    line_color: str | None = None, show_dot: bool = True,
    show_legend: bool = True,
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

    # right_pad reserves room for the endpoint price on single-line charts.
    # At 10px the last data point sat almost at the frame edge, so the label
    # had to flip left and landed ON the line -- unreadable exactly where the
    # line is most interesting. Widening the gutter lets it always sit to the
    # RIGHT of the dot, on empty background.
    left_pad, bottom_pad = 36, 22
    # 30 existed to clear the two legend rows that used to sit inside the
    # plot. With the legend moved out to its own rail, the only thing left in
    # this band is the scrub timestamp, which needs about 20.
    top_pad = 30 if not show_legend else 24
    right_pad = 44 if not show_legend else 10
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

    def _book_odds_label(pct_frac: float, complement_frac: float) -> str | None:
        """
        Safe book-style American odds label for a probability, given its
        paired complement (real value for a legend endpoint, or 1-pct for a
        gridline where no specific opponent probability applies). Returns
        None on any edge case (0%, 100%, NaN) rather than raising -- a
        missing odds label for one row is far better than breaking the
        whole chart's generation over an edge value.
        """
        try:
            vig_p, _ = add_estimated_vig(pct_frac, complement_frac)
            return format_american_odds(implied_prob_to_american(min(0.995, vig_p)))
        except (ValueError, ZeroDivisionError):
            return None

    grid_svg = ""
    for pct in shown_gridlines:
        y = y_at(pct)
        odds_label = _book_odds_label(pct, 1 - pct)
        grid_svg += (
            f'<line x1="{left_pad}" y1="{y:.1f}" x2="{left_pad + plot_w}" y2="{y:.1f}" '
            f'stroke="#2e2e30" stroke-width="1" stroke-dasharray="2,3"/>'
            f'<text class="label-pct" x="{left_pad - 6}" y="{y + 3:.1f}" font-size="9" fill="#8a8f9a" text-anchor="end">{round(pct*100)}%</text>'
        )
        if odds_label:
            grid_svg += (
                f'<text class="label-odds" x="{left_pad - 6}" y="{y + 3:.1f}" font-size="9" fill="#8a8f9a" text-anchor="end">{odds_label}</text>'
            )

    axis_svg = (
        f'<line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{top_pad + plot_h}" stroke="#3a3f4a" stroke-width="1"/>'
        f'<line x1="{left_pad}" y1="{top_pad + plot_h}" x2="{left_pad + plot_w}" y2="{top_pad + plot_h}" stroke="#3a3f4a" stroke-width="1"/>'
    )

    start_label = datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime("%b %-d")
    end_label = datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime("%b %-d")
    # Tagged so a scrub can fade them: the cursor's own timestamp lands on
    # this same baseline, and these two static endpoints are exactly what it
    # would collide with there.
    x_labels_svg = (
        f'<text class="ml-xlabel" x="{left_pad}" y="{height - 4}" font-size="9" fill="#8a8f9a" text-anchor="start">{start_label}</text>'
        f'<text class="ml-xlabel" x="{left_pad + plot_w}" y="{height - 4}" font-size="9" fill="#8a8f9a" text-anchor="end">{end_label}</text>'
    )

    def render_line(points, color, price_label=None):
        if len(points) < 2:
            return "", None, None
        pts_sorted = sorted(points, key=lambda p: p[0])
        coords = " ".join(f"{x_at(t):.1f},{y_at(p):.1f}" for t, p in pts_sorted)
        last_t, last_p = pts_sorted[-1]
        end_x, end_y = x_at(last_t), y_at(last_p)
        svg = (
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linejoin="round" stroke-linecap="round" class="chart-draw-line"/>'
            # Dot AND halo, both revealed by the clip sweep reaching this
            # point rather than by an opacity delay.
            #
            # The halo was briefly pulled out of here while chasing a
            # duplicate emission, which silently took the pulse off the
            # DUAL moneyline chart too -- that chart draws both fighters
            # through this same function, and was the one place the pulse
            # had always been correct.
            #
            # The delay is per-LINE, so each fighter's dot pulses as its
            # own line lands rather than both waiting on the slower one.
            f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3.5" fill="{color}"/>'
            f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3.5" fill="none" stroke="{color}" '
            f'stroke-width="1.5" class="chart-endpoint-halo" '
            f'style="transform-box: fill-box; transform-origin: center; '
            f'animation-delay: {1.6 * (end_x - left_pad) / max(plot_w + 6, 1):.3f}s"/>'
        )
        return svg, round(last_p * 100), last_p

    # line_color lets a single-series chart take the colour of whatever it is
    # about -- a fighter's corner for his own prop, gold for a fight-level
    # market. Every secondary chart drew in the same colour before, so nothing
    # on screen said whose line was moving.
    _colour_a = line_color or LINE_COLOR_A
    line_a_svg, pct_a, raw_a = render_line(points_a, _colour_a)

    # Current price at the endpoint, for single-line charts. Dropping the
    # legend removed this along with the fighter name -- but the price was the
    # half worth keeping; what made the legend wrong was labelling a chart
    # about the FIGHT with a fighter's name.
    halo_svg = ""
    endpoint_price_svg = ""
    if not show_legend and raw_a is not None and points_a:
        _pts = sorted(points_a, key=lambda p: p[0])
        _ex, _ey = x_at(_pts[-1][0]), y_at(_pts[-1][1])
        _label = _book_odds_label(raw_a, 1 - raw_a)
        if _label:
            # ALWAYS to the right, never flipped. The gutter above guarantees
            # the space, so the label can't overlap the line no matter where
            # the series ends. Vertically centred on the point rather than
            # raised, so it doesn't collide with the top gridline on a chart
            # that finishes high.
            # Same class as the endpoint dot, so it inherits the delayed
            # fade and lands WITH the finished line. Without it the price sat
            # there from the first frame while the line crawled toward it --
            # the answer arriving before the working.
            # Delay computed from the GEOMETRY, not assumed.
            #
            # The clip is 6px wider than the line so the dot's edge isn't
            # shaved, which means the sweep uncovers the dot at ~96.8% of its
            # travel -- 1.548s, not 1.600s. A flat 1.6s delay on the price
            # therefore trailed the dot by ~50ms, which is small enough to
            # look like a mistake rather than a beat.
            # Deriving it removes the gap exactly and survives any change to
            # the padding or the chart width.
            _sweep = 1.6
            _reveal_at = _sweep * (_ex - left_pad) / max(plot_w + 6, 1)
            endpoint_price_svg = (
                f'<text x="{_ex + 7:.1f}" y="{_ey + 3.5:.1f}" font-size="10" font-weight="700" '
                f'fill="#eef0f2" text-anchor="start" class="chart-endpoint-late" '
                f'style="transition-delay: {_reveal_at:.3f}s">{_label}</text>'
            )
            # render_line already emits the halo alongside the dot, inside
            # the clip. Emitting a second one here is what caused the
            # duplicate that took three passes to track down.
            halo_svg = ""
    line_b_svg, pct_b, raw_b = render_line(points_b, LINE_COLOR_B)

    # Pair each side's raw probability with a real complement when both
    # lines exist, falling back to 1-pct (matching the gridline treatment)
    # when only one side has data -- same assumption already used elsewhere
    # in this pipeline for implied/derived sides.
    complement_a = raw_b if raw_b is not None else (1 - raw_a if raw_a is not None else None)
    complement_b = raw_a if raw_a is not None else (1 - raw_b if raw_b is not None else None)

    # THE LEGEND IS AN HTML RAIL NOW, not text pinned inside the plot.
    # In the plot it occupied the top right, which is exactly where a scrub
    # timestamp wants to be, and the two collided. Out here it gets a row of
    # its own under the heading, the whole top of the chart comes free, and
    # the values are selectable text at a real font size instead of 9px SVG.
    legend_rail_html = ""
    # A legend distinguishes SERIES. On a single-line chart there is nothing
    # to distinguish, so "Salkilld +259" floating over a Total Rounds chart
    # was labelling the wrong thing entirely -- and the chart's own title
    # already names the market.
    if not show_legend:
        pct_a = pct_b = None

    def _rail_item(colour, short_name, pct, odds, series):
        cell = (f'<span class="ml-rail-item">'
                f'<i class="ml-rail-dot" style="background:{colour}"></i>'
                f'<span class="ml-rail-name">{_html_escape(short_name)}</span>')
        if pct is not None:
            cell += (f'<b class="label-pct ml-legend" data-series="{series}">{pct}%</b>')
        if odds:
            cell += (f'<b class="label-odds ml-legend" data-series="{series}">{_html_escape(str(odds))}</b>')
        return cell + '</span>'

    rail_cells = ""
    if pct_a is not None:
        short_name_a = name_a.split()[-1] + (" ~" if implied_a else "")
        odds_a = _book_odds_label(raw_a, complement_a) if complement_a is not None else None
        rail_cells += _rail_item(_colour_a, short_name_a, pct_a, odds_a, "a")
    if pct_b is not None:
        short_name_b = name_b.split()[-1] + (" ~" if implied_b else "")
        odds_b = _book_odds_label(raw_b, complement_b) if complement_b is not None else None
        rail_cells += _rail_item(LINE_COLOR_B, short_name_b, pct_b, odds_b, "b")
    if rail_cells:
        legend_rail_html = f'<div class="ml-rail">{rail_cells}</div>'

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
    # CLIP THE LINES, don't cover the chart.
    #
    # This was an opaque panel-coloured rect painted LAST over the whole plot
    # area, shrinking to reveal what was beneath. It hid everything under it,
    # not just the lines -- so the gridlines, the axes and the labels all
    # animated in together and the effect read as a box sliding away rather
    # than a line being drawn.
    #
    # A clipPath applies to the line group only. Grid, axes and labels render
    # immediately and never move; the line is revealed left to right beneath a
    # clip rect whose scaleX animates -- still a transform, so it keeps the
    # hardware compositing the mask was chosen for.
    # ===== SCRUB TIMELINE =====================================================
    # A reader holding a finger on the chart wants both prices AT THAT MOMENT,
    # not the endpoint. The two series are sampled independently, so neither
    # one's timestamps can serve as the shared axis: this merges them and
    # forward-fills each side, which is the honest reading of a step chart --
    # a price holds until it next changes.
    #
    # Downsampled to SCRUB_MAX_POINTS. The CLOB history runs to ~850 points on
    # a 300px-wide chart, so three quarters of them cannot be landed on with a
    # fingertip, and shipping them all would trade page weight for precision
    # nobody can express.
    scrub_json = ""
    if points_a and points_b:
        merged_ts = sorted({t for t, _ in points_a} | {t for t, _ in points_b})
        if len(merged_ts) > SCRUB_MAX_POINTS:
            step = len(merged_ts) / SCRUB_MAX_POINTS
            merged_ts = [merged_ts[min(len(merged_ts) - 1, int(i * step))]
                         for i in range(SCRUB_MAX_POINTS)]
            merged_ts = sorted(set(merged_ts))

        def _fill(points):
            """Value of this series at each merged timestamp, carried forward."""
            ordered = sorted(points, key=lambda p: p[0])
            out, j, last = [], 0, ordered[0][1]
            for t in merged_ts:
                while j < len(ordered) and ordered[j][0] <= t:
                    last = ordered[j][1]
                    j += 1
                out.append(last)
            return out

        fill_a, fill_b = _fill(points_a), _fill(points_b)
        scrub = {
            "x": [round(x_at(t), 1) for t in merged_ts],
            "ya": [round(y_at(v), 1) for v in fill_a],
            "yb": [round(y_at(v), 1) for v in fill_b],
            "t": [datetime.fromtimestamp(t, tz=timezone.utc).strftime("%a %-I:%M %p").upper()
                  for t in merged_ts],
            "pa": [round(v * 100) for v in fill_a],
            "pb": [round(v * 100) for v in fill_b],
            # Both representations, because the chart's legend has a %/odds
            # toggle and the scrub has to answer in whichever unit is showing.
            "oa": [_book_odds_label(va, vb) or "" for va, vb in zip(fill_a, fill_b)],
            "ob": [_book_odds_label(vb, va) or "" for va, vb in zip(fill_a, fill_b)],
        }
        scrub_json = _html_escape(json.dumps(scrub, separators=(",", ":")))

    clip_id = f"reveal-{abs(hash((width, height, len(points_a), len(points_b), name_a, name_b))) % 10**8}"
    past_id, future_id = f"past-{clip_id}", f"future-{clip_id}"
    series_id = f"series-{clip_id}"
    clip_svg = (
        f'<defs><clipPath id="{clip_id}">'
        f'<rect x="{left_pad}" y="{top_pad - 4}" width="{plot_w + right_pad}" height="{plot_h + 8}" '
        f'class="chart-reveal-clip" style="transform-box: fill-box; transform-origin: left center;"/>'
        f'</clipPath>'
        # Scrub clips. Defaults chosen so an untouched chart is unchanged:
        # the past covers everything, the future covers nothing.
        f'<clipPath id="{past_id}"><rect class="ml-clip-past" x="{left_pad}" y="{top_pad - 4}" '
        f'width="{plot_w + right_pad}" height="{plot_h + 8}"/></clipPath>'
        f'<clipPath id="{future_id}"><rect class="ml-clip-future" x="{left_pad}" y="{top_pad - 4}" '
        f'width="0" height="{plot_h + 8}"/></clipPath>'
        f'</defs>'
    )

    # The rail rides ahead of the chart so it lands directly under the
    # heading the template puts above this block.
    return legend_rail_html + (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="dual-chart" role="img" '
        + (f'data-scrub=\'{scrub_json}\' data-plot-left="{left_pad}" '
           f'data-plot-right="{left_pad + plot_w}" data-plot-top="{top_pad - 4}" '
           f'data-plot-h="{plot_h + 8}" ' if scrub_json else '')
        + f'aria-label="{name_a} vs {name_b} probability over time{" (one side implied)" if (implied_a or implied_b) else ""}">'
        # Grid and axes first and UNCLIPPED -- they are the backdrop and are
        # there from the first frame. Only the lines sit inside the clip.
        # Grid and axes render immediately. The lines, their endpoint dots and
        # the price all sit INSIDE the clip, so each is uncovered exactly when
        # the sweep reaches it -- the dot as the line lands, the price a beat
        # later because it sits to the right of the dot. Structural rather
        # than timed, so no amount of earlier revealing can put them on screen
        # before the line.
        + clip_svg + grid_svg + axis_svg + x_labels_svg
        # DIM COPY FIRST, LIVE COPY OVER IT. Both are clipped: the dim one to
        # everything after the finger, the live one to everything before. With
        # no scrub the past rect spans the plot and the future rect is empty,
        # so this is exactly the old picture until someone touches it.
        # Opacity rather than a darker stroke, so it works whatever colour a
        # series happens to be.
        # DEFINED ONCE, DRAWN TWICE. Emitting the two polylines into both
        # groups doubled every chart: the series is 45KB of point data, so the
        # dim copy alone was adding 1.7MB across the 38 charts on the page.
        # A <use> reference costs a few dozen bytes and renders identically.
        + f'<defs><g id="{series_id}" class="ml-series">' + line_a_svg + line_b_svg + '</g></defs>'
        + f'<g clip-path="url(#{clip_id})">'
        + f'<g class="ml-past" clip-path="url(#{past_id})"><use href="#{series_id}"/>' + halo_svg + '</g>'
        + f'<g class="ml-future" clip-path="url(#{future_id})"><use href="#{series_id}"/></g>'
        + '</g>'
        + endpoint_price_svg +
        '</svg>'
    )


def build_snapshot_chart(name_a: str, name_b: str, full_snapshot: dict) -> dict | None:
    """
    A moneyline chart for one fight, from the accumulated snapshot alone.

    NO NETWORK. attach_charts_to_fight reaches for CLOB history first, which
    is right for the ~65 fights on the live cards and wrong for the ~90 in
    the graded record -- that would be 180 requests for a marketing preview.
    The snapshot already holds what those fights traded at, right up to the
    bell, which is the whole reason a finished fight makes the better
    picture: its arc is complete.

    Returns a fight-shaped dict carrying moneyline_chart and its metrics, or
    None when neither side has enough recorded price to draw.
    """
    entry_a = full_snapshot.get(f"{name_a}|Moneyline")
    entry_b = full_snapshot.get(f"{name_b}|Moneyline")
    points_a = _snapshot_points(entry_a.get("history") or []) if entry_a else []
    points_b = _snapshot_points(entry_b.get("history") or []) if entry_b else []
    if len(points_a) < 2 and len(points_b) < 2:
        return None
    fight = {"fighter_a": name_a, "fighter_b": name_b}
    _finish_moneyline_chart(fight, points_a, points_b, quiet=True)
    return fight if fight.get("moneyline_chart") else None


def _finish_moneyline_chart(fight: dict, points_a: list, points_b: list,
                            quiet: bool = False) -> None:
    """
    Reconcile two price series into one chart, and record what shape it is.

    Split out of attach_charts_to_fight so the landing page can build a
    chart for a fight that is already over -- those carry the complete
    arc from open to bell, which is the picture the marketing card claims
    and the one an in-progress fight cannot supply. Everything delicate
    about a two-way market lives here (vig, complement derivation, which
    side to trust when they disagree), so it stays in exactly one place.
    """
    fighter_a, fighter_b = fight["fighter_a"], fight["fighter_b"]
    _say = (lambda *a, **k: None) if quiet else print
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
        _say(f"[charts] {fighter_b}: derived as inverse of {fighter_a}'s real line (no independent data)")
    elif len(points_b) >= 2 and len(points_a) < 2:
        points_a = [(t, 1 - p) for t, p in points_b]
        implied_a = True
        _say(f"[charts] {fighter_a}: derived as inverse of {fighter_b}'s real line (no independent data)")
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
                _say(f"[charts] {fighter_a} vs {fighter_b}: independent sides didn't sum to 100% "
                      f"({latest_a*100:.0f}% + {latest_b*100:.0f}%) -- trusting {fighter_a}'s more "
                      f"recent point ({len(points_a)} pts, latest at {latest_ts_a:.0f}) over "
                      f"{fighter_b}'s ({len(points_b)} pts, latest at {latest_ts_b:.0f}), "
                      f"deriving {fighter_b} as its complement")
            else:
                points_a = [(t, 1 - p) for t, p in points_b]
                implied_a = True
                _say(f"[charts] {fighter_a} vs {fighter_b}: independent sides didn't sum to 100% "
                      f"({latest_a*100:.0f}% + {latest_b*100:.0f}%) -- trusting {fighter_b}'s more "
                      f"recent point ({len(points_b)} pts, latest at {latest_ts_b:.0f}) over "
                      f"{fighter_a}'s ({len(points_a)} pts, latest at {latest_ts_a:.0f}), "
                      f"deriving {fighter_a} as its complement")

    fight["moneyline_chart"] = build_dual_line_chart_svg(
        points_a, points_b, fighter_a, fighter_b, implied_a=implied_a, implied_b=implied_b
    )
    fight["moneyline_chart_has_implied"] = implied_a or implied_b

    # HOW FAR THE LINE ACTUALLY TRAVELLED, in probability points, and whether
    # it ever changed its mind about who wins. Recorded here because this is
    # the one place the plotted series exists in full.
    # Measured on side A only: the two lines are exact complements by the
    # time they reach here, so B's swing is A's swing and B's crossing is A's.
    fight["moneyline_swing_pp"] = None
    fight["moneyline_net_pp"] = None
    fight["moneyline_span_days"] = None
    fight["moneyline_max_step_pp"] = None
    fight["moneyline_flipped"] = False
    if fight["moneyline_chart"] and len(points_a) >= 2:
        _pts = sorted(points_a, key=lambda q: q[0])
        _probs = [p for _, p in _pts]
        _lo, _hi = min(_probs), max(_probs)
        fight["moneyline_swing_pp"] = round((_hi - _lo) * 100, 1)
        # NET is the story the chart tells; SWING is only its envelope. A
        # price that wandered 40 points and came back has a huge swing and
        # says nothing.
        fight["moneyline_net_pp"] = round((_probs[-1] - _probs[0]) * 100, 1)
        fight["moneyline_span_days"] = round((_pts[-1][0] - _pts[0][0]) / 86400.0, 2)
        # The largest single step between consecutive quotes. On a liquid
        # market this stays small; a double-digit jump is a thin book or a
        # gap in the history, and a chart built from those reads as erratic
        # rather than as a market moving.
        _steps = [abs(_probs[i + 1] - _probs[i]) * 100 for i in range(len(_probs) - 1)]
        fight["moneyline_max_step_pp"] = round(max(_steps), 1)
        # HOW MANY jumps, not how big the biggest one was. A single 16-point
        # step is a market repricing a fight -- interesting, and the reason
        # the chart is worth showing. Four of them is a feed with holes in it.
        # Judging on the max alone cannot tell those apart, and it scored
        # Tsuruya/Borjas (one jump, mean step 0.27, a clean 34-point climb)
        # exactly as harshly as two charts that were visibly noise.
        fight["moneyline_jumps"] = sum(1 for x in _steps if x >= 8.0)
        # HOW MUCH OF THE TRAVEL ARRIVED IN JUMPS. This is the number that
        # actually matters: if a 28-point move happens as one 30-point cliff,
        # the cliff IS the move and there is nothing to watch it do. Net
        # minus this is the part of the journey that happened gradually,
        # which is the only part a "watch the market move" card can show.
        fight["moneyline_jump_pp"] = round(sum(x for x in _steps if x >= 8.0), 1)
        # A REAL CHANGE OF MIND, not a graze. Testing _lo < 0.5 < _hi called
        # Xiaonan/Gomes a flip on a low of 0.4987 -- four thousandths under
        # even money, invisible on the chart and gone at the next quote. The
        # band asks the price to have actually traded on both sides.
        fight["moneyline_flipped"] = _lo < 0.47 and _hi > 0.53
    if points_a and points_b:
        final_a = sorted(points_a, key=lambda p: p[0])[-1][1]
        final_b = sorted(points_b, key=lambda p: p[0])[-1][1]
        _say(f"[charts] {fighter_a} vs {fighter_b}: final displayed values -- "
              f"{fighter_a}={final_a*100:.1f}% {fighter_b}={final_b*100:.1f}% (sum={round((final_a+final_b)*100)}%)")


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

    _finish_moneyline_chart(fight, points_a, points_b)

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

        if len(points) < 2:
            continue

        market = str(edge.get("market") or "")
        who = str(edge.get("fighter") or "")

        # COMPLEMENTS DROPPED. Polymarket lists both sides of every method
        # market, so "Not KO/TKO" arrived alongside "KO/TKO" -- the same
        # information inverted, doubling the list to say nothing new.
        blob = f"{market} {edge.get('selection', '')}".lower()
        if "not " in blob or "endsinfinish" in blob.replace(" ", ""):
            continue

        # COLOUR BY SUBJECT. Every chart drew in the same colour, so nothing
        # said whose line was moving. A fighter's own prop takes his corner;
        # anything about the FIGHT (rounds, method, distance) takes gold,
        # which is the site's colour for a model/market view of the bout
        # rather than of a man.
        is_fight_level = ("vs" in who.lower()) or market.lower().startswith(
            ("fight ", "total rounds", "round betting"))
        if is_fight_level:
            colour = "#d4af37"
        elif _normalize_name(who) == _normalize_name(fighter_a):
            colour = LINE_COLOR_A
        elif _normalize_name(who) == _normalize_name(fighter_b):
            colour = LINE_COLOR_B
        else:
            colour = "#d4af37"

        # THE MATCHUP NAME IS DROPPED from fight-level labels: you are already
        # inside that fight's card, so repeating both names costs a line of
        # width to restate what the surrounding context says.
        label = market
        if not is_fight_level and who:
            label = f"{who} — {market}"
        label = _clean_movement_label(label, fighter_a, fighter_b)

        svg = build_dual_line_chart_svg(points, [], edge["fighter"], "",
                                        width=260, height=90,
                                        line_color=colour, show_dot=False,
                                        show_legend=False)
        if svg:
            other_charts.append({"label": label, "svg": svg})
    fight["other_charts"] = other_charts
