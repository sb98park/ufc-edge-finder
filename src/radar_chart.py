"""
Radar/spider chart for the Tale of the Tape: overlays both fighters' method
profiles on one chart so the SHAPE of the likely fight reads at a glance.

Six axes, every one computed from the fight RECORD:
  - KO Threat              (ko_wins / wins)
  - Submission Threat      (sub_wins / wins)
  - Distance Rate          ((dec_wins + dec_losses) / total fights)
  - KO Resistance          (1 - ko_losses / losses, shrunk toward the mean)
  - Submission Resistance  (1 - sub_losses / losses, shrunk toward the mean)
  - Experience             (career fight count on a veteran curve)

WHY THESE AND NOT THE PREVIOUS SET. The chart used to plot Striking Accuracy,
Grappling Offense and Grappling Defense, sourced from strike_accuracy_pct /
td_accuracy_pct / control_time_pct / td_defense_pct. A coverage audit
(scripts/audit_radar_coverage.py) found those columns present for 0 of 25
fighters on a live card and roughly 29% of the roster -- and NOT because of
debutants: established names sat at zero too, because nothing in the automated
pipeline writes them (src/scraper.py does, and it is explicitly manual).
Half the chart was therefore drawn at zero for both fighters in every bout,
which conveys nothing while looking like it conveys something.

Record-derived axes have near-total coverage by construction: anyone with a
Wikipedia page has a W-L record and its method splits. Verified on the same
card at 24-25 of 25 for every axis here.

IT ALSO STOPS DUPLICATING THE WATERFALL ABOVE IT. "Why the model likes X"
already decomposes rating gap, striking, wrestling, sub threat, recent form,
height and durability -- i.e. WHO WINS. These axes answer a different
question, HOW THE FIGHT GOES, which is what method, round and total props
actually price.

MISSING DATA IS RETURNED AS None, NEVER ZERO. The previous version coerced
absent inputs with `or 0`, so a fighter nobody has data on was drawn at the
origin -- indistinguishable from, and read as, the worst fighter on the card.
A debutant scored 0 for striking AND 100 for durability simultaneously, both
purely from absence. None renders as a break in the polygon and a greyed
axis label instead.
"""

import math

AXIS_LABELS = ["KO Threat", "Submission Threat", "Distance Rate",
               "KO Resistance", "Submission Resistance", "Experience"]

# Shrinkage for the two resistance axes. Without it, one decision loss scores
# a perfect 100 chin and outranks a fighter with a real sample -- the single
# noisiest thing in the old chart. Equivalent to SHRINK_K notional prior
# fights at the league-ish finish-loss rate, so small samples are pulled
# toward the middle and only a real record moves the needle.
SHRINK_K = 3.0
PRIOR_FINISH_LOSS_RATE = 0.5


def _pct(numerator, denominator) -> float | None:
    if denominator in (None, "") or float(denominator) <= 0:
        return None
    return round(float(numerator) / float(denominator) * 100, 1)


def _num(row: dict, key: str):
    """Value, or None. Deliberately does NOT default to 0 -- see module docstring."""
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f          # NaN


def _experience_score(row: dict) -> float | None:
    w, l = _num(row, "wins"), _num(row, "losses")
    if w is None and l is None:
        return None
    total_fights = (w or 0) + (l or 0)
    return round(min(100.0, total_fights * 4.0), 1)  # ~25 fights = veteran-level 100


def _resistance_score(row: dict, loss_key: str) -> float | None:
    """
    How rarely they're finished THIS way, shrunk toward the prior by sample.

    Returns None rather than a number when the split isn't known -- a fighter
    with recorded losses but no method breakdown is unmeasured, not durable.
    """
    losses = _num(row, "losses")
    finished = _num(row, loss_key)
    if losses is None or finished is None:
        return None
    if losses <= 0:
        # Undefeated: genuinely no evidence either way. The prior IS the
        # honest answer here, not 100 -- an unbeaten fighter has not
        # demonstrated a chin, they have demonstrated not having been tested.
        return round((1 - PRIOR_FINISH_LOSS_RATE) * 100, 1)
    rate = (finished + SHRINK_K * PRIOR_FINISH_LOSS_RATE) / (losses + SHRINK_K)
    return round((1 - rate) * 100, 1)


def _distance_rate(row: dict) -> float | None:
    dw, dl = _num(row, "dec_wins"), _num(row, "dec_losses")
    w, l = _num(row, "wins"), _num(row, "losses")
    if dw is None or dl is None or w is None or l is None:
        return None
    total = w + l
    return _pct(dw + dl, total)


def compute_radar_metrics(row: dict) -> list[float | None]:
    """
    [ko_threat, sub_threat, distance_rate, ko_resistance, sub_resistance, experience].

    Each 0-100, or None where the underlying record doesn't support a value.
    Callers MUST handle None rather than coercing it -- that coercion is the
    bug this rewrite exists to remove.
    """
    wins = _num(row, "wins")
    ko_threat = _pct(_num(row, "ko_wins"), wins) if _num(row, "ko_wins") is not None else None
    sub_threat = _pct(_num(row, "sub_wins"), wins) if _num(row, "sub_wins") is not None else None

    return [
        ko_threat,
        sub_threat,
        _distance_rate(row),
        _resistance_score(row, "ko_losses"),
        _resistance_score(row, "sub_losses"),
        _experience_score(row),
    ]


def build_radar_chart_svg(
    metrics_a: list[float], metrics_b: list[float], name_a: str, name_b: str,
    size: int = 280,
) -> str:
    """Renders a 5-axis radar chart overlaying both fighters' metrics as translucent polygons."""
    n = len(AXIS_LABELS)
    cx = cy = size / 2
    max_r = size * 0.24
    label_r = size * 0.32

    def angle(i):
        return -math.pi / 2 + i * (2 * math.pi / n)

    def point(value, i):
        r = max_r * max(0.0, min(100.0, value)) / 100.0
        a = angle(i)
        return cx + r * math.cos(a), cy + r * math.sin(a)

    def polygon_points(metrics):
        # None vertices are SKIPPED, not plotted at zero. The polygon closes
        # across the gap, which reads as "this axis isn't measured for this
        # fighter" rather than "this fighter scores zero here". Plotting the
        # origin instead is precisely the bug this rewrite removes: it made
        # absence indistinguishable from the worst possible score.
        pts = [point(v, i) for i, v in enumerate(metrics) if v is not None]
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    # Gridlines at 25/50/75/100%
    grid_svg = ""
    for pct in (25, 50, 75, 100):
        pts = [point(pct, i) for i in range(n)]
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        grid_svg += f'<polygon points="{pts_str}" fill="none" stroke="#2e2e30" stroke-width="1"/>'

    # Spoke lines from center to each axis
    spokes_svg = ""
    for i in range(n):
        x, y = point(100, i)
        spokes_svg += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#2e2e30" stroke-width="1"/>'

    # Axis labels, positioned just outside the outer gridline
    labels_svg = ""
    for i, label in enumerate(AXIS_LABELS):
        a = angle(i)
        lx, ly = cx + label_r * math.cos(a), cy + label_r * math.sin(a)
        anchor = "middle"
        if math.cos(a) > 0.3:
            anchor = "start"
        elif math.cos(a) < -0.3:
            anchor = "end"
        # An axis neither fighter has data for is dimmed and marked, so the
        # gap in the polygons above is explained rather than looking like a
        # rendering fault. Half-known axes keep the normal colour -- the
        # break in one polygon already carries that.
        both_missing = metrics_a[i] is None and metrics_b[i] is None
        fill = "#4a4d54" if both_missing else "#8a8f9a"
        text = f"{label} \u2014" if both_missing else label
        labels_svg += f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="8.5" fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle">{text}</text>'

    poly_a = polygon_points(metrics_a)
    poly_b = polygon_points(metrics_b)

    # CORNER COLOURS, matching the waterfall and the movement charts.
    # Fighter A (listed first) is the red corner, B the blue.
    #
    # Gold/grey was worse than it looked: grey is the least separable colour
    # on a charcoal panel, so fighter B effectively had no identity, and gold
    # meant "model" everywhere else on the site.
    #
    # Overlap is fine here because the fills are 0.18/0.22 with solid 2px
    # strokes -- at that opacity two strongly separated hues read as a mixed
    # region rather than mud, and the strokes keep both outlines legible
    # wherever they cross. If the fills were near-opaque this swap would have
    # made overlap worse whatever colours were chosen.
    color_a, color_b = "#e53935", "#3b82f6"

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" class="radar-chart" role="img" '
        f'style="overflow: visible;" '
        f'aria-label="Style matchup radar comparing {name_a} and {name_b}">'
        + grid_svg + spokes_svg +
        f'<polygon points="{poly_b}" fill="{color_b}" fill-opacity="0.18" stroke="{color_b}" stroke-width="2" '
        f'class="radar-polygon" style="transform-origin: {cx}px {cy}px;"/>'
        f'<polygon points="{poly_a}" fill="{color_a}" fill-opacity="0.22" stroke="{color_a}" stroke-width="2" '
        f'class="radar-polygon" style="transform-origin: {cx}px {cy}px; transition-delay: 0.12s;"/>'
        + labels_svg +
        '</svg>'
    )
