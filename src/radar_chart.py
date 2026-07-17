"""
Radar/spider chart for the Tale of the Tape: overlays both fighters' core
model metrics on one chart so the stylistic matchup reads at a glance.

Six axes, all fully populated for the whole roster:
  - Striking Accuracy    (strike_accuracy_pct)
  - Grappling Offense    (control_time_pct if populated, else td_accuracy_pct)
  - Grappling Defense    (td_defense_pct)
  - Finishing Ability    (career finish rate: KO+Sub wins / total wins)
  - Experience           (career fight count, scaled to a 0-100 veteran curve)
  - Durability           (how rarely they've been finished, i.e. inverse finish-loss rate)

Experience and Durability were originally a single averaged "Exp/Dur" axis;
split into two separate axes on request, since the underlying calculation
was already computing them independently before averaging them together --
no new data or logic was needed, just exposing both instead of blending them.

Axis labels are spelled out in full rather than abbreviated (e.g.
"Grappling Offense" instead of "Gr. Off.") -- the chart is rendered at a
fixed size on every single fight (see model_preview.py's one call site,
which never overrides the default `size`), so there's no real space
constraint forcing abbreviation, and the shorthand risked being unclear to
users without an MMA background.

Striking Defense and Striking Volume (SLpM/SApM) were both considered and
left out deliberately -- neither is real data this roster has. Striking
Defense specifically would need opponent-strikes-landed-against data no
source here provides; faking that axis would be worse than leaving it out.
"""

import math

AXIS_LABELS = ["Striking Accuracy", "Grappling Offense", "Grappling Defense", "Finishing Ability", "Experience", "Durability"]


def _experience_score(row: dict) -> float:
    total_fights = (row.get("wins") or 0) + (row.get("losses") or 0)
    return round(min(100.0, total_fights * 4.0), 1)  # ~25 fights = veteran-level 100


def _durability_score(row: dict) -> float:
    losses = row.get("losses") or 0
    if losses <= 0:
        return 100.0  # undefeated -- no data on how they take a loss, don't penalize
    finish_loss_rate = ((row.get("ko_losses") or 0) + (row.get("sub_losses") or 0)) / losses
    return round((1 - finish_loss_rate) * 100, 1)


def _finish_rate_score(row: dict) -> float:
    wins = row.get("wins") or 0
    if wins <= 0:
        return 0.0
    finishes = (row.get("ko_wins") or 0) + (row.get("sub_wins") or 0)
    return round(finishes / wins * 100, 1)


def compute_radar_metrics(row: dict) -> list[float]:
    """Returns [striking_acc, grappling_off, grappling_def, finishing, experience, durability], each 0-100."""
    striking_acc = float(row.get("strike_accuracy_pct") or 0)

    control_time = row.get("control_time_pct")
    grappling_off = float(control_time) if control_time not in (None, "") else float(row.get("td_accuracy_pct") or 0)

    grappling_def = float(row.get("td_defense_pct") or 0)
    finishing = _finish_rate_score(row)
    experience = _experience_score(row)
    durability = _durability_score(row)

    return [striking_acc, grappling_off, grappling_def, finishing, experience, durability]


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
        pts = [point(v, i) for i, v in enumerate(metrics)]
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    # Gridlines at 25/50/75/100%
    grid_svg = ""
    for pct in (25, 50, 75, 100):
        pts = [point(pct, i) for i in range(n)]
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        grid_svg += f'<polygon points="{pts_str}" fill="none" stroke="#262b36" stroke-width="1"/>'

    # Spoke lines from center to each axis
    spokes_svg = ""
    for i in range(n):
        x, y = point(100, i)
        spokes_svg += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#262b36" stroke-width="1"/>'

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
        labels_svg += f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="8.5" fill="#8a8f9a" text-anchor="{anchor}" dominant-baseline="middle">{label}</text>'

    poly_a = polygon_points(metrics_a)
    poly_b = polygon_points(metrics_b)

    color_a, color_b = "#d4af37", "#8a8f9a"

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
