"""
Radar/spider chart for the Tale of the Tape: overlays both fighters' core
model metrics on one chart so the stylistic matchup reads at a glance.

Five axes, all fully populated for the whole roster:
  - Striking Accuracy    (strike_accuracy_pct)
  - Grappling Offense    (control_time_pct if populated, else td_accuracy_pct)
  - Grappling Defense    (td_defense_pct)
  - Finishing Ability     (career finish rate: KO+Sub wins / total wins)
  - Experience/Durability (blend of career fight count and how rarely they've been finished)

Striking Defense and Striking Volume (SLpM/SApM) were both considered and
left out deliberately -- neither is real data this roster has. Striking
Defense specifically would need opponent-strikes-landed-against data no
source here provides; faking that axis would be worse than leaving it out.
"""

import math

AXIS_LABELS = ["Str. Acc.", "Gr. Off.", "Gr. Def.", "Finish", "Exp/Dur"]


def _experience_durability_score(row: dict) -> float:
    total_fights = (row.get("wins") or 0) + (row.get("losses") or 0)
    experience_score = min(100.0, total_fights * 4.0)  # ~25 fights = veteran-level 100

    losses = row.get("losses") or 0
    if losses > 0:
        finish_loss_rate = ((row.get("ko_losses") or 0) + (row.get("sub_losses") or 0)) / losses
        durability_score = (1 - finish_loss_rate) * 100
    else:
        durability_score = 100.0  # undefeated -- no data on how they take a loss, don't penalize

    return round((experience_score + durability_score) / 2, 1)


def _finish_rate_score(row: dict) -> float:
    wins = row.get("wins") or 0
    if wins <= 0:
        return 0.0
    finishes = (row.get("ko_wins") or 0) + (row.get("sub_wins") or 0)
    return round(finishes / wins * 100, 1)


def compute_radar_metrics(row: dict) -> list[float]:
    """Returns [striking_acc, grappling_off, grappling_def, finishing, exp_durability], each 0-100."""
    striking_acc = float(row.get("strike_accuracy_pct") or 0)

    control_time = row.get("control_time_pct")
    grappling_off = float(control_time) if control_time not in (None, "") else float(row.get("td_accuracy_pct") or 0)

    grappling_def = float(row.get("td_defense_pct") or 0)
    finishing = _finish_rate_score(row)
    exp_durability = _experience_durability_score(row)

    return [striking_acc, grappling_off, grappling_def, finishing, exp_durability]


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
        f'<polygon points="{poly_b}" fill="{color_b}" fill-opacity="0.18" stroke="{color_b}" stroke-width="2"/>'
        f'<polygon points="{poly_a}" fill="{color_a}" fill-opacity="0.22" stroke="{color_a}" stroke-width="2"/>'
        + labels_svg +
        '</svg>'
    )
