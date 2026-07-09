"""
Edge-vs-Confidence scatter for the Standout Props section: plots every
standout prop as a dot, model probability on one axis and edge size on the
other, so the best plays (high confidence AND high edge) visually separate
from the "big edge but the model itself isn't that sure" ones -- a
distinction the props list alone doesn't make visually obvious.
"""

import math


def build_scatter_svg(props: list[dict], width: int = 300, height: int = 180) -> str:
    if not props:
        return ""

    pad_left, pad_bottom, pad_top, pad_right = 34, 24, 14, 14
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    max_edge = max(abs(p["edge_pct"]) for p in props)
    y_max = max(20.0, math.ceil(max_edge / 5) * 5)  # round up to a clean gridline, floor of 20%

    def x_at(prob: float) -> float:
        return pad_left + (prob) * plot_w  # prob is 0-1

    def y_at(edge: float) -> float:
        return pad_top + plot_h - (min(abs(edge), y_max) / y_max) * plot_h

    # Gridlines: probability at 25/50/75%, edge at quarter marks
    grid_svg = ""
    for p in (0.25, 0.5, 0.75):
        x = x_at(p)
        grid_svg += f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{pad_top+plot_h}" stroke="#1c2028" stroke-width="1"/>'
        grid_svg += f'<text x="{x:.1f}" y="{height-8}" font-size="8" fill="#5a5f6a" text-anchor="middle">{round(p*100)}%</text>'
    for frac in (0.5, 1.0):
        y = y_at(y_max * frac)
        grid_svg += f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left+plot_w}" y2="{y:.1f}" stroke="#1c2028" stroke-width="1"/>'
        grid_svg += f'<text x="{pad_left-5}" y="{y+3:.1f}" font-size="8" fill="#5a5f6a" text-anchor="end">{round(y_max*frac)}%</text>'

    # Quadrant highlight: top-right (high prob, high edge) is the sweet spot
    quad_x = x_at(0.5)
    quad_svg = f'<rect x="{quad_x:.1f}" y="{pad_top}" width="{pad_left+plot_w-quad_x:.1f}" height="{plot_h/2:.1f}" fill="#3ddc84" opacity="0.05"/>'

    dots_svg = ""
    for p in props:
        cx = x_at(p["model_prob"])
        cy = y_at(p["edge_pct"])
        heat = "3" if abs(p["edge_pct"]) >= 20 else ("2" if abs(p["edge_pct"]) >= 10 else "1")
        radius = {"1": 3.5, "2": 4.5, "3": 5.5}[heat]
        color = {"1": "#8a8f9a", "2": "#e8c766", "3": "#3ddc84"}[heat]
        label = f"{p['fighter']}: {round(p['model_prob']*100)}% model, {p['edge_pct']:+.1f}% edge"
        dots_svg += (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{color}" fill-opacity="0.85" '
            f'stroke="#0a0c10" stroke-width="1"><title>{label}</title></circle>'
        )

    axis_labels = (
        f'<text x="{pad_left+plot_w/2:.1f}" y="{height-1}" font-size="8" fill="#5a5f6a" text-anchor="middle">Model Confidence</text>'
        f'<text x="8" y="{pad_top+plot_h/2:.1f}" font-size="8" fill="#5a5f6a" text-anchor="middle" '
        f'transform="rotate(-90 8 {pad_top+plot_h/2:.1f})">Edge Size</text>'
    )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="scatter-chart" role="img" '
        f'aria-label="Edge versus model confidence scatter for standout props">'
        + quad_svg + grid_svg + dots_svg + axis_labels +
        '</svg>'
    )
