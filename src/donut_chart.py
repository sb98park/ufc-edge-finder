"""
Donut ring showing landed/attempted plus percentage in the center -- used
for the post-fight Significant Strikes comparison. Deliberately a plain
function (not a class) matching sparkline_chart.py / calibration_chart.py's
style: one pure function in, one SVG string out, easy to unit test.
"""

import math


def build_donut_svg(landed: int, attempted: int, color: str, size: int = 108, stroke_width: int = 11) -> str:
    if attempted <= 0:
        pct = 0.0
    else:
        pct = max(0.0, min(1.0, landed / attempted))

    r = (size - stroke_width) / 2
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    dash = circumference * pct

    return f"""<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" class="donut-svg">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#262b36" stroke-width="{stroke_width}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke_width}"
    stroke-dasharray="{dash:.1f} {circumference:.1f}" stroke-linecap="round"
    transform="rotate(-90 {cx} {cy})"/>
  <text x="{cx}" y="{cy - 6}" text-anchor="middle" class="donut-center-value">{landed}/{attempted}</text>
  <text x="{cx}" y="{cy + 14}" text-anchor="middle" class="donut-center-pct">{round(pct*100)}%</text>
</svg>"""
