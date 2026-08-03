"""
Donut ring showing landed/attempted plus percentage in the center -- used
for the post-fight Significant Strikes comparison. Deliberately a plain
function (not a class) matching sparkline_chart.py / calibration_chart.py's
style: one pure function in, one SVG string out, easy to unit test.
"""

import math


def build_donut_svg(landed: int, attempted: int, color: str, size: int = 108, stroke_width: int = 11,
                     animate: bool = False, delay: float = 0.0) -> str:
    """
    animate=True adds a radial fill-in reveal (the ring sweeps from empty
    to its real percentage) via CSS custom properties + a shared keyframe
    (see templates/site.html's .donut-fill-ring rule) -- the standard,
    off-by-one-safe technique: dasharray is symmetric (circumference,
    circumference), and dashoffset animates from "fully hidden" to
    "exactly the real percentage visible", rather than fiddling with an
    asymmetric dash/gap pair. `delay` (seconds) staggers multiple donuts
    so they don't all sweep in unison. Reduced-motion users get the final
    state immediately (see the site's existing prefers-reduced-motion
    handling). The center percentage carries the site's existing generic
    .countup class (see templates/site.html) rather than a parallel
    count-up implementation -- it already knows how to animate any
    rendered number by parsing the text itself.
    """
    if attempted <= 0:
        pct = 0.0
    else:
        pct = max(0.0, min(1.0, landed / attempted))

    r = (size - stroke_width) / 2
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    dash = circumference * pct
    end_offset = circumference - dash

    ring_class = "donut-svg"
    progress_class = "donut-fill-ring" if animate else ""
    progress_style = (
        f'style="--donut-circ:{circumference:.1f}px; --donut-end:{end_offset:.1f}px; '
        f'animation-delay:{delay:.2f}s;"' if animate else
        f'stroke-dasharray="{dash:.1f} {circumference:.1f}"'
    )
    pct_class = "donut-center-pct countup" if animate else "donut-center-pct"

    return f"""<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" class="{ring_class}">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#383838" stroke-width="{stroke_width}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke_width}"
    class="{progress_class}" {progress_style} stroke-linecap="round"
    transform="rotate(-90 {cx} {cy})"/>
  <text x="{cx}" y="{cy - 6}" text-anchor="middle" class="donut-center-value">{landed}/{attempted}</text>
  <text x="{cx}" y="{cy + 14}" text-anchor="middle" class="{pct_class}">{round(pct*100)}%</text>
</svg>"""
