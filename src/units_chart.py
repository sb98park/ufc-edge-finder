"""
Full time-series chart for cumulative units over time -- richer than the
small glanceable sparkline elsewhere on the page (real axis labels, a
zero-reference line, gradient fill), for when someone actually wants to
read "where were we after fight N," not just see the shape of the trend.

Reuses the exact reveal-mask animation technique already proven reliable
on iOS in line_movement.py's moneyline chart: a covering rect that
shrinks away via CSS transform (scaleX), not a stroke-property animation.
That choice wasn't arbitrary -- stroke-dasharray/dashoffset animation
(both CSS-transitioned and JS-rAF-driven) was tested and found unreliable
specifically on iOS Safari across multiple rounds in this project, which
matches a known WebKit gap (transform gets real hardware compositing,
direct SVG stroke animation often doesn't). No reason to relitigate that
here -- just reuse the same mechanism, and it hooks into the *existing*
reveal observer automatically via the shared .chart-block wrapper class,
no new JS required.
"""


def build_units_timeseries_svg(running_total: list[float], width: int = 300, height: int = 180) -> str:
    """
    running_total should already include the 0 baseline as its first
    element (the model's starting point before any tracked results) --
    this function doesn't prepend it, since the caller knows whether
    that's already been done.
    """
    if not running_total or len(running_total) < 2:
        return ""

    pad_left, pad_bottom, pad_top, pad_right = 34, 20, 14, 12
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Range always includes 0, even if every result so far is positive
    # (or negative) -- the zero line is the whole point of reference here,
    # it shouldn't be able to drift off-chart.
    lo = min(0, min(running_total))
    hi = max(0, max(running_total))
    span = (hi - lo) or 1.0
    # A little headroom above/below so the line and endpoint dot aren't
    # pinned right against the plot edges.
    lo -= span * 0.12
    hi += span * 0.12
    span = hi - lo

    def x_at(i: int) -> float:
        return pad_left + (i / (len(running_total) - 1)) * plot_w

    def y_at(v: float) -> float:
        return pad_top + plot_h - ((v - lo) / span) * plot_h

    # Y-axis gridlines: 0 always included, plus the min/max rounded to a
    # clean step so the labels read as real numbers, not float noise.
    step = max(1, round(span / 4))
    grid_values = sorted(set([0] + [round(lo / step) * step + i * step for i in range(6)]))
    grid_values = [v for v in grid_values if lo - step <= v <= hi + step]

    grid_svg = ""
    for v in grid_values:
        y = y_at(v)
        is_zero = v == 0
        grid_svg += (
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left+plot_w}" y2="{y:.1f}" '
            f'stroke="{"#4a4f5a" if is_zero else "#1c2028"}" stroke-width="1" '
            f'{"stroke-dasharray=\"3,3\"" if is_zero else ""}/>'
        )
        grid_svg += f'<text x="{pad_left-6}" y="{y+3:.1f}" font-size="8" fill="#5a5f6a" text-anchor="end">{"+" if v > 0 else ""}{v:g}U</text>'

    x_labels_svg = (
        f'<text x="{x_at(0):.1f}" y="{height-4}" font-size="8" fill="#5a5f6a" text-anchor="start">Start</text>'
        f'<text x="{x_at(len(running_total)-1):.1f}" y="{height-4}" font-size="8" fill="#5a5f6a" text-anchor="end">Now</text>'
    )

    points = [(x_at(i), y_at(v)) for i, v in enumerate(running_total)]
    poly_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    last_x, last_y = points[-1]
    final_value = running_total[-1]
    trend_color = "#3ddc84" if final_value >= 0 else "#ff5c5c"

    fill_path = (
        f"M{points[0][0]:.1f},{y_at(0):.1f} "
        + " ".join(f"L{x:.1f},{y:.1f}" for x, y in points)
        + f" L{last_x:.1f},{y_at(0):.1f} Z"
    )

    mask_svg = (
        f'<rect x="{pad_left}" y="{pad_top}" width="{plot_w}" height="{plot_h}" fill="#1a1e28" '
        f'class="chart-reveal-mask" style="transform-box: fill-box; transform-origin: right center;"/>'
    )

    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="units-timeseries-chart" role="img"
  aria-label="Cumulative units over time, starting from a zero baseline">
  {grid_svg}
  {x_labels_svg}
  <defs>
    <linearGradient id="units-ts-fill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{trend_color}" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="{trend_color}" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <path d="{fill_path}" fill="url(#units-ts-fill)" stroke="none"/>
  <polyline points="{poly_points}" fill="none" stroke="{trend_color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="{trend_color}" class="chart-draw-endpoint"/>
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="none" stroke="{trend_color}" stroke-width="1.5" class="chart-endpoint-halo"/>
  {mask_svg}
</svg>"""
