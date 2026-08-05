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

    X-axis is even index spacing (each pick gets equal width regardless
    of real elapsed time), labeled "Start"/"Now" -- a real-date-based
    version of this was tried and reverted per explicit user preference:
    it made the shape of the line/slope at each step less readable, which
    was specifically the point of this chart. Kept simple and index-based
    on purpose.
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
    # Clamp to the ACTUAL domain. This allowed values up to hi + step -- a
    # full gridline above the top of the plot -- so its label was drawn above
    # y=0 and the svg clipped it in half. Only the topmost label ever showed
    # the symptom, which is why it looked like a one-off rather than an
    # off-by-one in the range.
    grid_values = [v for v in grid_values if lo <= v <= hi]

    grid_svg = ""
    for v in grid_values:
        y = y_at(v)
        is_zero = v == 0
        line_color = "#4a4f5a" if is_zero else "#1c2028"
        dasharray_attr = ' stroke-dasharray="3,3"' if is_zero else ""
        grid_svg += (
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left+plot_w}" y2="{y:.1f}" '
            f'stroke="{line_color}" stroke-width="1"{dasharray_attr}/>'
        )
        sign = "+" if v > 0 else ""
        grid_svg += f'<text x="{pad_left-6}" y="{y+3:.1f}" font-size="8" fill="#5a5f6a" text-anchor="end">{sign}{v:g}U</text>'

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

    # FILL FROM THE VARIABLE, not a literal. This mask works by matching the
    # background exactly and shrinking, so the chart appears to draw itself in.
    # The moment the surrounding block's colour changed -- .units-block moved
    # from a gold-washed gradient to flat var(--panel) -- the hardcoded
    # #242426 stopped matching and the mask became a visible grey rectangle
    # sliding across the graph.
    # A CSS variable can't drift from the palette the way a literal can.
    # CLIP THE LINE, don't cover the chart -- same change as the movement
    # charts. The mask was an opaque rect painted over the whole plot area,
    # so the gridlines and axis labels animated in with the line and the
    # effect read as a box sliding away rather than a line being drawn.
    mask_svg = (
        f'<defs><clipPath id="units-reveal">'
        f'<rect x="{pad_left}" y="{pad_top - 6}" width="{plot_w + 12}" height="{plot_h + 12}" '
        f'class="chart-reveal-clip" style="transform-box: fill-box; transform-origin: left center;"/>'
        f'</clipPath></defs>'
    )

    # Expose the plotted coordinates so the client can map a finger or cursor
    # position to a point WITHOUT re-deriving the geometry. Re-deriving it in
    # JS would mean two copies of the same maths that could drift apart the
    # moment padding or sizing changed here.
    scrub_xs = ",".join(f"{x:.1f}" for x, _ in points)
    scrub_ys = ",".join(f"{y:.1f}" for _, y in points)

    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-scrub-xs="{scrub_xs}" data-scrub-ys="{scrub_ys}" class="units-timeseries-chart" role="img"
  aria-label="Cumulative units over time, starting from a zero baseline">
  {grid_svg}
  {x_labels_svg}
  <defs>
    <linearGradient id="units-ts-fill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{trend_color}" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="{trend_color}" stop-opacity="0"/>
    </linearGradient>
  </defs>
  {mask_svg}
  <g clip-path="url(#units-reveal)">
    <path d="{fill_path}" fill="url(#units-ts-fill)" stroke="none"/>
    <polyline points="{poly_points}" fill="none" stroke="{trend_color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
    <!-- INSIDE the clip. The dot was outside it, hidden only by an opacity
         delay -- which works right up until the block was revealed earlier
         than the reader scrolled to it, and then the dot simply sat there
         waiting for the line. Inside, it is uncovered by the clip edge
         reaching its x position, which IS the moment the line arrives.
         No timing, no class state, nothing to get out of sync. -->
    <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="{trend_color}"/>
  </g>
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="none" stroke="{trend_color}" stroke-width="1.5" class="chart-endpoint-halo" style="transform-box: fill-box; transform-origin: center;"/>
</svg>"""
