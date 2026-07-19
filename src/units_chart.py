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

import datetime as dt


def build_units_timeseries_svg(running_total: list[float], running_dates: list[str] | None = None,
                                width: int = 300, height: int = 180) -> str:
    """
    running_total should already include the 0 baseline as its first
    element (the model's starting point before any tracked results) --
    this function doesn't prepend it, since the caller knows whether
    that's already been done.

    running_dates, if provided, must be the same length as running_total
    (one date string per point, "YYYY-MM-DD") and is what makes the
    x-axis honest: points are positioned by REAL elapsed time between
    dates, not evenly by index. Without this, a multi-day idle gap
    between one event ending and the next starting (nothing moves, since
    there's nothing to grade in between) looked visually identical to
    the tight spacing between picks logged hours apart on the same
    card -- implying a steady, continuous climb that never actually
    happened. Falls back to the old even-index spacing if dates are
    missing, unparseable, or all the same day (can't derive a meaningful
    time span from a single day), so this never divides by zero or
    breaks the chart over a data quirk -- it just loses the extra
    honesty for that one render.
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

    parsed_dates = None
    if running_dates and len(running_dates) == len(running_total):
        candidates = []
        for d in running_dates:
            try:
                candidates.append(dt.datetime.strptime(str(d), "%Y-%m-%d"))
            except (ValueError, TypeError):
                candidates = None
                break
        if candidates and len({c.date() for c in candidates}) > 1:
            # Same-day points get a small artificial spread so they don't
            # render exactly on top of each other -- date_added only has
            # day granularity, but these are already in true chronological
            # order (same date-sort used everywhere else), so this just
            # preserves that real relative order visually. Capped at a
            # small fraction of a day specifically so it can never rival a
            # genuine multi-day gap between events.
            day_counts: dict = {}
            for c in candidates:
                day_counts[c.date()] = day_counts.get(c.date(), 0) + 1
            day_seen: dict = {}
            spread_candidates = []
            for c in candidates:
                day = c.date()
                idx = day_seen.get(day, 0)
                day_seen[day] = idx + 1
                total_that_day = day_counts[day]
                # Spread across at most 6 hours of the day, evenly, so
                # multiple same-day picks fan out left-to-right in order
                # without implying real clock times that aren't known.
                offset_hours = 0 if total_that_day <= 1 else (idx / (total_that_day - 1)) * 6
                spread_candidates.append(c + dt.timedelta(hours=offset_hours))
            parsed_dates = spread_candidates

    if parsed_dates:
        first_ts = parsed_dates[0].timestamp()
        last_ts = parsed_dates[-1].timestamp()
        time_span = (last_ts - first_ts) or 1.0

        def x_at(i: int) -> float:
            return pad_left + ((parsed_dates[i].timestamp() - first_ts) / time_span) * plot_w

        start_label = parsed_dates[0].strftime("%b %-d")
        end_label = parsed_dates[-1].strftime("%b %-d")
    else:
        def x_at(i: int) -> float:
            return pad_left + (i / (len(running_total) - 1)) * plot_w

        start_label, end_label = "Start", "Now"

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
        line_color = "#4a4f5a" if is_zero else "#1c2028"
        dasharray_attr = ' stroke-dasharray="3,3"' if is_zero else ""
        grid_svg += (
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left+plot_w}" y2="{y:.1f}" '
            f'stroke="{line_color}" stroke-width="1"{dasharray_attr}/>'
        )
        sign = "+" if v > 0 else ""
        grid_svg += f'<text x="{pad_left-6}" y="{y+3:.1f}" font-size="8" fill="#5a5f6a" text-anchor="end">{sign}{v:g}U</text>'

    x_labels_svg = (
        f'<text x="{x_at(0):.1f}" y="{height-4}" font-size="8" fill="#5a5f6a" text-anchor="start">{start_label}</text>'
        f'<text x="{x_at(len(running_total)-1):.1f}" y="{height-4}" font-size="8" fill="#5a5f6a" text-anchor="end">{end_label}</text>'
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
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="none" stroke="{trend_color}" stroke-width="1.5" class="chart-endpoint-halo" style="transform-box: fill-box; transform-origin: center;"/>
  {mask_svg}
</svg>"""
