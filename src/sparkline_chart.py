"""
A minimal sparkline for the accuracy-over-time trend -- deliberately no
axis labels or gridlines (that's the point of a sparkline: a glanceable
shape, not a chart to study). The single aggregate accuracy% already
shown elsewhere answers "how accurate is the model right now"; this
answers a different question -- "is that number trending up, down, or
flat" -- which a single point-in-time figure can never show on its own.
"""


def build_sparkline_svg(values: list[float], width: int = 280, height: int = 56) -> str:
    if not values or len(values) < 2:
        return ""

    pad = 6
    plot_w = width - pad * 2
    plot_h = height - pad * 2

    lo, hi = min(values), max(values)
    # Flat series (identical accuracy every snapshot so far) would divide
    # by zero mapping to a y-range -- fall back to a fixed small band
    # centered on the value so it renders as a flat line, not a crash.
    span = (hi - lo) or 1.0

    def x_at(i: int) -> float:
        return pad + (i / (len(values) - 1)) * plot_w

    def y_at(v: float) -> float:
        return pad + plot_h - ((v - lo) / span) * plot_h

    points = [(x_at(i), y_at(v)) for i, v in enumerate(values)]
    poly_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    last_x, last_y = points[-1]

    fill_path = (
        f"M{points[0][0]:.1f},{height - pad} "
        + " ".join(f"L{x:.1f},{y:.1f}" for x, y in points)
        + f" L{last_x:.1f},{height - pad} Z"
    )

    trend_color = "#3ddc84" if values[-1] >= values[0] else "#ff5c5c"

    return f"""<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none" class="sparkline-svg">
  <defs>
    <linearGradient id="sparkline-fill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{trend_color}" stop-opacity="0.28"/>
      <stop offset="100%" stop-color="{trend_color}" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <path d="{fill_path}" fill="url(#sparkline-fill)" stroke="none"/>
  <polyline points="{poly_points}" fill="none" stroke="{trend_color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
  <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3" fill="{trend_color}"/>
</svg>"""
