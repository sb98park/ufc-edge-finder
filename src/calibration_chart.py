"""
Calibration, redesigned as grouped bars instead of a scatter-plot-against-
a-diagonal-line. The original scatter chart was mathematically correct but
genuinely hard to read without some statistics background -- "is this dot
above or below an imaginary diagonal" asks the viewer to do spatial
reasoning most people don't do casually. A bar chart doesn't: "here's what
we claimed, here's what actually happened, compare the two bar lengths" is
about as close to universally readable as a chart gets, and it's the exact
same underlying data (predicted probability per confidence bucket vs. the
actual win rate in that bucket) with none of the interpretation lost --
if anything, seeing the gap as a literal length difference between two
bars makes the size of any over/underconfidence MORE viscerally obvious
than a dot's distance from a line.
"""


def build_calibration_svg(points: list[dict], width: int = 300) -> str:
    if not points:
        return ""

    # Highest confidence first -- the most headline-relevant row at the
    # top, matching the "most important first" convention used elsewhere
    # on this site (billing order, etc.), not just the order buckets
    # happen to be computed in.
    rows = sorted(points, key=lambda p: p["predicted"], reverse=True)

    pad_left, pad_right, pad_top = 74, 34, 6
    row_h = 40
    bar_h = 9
    bar_gap = 3
    label_h = 14
    height = pad_top + label_h + len(rows) * row_h + 8

    plot_left, plot_right = pad_left, width - pad_right
    plot_w = plot_right - plot_left

    def x_at(prob: float) -> float:
        return plot_left + prob * plot_w

    grid_svg = ""
    for pct in (0, 25, 50, 75, 100):
        x = x_at(pct / 100)
        grid_svg += f'<line x1="{x:.1f}" y1="{pad_top+label_h}" x2="{x:.1f}" y2="{height-6}" stroke="#1c2028" stroke-width="1"/>'
    grid_svg += (
        f'<text x="{plot_left}" y="{pad_top+10}" font-size="8" fill="#5a5f6a" text-anchor="start">0%</text>'
        f'<text x="{x_at(0.5):.1f}" y="{pad_top+10}" font-size="8" fill="#5a5f6a" text-anchor="middle">50%</text>'
        f'<text x="{plot_right}" y="{pad_top+10}" font-size="8" fill="#5a5f6a" text-anchor="end">100%</text>'
    )

    rows_svg = ""
    for i, p in enumerate(rows):
        row_top = pad_top + label_h + i * row_h
        predicted_bar_y = row_top + 8
        actual_bar_y = predicted_bar_y + bar_h + bar_gap
        row_mid = (predicted_bar_y + actual_bar_y + bar_h) / 2

        diff = p["predicted"] - p["actual"]
        if abs(diff) < 0.10:
            actual_color = "#3ddc84"
            verdict = "on target"
        elif diff >= 0.20:
            actual_color = "#ff5c5c"
            verdict = "overconfident"
        elif diff <= -0.20:
            actual_color = "#5fb8c9"
            verdict = "underconfident"
        else:
            actual_color = "#e8c766"
            verdict = "mild drift"

        bucket_lo = int(round((p["predicted"] - 0.05) * 100 / 10) * 10)
        bucket_label = f'~{round(p["predicted"]*100)}%'

        rows_svg += f'<text x="{pad_left-8}" y="{row_mid+3:.1f}" font-size="9.5" font-weight="700" fill="#e8e8ec" text-anchor="end">{bucket_label}</text>'
        rows_svg += f'<text x="{pad_left-8}" y="{row_mid+15:.1f}" font-size="7.5" fill="#5a5f6a" text-anchor="end">n={p["n"]}</text>'

        # Predicted bar (gold, matching this site's "model" color language)
        rows_svg += (
            f'<rect x="{plot_left}" y="{predicted_bar_y:.1f}" width="{(x_at(p["predicted"])-plot_left):.1f}" '
            f'height="{bar_h}" rx="2" fill="#d4af37" fill-opacity="0.85"/>'
        )
        rows_svg += f'<text x="{x_at(p["predicted"])+4:.1f}" y="{predicted_bar_y+bar_h-1.5:.1f}" font-size="8" fill="#d4af37" font-weight="700">{round(p["predicted"]*100)}%</text>'

        # Actual bar (colored by calibration quality)
        rows_svg += (
            f'<rect x="{plot_left}" y="{actual_bar_y:.1f}" width="{(x_at(p["actual"])-plot_left):.1f}" '
            f'height="{bar_h}" rx="2" fill="{actual_color}" fill-opacity="0.85"/>'
        )
        rows_svg += f'<text x="{x_at(p["actual"])+4:.1f}" y="{actual_bar_y+bar_h-1.5:.1f}" font-size="8" fill="{actual_color}" font-weight="700">{round(p["actual"]*100)}%</text>'

    legend_y = height + 2
    legend_svg = (
        f'<rect x="{plot_left}" y="{legend_y-7}" width="10" height="7" rx="1.5" fill="#d4af37" fill-opacity="0.85"/>'
        f'<text x="{plot_left+14}" y="{legend_y-1}" font-size="8" fill="#8a8f9a">We said</text>'
        f'<rect x="{plot_left+58}" y="{legend_y-7}" width="10" height="7" rx="1.5" fill="#3ddc84" fill-opacity="0.85"/>'
        f'<text x="{plot_left+72}" y="{legend_y-1}" font-size="8" fill="#8a8f9a">What happened</text>'
    )
    height += 14

    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="calibration-chart" role="img"
  aria-label="Predicted confidence versus actual win rate, per confidence bucket">
  {grid_svg}
  {rows_svg}
  {legend_svg}
</svg>"""
