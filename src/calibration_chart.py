"""
Calibration curve: plots the model's predicted probability against the
actual observed accuracy for picks in that probability range, against a
diagonal "perfect calibration" reference line. This is a genuinely
different, more rigorous check than a single accuracy number -- a model
that's right 70% of the time overall could still be badly overconfident on
its "90% sure" picks and underconfident on its "55% sure" ones, and a
single accuracy figure would never reveal that.
"""


def build_calibration_svg(points: list[dict], width: int = 280, height: int = 200) -> str:
    if not points:
        return ""

    pad_left, pad_bottom, pad_top, pad_right = 32, 24, 14, 10
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Fixed 50-100% range on both axes -- predictions are always >=50% by
    # definition (picking a "favorite"), so this is the meaningful range.
    def x_at(prob: float) -> float:
        return pad_left + ((prob - 0.5) / 0.5) * plot_w

    def y_at(prob: float) -> float:
        return pad_top + plot_h - ((prob - 0.5) / 0.5) * plot_h

    grid_svg = ""
    for pct in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        x, y = x_at(pct), y_at(pct)
        grid_svg += f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" y2="{pad_top+plot_h}" stroke="#1c2028" stroke-width="1"/>'
        grid_svg += f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left+plot_w}" y2="{y:.1f}" stroke="#1c2028" stroke-width="1"/>'
        grid_svg += f'<text x="{x:.1f}" y="{height-8}" font-size="7.5" fill="#5a5f6a" text-anchor="middle">{round(pct*100)}</text>'
        grid_svg += f'<text x="{pad_left-5}" y="{y+3:.1f}" font-size="7.5" fill="#5a5f6a" text-anchor="end">{round(pct*100)}</text>'

    # Perfect-calibration diagonal reference line
    diag_svg = (
        f'<line x1="{x_at(0.5):.1f}" y1="{y_at(0.5):.1f}" x2="{x_at(1.0):.1f}" y2="{y_at(1.0):.1f}" '
        f'stroke="#5a5f6a" stroke-width="1.5" stroke-dasharray="3,3"/>'
    )

    dots_svg = ""
    for p in points:
        cx, cy = x_at(p["predicted"]), y_at(p["actual"])
        radius = min(4 + p["n"] * 0.8, 9)  # bigger dot = more picks in that bucket
        # Direction matters, not just magnitude: actual BELOW predicted
        # means the model claimed more confidence than its picks earned
        # (genuinely overconfident, worth flagging). Actual ABOVE
        # predicted means the picks won more than the model even claimed
        # -- underconfidence, and a good problem to have, not the same
        # thing as overconfidence even though a naive abs() diff would
        # color them identically.
        diff = p["predicted"] - p["actual"]
        if abs(diff) < 0.10:
            color = "#3ddc84"  # well-calibrated
        elif diff >= 0.20:
            color = "#ff5c5c"  # overconfident: predicted well above what it earned
        elif diff <= -0.20:
            color = "#5fb8c9"  # underconfident: earned more than it claimed -- high value zone
        else:
            color = "#e8c766"  # mild drift, either direction
        dots_svg += (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{color}" fill-opacity="0.8" stroke="#0a0c10" stroke-width="1">'
            f'<title>Predicted {round(p["predicted"]*100)}% / Actual {round(p["actual"]*100)}% (n={p["n"]})</title></circle>'
        )

    axis_labels = (
        f'<text x="{pad_left+plot_w/2:.1f}" y="{height-1}" font-size="7.5" fill="#5a5f6a" text-anchor="middle">Predicted %</text>'
        f'<text x="8" y="{pad_top+plot_h/2:.1f}" font-size="7.5" fill="#5a5f6a" text-anchor="middle" '
        f'transform="rotate(-90 8 {pad_top+plot_h/2:.1f})">Actual %</text>'
    )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="calibration-chart" role="img" '
        f'aria-label="Model calibration: predicted probability versus actual accuracy">'
        + grid_svg + diag_svg + dots_svg + axis_labels +
        '</svg>'
    )
