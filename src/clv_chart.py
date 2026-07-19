"""
"Beat the Closing Line" as a dumbbell/slope chart: one row per pick,
showing the market's implied win probability at two points in time --
when the model made the pick, and where the market ended up by fight
night -- connected by a line. The direction the line points IS the
whole concept: if it moves right (toward more confident in the model's
side), the market came around to agreeing with the model after the
pick was already made. That's the entire idea CLV is trying to capture,
and a two-dot-and-a-line visual shows it directly instead of requiring
someone to already understand "closing line value" as a term before a
single percentage number means anything to them.

Rows are capped (most recent first) rather than growing forever as more
picks accumulate -- this is meant to be a glanceable, impressive
visual, not an exhaustive record (the exhaustive one already exists in
the Every Tracked Pick / Past Events lists elsewhere on the page).
"""

MAX_ROWS = 8


def build_clv_dumbbell_svg(clv_eligible: list[dict], width: int = 300) -> str:
    """
    clv_eligible: matched-result dicts that have a non-None "clv" field
    (see _clv_result in track_record.py), most-recent-first. Each needs
    fighter_a, fighter_b, predicted_favorite, and clv.pick_prob /
    clv.closing_prob / clv.beat_clv.
    """
    rows = clv_eligible[:MAX_ROWS]
    if not rows:
        return ""

    pad_left, pad_right, pad_top = 8, 8, 6
    row_h = 34
    label_h = 16
    height = pad_top + label_h + len(rows) * row_h + 10

    plot_left, plot_right = pad_left, width - pad_right
    plot_w = plot_right - plot_left

    def x_at(prob: float) -> float:
        return plot_left + prob * plot_w

    # Gridlines at 0/25/50/75/100% -- faint, just enough to anchor the
    # eye to "this is a probability scale," not a precise reading tool.
    grid_svg = ""
    for pct in (0, 25, 50, 75, 100):
        x = x_at(pct / 100)
        grid_svg += f'<line x1="{x:.1f}" y1="{pad_top+label_h}" x2="{x:.1f}" y2="{height-6}" stroke="#1c2028" stroke-width="1"/>'
    grid_svg += (
        f'<text x="{plot_left}" y="{pad_top+11}" font-size="8" fill="#5a5f6a" text-anchor="start">0%</text>'
        f'<text x="{x_at(0.5):.1f}" y="{pad_top+11}" font-size="8" fill="#5a5f6a" text-anchor="middle">50%</text>'
        f'<text x="{plot_right}" y="{pad_top+11}" font-size="8" fill="#5a5f6a" text-anchor="end">100%</text>'
    )

    rows_svg = ""
    for i, m in enumerate(rows):
        y = pad_top + label_h + i * row_h + row_h / 2
        clv = m["clv"]
        pick_x = x_at(clv["pick_prob"])
        close_x = x_at(clv["closing_prob"])
        beat = clv["beat_clv"]
        line_color = "#3ddc84" if beat else "#5a5f6a"
        opponent = m["fighter_b"] if m["predicted_favorite"] == m["fighter_a"] else m["fighter_a"]

        rows_svg += f'<text x="{plot_left}" y="{y-11:.1f}" font-size="9.5" font-weight="700" fill="#e8e8ec">{m["predicted_favorite"]}</text>'
        # Connecting line between the two probability points
        rows_svg += f'<line x1="{pick_x:.1f}" y1="{y:.1f}" x2="{close_x:.1f}" y2="{y:.1f}" stroke="{line_color}" stroke-width="2.5" stroke-linecap="round"/>'
        # Small arrowhead at the closing end, in the direction of travel, so the
        # line reads as "moved from here to here" rather than just "a segment"
        direction = 1 if close_x >= pick_x else -1
        arrow_x = close_x - direction * 5
        rows_svg += (
            f'<polygon points="{close_x:.1f},{y:.1f} {arrow_x:.1f},{y-3.5:.1f} {arrow_x:.1f},{y+3.5:.1f}" fill="{line_color}"/>'
        )
        # Pick-time point: hollow ring (cyan, matching the site's "market" color language)
        rows_svg += f'<circle cx="{pick_x:.1f}" cy="{y:.1f}" r="4.5" fill="#0a0c10" stroke="#5fb8c9" stroke-width="2"/>'
        rows_svg += (
            f'<text x="{plot_left}" y="{y+15:.1f}" font-size="8" fill="#8a8f9a">vs. {opponent} · '
            f'{"beat the close" if beat else "line moved away"} '
            f'({"+" if clv["clv_pct"] >= 0 else ""}{clv["clv_pct"]}pp)</text>'
        )

    legend_svg = (
        f'<circle cx="{plot_left+4}" cy="{height-2}" r="3" fill="#0a0c10" stroke="#5fb8c9" stroke-width="1.5"/>'
        f'<text x="{plot_left+11}" y="{height+1}" font-size="8" fill="#8a8f9a">When picked</text>'
        f'<line x1="{plot_left+68}" y1="{height-2}" x2="{plot_left+82}" y2="{height-2}" stroke="#3ddc84" stroke-width="2.5"/>'
        f'<text x="{plot_left+86}" y="{height+1}" font-size="8" fill="#8a8f9a">→ At close</text>'
    )
    height += 12

    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="clv-dumbbell-chart" role="img"
  aria-label="Market probability when picked versus at closing, per pick">
  {grid_svg}
  {rows_svg}
  {legend_svg}
</svg>"""
