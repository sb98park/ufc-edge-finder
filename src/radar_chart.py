"""
Radar/spider chart for the Tale of the Tape: overlays both fighters' profiles
so the SHAPE of the likely fight reads at a glance.

Six axes, three MEASURED from ESPN per-fight statistics and three derived from
the fight record:

  Knockdown Rate         knockdowns_per_fight            percentile
  Submission Threat      sub_wins / wins                 record
  Striking Pace          sig_strikes_att_per_fight       percentile
  Damage Resistance      sig_strikes_absorbed_per_fight  percentile, INVERTED
  Submission Resistance  1 - sub_losses/losses, shrunk   record
  Distance Rate          (dec_wins + dec_losses) / total record

WHY THESE. The previous set inferred power from ko_wins/wins and durability
from ko_losses/losses -- both hostage to matchmaking, both moving in huge
steps for a fighter with few bouts (a 7-1 record can only express KO threat in
14-point increments), and durability carrying no information at all for anyone
undefeated. ESPN's per-fight data measures the underlying things directly:
knockdowns COUNT the power event, and strikes absorbed measures damage taken
every fight rather than inferring it from the handful a fighter has lost.
Striking Pace replaces Experience, which could not distinguish a 25-fight
regional journeyman from a 25-fight UFC veteran and priced nothing; pace drives
totals and round props, where the softer lines are.

Submission Threat and Resistance stay record-derived because ESPN publishes no
submission-attempt data -- there is nothing better to switch to. Distance Rate
stays because it answers a question none of the measured stats do: does this
fighter's fight reach the judges.

"RESISTANCE", NOT "DEFENSE", deliberately. In MMA stats "defense" means the
share of ATTEMPTS AGAINST YOU that fail, which is what td_defense_pct genuinely
measures and what a reader will assume. These are outcome-based, not
attempt-based, so borrowing the word would imply a parity that does not exist.

PERCENTILES, NOT RAW VALUES, for the three measured axes. They are rates on
incompatible scales (knockdowns ~0-1.5 per fight, strikes attempted ~20-150),
so plotting them raw would make the chart meaningless. Percentile-ranking
against the roster also answers the question a bettor actually has -- is this
fighter dangerous RELATIVE to the division -- rather than against a cap someone
invented. The cost, accepted knowingly: a fighter's shape can shift as the
roster changes, without them fighting.

DAMAGE RESISTANCE IS INVERTED so that outward always means better on every
axis. One spoke reading backwards would make the overall shape actively
misleading, which is worse than omitting it.

MISSING DATA IS None, NEVER ZERO -- see the polygon and label handling below.
The measured axes are UFC-only, so a debutant renders a partial chart. That is
the honest cost of using measurements instead of inferences.
"""

import math

AXIS_LABELS = ["Knockdown Rate", "Submission Threat", "Striking Pace",
               "Damage Resistance", "Submission Resistance", "Distance Rate"]

# Columns that get percentile-ranked, and whether higher is better.
PERCENTILE_AXES = {
    "knockdowns_per_fight": True,
    "sig_strikes_att_per_fight": True,
    "sig_strikes_absorbed_per_fight": False,     # inverted: less damage taken is better
}

SHRINK_K = 3.0
PRIOR_FINISH_LOSS_RATE = 0.5

# A fighter with almost no tracked fights would rank on noise. Below this the
# measured axes stay None rather than plotting a percentile built from one bout.
MIN_ESPN_FIGHTS = 3


def build_percentile_index(fighters_df) -> dict:
    """
    {column: sorted list of values} for percentile ranking.

    Built once per site build from the whole roster and passed into
    compute_radar_metrics. Computing it per fighter would be both slow and
    wrong -- the ranking has to be against a fixed population, not against
    whoever happens to be on the card.
    """
    index = {}
    for col in PERCENTILE_AXES:
        if col not in getattr(fighters_df, "columns", []):
            continue
        vals = []
        for v in fighters_df[col]:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:                      # not NaN
                vals.append(f)
        if len(vals) >= 20:                 # too few to rank against meaningfully
            index[col] = sorted(vals)
    return index


def _percentile(value, sorted_vals, higher_is_better: bool):
    if value is None or not sorted_vals:
        return None
    lo, hi = 0, len(sorted_vals)
    while lo < hi:                          # bisect_left, stdlib-free
        mid = (lo + hi) // 2
        if sorted_vals[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    pct = lo / len(sorted_vals) * 100.0
    return round(pct if higher_is_better else 100.0 - pct, 1)


def _pct(numerator, denominator) -> float | None:
    if denominator in (None, "") or float(denominator) <= 0:
        return None
    return round(float(numerator) / float(denominator) * 100, 1)


def _num(row: dict, key: str):
    """Value, or None. Deliberately does NOT default to 0."""
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _resistance_score(row: dict, loss_key: str) -> float | None:
    """How rarely they're finished this way, shrunk toward the prior by sample."""
    losses = _num(row, "losses")
    finished = _num(row, loss_key)
    if losses is None or finished is None:
        return None
    if losses <= 0:
        # Undefeated is untested, not proven. The prior is the honest answer.
        return round((1 - PRIOR_FINISH_LOSS_RATE) * 100, 1)
    rate = (finished + SHRINK_K * PRIOR_FINISH_LOSS_RATE) / (losses + SHRINK_K)
    return round((1 - rate) * 100, 1)


def _distance_rate(row: dict) -> float | None:
    dw, dl = _num(row, "dec_wins"), _num(row, "dec_losses")
    w, l = _num(row, "wins"), _num(row, "losses")
    if dw is None or dl is None or w is None or l is None:
        return None
    return _pct(dw + dl, w + l)


def compute_radar_metrics(row: dict, pct_index: dict | None = None) -> list[float | None]:
    """
    Six axes, each 0-100 or None. Callers MUST handle None rather than coerce.

    Without pct_index the three measured axes return None -- ranking needs a
    population, and inventing one from a single fighter would be worse than
    admitting the axis can't be drawn.
    """
    pct_index = pct_index or {}
    wins = _num(row, "wins")
    espn_fights = _num(row, "espn_fights")
    enough = espn_fights is not None and espn_fights >= MIN_ESPN_FIGHTS

    def measured(col):
        if not enough:
            return None
        return _percentile(_num(row, col), pct_index.get(col), PERCENTILE_AXES[col])

    sub_wins = _num(row, "sub_wins")
    return [
        measured("knockdowns_per_fight"),
        _pct(sub_wins, wins) if sub_wins is not None else None,
        measured("sig_strikes_att_per_fight"),
        measured("sig_strikes_absorbed_per_fight"),
        _resistance_score(row, "sub_losses"),
        _distance_rate(row),
    ]


def build_radar_chart_svg(
    metrics_a: list[float], metrics_b: list[float], name_a: str, name_b: str,
    size: int = 280,
) -> str:
    """Renders a 5-axis radar chart overlaying both fighters' metrics as translucent polygons."""
    n = len(AXIS_LABELS)
    cx = cy = size / 2
    max_r = size * 0.24
    label_r = size * 0.32

    def angle(i):
        return -math.pi / 2 + i * (2 * math.pi / n)

    def point(value, i):
        r = max_r * max(0.0, min(100.0, value)) / 100.0
        a = angle(i)
        return cx + r * math.cos(a), cy + r * math.sin(a)

    def polygon_points(metrics):
        # None vertices are SKIPPED, not plotted at zero. The polygon closes
        # across the gap, which reads as "this axis isn't measured for this
        # fighter" rather than "this fighter scores zero here". Plotting the
        # origin instead is precisely the bug this rewrite removes: it made
        # absence indistinguishable from the worst possible score.
        pts = [point(v, i) for i, v in enumerate(metrics) if v is not None]
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    # Gridlines at 25/50/75/100%
    grid_svg = ""
    for pct in (25, 50, 75, 100):
        pts = [point(pct, i) for i in range(n)]
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        grid_svg += f'<polygon points="{pts_str}" fill="none" stroke="#2e2e30" stroke-width="1"/>'

    # Spoke lines from center to each axis
    spokes_svg = ""
    for i in range(n):
        x, y = point(100, i)
        spokes_svg += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#2e2e30" stroke-width="1"/>'

    # Axis labels, positioned just outside the outer gridline
    labels_svg = ""
    for i, label in enumerate(AXIS_LABELS):
        a = angle(i)
        lx, ly = cx + label_r * math.cos(a), cy + label_r * math.sin(a)
        anchor = "middle"
        if math.cos(a) > 0.3:
            anchor = "start"
        elif math.cos(a) < -0.3:
            anchor = "end"
        # An axis neither fighter has data for is dimmed and marked, so the
        # gap in the polygons above is explained rather than looking like a
        # rendering fault. Half-known axes keep the normal colour -- the
        # break in one polygon already carries that.
        both_missing = metrics_a[i] is None and metrics_b[i] is None
        fill = "#4a4d54" if both_missing else "#8a8f9a"
        text = f"{label} \u2014" if both_missing else label
        labels_svg += f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="8.5" fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle">{text}</text>'

    poly_a = polygon_points(metrics_a)
    poly_b = polygon_points(metrics_b)

    # CORNER COLOURS, matching the waterfall and the movement charts.
    # Fighter A (listed first) is the red corner, B the blue.
    #
    # Gold/grey was worse than it looked: grey is the least separable colour
    # on a charcoal panel, so fighter B effectively had no identity, and gold
    # meant "model" everywhere else on the site.
    #
    # Overlap is fine here because the fills are 0.18/0.22 with solid 2px
    # strokes -- at that opacity two strongly separated hues read as a mixed
    # region rather than mud, and the strokes keep both outlines legible
    # wherever they cross. If the fills were near-opaque this swap would have
    # made overlap worse whatever colours were chosen.
    color_a, color_b = "#e53935", "#3b82f6"

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" class="radar-chart" role="img" '
        f'style="overflow: visible;" '
        f'aria-label="Style matchup radar comparing {name_a} and {name_b}">'
        + grid_svg + spokes_svg +
        f'<polygon points="{poly_b}" fill="{color_b}" fill-opacity="0.18" stroke="{color_b}" stroke-width="2" '
        f'class="radar-polygon" style="transform-origin: {cx}px {cy}px;"/>'
        f'<polygon points="{poly_a}" fill="{color_a}" fill-opacity="0.22" stroke="{color_a}" stroke-width="2" '
        f'class="radar-polygon" style="transform-origin: {cx}px {cy}px; transition-delay: 0.12s;"/>'
        + labels_svg +
        '</svg>'
    )
