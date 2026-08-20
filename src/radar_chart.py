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


# build_radar_chart_svg lived here and drew the six raw axes. It has no
# consumers left: the Tale of the Tape now plots the five categories from
# fighter_profile, because two of the old axes ("Knockdown Rate", "Damage
# Resistance") were being drawn a second time by the scout rails from the same
# underlying numbers. compute_radar_metrics above is kept -- it still backs
# scripts/audit_radar_coverage.py.


# ============================================================================
# CATEGORY RADAR
# ----------------------------------------------------------------------------
# Replaces the six raw axes above with the five categories from
# fighter_profile. WHY: "Knockdown Rate" was one of the old axes, and the scout
# row under the fighter buttons now draws "Knockdowns" from the same underlying
# measure -- the card was plotting one thing twice, in two panels, from two
# sources. "Damage Resistance" overlapped "Chin" the same way.
#
# Categories also give the chart a job the rails cannot do: the rails answer
# "how does this fighter rank", the radar answers "what SHAPE is he", and the
# five-sided figure carries that at a glance in a way nine bars never will.
# ============================================================================

CATEGORY_PAD = 40  # viewBox breathing room; the side labels overrun otherwise


def build_category_radar_svg(rows, size: int = 250) -> str:
    """
    Pentagon radar from [{label, a, b}, ...] of 0-100 category scores.

    Returns "" when neither fighter has a score -- the caller renders its own
    explanation rather than an empty five-sided outline, which would read as
    two fighters who are zero at everything.
    """
    rows = [r for r in (rows or []) if r.get("a") is not None or r.get("b") is not None]
    if not rows:
        return ""

    n = len(rows)
    cx = cy = size / 2.0
    radius = size * 0.30

    def angle(i):
        return -math.pi / 2 + i * (2 * math.pi / n)

    def point(v, i):
        r = radius * max(0.0, min(100.0, v)) / 100.0
        return cx + r * math.cos(angle(i)), cy + r * math.sin(angle(i))

    def polygon(key):
        # Same rule as the six-axis chart: a missing value is skipped, not
        # plotted at the origin, so absence never masquerades as a zero.
        pts = [point(r[key], i) for i, r in enumerate(rows) if r.get(key) is not None]
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    out = []
    for ring in (25, 50, 75, 100):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(ring, i) for i in range(n)))
        out.append(f'<polygon points="{pts}" fill="none" stroke="#2e2e30" stroke-width="1"/>')
    for i in range(n):
        x, y = point(100, i)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#2e2e30" stroke-width="1"/>')
    for i, r in enumerate(rows):
        a = angle(i)
        lx, ly = cx + (radius + 16) * math.cos(a), cy + (radius + 16) * math.sin(a)
        anchor = "middle"
        if math.cos(a) > 0.3:
            anchor = "start"
        elif math.cos(a) < -0.3:
            anchor = "end"
        out.append(
            f'<text x="{lx:.1f}" y="{ly + 3:.1f}" text-anchor="{anchor}" font-size="8" '
            f'font-weight="700" letter-spacing="0.5" fill="#8a8f9a">'
            f'{r["label"].upper()}</text>')
    # B under A so the red corner reads on top, matching every other paired
    # element on the card.
    for key, stroke, fill in (("b", "#3b82f6", "rgba(59,130,246,0.22)"),
                              ("a", "#e53935", "rgba(229,57,53,0.22)")):
        pts = polygon(key)
        if pts:
            out.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
                       f'stroke-width="1.6" stroke-linejoin="round"/>')

    # A TIGHT viewBox, computed rather than assumed. A square box around a
    # pentagon leaves roughly 50px of empty canvas under the two bottom
    # vertices, which pushed the legend and the category rows down the card
    # for nothing. The label ring is radius + 16, and the font's ascender and
    # descender need ~8px either side of the text baseline.
    label_r = radius + 16
    sins = [math.sin(angle(i)) for i in range(n)]
    top = cy + label_r * min(sins) - 8
    bottom = cy + label_r * max(sins) + 8
    w = size + 2 * CATEGORY_PAD
    h = bottom - top
    return (f'<svg class="radar-chart" viewBox="{-CATEGORY_PAD} {top:.1f} {w} {h:.1f}" '
            f'width="{w}" height="{h:.1f}" role="img" '
            f'aria-label="Category comparison">{"".join(out)}</svg>')
