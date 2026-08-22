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

# WHAT IS LEFT HERE, AND WHY. This module used to compute and draw a six-axis
# radar from fighters.csv. That chart is gone: the Tale of the Tape now plots
# five categories from src/fighter_profile.py, built on pit_stats, and two of
# the old axes ("Knockdown Rate", "Damage Resistance") were being drawn a
# second time by the scout rails from the same underlying measures.
#
# Removed with it: AXIS_LABELS, compute_radar_metrics and the helpers only it
# used -- _percentile, _pct, _num, _resistance_score, _distance_rate, SHRINK_K,
# PRIOR_FINISH_LOSS_RATE, MIN_ESPN_FIGHTS. build_percentile_index survives
# because build_spotlight_chips still ranks against it.

PERCENTILE_AXES = {
    "knockdowns_per_fight": True,
    "sig_strikes_att_per_fight": True,
    "sig_strikes_absorbed_per_fight": False,     # inverted: less damage taken is better
}

def build_percentile_index(fighters_df) -> dict:
    """
    {column: sorted list of values} for percentile ranking.

    Built once per site build from the whole roster and passed into
    build_spotlight_chips, its only remaining caller. Computing it per
    fighter would be both slow and wrong -- the ranking has to be against a
    fixed population, not against whoever happens to be on the card.
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
    # Held back rather than appended: they go inside the .radar-polygon group
    # below, which needs the viewBox to have been computed first.
    data_polys = []
    for key, stroke, fill in (("b", "#3b82f6", "rgba(59,130,246,0.22)"),
                              ("a", "#e53935", "rgba(229,57,53,0.22)")):
        pts = polygon(key)
        if pts:
            data_polys.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
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

    # THE SCROLL-IN REVEAL. .radar-polygon is what the stylesheet scales from
    # 0 to 1 once the block enters view; rebuilding this chart for the five
    # categories dropped the class, so the polygons were simply painted at
    # full size and the animation had nothing to act on.
    #
    # One group around BOTH polygons, not the class on each: the two have
    # different bounding boxes, so per-polygon origins would grow them from
    # two different points. A radar expands from its centre.
    #
    # transform-origin is written explicitly because the default (50% 50%)
    # is the centre of the viewBox, and this viewBox is deliberately tight
    # and vertically asymmetric -- its centre is not the centre of the
    # pentagon.
    #
    # PLAIN cx/cy, with no viewBox offset applied. Under
    # transform-box: view-box these lengths resolve in the same user
    # coordinates the polygon points are already written in, so the centre is
    # simply (cx, cy). Offsetting them by the viewBox min corner first --
    # which looks right and is not -- put the growth origin 37px off, and the
    # radar unfolded from a point outside itself. Measured: an origin of
    # 0px 0px fixes the scale at user (0, 0), and 125px is the pentagon
    # centre of a 250px chart.
    if data_polys:
        out.append(
            f'<g class="radar-polygon" style="transform-box: view-box; '
            f'transform-origin: {cx:.1f}px {cy:.1f}px;">'
            + "".join(data_polys) + '</g>')

    return (f'<svg class="radar-chart" viewBox="{-CATEGORY_PAD} {top:.1f} {w} {h:.1f}" '
            f'width="{w}" height="{h:.1f}" role="img" '
            f'aria-label="Category comparison">{"".join(out)}</svg>')
