"""
Donut ring showing landed/attempted plus percentage in the center -- used
for the post-fight Significant Strikes comparison. Deliberately a plain
function (not a class) matching sparkline_chart.py / calibration_chart.py's
style: one pure function in, one SVG string out, easy to unit test.
"""

import math


def build_donut_svg(landed: int, attempted: int, color: str, size: int = 108, stroke_width: int = 11,
                     animate: bool = False, delay: float = 0.0) -> str:
    """
    animate=True adds a radial fill-in reveal (the ring sweeps from empty
    to its real percentage) via CSS custom properties + a shared keyframe
    (see templates/site.html's .donut-fill-ring rule) -- the standard,
    off-by-one-safe technique: dasharray is symmetric (circumference,
    circumference), and dashoffset animates from "fully hidden" to
    "exactly the real percentage visible", rather than fiddling with an
    asymmetric dash/gap pair. `delay` (seconds) staggers multiple donuts
    so they don't all sweep in unison. Reduced-motion users get the final
    state immediately (see the site's existing prefers-reduced-motion
    handling). The center percentage carries the site's existing generic
    .countup class (see templates/site.html) rather than a parallel
    count-up implementation -- it already knows how to animate any
    rendered number by parsing the text itself.
    """
    if attempted <= 0:
        pct = 0.0
    else:
        pct = max(0.0, min(1.0, landed / attempted))

    r = (size - stroke_width) / 2
    cx = cy = size / 2
    circumference = 2 * math.pi * r
    dash = circumference * pct
    end_offset = circumference - dash

    ring_class = "donut-svg"
    progress_class = "donut-fill-ring" if animate else ""
    progress_style = (
        f'style="--donut-circ:{circumference:.1f}px; --donut-end:{end_offset:.1f}px; '
        f'animation-delay:{delay:.2f}s;"' if animate else
        f'stroke-dasharray="{dash:.1f} {circumference:.1f}"'
    )
    pct_class = "donut-center-pct countup" if animate else "donut-center-pct"

    return f"""<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" class="{ring_class}">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="var(--chart-grid)" stroke-width="{stroke_width}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke_width}"
    class="{progress_class}" {progress_style} stroke-linecap="round"
    transform="rotate(-90 {cx} {cy})"/>
  <text x="{cx}" y="{cy - 6}" text-anchor="middle" class="donut-center-value">{landed}/{attempted}</text>
  <text x="{cx}" y="{cy + 14}" text-anchor="middle" class="{pct_class}">{round(pct*100)}%</text>
</svg>"""


def build_split_donut_svg(stats: dict, color: str, size: int = 90, stroke_width: int = 11,
                          delay: float = 0.0) -> str:
    """
    A confidence-tier ring that can be filtered to the picks where we backed
    the market's favourite, or the ones where we faded it.

    THREE LAYERS, ONE DENOMINATOR -- every pick in the tier is the full
    circle, and that never changes as you filter. The dim arc is every win in
    the tier and is likewise FIXED; only the bright arc moves, from "all the
    wins" to "the wins that came from this cohort". So the question the ring
    answers is a composition one: of the ones we hit, how many were favourites?

    That the dim arc is constant is why this needs no second animated ring.
    The alternative reading -- dim arc = the cohort's share of the tier, which
    makes bright/dim the cohort's hit RATE -- was rendered alongside this one
    and not chosen.

    The bright arc keeps .donut-fill-ring, so the scroll-into-view sweep in
    templates/site.html is untouched and the unfiltered state is pixel-identical
    to what shipped before this existed. Retargeting on a filter tap is a
    STATE CHANGE, not a reveal, so the template hands it to a transition in the
    site's 0.55s vocabulary rather than restarting the 1.6s reveal keyframe.

    THE COHORTS NEED NOT SUM TO THE DIM ARC. A pick with no recorded price
    belongs to neither, so a tier can hold a win that no filtered state will
    ever cover. The counts are printed rather than only the arcs precisely so
    that gap is legible instead of looking like a rounding error.
    """
    total = int(stats.get("total") or 0)
    hit = int(stats.get("correct") or 0)
    fav = int((stats.get("favorite") or {}).get("correct") or 0)
    dog = int((stats.get("underdog") or {}).get("correct") or 0)

    r = (size - stroke_width) / 2
    cx = cy = size / 2
    circ = 2 * math.pi * r

    def offset(n):
        """Dashoffset for n wins out of the tier -- 0 wins hides the arc."""
        frac = max(0.0, min(1.0, n / total)) if total > 0 else 0.0
        return circ - circ * frac

    pct = round(hit / total * 100) if total > 0 else 0
    return f"""<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" class="donut-svg split-donut"
  data-total="{total}" data-hit="{hit}" data-fav="{fav}" data-dog="{dog}"
  data-circ="{circ:.1f}" data-off-all="{offset(hit):.1f}"
  data-off-fav="{offset(fav):.1f}" data-off-dog="{offset(dog):.1f}">
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="var(--chart-grid)" stroke-width="{stroke_width}"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke_width}"
    class="donut-ghost-ring" stroke-dasharray="{circ - offset(hit):.1f} {circ:.1f}"
    stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke_width}"
    class="donut-fill-ring" style="--donut-circ:{circ:.1f}px; --donut-end:{offset(hit):.1f}px; animation-delay:{delay:.2f}s;"
    stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/>
  <text x="{cx}" y="{cy - 6}" text-anchor="middle" class="donut-center-value">{hit}/{total}</text>
  <text x="{cx}" y="{cy + 14}" text-anchor="middle" class="donut-center-pct countup">{pct}%</text>
</svg>"""
