"""
Striking profile: WHERE a fighter operates and WHAT he hits.

THE GAP THIS FILLS. Nothing on the site expresses a fighter's style. The
waterfall says why he wins, the radar says how he finishes, the props say when
it ends -- none of them say that he lives at range and head-hunts, or that he
drags people down and works the body. That is the first thing a bettor
actually asks about a matchup, and ESPN has been returning the data all along
in the same payload the totals come from.

PERCENTILE, NOT RAW SHARE, and the roster proves why. A distance head-hunter
and a ground-and-pound grappler both land ~68.5% of their strikes to the head
-- head share barely separates them. Colour a silhouette by raw share and
every fighter on the card gets the same red head. Ranked against the roster
instead, the same numbers separate cleanly, and the informative zones turn out
to be legs (mean 14.6%, sd 7.7 -- the most variable trait relative to its own
mean) and position.

CORNER COLOURS, NOT A RED HEAT RAMP. ESPN shades its damage maps red. This
site cannot: red is reserved for an incorrect pick and carries the cancelled
banner, so a red silhouette would collide with a meaning that already exists.
The radar already establishes red for corner A and blue for corner B, so each
silhouette shades in ITS OWN fighter's corner colour with intensity carrying
the percentile. No new hue enters the system, and the two fighters stay
instantly distinguishable side by side.

"AT RANGE", NEVER "DISTANCE", in anything the reader sees. In striking stats
"distance" means standing at range; in betting "goes the distance" means
reaching the judges. Those would sit inches apart on the same card meaning
opposite things.
"""

ZONES = ("head", "body", "leg")
POSITIONS = ("distance", "clinch", "ground")

# Percentile bands -> alpha. Deliberately coarse: a continuous ramp invites
# reading a 3-point percentile gap as meaningful, which it is not at this
# sample size. Five steps say "unusually low / low / typical / high /
# unusually high" and nothing finer.
# A CONTINUOUS, GAMMA-CURVED RAMP rather than five flat steps.
# The stepped version compressed the middle: a 88th-percentile head landed at
# 1.0 and a 40th-percentile body at 0.52, which is under 2x apart and, once
# the gradient blends them, not enough for the eye to pick the dominant zone
# out at a glance -- the one job this figure has.
# Raising the percentile to a power >1 pushes everything below the top down
# hard, so the same pair now separates ~5x. The floor stays clear of the
# unshaded figure so "rarely goes there" still never reads as "no data".
# THREE TIERS, not a smooth ramp -- and colour varies, not just opacity.
# Opacity alone compresses into a narrow luminance band on a near-black card,
# so no gamma curve separates the zones much further; changing the COLOUR per
# tier is the lever that actually works. The bands also read better than a
# continuous ramp at 78px, where a 5-point percentile difference is invisible
# anyway and pretending otherwise is false precision.
#   low  -- a quiet tint, "he rarely goes here"
#   mid  -- the corner colour proper
#   high -- brighter and shifted toward the hot end of its own hue (red ->
#           orange, blue -> cyan), plus a glow layer. Deliberately NOT a
#           solid corner colour at mid: that leaves the top tier nowhere to
#           go except the glow.
# The jump from mid's ceiling (0.60) to high's floor (0.80) is intentional --
# a visible step is the point.
TIER_LOW, TIER_HIGH = 30.0, 70.0


def _pct_rank(value, sorted_vals) -> float | None:
    """Midrank percentile -- ties share a midpoint rather than all reading 0."""
    if value is None or not sorted_vals:
        return None
    below = sum(1 for x in sorted_vals if x < value)
    equal = sum(1 for x in sorted_vals if x == value)
    return (below + equal / 2) / len(sorted_vals) * 100


def build_zone_index(fighters_df) -> dict:
    """{column: sorted values} for every zone share, built once per site build."""
    index = {}
    cols = [f"strikes_{z}_share" for z in ZONES] + \
           [f"strikes_{p}_share" for p in POSITIONS] + \
           [f"absorbed_{z}_share" for z in ZONES]
    for col in cols:
        if col not in getattr(fighters_df, "columns", []):
            continue
        vals = []
        for v in fighters_df[col]:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:
                vals.append(f)
        if len(vals) >= 20:
            index[col] = sorted(vals)
    return index


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _tier(pct: float | None) -> str:
    if pct is None:
        return "low"
    if pct >= TIER_HIGH:
        return "high"
    return "mid" if pct >= TIER_LOW else "low"


def _alpha(pct: float | None) -> float:
    """Alpha within the tier, so ranking survives inside a band."""
    if pct is None:
        return 0.0
    p = max(0.0, min(100.0, pct))
    if p >= TIER_HIGH:
        return round(0.80 + 0.20 * (p - TIER_HIGH) / (100.0 - TIER_HIGH), 3)
    if p >= TIER_LOW:
        return round(0.38 + 0.22 * (p - TIER_LOW) / (TIER_HIGH - TIER_LOW), 3)
    return round(0.08 + 0.10 * (p / TIER_LOW), 3)


def _glow(pct: float | None) -> float:
    """Halo strength -- zero below the high tier, so only standouts bloom."""
    if pct is None or pct < TIER_HIGH:
        return 0.0
    return round(0.5 + 0.35 * (pct - TIER_HIGH) / (100.0 - TIER_HIGH), 3)


def zone_profile(row: dict, zone_index: dict, prefix: str = "strikes") -> dict | None:
    """
    {zone: {share, percentile, alpha}} or None when the fighter has no profile.

    `prefix` selects offense ("strikes", where he lands) or defense
    ("absorbed", where he gets hit) -- the defensive side costs nothing
    because the opponent's row was already fetched for takedown defense.
    """
    out = {}
    for zone in ZONES:
        col = f"{prefix}_{zone}_share"
        raw = row.get(col)
        try:
            share = float(raw)
        except (TypeError, ValueError):
            return None
        if share != share:
            return None
        pct = _pct_rank(share, zone_index.get(col))
        out[zone] = {
            "share": round(share, 1),
            "percentile": None if pct is None else int(round(pct)),
            "alpha": _alpha(pct),
            "tier": _tier(pct),
            "glow": _glow(pct),
            # THE LABEL MUST CARRY THE PERCENTILE, because that is what the
            # COLOUR encodes. Showing the raw share alone made the figure look
            # broken: Makhachev absorbs 49.8% to the head and it renders dim,
            # while 32.8% to the body renders bright -- correct (head is ~8th
            # percentile, body ~96th, since the division averages ~65% head)
            # but indefensible when the tooltip only shows the raw number.
            # Raw share cannot drive the colour: head is ~65% for nearly
            # everyone, so every fighter would look identical.
            "pct_label": None if pct is None else _ordinal(int(round(pct))),
        }
    return out


def position_profile(row: dict, zone_index: dict) -> dict | None:
    out = {}
    for pos in POSITIONS:
        col = f"strikes_{pos}_share"
        raw = row.get(col)
        try:
            share = float(raw)
        except (TypeError, ValueError):
            return None
        if share != share:
            return None
        pct = _pct_rank(share, zone_index.get(col))
        out[pos] = {"share": round(share, 1),
                    "percentile": None if pct is None else int(round(pct))}
    return out


# Takedowns per 15 min, converted to the same scale as a ground-strike share.
# 5 points of "grappling intent" per takedown: Makhachev's 3.1 TD/15 adds ~15,
# which is what lifts him from looking like a striker to reading as the
# grappler he is.
TD_TO_INTENT = 5.0


def grappling_intent(row: dict) -> float | None:
    """
    Ground-strike share PLUS takedown rate, because strike share alone
    misreads control-based wrestlers.

    THE FIGHT THAT EXPOSED THIS: Makhachev vs Garry came back "Mostly
    standing". Makhachev is arguably the best grappler in the sport facing a
    pure kickboxer, and the panel called it a striking match -- because his
    ground-STRIKE share is only 24.3%. He wins minutes by takedown, control
    and positional advance without necessarily landing a high PROPORTION of
    his strikes there, and ESPN publishes no control time, so a
    control-based wrestler is invisible in strike distribution alone.
    His takedown rate (3.10 per 15) is the signal that was sitting unused.
    """
    ground = row.get("strikes_ground_share")
    try:
        ground = float(ground)
    except (TypeError, ValueError):
        return None
    if ground != ground:
        return None
    td = row.get("td_landed_per_15")
    try:
        td = float(td)
        if td != td:
            td = 0.0
    except (TypeError, ValueError):
        td = 0.0
    return ground + td * TD_TO_INTENT


def fight_shape(pos_a: dict | None, pos_b: dict | None,
                row_a: dict | None = None, row_b: dict | None = None) -> dict | None:
    """
    One combined read on WHERE this fight happens.

    Deliberately says nothing about whether it reaches the judges. The round
    and method props already answer that, and a second answer that disagreed
    would be worse than no answer at all -- so this stays strictly positional.

    Requires BOTH fighters: a fight's shape is a property of the pairing, and
    describing it from one man's habits would be a guess wearing a fact's
    clothes.
    """
    if not pos_a or not pos_b:
        return None
    # Grappling INTENT where the rows are available, ground-strike share
    # otherwise -- so an older caller degrades rather than breaking.
    ga = grappling_intent(row_a) if row_a else None
    gb = grappling_intent(row_b) if row_b else None
    if ga is None:
        ga = pos_a["ground"]["share"]
    if gb is None:
        gb = pos_b["ground"]["share"]
    ground = (ga + gb) / 2
    at_range = (pos_a["distance"]["share"] + pos_b["distance"]["share"]) / 2
    clinch = (pos_a["clinch"]["share"] + pos_b["clinch"]["share"]) / 2

    # A STYLE CLASH IS NOT AN AVERAGE. One man at 63% ground against another
    # at 1% averages to 32% and reads as "mat fight likely" -- a confident
    # claim about the one matchup where the shape is genuinely CONTESTED, and
    # the most interesting fight on any card. Averaging erases exactly the
    # thing worth seeing. When the two disagree sharply, say so instead of
    # picking a side: whose game it becomes is decided by takedown entries
    # against takedown defense, which the wrestling term already prices and
    # this panel has no business restating.
    if abs(ga - gb) >= 25:
        grappler = "A" if ga > gb else "B"
        # SHORT. The first version ran to 18 words and had to be READ; a
        # label under a picture gets a glance, not a sentence, and anything
        # that needs parsing is worse than nothing there.
        return {"label": "Style clash",
                "detail": "grappler vs striker",
                "at_range": round(at_range, 1), "clinch": round(clinch, 1),
                "ground": round(ground, 1), "clash_side": grappler}

    # Thresholds from the real roster distribution: ground share has median
    # 10.0 and a right tail to 46, so 25 is genuinely unusual rather than a
    # round number. At-range median is 79.8, so 88 is the top quartile.
    # Thresholds re-fitted to the intent scale (ground share + 5x TD/15),
    # checked against a real card rather than chosen as round numbers.
    if ground >= 45:
        label, detail = "Mat fight", "both grapple heavily"
    elif at_range >= 88 and ground <= 12:
        label, detail = "Range fight", "both strike at range"
    elif ground >= 25:
        label, detail = "Grappling both ways", "ground work on both sides"
    elif clinch >= 20:
        label, detail = "Clinch heavy", "unusual clinch volume"
    else:
        label, detail = "Mostly standing", "typical striking mix"
    return {"label": label, "detail": detail,
            "at_range": round(at_range, 1), "clinch": round(clinch, 1),
            "ground": round(ground, 1)}
