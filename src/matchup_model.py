"""
'Styles make fights.' A raw rating gap between two fighters misses a real
dynamic: a strong wrestler with good takedown accuracy against a striker
with weak takedown defense has a stylistic advantage the base rating alone
won't capture, and a fighter who's been finished by strikes repeatedly
brings real durability risk into their next fight, independent of their
overall record.

This layer takes the Elo/stats blended rating from power_rating.py and
nudges it based on:
  1. Takedown accuracy vs. opponent's takedown defense (wrestling advantage)
  2. Striking accuracy differential (volume/precision advantage)
  3. Durability: how often each fighter has been finished before, and by
     which method -- a proxy for whether a given attack is likely to work
     against them specifically, not just in general

None of this replaces real film study or a trained analyst's eye -- it's a
systematic way to weight publicly available stats a bit closer to how
people actually reason about matchups, instead of just comparing records.
"""

import datetime as dt
import math

import pandas as pd

# How many Elo-equivalent rating points a fully-realized stylistic
# advantage is worth. Tuned to be meaningful but not dominate the base
# rating gap entirely -- these are secondary signals, not the headline.
WRESTLING_ADVANTAGE_SCALE = 300.0  # legacy fallback path only (see wrestling_adj)
# Rating points per 1.0 takedown-per-15-min differential. Set so the term's
# typical magnitude matches what head_to_head_adjustment.py validated (its
# TD-rate coefficient of 10.0 carried a tuned blend weight of 1.5).
TD_RATE_ADVANTAGE_SCALE = 15.0
STRIKING_ADVANTAGE_SCALE = 150.0
DURABILITY_SCALE = 120.0
VOLUME_DIFFERENTIAL_SCALE = 40.0  # rating points per 1.0 SLpM-SApM differential gap

# Cage time at which a fighter's RATE statistics earn full weight in the
# style layer. Below it, the striking and wrestling terms are scaled down
# toward zero, handing the fight back to the rating gap.
#
# WHY. Rate stats are per-minute and per-fight averages, so a short sample
# does not merely make them uncertain -- it makes them EXTREME. A fighter
# with four bouts and 38 minutes of tracked time can read as 1.42 strikes
# absorbed per minute with zero knockdowns absorbed, numbers no established
# fighter posts, because he has not yet had the bad night that would move
# them. The style layer then reads that as an enormous edge.
#
# The both-corners gating below already stops a real number being compared
# against a DEFAULT. It does nothing about a real number computed from
# almost nothing, which is a different failure and the one that bit on
# UFC 330: the model's three disagreements with the market all went against
# it, and two were built on 38 and 15 minutes of cage time respectively
# (Kaue Fernandes at 70% against a market 40%, Eduardo Chapolin at 64%
# against 46%). Both lost.
#
# 90 minutes is six full three-round fights. Deliberately a smooth ramp
# rather than a cliff: radar_chart draws nothing below MIN_ESPN_FIGHTS = 3,
# which is right for a chart that must either show a number or not, but a
# hard threshold in a continuous model just relocates the discontinuity.
#
# Set to 0 to disable the scaling entirely -- used by the backtest harness
# to run the unweighted model as a control arm.
#
# COUNTED IN FIGHTS, NOT MINUTES, and that choice is load-bearing. Minutes
# are the better measure in principle -- four five-round fights carry more
# information than four first-round finishes -- but they cannot be
# reconstructed point-in-time from the cached ESPN data: fight duration comes
# from a per-competition status endpoint, and 73% of those were never cached.
# A first version of this keyed on fight_minutes_total and its backtest was
# meaningless as a result: the reconstruction returned 0 for most fights, the
# weight collapsed to ~0 nearly everywhere, and what actually got measured
# was "switch the style layer off", which this project already established is
# worse (57.7% with adjustments vs 55.9% Elo-only).
#
# Fight COUNT is reconstructable exactly -- it is just the timeline entries
# before a date -- so the validation can test what production runs. It is
# also what radar_chart already gates on (MIN_ESPN_FIGHTS = 3), which keeps
# one notion of "enough data" across the codebase rather than two.
# OFF (0.0) -- IMPLEMENTED, MEASURED, NOT JUSTIFIED YET.
#
# scripts/validate_rate_stat_shrinkage.py walks 3,181 point-in-time fights
# and compares this against an unweighted control. Every threshold, every
# subset, the paired bootstrap comes back NOT significant:
#
#     subset                     best threshold   Brier      p
#     all 3,181 fights                  3        -0.0001   0.248
#     corner under 3 prior fights       3        -0.0009   0.263
#     corner under 6 prior fights       3        -0.0003   0.252
#     corner under 10 prior fights      3        -0.0002   0.242
#
# The direction is consistently right -- threshold 3 never hurts, and helps
# most on exactly the thin-sample fights the theory points at -- but 377
# fights cannot separate that from luck, and 0.248 is not close.
#
# Notably 6.0 (the value first proposed, from reasoning rather than
# measurement) is NEUTRAL overall and beaten by 3.0 everywhere. That is the
# same failure mode as the picks this was meant to fix: a confident number
# derived from a plausible story rather than from data.
#
# Left in place rather than deleted because the mechanism is written and
# tested, and the sample only grows. Re-run the harness after another
# season; if the thin-fight subset reaches significance at threshold 3, set
# this to 3.0 and nothing else needs to change. Setting it to any positive
# value enables the scaling.
STYLE_FULL_TRUST_FIGHTS = 0.0


def rate_stat_confidence(row_a: pd.Series, row_b: pd.Series) -> float:
    """
    0.0-1.0 multiplier for style terms built from rate statistics.

    Governed by the THINNER corner. A differential is only as trustworthy as
    its weaker side: pairing 24 tracked fights against 1 does not average out
    to a reliable comparison, it produces a confident-looking number about a
    fighter nobody has measured.
    """
    if STYLE_FULL_TRUST_FIGHTS <= 0:
        return 1.0

    def _fights(row) -> float:
        v = row.get("espn_fights")
        try:
            return float(v) if pd.notna(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    thinner = min(_fights(row_a), _fights(row_b))
    return max(0.0, min(1.0, thinner / STYLE_FULL_TRUST_FIGHTS))
SUBMISSION_THREAT_SCALE = 60.0  # rating points per 1.0 sub-win-rate differential

# Southpaw-vs-orthodox is a real, documented edge in striking sports --
# most fighters train far more often against orthodox opponents, so a
# southpaw sees a less familiar look more often than the reverse. It's a
# real but modest effect in the research, not a dominant one, so this is
# calibrated well below the primary factors above (roughly comparable to
# a single year past a fighter's age-decline threshold, not a heavy
# thumb on the scale). Switch-stance fighters can choose their angle, so
# they get the same small edge against either pure stance.
STANCE_MISMATCH_BONUS = 18.0

# Base width (in probability points) for the uncertainty band before
# dividing by sqrt(thinner_record + 1) -- 0.30 means a 0-fight matchup
# gets roughly a +/-30pp band, narrowing to roughly +/-6pp by the time
# the thinner record reaches 20+ fights.
UNCERTAINTY_BASE = 0.30

# A recent win/loss (from fight_history.csv specifically, not the
# aggregate career record) gets a small rating nudge that decays to zero
# over RECENT_FORM_DECAY_YEARS -- deliberately modest since this is a
# single data point, not an aggregate trend.
RECENT_FORM_SCALE = 10.0  # per decayed win/loss unit, summed over last 3 fights (backtest-validated, July 2026)
RECENT_FORM_DECAY_YEARS = 2.0
RECENT_FORM_LOOKBACK = 3

# Height advantage in the adjustment layer. Deliberately small: reach
# (which correlates strongly with height) is already worth 4 pts/inch in
# the stats-based power rating, so a large height weight here would
# double-count the same physical edge. NOT backtest-validatable with
# current data (fight_history.csv has no physical stats), so this is a
# documented heuristic, kept conservative for exactly that reason.
HEIGHT_ADVANTAGE_SCALE = 2.0

# Short-notice fights: fighters accepting a bout on short notice
# (typically < ~30 days, no full camp) win measurably less often --
# roughly a 10 percentage-point drop is the commonly reported figure in
# published analyses of UFC short-notice outcomes. ~10pp at even odds
# corresponds to ~70 Elo points (dP/dgap near 50% is ~0.144pp per
# point). NOT backtest-validatable (fight_history.csv has no notice
# data); literature-derived rather than fitted, flagged per-fighter via
# the short_notice column in fighters.csv at card-research time and
# cleared automatically once that fighter's bout result is logged.
SHORT_NOTICE_PENALTY = 70.0

# Safety rail on the ADJUSTMENT LAYER (style factors + recent form), NOT
# on the Elo/base rating gap. Evidence basis: the walk-forward backtest
# (3,713 point-in-time fights) showed the Elo core is well-calibrated
# but rarely confident -- it said 70%+ only 16 times and never said
# 80%+. Extreme final numbers therefore come almost entirely from the
# adjustment stack, which is exactly the part that CANNOT be backtested
# yet (needs historical stat snapshots). Until those magnitudes are
# fitted to evidence rather than hand-tuned, the stack's total influence
# is clipped: +/-150 rating points can move a coin flip to at most
# ~70/30, so no pile of unvalidated modifiers can manufacture extreme
# confidence on its own. Revisit after the extended walk-forward fits
# the layoff/age magnitudes (the two factors this bites most).
ADJUSTMENT_TOTAL_CAP = 150.0

# Ring rust: no penalty for a normal 6-12 month camp cycle. Beyond a year
# away, each additional year away costs more. WEIGHT REVISED by the July
# 2026 walk-forward backtest: the previous 60 pts/yr scored WORSE than no
# layoff penalty at all on both the pre-2019 train split and the 2019+
# held-out split -- the penalty as sized was actively hurting predictions.
# The data tolerates 0-20 pts/yr (differences within noise across that
# range); 20 keeps the domain-motivated factor alive at the highest
# level the evidence supports rather than the loudest level intuition
# suggested.
LAYOFF_GRACE_YEARS = 1.0
LAYOFF_PENALTY_PER_YEAR = 20.0
LAYOFF_PENALTY_CAP = 300.0


def _get(row: pd.Series, col: str, default: float) -> float:
    return float(row[col]) if col in row and pd.notna(row[col]) else default


def layoff_years(row: pd.Series, reference_date: dt.date | None = None) -> float | None:
    if "last_fight_date" not in row or pd.isna(row["last_fight_date"]):
        return None
    reference_date = reference_date or dt.date.today()
    last_fight = pd.to_datetime(row["last_fight_date"]).date()
    return (reference_date - last_fight).days / 365.25


def layoff_penalty(row: pd.Series, reference_date: dt.date | None = None) -> float:
    years_away = layoff_years(row, reference_date)
    if years_away is None or years_away <= LAYOFF_GRACE_YEARS:
        return 0.0
    penalty = LAYOFF_PENALTY_PER_YEAR * (years_away - LAYOFF_GRACE_YEARS)
    return -min(penalty, LAYOFF_PENALTY_CAP)


# Coming back too SOON after being finished carries real, documented risk --
# the opposite problem from ring rust. Six months is a rough dividing line.
QUICK_RETURN_THRESHOLD_YEARS = 0.5
QUICK_RETURN_PENALTY_CAP = 150.0


def quick_return_penalty(row: pd.Series, reference_date: dt.date | None = None) -> float:
    if row.get("last_fight_result") != "L" or row.get("last_fight_method") not in ("KO/TKO", "SUB"):
        return 0.0  # only a finish loss carries this specific risk, not a decision loss
    years_away = layoff_years(row, reference_date)
    if years_away is None or years_away >= QUICK_RETURN_THRESHOLD_YEARS:
        return 0.0
    severity = (QUICK_RETURN_THRESHOLD_YEARS - years_away) / QUICK_RETURN_THRESHOLD_YEARS
    return -severity * QUICK_RETURN_PENALTY_CAP


# The age cliff hits divisions very differently. Speed/output-dependent
# lighter divisions (where reflexes and recovery matter most) see a steep
# decline past 35; heavyweight and light heavyweight fighters, where power
# and experience matter more than raw speed, often peak or sustain well
# into their late 30s.
AGE_CLIFF_START = {
    "Strawweight": 35, "Flyweight": 35, "Bantamweight": 35, "Featherweight": 35,
    "Lightweight": 37, "Welterweight": 37, "Middleweight": 37,
    "Light Heavyweight": 39, "Heavyweight": 40,
}
AGE_CLIFF_DEFAULT_START = 37  # for any weight class not explicitly listed
AGE_CLIFF_PENALTY_PER_YEAR = 25.0
AGE_CLIFF_PENALTY_CAP = 200.0


def age_cliff_penalty(row: pd.Series) -> float:
    age = row.get("age")
    weight_class = row.get("weight_class")
    if pd.isna(age) or not weight_class:
        return 0.0  # no penalty when age isn't known -- better than guessing wrong
    cliff_age = AGE_CLIFF_START.get(weight_class, AGE_CLIFF_DEFAULT_START)
    years_past_cliff = float(age) - cliff_age
    if years_past_cliff <= 0:
        return 0.0
    return -min(AGE_CLIFF_PENALTY_PER_YEAR * years_past_cliff, AGE_CLIFF_PENALTY_CAP)


# Missing weight is a documented red flag -- often reflecting a rushed or
# broken training camp, not just a one-off scale mistake -- and it also
# means the opponent gets an automatic strength/size advantage on fight
# night after rehydration. Data note: this field defaults to 0 (no known
# instances) for the current roster; populating real history requires
# per-fighter weigh-in research this build doesn't have time to do exhaustively.
MISSED_WEIGHT_PENALTY_PER_INSTANCE = 20.0
MISSED_WEIGHT_PENALTY_CAP = 80.0


def missed_weight_penalty(row: pd.Series) -> float:
    count = row.get("missed_weight_count")
    if pd.isna(count) or count <= 0:
        return 0.0
    return -min(MISSED_WEIGHT_PENALTY_PER_INSTANCE * float(count), MISSED_WEIGHT_PENALTY_CAP)


# Canonical lightest-to-heaviest ordering, used to measure how many
# divisions a fighter is jumping -- a one-division move (e.g. Welterweight
# to Middleweight) and a two-division move (e.g. Lightweight to
# Middleweight) are not the same risk, and shouldn't score identically.
DIVISION_ORDER = [
    "Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
]

# Deliberately modest magnitudes -- this factor is NOT backtested against
# historical outcomes the way e.g. the durability calibration was (that
# would require weight-class data per historical fight, which
# fight_history.csv doesn't have). Moving up is scaled by how many
# divisions are jumped; moving down gets a flat, smaller bonus, since a
# clean cut's advantage is real but shouldn't be overstated relative to
# an unvalidated theory. Deliberately asymmetric: moving up is riskier
# than moving down is advantageous, per the reasoning discussed with the
# user (power/leverage disadvantage facing larger opponents, versus a
# capped upside from cutting into a smaller field).
WEIGHT_CLASS_UP_PENALTY_PER_DIVISION = 10.0
WEIGHT_CLASS_UP_PENALTY_CAP = 30.0
WEIGHT_CLASS_DOWN_BONUS = 8.0


def recent_weight_class(name: str, history_df: pd.DataFrame | None) -> str | None:
    """
    A fighter's settled recent division: simply their single most recent
    fight's weight class. This deliberately does NOT average over a
    window of several fights -- by the time we're predicting a fighter's
    CURRENT upcoming fight, any past one-off aberration (a short-notice
    or catchweight booking at an unusual weight) has already been
    superseded by whatever they actually fought at next, so the most
    recent entry already reflects reality. Concretely: Usman's one-off
    2023 Middleweight fight is correctly ignored because his following
    fight (2025, Welterweight) is now the most recent entry -- and
    Makhachev's genuine, deliberate Welterweight title win is correctly
    recognized immediately, rather than needing a second Welterweight
    fight to outvote two years of Lightweight history first. Returns
    None with no history at all, which the caller treats as "unknown,
    don't penalize" rather than guessing.
    """
    if history_df is None or history_df.empty:
        return None
    rows = history_df[history_df["name"] == name].sort_values("date", ascending=False)
    if rows.empty:
        return None
    return rows["weight_class"].iloc[0]


def weight_class_change_penalty(name: str, this_fight_weight_class: str | None, history_df: pd.DataFrame | None) -> float:
    """
    Penalizes a fighter moving up in weight (scaled by how many divisions),
    rewards moving down (flat, smaller bonus). Returns 0.0 -- no penalty,
    no bonus -- whenever there isn't enough information to say anything:
    no history, no current division, an unrecognized division name, or no
    settled division different from this fight's. Silence over a guess is
    deliberate here, same as elsewhere in this module.
    """
    settled = recent_weight_class(name, history_df)
    if not settled or not this_fight_weight_class or settled == this_fight_weight_class:
        return 0.0
    if settled not in DIVISION_ORDER or this_fight_weight_class not in DIVISION_ORDER:
        return 0.0

    division_distance = DIVISION_ORDER.index(this_fight_weight_class) - DIVISION_ORDER.index(settled)
    if division_distance > 0:  # moving up (this fight's division is heavier)
        return -min(WEIGHT_CLASS_UP_PENALTY_PER_DIVISION * division_distance, WEIGHT_CLASS_UP_PENALTY_CAP)
    else:  # moving down
        return WEIGHT_CLASS_DOWN_BONUS


def compute_divisional_method_priors(fighters_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Divisional average method-of-victory rates, computed from the roster's
    own aggregate data. A heavyweight fight has an inherently higher
    baseline finish-by-KO rate than a strawweight fight, which leans
    heavily toward decisions -- a flat blend for every division ignores
    this real, well-documented difference between weight classes.
    """
    priors = {}
    for wc, group in fighters_df.groupby("weight_class"):
        total_wins = group["wins"].sum()
        if total_wins <= 0:
            continue
        priors[wc] = {
            "KO/TKO": group["ko_wins"].sum() / total_wins,
            "SUB": group["sub_wins"].sum() / total_wins,
            "DEC": group["dec_wins"].sum() / total_wins,
        }
    return priors


def blend_method_probability(
    divisional_prior: float, fighter_own_rate: float, opponent_vulnerability: float, fighter_total_wins: int,
) -> float:
    """
    Prior-informed blend: starts at the divisional baseline, then shifts
    toward the fighter's own observed tendency -- weighted by how much
    career sample size backs it up, so a 3-fight newcomer's personal rate
    doesn't override the divisional prior as hard as a proven veteran's
    would -- then further incorporates the specific opponent's vulnerability.
    """
    experience_weight = min(1.0, fighter_total_wins / 10.0)
    fighter_adjusted = divisional_prior + (fighter_own_rate - divisional_prior) * experience_weight
    return 0.7 * fighter_adjusted + 0.3 * opponent_vulnerability


def _stat(row, field):
    """
    Prefer the RECENCY-WEIGHTED value, fall back to the all-time one.

    Career means treat a fighter's 2016 form as equal to their 2025 form.
    Measured across 1,061 fighters with 6+ tracked fights, the median
    within-career drift in strikes/min is 0.62 of the BETWEEN-fighter
    standard deviation, and 32% drift by more than a full sd -- so an
    all-time mean often describes a blend of two different fighters.

    Validated before wiring (research_recency_weighting.py): an 18-month
    half-life was swept on the tuning split with all-time included as a
    control, and the control lost MONOTONICALLY. Frozen holdout, n=1747:
    58.6% -> 59.9% accuracy, Brier 0.2345 -> 0.2329.

    The fallback matters as much as the preference: scripts/backfill_recency_stats.py
    only writes a weighted value for fighters with enough dated cage time, so
    debutants and thin records keep the all-time number rather than dropping
    out of the adjustment entirely.
    """
    v = row.get(field + "_r")
    if v is not None and pd.notna(v) and str(v) != "":
        return v
    return row.get(field)


def build_probability_waterfall(matchup: dict) -> dict | None:
    """
    Decomposes the final probability into a running journey a reader can
    follow: start from an even fight at 50%, then watch each factor move
    the number, ending exactly on the model's real pick percentage.

    WHY PERCENTAGE POINTS AND A RUNNING TOTAL. The model works internally
    in rating points, which are exactly additive -- but "+149.4" means
    nothing at a glance, which made the first version of this panel
    unreadable. Percentage points are the unit the reader already thinks
    in, and showing the running probability after every step means no
    arithmetic is required to follow it: you can see the pick move from
    50% to its final number, one factor at a time.

    The deltas still sum EXACTLY to (final - 50%), because each is
    measured as the change in probability caused by adding that factor to
    everything before it. The one honest caveat is that with a non-linear
    logistic link, a factor's share depends slightly on where in the
    order it lands. Factors are applied largest-first (after the base
    rating gap), which is both a fixed, disclosed order and the one that
    reads most naturally -- biggest movers at the top.

    Rating points are still carried on every row as a secondary number,
    for anyone who wants the model's native units.

    ORIENTATION. predict_matchup computes everything as "positive favors
    fighter A". A reader wants "why do we like the pick", so signs flip
    when the favorite is fighter B. After this, positive always means
    "helps the favorite".

    EXACTNESS. Factors below DISPLAY_THRESHOLD fold into one "other
    factors" row rather than being dropped, and the cap gets its own row
    when it bites, so the journey always lands on the true final number.
    """
    if not matchup:
        return None

    prob_a = matchup.get("prob_a")
    base_a, base_b = matchup.get("base_rating_a"), matchup.get("base_rating_b")
    if prob_a is None or base_a is None or base_b is None:
        return None

    favorite_is_a = prob_a >= 0.5
    sign = 1.0 if favorite_is_a else -1.0
    favorite = matchup["fighter_a"] if favorite_is_a else matchup["fighter_b"]
    underdog = matchup["fighter_b"] if favorite_is_a else matchup["fighter_a"]

    base_gap = (base_a - base_b) * sign
    raw_layer = matchup.get("adjustment_layer_raw", 0.0) * sign
    applied_layer = matchup.get("adjustment_layer_applied", 0.0) * sign

    # (key in matchup, display label, plain-language explanation)
    FACTORS = [
        ("wrestling_adjustment", "Wrestling", "takedowns landed per 15 minutes, head to head"),
        ("striking_adjustment", "Striking", "significant-strike accuracy and defense"),
        ("durability_adjustment", "Durability", "how often each has been finished"),
        ("recent_form_adjustment", "Recent form", "last three fights, recent ones counting more"),
        ("submission_threat_adjustment", "Sub threat", "share of wins by submission"),
        ("stance_adjustment", "Stance", "orthodox vs southpaw matchup"),
        ("height_adjustment", "Height", "height difference"),
        ("layoff_adjustment", "Layoff", "time since last fight"),
        ("quick_return_adjustment", "Quick turnaround", "unusually short rest since the last fight"),
        ("age_cliff_adjustment", "Age", "age vs the division's typical decline point"),
        ("missed_weight_adjustment", "Missed weight", "history of missing weight"),
        ("weight_class_change_adjustment", "Division change", "moving up or down in weight"),
        ("short_notice_adjustment", "Short notice", "took the fight on short notice"),
    ]

    DISPLAY_THRESHOLD = 2.0  # rating points -- below this it's noise, not a driver

    def prob_at(gap):
        return 1.0 / (1.0 + 10 ** (-gap / 400.0))

    # Collect factors, fold the negligible ones together.
    collected, folded = [], 0.0
    for key, label, why in FACTORS:
        pts = matchup.get(key, 0.0) * sign
        if abs(pts) < DISPLAY_THRESHOLD:
            folded += pts
            continue
        collected.append({"label": label, "why": why, "points": pts})
    collected.sort(key=lambda r: abs(r["points"]), reverse=True)
    if abs(folded) >= 0.05:
        collected.append({"label": "Other factors",
                          "why": "everything else, each too small to list on its own",
                          "points": folded})

    # Walk the journey, recording the running probability after each step.
    rows = []
    running_gap = 0.0
    running_prob = prob_at(running_gap)  # 50%

    def step(label, why, pts, kind="factor"):
        nonlocal running_gap, running_prob
        before = running_prob
        running_gap += pts
        running_prob = prob_at(running_gap)
        rows.append({
            "label": label, "why": why, "kind": kind,
            "points": round(pts, 1),
            "delta_pp": round((running_prob - before) * 100, 1),
            "running_pct": round(running_prob * 100, 1),
            "favors": favorite if pts > 0 else underdog,
        })

    step("Rating gap", "career record and quality of opposition", base_gap, kind="base")
    for c in collected:
        step(c["label"], c["why"], c["points"])
    if matchup.get("adjustment_capped"):
        step("Adjustment cap",
             f"total adjustment held to {ADJUSTMENT_TOTAL_CAP:.0f} rating points, so no pile-up "
             f"of factors can outweigh the rating gap itself",
             applied_layer - raw_layer, kind="cap")

    favorite_prob = prob_a if favorite_is_a else 1.0 - prob_a
    return {
        "favorite": favorite,
        "underdog": underdog,
        "rows": rows,
        "favorite_pct": round(favorite_prob * 100, 1),
        "underdog_pct": round((1.0 - favorite_prob) * 100, 1),
        "total_points": round(base_gap + applied_layer, 1),
        # Largest single move, for scaling the bars.
        "scale": max([abs(r["delta_pp"]) for r in rows] + [1.0]),
    }


BADGE_THRESHOLD = 15.0  # rating points -- below this, a factor isn't worth calling out as a driver


def build_factor_badges(matchup: dict) -> dict:
    """
    Translates the raw adjustment numbers already computed in predict_matchup
    into small labeled badges per fighter, e.g. "+ Durability" or "- Layoff",
    so the model's reasoning is scannable at a glance instead of only living
    in the prose narrative.

    Advantage-style factors (wrestling/striking/durability) are POSITIVE
    when they favor fighter A -- badge goes on whichever fighter has the
    edge, framed as a plus for them.

    Penalty-style factors (layoff/quick-return/age-cliff/missed-weight) are
    computed as (a's own penalty - b's own penalty) -- badge goes on
    whichever fighter is actually carrying that specific risk, framed as a
    minus for them, since the badge should describe what's true about the
    fighter it's attached to.
    """
    badges_a, badges_b = [], []

    def add_advantage(value: float, label: str):
        if value > BADGE_THRESHOLD:
            badges_a.append({"label": label, "direction": "+"})
        elif value < -BADGE_THRESHOLD:
            badges_b.append({"label": label, "direction": "+"})

    add_advantage(matchup.get("wrestling_adjustment", 0), "Wrestling")
    add_advantage(matchup.get("striking_adjustment", 0), "Striking")
    add_advantage(matchup.get("durability_adjustment", 0), "Durability")
    add_advantage(matchup.get("stance_adjustment", 0), "Stance")
    add_advantage(matchup.get("submission_threat_adjustment", 0), "Sub Threat")
    add_advantage(matchup.get("height_adjustment", 0), "Height")

    if matchup.get("short_notice_flag_a"):
        badges_a.append({"label": "Short Notice", "direction": "-"})
    if matchup.get("short_notice_flag_b"):
        badges_b.append({"label": "Short Notice", "direction": "-"})

    layoff_adj = matchup.get("layoff_adjustment", 0)
    if layoff_adj < -BADGE_THRESHOLD:
        badges_a.append({"label": "Layoff", "direction": "-"})
    elif layoff_adj > BADGE_THRESHOLD:
        badges_b.append({"label": "Layoff", "direction": "-"})

    if matchup.get("quick_return_flag_a"):
        badges_a.append({"label": "Quick Return", "direction": "-"})
    if matchup.get("quick_return_flag_b"):
        badges_b.append({"label": "Quick Return", "direction": "-"})

    if matchup.get("age_cliff_flag_a"):
        badges_a.append({"label": "Age Cliff", "direction": "-"})
    if matchup.get("age_cliff_flag_b"):
        badges_b.append({"label": "Age Cliff", "direction": "-"})

    missed_weight_adj = matchup.get("missed_weight_adjustment", 0)
    if missed_weight_adj < -BADGE_THRESHOLD / 3:  # smaller threshold - even one instance should show
        badges_a.append({"label": "Missed Weight", "direction": "-"})
    elif missed_weight_adj > BADGE_THRESHOLD / 3:
        badges_b.append({"label": "Missed Weight", "direction": "-"})

    if matchup.get("weight_class_change_flag_a"):
        direction = matchup.get("weight_class_change_direction_a")
        label = "Moving Up" if direction == "up" else "Moving Down"
        badges_a.append({"label": label, "direction": "-" if direction == "up" else "+"})
    if matchup.get("weight_class_change_flag_b"):
        direction = matchup.get("weight_class_change_direction_b")
        label = "Moving Up" if direction == "up" else "Moving Down"
        badges_b.append({"label": label, "direction": "-" if direction == "up" else "+"})

    return {"a": badges_a, "b": badges_b}


def classify_style(row: pd.Series) -> str:
    td_acc = _get(row, "td_accuracy_pct", 20)
    strike_acc = _get(row, "strike_accuracy_pct", 45)
    if td_acc >= 40:
        return "Wrestler/Grappler"
    elif strike_acc >= 47:
        return "Striker"
    return "Balanced"


def stance_matchup_adjustment(row_a: pd.Series, row_b: pd.Series) -> float:
    """
    Southpaw (or switch) gets a modest bonus against a pure-orthodox
    opponent, reflecting the real "unfamiliar look" edge -- two fighters
    sharing the same stance (including two southpaws) is neutral, since
    neither has the familiarity advantage over the other.
    """
    stance_a = str(row_a.get("stance", "Orthodox") or "Orthodox").strip()
    stance_b = str(row_b.get("stance", "Orthodox") or "Orthodox").strip()
    a_unorthodox = stance_a in ("Southpaw", "Switch")
    b_unorthodox = stance_b in ("Southpaw", "Switch")
    if a_unorthodox and not b_unorthodox:
        return STANCE_MISMATCH_BONUS
    if b_unorthodox and not a_unorthodox:
        return -STANCE_MISMATCH_BONUS
    return 0.0


def submission_threat_adjustment(row_a: pd.Series, row_b: pd.Series) -> float:
    """
    A fighter's rate of finishing wins by submission is a distinct skill
    from wrestling_adjustment's takedown-accuracy/control-time focus --
    a fighter can have modest takedown numbers but a live submission
    threat off scrambles, guard, or clinch entries, which wrestling stats
    alone don't capture. Motivated by two real misses on the same card
    where the eventual winner's submission win ended the fight despite
    wrestling_adjustment reading 0.0 for both matchups (neither fighter
    stood out on raw takedown/control numbers, even though a submission
    is exactly how each fight was actually decided) -- a gap this factor
    is meant to close, not a guess without a specific motivating case.
    """
    wins_a = int(_get(row_a, "wins", 0))
    wins_b = int(_get(row_b, "wins", 0))
    # A MISSING sub_wins IS NOT ZERO SUBMISSIONS. Defaulting it to 0 scores a
    # fighter whose method splits never backfilled as having no submission
    # game at all, and hands the difference to the opponent -- so a grappler
    # with an incomplete row is read as the LESS dangerous grappler. Both
    # corners need the split or the term says nothing.
    have_a = row_a.get("sub_wins") is not None and pd.notna(row_a.get("sub_wins"))
    have_b = row_b.get("sub_wins") is not None and pd.notna(row_b.get("sub_wins"))
    if not (have_a and have_b and wins_a and wins_b):
        return 0.0
    sub_rate_a = _get(row_a, "sub_wins", 0) / wins_a
    sub_rate_b = _get(row_b, "sub_wins", 0) / wins_b
    return (sub_rate_a - sub_rate_b) * SUBMISSION_THREAT_SCALE


def style_matchup_adjustment(
    row_a: pd.Series, row_b: pd.Series,
    weight_class_history_df: pd.DataFrame | None = None, this_fight_weight_class: str | None = None,
) -> dict:
    """
    Returns a rating-point adjustment (in favor of fighter A, can be
    negative) plus a breakdown of what drove it, for transparency.
    """
    td_acc_a = _get(row_a, "td_accuracy_pct", 20)
    td_acc_b = _get(row_b, "td_accuracy_pct", 20)
    td_def_a = _get(row_a, "td_defense_pct", 65)
    td_def_b = _get(row_b, "td_defense_pct", 65)
    strike_acc_a = _get(row_a, "strike_accuracy_pct", 45)
    strike_acc_b = _get(row_b, "strike_accuracy_pct", 45)

    def _has(row, col) -> bool:
        """True only when the column holds a real number for this fighter."""
        v = row.get(col)
        return v is not None and v != "" and pd.notna(v)

    # A DIFFERENTIAL BETWEEN A REAL NUMBER AND A DEFAULT IS NOT A DIFFERENTIAL.
    #
    # The _get defaults above exist so a missing column can't crash the model,
    # and when NEITHER fighter has the data they cancel harmlessly: 45 - 45 and
    # max(0, 20 - 65) are both zero. The dangerous case is asymmetry. An
    # established fighter at 52% accuracy against a debutant defaulted to 45%
    # produces a 7-point "striking edge" that is an artifact of who has a data
    # file, not of anything either man does in a cage -- and it always favours
    # the fighter with more history, which is a bias dressed as a signal.
    #
    # That case is not hypothetical: strike_accuracy_pct / td_accuracy_pct /
    # td_defense_pct sit at ~29% roster coverage (0 of 25 on one live card),
    # because their only source is the manual ufcstats scraper, and ufcstats
    # now sits behind a JavaScript challenge that requests+BeautifulSoup
    # cannot pass. Coverage will stay partial even if that is ever solved:
    # ufcstats records UFC bouts only, so debutants have nothing there by
    # definition, and every card has debutants.
    #
    # So each term requires BOTH fighters to have real data or it contributes
    # nothing. Today that makes these terms inert, exactly as they already
    # were in practice. If the data ever returns they light up on their own
    # for the fights that can support them, and stay correctly silent on the
    # ones that can't -- no follow-up change needed.
    striking_data_ok = _has(row_a, "strike_accuracy_pct") and _has(row_b, "strike_accuracy_pct")
    td_acc_data_ok = _has(row_a, "td_accuracy_pct") and _has(row_b, "td_accuracy_pct")
    td_def_data_ok = _has(row_a, "td_defense_pct") and _has(row_b, "td_defense_pct")

    # Striking: accuracy differential, PLUS volume differential (SLpM - SApM)
    # when that data exists. A high-output fighter who lands 45% of a high
    # volume typically outpoints a low-output 60%-accurate fighter on
    # judges' cards -- accuracy alone misses this real, well-documented
    # dynamic. Falls back to accuracy-only when strike-volume data isn't
    # populated yet (graceful no-op, not a guessed number).
    striking_adj = (((strike_acc_a - strike_acc_b) / 100.0) * STRIKING_ADVANTAGE_SCALE
                    if striking_data_ok else 0.0)
    slpm_a, sapm_a = _stat(row_a, "slpm"), _stat(row_a, "sapm")
    slpm_b, sapm_b = _stat(row_b, "slpm"), _stat(row_b, "sapm")
    volume_adj = 0.0
    if pd.notna(slpm_a) and pd.notna(sapm_a) and pd.notna(slpm_b) and pd.notna(sapm_b):
        volume_diff_a = float(slpm_a) - float(sapm_a)
        volume_diff_b = float(slpm_b) - float(sapm_b)
        volume_adj = (volume_diff_a - volume_diff_b) * VOLUME_DIFFERENTIAL_SCALE
    striking_adj += volume_adj

    # Wrestling: prefer CONTROL TIME PERCENTAGE when available -- a fighter
    # who goes 1-for-5 on takedowns but holds 4 minutes of control along the
    # fence is far more effective than raw takedown accuracy alone implies.
    # Falls back to takedown-accuracy-vs-defense when control time isn't
    # populated yet.
    ctrl_a, ctrl_b = row_a.get("control_time_pct"), row_b.get("control_time_pct")
    td_rate_a, td_rate_b = _stat(row_a, "td_per_15"), _stat(row_b, "td_per_15")
    if pd.notna(td_rate_a) and pd.notna(td_rate_b):
        # Takedown RATE differential -- validated on a frozen 2019+ holdout in
        # head_to_head_adjustment.py. Holding every other term fixed and
        # swapping only this one: 57.7% / Brier 0.2365 -> 58.2% / 0.2354, and
        # it won at every blend weight tested from 1.0 to 3.0.
        #
        # Why rate beats the accuracy-vs-defense form it replaces: accuracy is
        # a percentage with no volume in it. A fighter who goes 1-for-1 reads
        # as 100% accurate with almost no wrestling output; one who goes 6-for-12
        # reads as half as good while actually controlling where the fight
        # happens far more. The old form also clipped at zero (max(0, acc - def)),
        # discarding the sign -- so being clearly WORSE at takedowns than the
        # opponent's defence registered identically to being merely equal.
        # Testing the clipping removal on its own made things slightly worse,
        # so the clipping was never the real problem; the input was.
        wrestling_adj = (float(td_rate_a) - float(td_rate_b)) * TD_RATE_ADVANTAGE_SCALE
    elif pd.notna(ctrl_a) and pd.notna(ctrl_b) and td_def_data_ok:
        # td_def_data_ok is required as well: this branch compares each
        # fighter's REAL control time against the OTHER's takedown defence,
        # so a defaulted 65 on one side manufactures the same phantom edge
        # the striking term above guards against.
        # Fallback for fighters with no tracked cage time yet (debutants, and
        # anyone the stats backfill couldn't name-match). Unchanged prior
        # behaviour rather than a guessed rate.
        wrestling_edge_a = max(0.0, float(ctrl_a) - td_def_b) / 100.0
        wrestling_edge_b = max(0.0, float(ctrl_b) - td_def_a) / 100.0
        wrestling_adj = (wrestling_edge_a - wrestling_edge_b) * WRESTLING_ADVANTAGE_SCALE
    elif td_acc_data_ok and td_def_data_ok:
        # Wrestling: A's takedown accuracy vs. B's takedown defense, and vice versa.
        # Only counts as an "edge" if the attacker's accuracy actually exceeds
        # the defender's defense rate -- otherwise no stylistic advantage either way.
        wrestling_edge_a = max(0.0, td_acc_a - td_def_b) / 100.0
        wrestling_edge_b = max(0.0, td_acc_b - td_def_a) / 100.0
        wrestling_adj = (wrestling_edge_a - wrestling_edge_b) * WRESTLING_ADVANTAGE_SCALE
    else:
        # No wrestling data either side can support. Zero, not a guess.
        wrestling_adj = 0.0

    # Durability: how often has each been finished before (by any method)?
    # A high finish-loss rate against someone with strong finishing tools
    # is a real, specific risk -- not just "durability" in the abstract.
    losses_a = max(int(_get(row_a, "losses", 0)), 1) if _get(row_a, "losses", 0) else 1
    losses_b = max(int(_get(row_b, "losses", 0)), 1) if _get(row_b, "losses", 0) else 1
    finish_loss_rate_a = (_get(row_a, "ko_losses", 0) + _get(row_a, "sub_losses", 0)) / losses_a if _get(row_a, "losses", 0) else 0
    finish_loss_rate_b = (_get(row_b, "ko_losses", 0) + _get(row_b, "sub_losses", 0)) / losses_b if _get(row_b, "losses", 0) else 0
    # Same guard, and this one was the most backwards of the three: with
    # ko_losses/sub_losses defaulting to 0, a fighter whose method splits are
    # simply unknown computes a finish-loss rate of zero -- a PERFECT chin --
    # and is handed a durability edge over an opponent with a real, honest
    # record of having been stopped. Unmeasured was scoring better than
    # measured.
    durability_data_ok = all(
        _has(r, c) for r in (row_a, row_b) for c in ("losses", "ko_losses", "sub_losses")
    )
    durability_adj = ((finish_loss_rate_b - finish_loss_rate_a) * DURABILITY_SCALE
                      if durability_data_ok else 0.0)

    layoff_adj_a = layoff_penalty(row_a)
    layoff_adj_b = layoff_penalty(row_b)
    layoff_adj = layoff_adj_a - layoff_adj_b  # penalize A if A has the longer layoff, and vice versa

    quick_return_adj_a = quick_return_penalty(row_a)
    quick_return_adj_b = quick_return_penalty(row_b)
    quick_return_adj = quick_return_adj_a - quick_return_adj_b

    age_cliff_adj_a = age_cliff_penalty(row_a)
    age_cliff_adj_b = age_cliff_penalty(row_b)
    age_cliff_adj = age_cliff_adj_a - age_cliff_adj_b

    missed_weight_adj_a = missed_weight_penalty(row_a)
    missed_weight_adj_b = missed_weight_penalty(row_b)
    missed_weight_adj = missed_weight_adj_a - missed_weight_adj_b

    weight_class_change_adj_a = weight_class_change_penalty(row_a.get("name"), this_fight_weight_class, weight_class_history_df)
    weight_class_change_adj_b = weight_class_change_penalty(row_b.get("name"), this_fight_weight_class, weight_class_history_df)
    weight_class_change_adj = weight_class_change_adj_a - weight_class_change_adj_b

    stance_adj = stance_matchup_adjustment(row_a, row_b)
    submission_threat_adj = submission_threat_adjustment(row_a, row_b)

    height_a = _get(row_a, "height_in", 70)
    height_b = _get(row_b, "height_in", 70)
    height_adj = (height_a - height_b) * HEIGHT_ADVANTAGE_SCALE

    short_notice_a = bool(_get(row_a, "short_notice", 0))
    short_notice_b = bool(_get(row_b, "short_notice", 0))
    short_notice_adj = (SHORT_NOTICE_PENALTY if short_notice_b else 0.0) - (SHORT_NOTICE_PENALTY if short_notice_a else 0.0)

    # SCALED BY SAMPLE SIZE -- but only the two terms actually built from rate
    # statistics. Height, age, layoff, short notice, missed weight and stance
    # are biographical facts that a debutant knows about himself as precisely
    # as a veteran does; discounting them for inexperience would be wrong.
    # Durability and submission threat come from the win/loss RECORD, which is
    # a count rather than a per-minute rate, so a short career makes them
    # noisy in the ordinary way rather than systematically extreme -- a
    # different problem, left alone here rather than swept in.
    rate_conf = rate_stat_confidence(row_a, row_b)
    striking_adj *= rate_conf
    wrestling_adj *= rate_conf

    total_adj = (
        wrestling_adj + striking_adj + durability_adj + layoff_adj
        + quick_return_adj + age_cliff_adj + missed_weight_adj + weight_class_change_adj
        + stance_adj + submission_threat_adj + height_adj + short_notice_adj
    )

    return {
        "total_adjustment": total_adj,
        # Exposed so the waterfall can say WHY a style edge is small on a
        # thin-sample fight, rather than the number just quietly being
        # unimpressive with no explanation.
        "rate_stat_confidence": rate_conf,
        "wrestling_adjustment": wrestling_adj,
        "striking_adjustment": striking_adj,
        "durability_adjustment": durability_adj,
        "submission_threat_adjustment": submission_threat_adj,
        "height_adjustment": height_adj,
        "short_notice_adjustment": short_notice_adj,
        "short_notice_flag_a": short_notice_a,
        "short_notice_flag_b": short_notice_b,
        "layoff_adjustment": layoff_adj,
        "layoff_years_a": layoff_years(row_a),
        "layoff_years_b": layoff_years(row_b),
        "quick_return_adjustment": quick_return_adj,
        "quick_return_flag_a": quick_return_adj_a < 0,
        "quick_return_flag_b": quick_return_adj_b < 0,
        "age_cliff_adjustment": age_cliff_adj,
        "age_cliff_flag_a": age_cliff_adj_a < 0,
        "age_cliff_flag_b": age_cliff_adj_b < 0,
        "missed_weight_adjustment": missed_weight_adj,
        "weight_class_change_adjustment": weight_class_change_adj,
        "weight_class_change_flag_a": weight_class_change_adj_a != 0,
        "weight_class_change_flag_b": weight_class_change_adj_b != 0,
        "weight_class_change_direction_a": "up" if weight_class_change_adj_a < 0 else ("down" if weight_class_change_adj_a > 0 else None),
        "weight_class_change_direction_b": "up" if weight_class_change_adj_b < 0 else ("down" if weight_class_change_adj_b > 0 else None),
        "stance_adjustment": stance_adj,
        "style_a": classify_style(row_a),
        "style_b": classify_style(row_b),
    }


def recent_form_adjustment(
    fighter_a: str, fighter_b: str, fight_history_df: pd.DataFrame | None,
    reference_date: dt.date | None = None,
) -> float:
    """
    A genuine but partial recency signal: fighters.csv only tracks
    aggregate career win/loss counts, not dated per-fight records, so a
    fully recency-weighted career rating isn't possible for the roster as
    a whole with current data. This instead looks at each fighter's most
    recent entry in fight_history.csv specifically (if they have one) --
    a small, decaying bonus for a recent win, a small penalty for a
    recent loss, weighted by how long ago it was. Fighters with no
    tracked history get exactly 0 here, same as every other graceful
    fallback in this file -- this is a real but limited signal, not a
    substitute for the full recency-weighted system a richer dataset
    would support.
    """
    if fight_history_df is None or fight_history_df.empty:
        return 0.0
    reference_date = reference_date or dt.date.today()

    def fighter_signal(name: str) -> float:
        rows = fight_history_df[
            (fight_history_df["fighter_a"] == name) | (fight_history_df["fighter_b"] == name)
        ].copy()
        if rows.empty:
            return 0.0
        rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
        rows = rows.dropna(subset=["date"]).sort_values("date")
        if rows.empty:
            return 0.0
        # Last-3-fights decayed form, VALIDATED on the July 2026 walk-forward
        # backtest: formulation + scale selected on pre-2019 fights, confirmed
        # on held-out 2019+ fights, where it beat both no-form-at-all and the
        # previous single-most-recent-fight version this replaces.
        signal = 0.0
        for _, last in rows.tail(RECENT_FORM_LOOKBACK).iterrows():
            won = last["winner"] == name
            years_ago = max((reference_date - last["date"].date()).days / 365.25, 0.0)
            decay = max(0.0, 1.0 - years_ago / RECENT_FORM_DECAY_YEARS)
            signal += (1.0 if won else -1.0) * decay * RECENT_FORM_SCALE
        return signal

    return fighter_signal(fighter_a) - fighter_signal(fighter_b)


def predict_matchup(
    fighter_a: str, fighter_b: str,
    fighters_df: pd.DataFrame,
    effective_ratings: dict[str, float],
    fight_history_df: pd.DataFrame | None = None,
    weight_class_history_df: pd.DataFrame | None = None,
    fight_weight_class: str | None = None,
) -> dict | None:
    """
    Full pairwise prediction: base rating gap + style-matchup adjustment,
    converted to a win probability, with a breakdown for the UI to explain.
    """
    match_a = fighters_df[fighters_df["name"] == fighter_a]
    match_b = fighters_df[fighters_df["name"] == fighter_b]
    if match_a.empty or match_b.empty:
        return None
    row_a, row_b = match_a.iloc[0], match_b.iloc[0]

    base_r_a = effective_ratings.get(fighter_a, 1500.0)
    base_r_b = effective_ratings.get(fighter_b, 1500.0)

    # Prefer the caller's card-specific division when they have it (the
    # actual booked weight class for this fight) over fighters_df's own
    # weight_class column, which reflects "most recently known division"
    # and could in principle drift from what a specific card says.
    this_fight_wc = fight_weight_class or row_a.get("weight_class")
    style = style_matchup_adjustment(row_a, row_b, weight_class_history_df, this_fight_wc)
    recent_form_adj = recent_form_adjustment(fighter_a, fighter_b, fight_history_df)
    raw_layer = style["total_adjustment"] + recent_form_adj
    applied_layer = max(-ADJUSTMENT_TOTAL_CAP, min(ADJUSTMENT_TOTAL_CAP, raw_layer))
    adjusted_gap = (base_r_a - base_r_b) + applied_layer
    prob_a = 1.0 / (1.0 + 10 ** (-adjusted_gap / 400.0))

    # Uncertainty band: this is a heuristic, not a fitted confidence
    # interval (that would need a proper Bayesian treatment or bootstrap
    # over historical outcomes, which the current data doesn't support).
    # It scales down as the THINNER of the two records grows -- a matchup
    # where one side has 2 fights should visibly carry more uncertainty
    # than one where both have 25, even if the point estimate is
    # identical. Floors around ~5pp even for deep records, since MMA has
    # real irreducible variance no amount of data fully removes.
    fights_a = int(_get(row_a, "wins", 0)) + int(_get(row_a, "losses", 0))
    fights_b = int(_get(row_b, "wins", 0)) + int(_get(row_b, "losses", 0))
    thinner_record = min(fights_a, fights_b)
    uncertainty = UNCERTAINTY_BASE / math.sqrt(thinner_record + 1)
    prob_low = max(0.01, prob_a - uncertainty)
    prob_high = min(0.99, prob_a + uncertainty)

    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "prob_a": prob_a,
        "prob_b": 1 - prob_a,
        "prob_low": prob_low,
        "prob_high": prob_high,
        "base_rating_a": base_r_a,
        "base_rating_b": base_r_b,
        "recent_form_adjustment": recent_form_adj,
        "adjustment_layer_raw": raw_layer,
        "adjustment_layer_applied": applied_layer,
        "adjustment_capped": applied_layer != raw_layer,
        **style,
    }
