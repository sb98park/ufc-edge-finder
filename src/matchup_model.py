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

import pandas as pd

# How many Elo-equivalent rating points a fully-realized stylistic
# advantage is worth. Tuned to be meaningful but not dominate the base
# rating gap entirely -- these are secondary signals, not the headline.
WRESTLING_ADVANTAGE_SCALE = 300.0
STRIKING_ADVANTAGE_SCALE = 150.0
DURABILITY_SCALE = 120.0
VOLUME_DIFFERENTIAL_SCALE = 40.0  # rating points per 1.0 SLpM-SApM differential gap

# Ring rust: no penalty for a normal 6-12 month camp cycle. Beyond a year
# away, each additional year away costs more -- extended layoffs (multi-year,
# often tied to serious injury) are a real, well-documented risk factor in
# combat sports, not just "conventional wisdom."
LAYOFF_GRACE_YEARS = 1.0
LAYOFF_PENALTY_PER_YEAR = 60.0
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


def classify_style(row: pd.Series) -> str:
    td_acc = _get(row, "td_accuracy_pct", 20)
    strike_acc = _get(row, "strike_accuracy_pct", 45)
    if td_acc >= 40:
        return "Wrestler/Grappler"
    elif strike_acc >= 47:
        return "Striker"
    return "Balanced"


def style_matchup_adjustment(row_a: pd.Series, row_b: pd.Series) -> dict:
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

    # Striking: accuracy differential, PLUS volume differential (SLpM - SApM)
    # when that data exists. A high-output fighter who lands 45% of a high
    # volume typically outpoints a low-output 60%-accurate fighter on
    # judges' cards -- accuracy alone misses this real, well-documented
    # dynamic. Falls back to accuracy-only when strike-volume data isn't
    # populated yet (graceful no-op, not a guessed number).
    striking_adj = ((strike_acc_a - strike_acc_b) / 100.0) * STRIKING_ADVANTAGE_SCALE
    slpm_a, sapm_a = row_a.get("slpm"), row_a.get("sapm")
    slpm_b, sapm_b = row_b.get("slpm"), row_b.get("sapm")
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
    if pd.notna(ctrl_a) and pd.notna(ctrl_b):
        wrestling_edge_a = max(0.0, float(ctrl_a) - td_def_b) / 100.0
        wrestling_edge_b = max(0.0, float(ctrl_b) - td_def_a) / 100.0
    else:
        # Wrestling: A's takedown accuracy vs. B's takedown defense, and vice versa.
        # Only counts as an "edge" if the attacker's accuracy actually exceeds
        # the defender's defense rate -- otherwise no stylistic advantage either way.
        wrestling_edge_a = max(0.0, td_acc_a - td_def_b) / 100.0
        wrestling_edge_b = max(0.0, td_acc_b - td_def_a) / 100.0
    wrestling_adj = (wrestling_edge_a - wrestling_edge_b) * WRESTLING_ADVANTAGE_SCALE

    # Durability: how often has each been finished before (by any method)?
    # A high finish-loss rate against someone with strong finishing tools
    # is a real, specific risk -- not just "durability" in the abstract.
    losses_a = max(int(row_a.get("losses", 0)), 1) if row_a.get("losses", 0) else 1
    losses_b = max(int(row_b.get("losses", 0)), 1) if row_b.get("losses", 0) else 1
    finish_loss_rate_a = (row_a.get("ko_losses", 0) + row_a.get("sub_losses", 0)) / losses_a if row_a.get("losses", 0) else 0
    finish_loss_rate_b = (row_b.get("ko_losses", 0) + row_b.get("sub_losses", 0)) / losses_b if row_b.get("losses", 0) else 0
    durability_adj = (finish_loss_rate_b - finish_loss_rate_a) * DURABILITY_SCALE

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

    total_adj = (
        wrestling_adj + striking_adj + durability_adj + layoff_adj
        + quick_return_adj + age_cliff_adj + missed_weight_adj
    )

    return {
        "total_adjustment": total_adj,
        "wrestling_adjustment": wrestling_adj,
        "striking_adjustment": striking_adj,
        "durability_adjustment": durability_adj,
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
        "style_a": classify_style(row_a),
        "style_b": classify_style(row_b),
    }


def predict_matchup(
    fighter_a: str, fighter_b: str,
    fighters_df: pd.DataFrame,
    effective_ratings: dict[str, float],
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

    style = style_matchup_adjustment(row_a, row_b)
    adjusted_gap = (base_r_a - base_r_b) + style["total_adjustment"]
    prob_a = 1.0 / (1.0 + 10 ** (-adjusted_gap / 400.0))

    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "prob_a": prob_a,
        "prob_b": 1 - prob_a,
        "base_rating_a": base_r_a,
        "base_rating_b": base_r_b,
        **style,
    }
