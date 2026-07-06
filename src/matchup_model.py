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

import pandas as pd

# How many Elo-equivalent rating points a fully-realized stylistic
# advantage is worth. Tuned to be meaningful but not dominate the base
# rating gap entirely -- these are secondary signals, not the headline.
WRESTLING_ADVANTAGE_SCALE = 300.0
STRIKING_ADVANTAGE_SCALE = 150.0
DURABILITY_SCALE = 120.0


def _get(row: pd.Series, col: str, default: float) -> float:
    return float(row[col]) if col in row and pd.notna(row[col]) else default


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

    # Wrestling: A's takedown accuracy vs. B's takedown defense, and vice versa.
    # Only counts as an "edge" if the attacker's accuracy actually exceeds
    # the defender's defense rate -- otherwise no stylistic advantage either way.
    wrestling_edge_a = max(0.0, td_acc_a - td_def_b) / 100.0
    wrestling_edge_b = max(0.0, td_acc_b - td_def_a) / 100.0
    wrestling_adj = (wrestling_edge_a - wrestling_edge_b) * WRESTLING_ADVANTAGE_SCALE

    # Striking: simple accuracy differential
    striking_adj = ((strike_acc_a - strike_acc_b) / 100.0) * STRIKING_ADVANTAGE_SCALE

    # Durability: how often has each been finished before (by any method)?
    # A high finish-loss rate against someone with strong finishing tools
    # is a real, specific risk -- not just "durability" in the abstract.
    losses_a = max(int(row_a.get("losses", 0)), 1) if row_a.get("losses", 0) else 1
    losses_b = max(int(row_b.get("losses", 0)), 1) if row_b.get("losses", 0) else 1
    finish_loss_rate_a = (row_a.get("ko_losses", 0) + row_a.get("sub_losses", 0)) / losses_a if row_a.get("losses", 0) else 0
    finish_loss_rate_b = (row_b.get("ko_losses", 0) + row_b.get("sub_losses", 0)) / losses_b if row_b.get("losses", 0) else 0
    durability_adj = (finish_loss_rate_b - finish_loss_rate_a) * DURABILITY_SCALE

    total_adj = wrestling_adj + striking_adj + durability_adj

    return {
        "total_adjustment": total_adj,
        "wrestling_adjustment": wrestling_adj,
        "striking_adjustment": striking_adj,
        "durability_adjustment": durability_adj,
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
