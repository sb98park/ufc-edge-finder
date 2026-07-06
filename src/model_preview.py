"""
DraftKings doesn't always return every market, and The Odds API only has
moneylines for MMA at all. So instead of method-of-victory and round-total
insight disappearing whenever live odds don't cover it, this builds a
model-only projection straight from career stats -- always available,
clearly labeled as a projection rather than a live-market edge.
"""

import pandas as pd

from src.matchup_model import predict_matchup


def _fighter_row(fighters_df: pd.DataFrame, name: str) -> pd.Series | None:
    match = fighters_df[fighters_df["name"] == name]
    return match.iloc[0] if not match.empty else None


def build_fight_preview(
    fighter_a: str, fighter_b: str,
    fighters_df: pd.DataFrame,
    effective_ratings: dict[str, float],
) -> dict | None:
    row_a, row_b = _fighter_row(fighters_df, fighter_a), _fighter_row(fighters_df, fighter_b)
    if row_a is None or row_b is None:
        return None

    matchup = predict_matchup(fighter_a, fighter_b, fighters_df, effective_ratings)
    prob_a = matchup["prob_a"]

    favorite, favorite_prob, underdog = (
        (fighter_a, prob_a, fighter_b) if prob_a >= 0.5 else (fighter_b, 1 - prob_a, fighter_a)
    )
    favorite_row = row_a if favorite == fighter_a else row_b

    total_wins = max(int(favorite_row["wins"]), 1)
    method_rates = {
        "KO/TKO": favorite_row["ko_wins"] / total_wins,
        "Submission": favorite_row["sub_wins"] / total_wins,
        "Decision": favorite_row["dec_wins"] / total_wins,
    }
    likely_method = max(method_rates, key=method_rates.get)

    combined_finish_rate = (
        (row_a["ko_wins"] + row_a["sub_wins"]) / max(int(row_a["wins"]), 1)
        + (row_b["ko_wins"] + row_b["sub_wins"]) / max(int(row_b["wins"]), 1)
    ) / 2
    rounds_lean = "Under" if combined_finish_rate >= 0.5 else "Over"

    style_note = ""
    if abs(matchup["wrestling_adjustment"]) > 15:
        stronger_wrestler = fighter_a if matchup["wrestling_adjustment"] > 0 else fighter_b
        style_note = (
            f" Style note: {stronger_wrestler}'s takedown accuracy vs. the opponent's takedown "
            f"defense gives a real wrestling-based edge here."
        )
    elif abs(matchup["durability_adjustment"]) > 15:
        more_durable = fighter_a if matchup["durability_adjustment"] > 0 else fighter_b
        style_note = f" Style note: {more_durable} has been finished less often historically, a durability factor."

    narrative = (
        f"Model favors {favorite} at {favorite_prob*100:.0f}% to beat {underdog} "
        f"({matchup['style_a']} vs. {matchup['style_b']}). "
        f"If {favorite} wins, {likely_method.lower()} is the most likely path "
        f"({method_rates[likely_method]*100:.0f}% of their career wins have come that way). "
        f"Combined finish rate between both fighters is {combined_finish_rate*100:.0f}%, "
        f"leaning {rounds_lean.lower()} on total rounds.{style_note}"
    )

    return {
        "favorite": favorite,
        "favorite_prob": round(favorite_prob, 3),
        "underdog": underdog,
        "likely_method": likely_method,
        "likely_method_rate": round(method_rates[likely_method], 3),
        "rounds_lean": rounds_lean,
        "combined_finish_rate": round(combined_finish_rate, 3),
        "style_a": matchup["style_a"],
        "style_b": matchup["style_b"],
        "narrative": narrative,
    }
