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


def _method_vulnerability_blend(fighter_row: pd.Series, opponent_row: pd.Series, method: str) -> float:
    """Same 65/35 blend used in edge_finder.compute_method_edges, but usable without a live line."""
    total_wins = max(int(fighter_row["wins"]), 1)
    rate_map = {
        "KO/TKO": fighter_row["ko_wins"] / total_wins,
        "Submission": fighter_row["sub_wins"] / total_wins,
        "Decision": fighter_row["dec_wins"] / total_wins,
    }
    own_rate = rate_map[method]

    opp_losses = max(int(opponent_row["losses"]), 1) if opponent_row["losses"] else 0
    if not opp_losses:
        return own_rate
    loss_col = {"KO/TKO": "ko_losses", "Submission": "sub_losses", "Decision": "dec_losses"}[method]
    opp_vulnerability = opponent_row[loss_col] / opp_losses
    return 0.65 * own_rate + 0.35 * opp_vulnerability


def build_full_market_projection(
    fighter_a: str, fighter_b: str,
    fighters_df: pd.DataFrame,
    effective_ratings: dict[str, float],
) -> dict | None:
    """
    Model-only projections for method-of-victory (both fighters, all three
    methods) and total rounds -- shown even when the live book doesn't
    happen to offer that market for this particular fight, clearly labeled
    as a projection rather than an odds comparison.
    """
    row_a, row_b = _fighter_row(fighters_df, fighter_a), _fighter_row(fighters_df, fighter_b)
    if row_a is None or row_b is None:
        return None

    matchup = predict_matchup(fighter_a, fighter_b, fighters_df, effective_ratings)
    prob_a, prob_b = matchup["prob_a"], matchup["prob_b"]

    method_rows = []
    for name, row, opp_row, win_prob in [
        (fighter_a, row_a, row_b, prob_a), (fighter_b, row_b, row_a, prob_b)
    ]:
        for method in ["KO/TKO", "Submission", "Decision"]:
            method_given_win = _method_vulnerability_blend(row, opp_row, method)
            combined_prob = win_prob * method_given_win
            method_rows.append({
                "fighter": name, "market": f"Method: {method}", "model_prob": round(combined_prob, 3),
            })

    combined_finish_rate = (
        (row_a["ko_wins"] + row_a["sub_wins"]) / max(int(row_a["wins"]), 1)
        + (row_b["ko_wins"] + row_b["sub_wins"]) / max(int(row_b["wins"]), 1)
    ) / 2

    first_round_rates = [
        float(r["first_round_finish_pct"]) for r in (row_a, row_b)
        if "first_round_finish_pct" in r and pd.notna(r["first_round_finish_pct"])
    ]
    combined_first_round_rate = sum(first_round_rates) / len(first_round_rates) if first_round_rates else combined_finish_rate * 0.5

    rounds_2_5 = 0.7 * combined_finish_rate + 0.3 * combined_first_round_rate
    rounds_rows = [
        {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Under 1.5", "model_prob": round(combined_first_round_rate, 3)},
        {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Over 1.5", "model_prob": round(1 - combined_first_round_rate, 3)},
        {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Under 2.5", "model_prob": round(rounds_2_5, 3)},
        {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Over 2.5", "model_prob": round(1 - rounds_2_5, 3)},
    ]

    dec_rate_a = row_a["dec_wins"] / max(int(row_a["wins"]), 1)
    dec_rate_b = row_b["dec_wins"] / max(int(row_b["wins"]), 1)
    goes_distance_prob = (dec_rate_a + dec_rate_b) / 2
    distance_rows = [
        {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Fight Outcome: Goes The Distance", "model_prob": round(goes_distance_prob, 3)},
        {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Fight Outcome: Ends In Finish", "model_prob": round(1 - goes_distance_prob, 3)},
    ]

    return {"method_rows": method_rows, "rounds_rows": rounds_rows, "distance_rows": distance_rows}


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
            f" {stronger_wrestler}'s takedown accuracy against the opponent's takedown defense "
            f"gives a real wrestling-based edge here."
        )
    elif abs(matchup["durability_adjustment"]) > 15:
        more_durable = fighter_a if matchup["durability_adjustment"] > 0 else fighter_b
        style_note = f" {more_durable} has been finished less often historically, a durability factor working in their favor."

    layoff_note = ""
    for name, yrs in [(fighter_a, matchup["layoff_years_a"]), (fighter_b, matchup["layoff_years_b"])]:
        if yrs and yrs > 1.0:
            layoff_note += (
                f" {name} is returning from a {yrs:.1f}-year layoff, which carries real ring-rust risk "
                f"regardless of what their career numbers say."
            )

    reach_diff = row_a["reach_in"] - row_b["reach_in"]
    reach_note = ""
    if abs(reach_diff) >= 4:
        longer = fighter_a if reach_diff > 0 else fighter_b
        reach_note = f" {longer} also holds a notable reach advantage ({abs(reach_diff):.0f} inches)."

    fast_finisher_note = ""
    for name, row in [(fighter_a, row_a), (fighter_b, row_b)]:
        rate = row.get("first_round_finish_pct")
        if pd.notna(rate) and rate >= 0.6:
            fast_finisher_note += (
                f" {name} is a genuine round-1 threat — {rate*100:.0f}% of their career wins have come "
                f"before the first round even ends, which should pull any rounds/distance line lower "
                f"regardless of who's favored to win outright."
            )

    narrative = (
        f"Model favors {favorite} at {favorite_prob*100:.0f}% over {underdog} "
        f"({matchup['style_a']} vs. {matchup['style_b']} stylistically). "
        f"Path to victory most likely runs through {likely_method.lower()} "
        f"({method_rates[likely_method]*100:.0f}% of {favorite.split()[-1]}'s career wins). "
        f"Combined finish rate between both fighters sits at {combined_finish_rate*100:.0f}%, "
        f"leaning {rounds_lean.lower()} on total rounds."
        f"{style_note}{reach_note}{layoff_note}{fast_finisher_note}"
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
