"""
Combines the Elo model + fighter finish-rate stats with sportsbook lines
to surface bets where the model disagrees most with the market.

This is a decision-SUPPORT tool, not a decision-maker. Edges are only as
good as the historical data feeding the model -- always sanity check
against recent form, injuries, weight cuts, camp changes, etc. that a
pure stats model can't see.
"""

import pandas as pd

from .odds_utils import american_to_implied_prob, remove_vig_two_way, edge_percent, kelly_fraction


def compute_moneyline_edges(upcoming_df: pd.DataFrame, elo_ratings: dict[str, float]) -> pd.DataFrame:
    rows = []
    ml = upcoming_df[upcoming_df["market"] == "Moneyline"]

    for fight_id, group in ml.groupby("fight_id"):
        if len(group) != 2:
            continue  # need both sides of the moneyline to devig

        a, b = group.iloc[0], group.iloc[1]
        elo_a = elo_ratings.get(a["selection"], 1500.0)
        elo_b = elo_ratings.get(b["selection"], 1500.0)

        model_prob_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
        model_prob_b = 1.0 - model_prob_a

        imp_a = american_to_implied_prob(a["odds_american"])
        imp_b = american_to_implied_prob(b["odds_american"])
        fair_a, fair_b = remove_vig_two_way(imp_a, imp_b)

        for fighter, model_p, fair_p, odds in [
            (a["selection"], model_prob_a, fair_a, a["odds_american"]),
            (b["selection"], model_prob_b, fair_b, b["odds_american"]),
        ]:
            rows.append({
                "fight_id": fight_id,
                "fighter": fighter,
                "market": "Moneyline",
                "odds_american": odds,
                "model_prob": round(model_p, 3),
                "book_fair_prob": round(fair_p, 3),
                "edge_pct": round(edge_percent(model_p, fair_p), 2),
                "half_kelly_stake_pct": round(kelly_fraction(model_p, odds) * 100, 2),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def compute_method_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Method-of-victory props (KO/TKO, Submission, Decision), priced off each
    fighter's historical finish rate. Simplified on purpose -- treat as a
    first-pass screen, not gospel.
    """
    rows = []
    props = upcoming_df[upcoming_df["market"] == "Method"]

    for _, row in props.iterrows():
        stats = fighters_df[fighters_df["name"] == row["selection"]]
        if stats.empty:
            continue
        f = stats.iloc[0]
        total_wins = max(int(f["wins"]), 1)

        rate_map = {
            "KO/TKO": f["ko_wins"] / total_wins,
            "SUB": f["sub_wins"] / total_wins,
            "DEC": f["dec_wins"] / total_wins,
        }
        model_p = rate_map.get(row["selection_method"])
        if model_p is None:
            continue

        imp = american_to_implied_prob(row["odds_american"])

        rows.append({
            "fight_id": row["fight_id"],
            "fighter": row["selection"],
            "market": f"Method: {row['selection_method']}",
            "odds_american": row["odds_american"],
            "model_prob": round(model_p, 3),
            "book_fair_prob": round(imp, 3),  # not devigged (single-sided prop)
            "edge_pct": round(edge_percent(model_p, imp), 2),
            "half_kelly_stake_pct": round(kelly_fraction(model_p, row["odds_american"]) * 100, 2),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def compute_total_rounds_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Over/Under total rounds props, estimated from each fighter's historical
    finish rate (frequent finishers -> fewer rounds; frequent decisions ->
    more rounds). Very simplified -- a good candidate to improve first with
    real fight-length data instead of just win-method proxies.
    """
    rows = []
    props = upcoming_df[upcoming_df["market"] == "TotalRounds"]

    for fight_id, group in props.groupby("fight_id"):
        fighters_in_fight = group["fighter_a"].iloc[0], group["fighter_b"].iloc[0]
        finish_rates = []
        for name in fighters_in_fight:
            stats = fighters_df[fighters_df["name"] == name]
            if stats.empty:
                continue
            f = stats.iloc[0]
            total_wins = max(int(f["wins"]), 1)
            finish_rate = (f["ko_wins"] + f["sub_wins"]) / total_wins
            finish_rates.append(finish_rate)

        if not finish_rates:
            continue

        combined_finish_rate = sum(finish_rates) / len(finish_rates)
        # crude heuristic: high combined finish rate -> lean Under
        model_prob_under = min(0.85, max(0.15, combined_finish_rate))

        for _, row in group.iterrows():
            model_p = model_prob_under if row["selection"] == "Under" else (1 - model_prob_under)
            imp = american_to_implied_prob(row["odds_american"])
            rows.append({
                "fight_id": fight_id,
                "fighter": f"{fighters_in_fight[0]} vs {fighters_in_fight[1]}",
                "market": f"Total Rounds {row['selection']}",
                "odds_american": row["odds_american"],
                "model_prob": round(model_p, 3),
                "book_fair_prob": round(imp, 3),
                "edge_pct": round(edge_percent(model_p, imp), 2),
                "half_kelly_stake_pct": round(kelly_fraction(model_p, row["odds_american"]) * 100, 2),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def find_all_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame, elo_ratings: dict[str, float]) -> pd.DataFrame:
    frames = [
        compute_moneyline_edges(upcoming_df, elo_ratings),
        compute_method_edges(upcoming_df, fighters_df),
        compute_total_rounds_edges(upcoming_df, fighters_df),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("edge_pct", ascending=False).reset_index(drop=True)
