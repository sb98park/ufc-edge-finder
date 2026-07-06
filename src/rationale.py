"""
Turns a computed edge row into a plain-English explanation of what's
actually driving the model's number -- so a flagged prop isn't just a
mystery percentage, you can see the stat behind it and judge for yourself
whether it's a real signal or a model blind spot.
"""

import pandas as pd


def _fighter_stats(fighters_df: pd.DataFrame, name: str) -> dict | None:
    row = fighters_df[fighters_df["name"] == name]
    if row.empty:
        return None
    r = row.iloc[0]
    total_wins = max(int(r["wins"]), 1)
    total_fights = max(int(r["wins"]) + int(r["losses"]), 1)
    return {
        "win_pct": r["wins"] / total_fights,
        "finish_rate": (r["ko_wins"] + r["sub_wins"]) / total_wins,
        "ko_rate": r["ko_wins"] / total_wins,
        "sub_rate": r["sub_wins"] / total_wins,
        "dec_rate": r["dec_wins"] / total_wins,
        "reach_in": r["reach_in"],
        "wins": int(r["wins"]),
        "losses": int(r["losses"]),
    }


from src.matchup_model import predict_matchup


def explain_moneyline(row: dict, fighters_df: pd.DataFrame) -> str:
    stats = _fighter_stats(fighters_df, row["fighter"])
    edge_dir = "higher" if row["edge_pct"] > 0 else "lower"
    base = (
        f"The model puts {row['fighter']}'s win probability at {row['model_prob']*100:.0f}%, "
        f"{edge_dir} than the market's {row['book_fair_prob']*100:.0f}% implied probability "
        f"at {row['odds_american']} ({row['edge_pct']:+.1f}% edge)."
    )

    opponent = row.get("opponent")
    if opponent:
        matchup = predict_matchup(row["fighter"], opponent, fighters_df, {})
        # predict_matchup needs effective_ratings for the base gap, but we don't
        # have that here -- just the style breakdown, which doesn't depend on it
        if matchup:
            drivers = []
            if abs(matchup["wrestling_adjustment"]) > 15:
                who = row["fighter"] if matchup["wrestling_adjustment"] > 0 else opponent
                drivers.append(f"{who}'s takedown accuracy vs. the opponent's takedown defense")
            if abs(matchup["striking_adjustment"]) > 10:
                who = row["fighter"] if matchup["striking_adjustment"] > 0 else opponent
                drivers.append(f"{who}'s striking accuracy edge")
            if abs(matchup["durability_adjustment"]) > 15:
                who = row["fighter"] if matchup["durability_adjustment"] > 0 else opponent
                drivers.append(f"{who} having been finished less often historically")
            layoff_a = matchup.get("layoff_years_a")
            layoff_b = matchup.get("layoff_years_b")
            if layoff_a and layoff_a > 1.0:
                drivers.append(f"{row['fighter']} coming off a {layoff_a:.1f}-year layoff (ring rust risk)")
            if layoff_b and layoff_b > 1.0:
                drivers.append(f"{opponent} coming off a {layoff_b:.1f}-year layoff (ring rust risk)")

            if drivers:
                base += f" Biggest factors in that number: {', '.join(drivers)}."
            elif stats:
                base += (
                    f" That's built on a {stats['wins']}-{stats['losses']} record "
                    f"({stats['win_pct']*100:.0f}% win rate) and a {stats['finish_rate']*100:.0f}% finish rate, "
                    f"with no major style, durability, or layoff mismatch pulling the number further."
                )
            return base

    if stats:
        base += (
            f" That's built on a {stats['wins']}-{stats['losses']} record "
            f"({stats['win_pct']*100:.0f}% win rate) and a {stats['finish_rate']*100:.0f}% finish rate."
        )
    return base


def explain_method(row: dict, fighters_df: pd.DataFrame) -> str:
    stats = _fighter_stats(fighters_df, row["fighter"])
    method = row["market"].replace("Method: ", "")
    base = (
        f"{row['fighter']} to win by {method} is priced at {row['odds_american']} "
        f"({row['book_fair_prob']*100:.0f}% implied), while the model estimates {row['model_prob']*100:.0f}% "
        f"({row['edge_pct']:+.1f}% edge)."
    )
    if stats:
        base += (
            f" That blends their career tendency (KO/TKO in {stats['ko_rate']*100:.0f}% of wins, "
            f"submission in {stats['sub_rate']*100:.0f}%, decision in {stats['dec_rate']*100:.0f}%) "
            f"with how often this specific opponent has actually lost that way before -- "
            f"a fighter's finishing rate matters less if the person across from them has never "
            f"been finished that way."
        )
    return base


def explain_total_rounds(row: dict, fighters_df: pd.DataFrame) -> str:
    names = row["fighter"].split(" vs ")
    finish_rates = []
    for name in names:
        s = _fighter_stats(fighters_df, name.strip())
        if s:
            finish_rates.append(s["finish_rate"])

    base = (
        f"{row['market']} at {row['odds_american']} implies {row['book_fair_prob']*100:.0f}%, "
        f"vs. the model's {row['model_prob']*100:.0f}% ({row['edge_pct']:+.1f}% edge)."
    )
    if finish_rates:
        avg_finish = sum(finish_rates) / len(finish_rates)
        base += (
            f" This leans on a combined {avg_finish*100:.0f}% finish rate between both fighters — "
            f"a simplified proxy for fight length, not a real per-round simulation."
        )
    return base


def explain_goes_the_distance(row: dict, fighters_df: pd.DataFrame) -> str:
    names = row["fighter"].split(" vs ")
    dec_rates = []
    for name in names:
        s = _fighter_stats(fighters_df, name.strip())
        if s:
            dec_rates.append(s["dec_rate"])

    base = (
        f"{row['market']} at {row['odds_american']} implies {row['book_fair_prob']*100:.0f}%, "
        f"vs. the model's {row['model_prob']*100:.0f}% ({row['edge_pct']:+.1f}% edge)."
    )
    if dec_rates:
        avg_dec = sum(dec_rates) / len(dec_rates)
        base += f" Based on both fighters' career decision rate averaging {avg_dec*100:.0f}%."
    return base


def explain_edge(row: dict, fighters_df: pd.DataFrame) -> str:
    if row["market"] == "Moneyline":
        return explain_moneyline(row, fighters_df)
    elif row["market"].startswith("Method"):
        return explain_method(row, fighters_df)
    elif row["market"].startswith("Total Rounds"):
        return explain_total_rounds(row, fighters_df)
    elif row["market"].startswith("Fight Outcome"):
        return explain_goes_the_distance(row, fighters_df)
    return f"{row['fighter']} — {row['market']}: {row['edge_pct']:+.1f}% edge vs. the market."
