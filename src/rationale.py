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


def explain_moneyline(row: dict, fighters_df: pd.DataFrame) -> str:
    stats = _fighter_stats(fighters_df, row["fighter"])
    edge_dir = "higher" if row["edge_pct"] > 0 else "lower"
    base = (
        f"The model puts {row['fighter']}'s win probability at {row['model_prob']*100:.0f}%, "
        f"{edge_dir} than the market's {row['book_fair_prob']*100:.0f}% implied probability "
        f"at {row['odds_american']} ({row['edge_pct']:+.1f}% edge)."
    )
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


def explain_edge(row: dict, fighters_df: pd.DataFrame) -> str:
    if row["market"] == "Moneyline":
        return explain_moneyline(row, fighters_df)
    elif row["market"].startswith("Method"):
        return explain_method(row, fighters_df)
    elif row["market"].startswith("Total Rounds"):
        return explain_total_rounds(row, fighters_df)
    return f"{row['fighter']} — {row['market']}: {row['edge_pct']:+.1f}% edge vs. the market."
