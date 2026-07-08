"""
Turns a computed edge row into a plain-English explanation of what's
actually driving the model's number -- so a flagged prop isn't just a
mystery percentage, you can see the stat behind it and judge for yourself
whether it's a real signal or a model blind spot.
"""

import pandas as pd

from src.odds_utils import format_american_odds


def _fighter_stats(fighters_df: pd.DataFrame, name: str) -> dict | None:
    row = fighters_df[fighters_df["name"] == name]
    if row.empty:
        return None
    r = row.iloc[0]
    total_wins = max(int(r["wins"]), 1)
    total_losses = max(int(r["losses"]), 1) if r["losses"] else 0
    total_fights = max(int(r["wins"]) + int(r["losses"]), 1)
    return {
        "win_pct": r["wins"] / total_fights,
        "finish_rate": (r["ko_wins"] + r["sub_wins"]) / total_wins,
        "ko_rate": r["ko_wins"] / total_wins,
        "sub_rate": r["sub_wins"] / total_wins,
        "dec_rate": r["dec_wins"] / total_wins,
        "ko_loss_rate": (r["ko_losses"] / total_losses) if total_losses else 0.0,
        "sub_loss_rate": (r["sub_losses"] / total_losses) if total_losses else 0.0,
        "dec_loss_rate": (r["dec_losses"] / total_losses) if total_losses else 0.0,
        "reach_in": r["reach_in"],
        "wins": int(r["wins"]),
        "losses": int(r["losses"]),
        "weight_class": r["weight_class"],
        "first_round_finish_pct": float(r["first_round_finish_pct"]) if "first_round_finish_pct" in r and pd.notna(r["first_round_finish_pct"]) else None,
    }


from src.matchup_model import predict_matchup


def explain_moneyline(row: dict, fighters_df: pd.DataFrame) -> str:
    stats = _fighter_stats(fighters_df, row["fighter"])
    edge_dir = "higher" if row["edge_pct"] > 0 else "lower"
    base = (
        f"The model puts {row['fighter']}'s win probability at {row['model_prob']*100:.0f}%, "
        f"{edge_dir} than the market's {row['book_fair_prob']*100:.0f}% implied probability "
        f"at {format_american_odds(row['odds_american'])} ({row['edge_pct']:+.1f}% edge)."
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
        f"{row['fighter']} to win by {method} is priced at {format_american_odds(row['odds_american'])} "
        f"({row['book_fair_prob']*100:.0f}% implied), while the model estimates {row['model_prob']*100:.0f}% "
        f"({row['edge_pct']:+.1f}% edge)."
    )
    if not stats:
        return base

    rate_key = {"KO/TKO": "ko_rate", "SUB": "sub_rate", "DEC": "dec_rate"}.get(method)
    loss_key = {"KO/TKO": "ko_loss_rate", "SUB": "sub_loss_rate", "DEC": "dec_loss_rate"}.get(method)
    win_col = {"KO/TKO": "ko_wins", "SUB": "sub_wins", "DEC": "dec_wins"}.get(method)
    if not rate_key:
        return base
    own_rate = stats[rate_key]

    opponent = row.get("opponent")
    opp_stats = _fighter_stats(fighters_df, opponent) if opponent else None
    opp_vulnerability = opp_stats[loss_key] if opp_stats else None

    weight_class = stats.get("weight_class")
    divisional_rate = own_rate
    if weight_class is not None and win_col:
        div_group = fighters_df[fighters_df["weight_class"] == weight_class]
        total_div_wins = div_group["wins"].sum()
        if total_div_wins > 0:
            divisional_rate = div_group[win_col].sum() / total_div_wins

    div_gap = own_rate - divisional_rate
    method_lower = method.lower().replace("ko/tko", "KO/TKO").replace("sub", "submission").replace("dec", "decision")

    # Pick whichever angle is actually most distinctive about THIS matchup,
    # rather than always leading with the same blended-factors sentence --
    # different fights genuinely have different "why" depending on the data.

    if opp_vulnerability is not None and opp_vulnerability < 0.08 and opp_stats["losses"] >= 2:
        # opponent has essentially never lost this way -- worth naming directly as the tension
        detail = (
            f" Worth flagging directly: {opponent} has never lost by {method_lower} across "
            f"{opp_stats['losses']} career loss(es), even though {row['fighter']} has finished "
            f"{own_rate*100:.0f}% of wins that way -- the model still leans toward it, but this "
            f"specific matchup history is a real headwind on the pick."
        )
    elif opp_vulnerability is not None and opp_vulnerability >= 0.45:
        # opponent is genuinely vulnerable to this specific method -- lead with that
        detail = (
            f" {opponent} has gone down by {method_lower} in {opp_vulnerability*100:.0f}% of their "
            f"career losses -- a real, specific vulnerability this matchup plays into, on top of "
            f"{row['fighter']}'s own {own_rate*100:.0f}% career rate finishing fights that way."
        )
    elif abs(div_gap) >= 0.15 and weight_class:
        # fighter's own rate is well off the divisional norm -- that's the interesting part
        comparison = "well above" if div_gap > 0 else "well below"
        detail = (
            f" {row['fighter']}'s {own_rate*100:.0f}% career rate by {method_lower} runs {comparison} "
            f"the {divisional_rate*100:.0f}% baseline for {weight_class} -- a real outlier for the "
            f"division, not just a generic tendency."
        )
    elif stats["wins"] < 6:
        # small sample -- worth being upfront that this leans on limited data
        detail = (
            f" Built on a smaller sample ({stats['wins']} career wins), so {row['fighter']}'s "
            f"{own_rate*100:.0f}% rate by {method_lower} carries more uncertainty than a longer "
            f"track record would."
        )
    else:
        # nothing sharply distinctive -- fall back to the blended explanation, but vary the wording
        detail = (
            f" No single factor dominates here -- it's a blend of {row['fighter']}'s own "
            f"{own_rate*100:.0f}% career rate by {method_lower} and how {opponent or 'their opponent'} "
            f"has historically fared against that specific type of finish."
        )

    return base + detail


def explain_total_rounds(row: dict, fighters_df: pd.DataFrame) -> str:
    names = [n.strip() for n in row["fighter"].split(" vs ")]
    fighter_stats = []
    fast_finishers = []
    for name in names:
        s = _fighter_stats(fighters_df, name)
        if s:
            fighter_stats.append((name, s))
            if s["first_round_finish_pct"] and s["first_round_finish_pct"] >= 0.6:
                fast_finishers.append((name, s["first_round_finish_pct"]))

    base = (
        f"{row['market']} at {format_american_odds(row['odds_american'])} implies {row['book_fair_prob']*100:.0f}%, "
        f"vs. the model's {row['model_prob']*100:.0f}% ({row['edge_pct']:+.1f}% edge)."
    )

    if len(fighter_stats) == 2:
        (name_a, s_a), (name_b, s_b) = fighter_stats
        rate_a, rate_b = s_a["finish_rate"], s_b["finish_rate"]
        avg_finish = (rate_a + rate_b) / 2
        gap = abs(rate_a - rate_b)

        if gap >= 0.30:
            # one fighter is a clear finisher, the other isn't -- the average would hide this, so name it directly
            higher_name, higher_rate = (name_a, rate_a) if rate_a > rate_b else (name_b, rate_b)
            lower_name, lower_rate = (name_b, rate_b) if rate_a > rate_b else (name_a, rate_a)
            base += (
                f" This one's lopsided on paper -- {higher_name} finishes {higher_rate*100:.0f}% of wins, "
                f"while {lower_name} sits at just {lower_rate*100:.0f}%, so the combined number undersells "
                f"how much this depends on which fighter's game show up."
            )
        elif avg_finish >= 0.65:
            base += f" Both fighters finish often ({rate_a*100:.0f}% and {rate_b*100:.0f}% of their wins), which is the real driver here."
        elif avg_finish <= 0.30:
            base += f" Neither fighter finishes much ({rate_a*100:.0f}% and {rate_b*100:.0f}% of wins) -- this leans toward distance almost by default."
        else:
            base += f" A fairly even {avg_finish*100:.0f}% combined finish rate between the two, nothing lopsided either way."

    if fast_finishers:
        for name, rate in fast_finishers:
            base += f" Worth flagging: {rate*100:.0f}% of {name}'s career wins have come in round 1 specifically."
    return base


def explain_goes_the_distance(row: dict, fighters_df: pd.DataFrame) -> str:
    names = row["fighter"].split(" vs ")
    dec_rates = []
    for name in names:
        s = _fighter_stats(fighters_df, name.strip())
        if s:
            dec_rates.append(s["dec_rate"])

    base = (
        f"{row['market']} at {format_american_odds(row['odds_american'])} implies {row['book_fair_prob']*100:.0f}%, "
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
