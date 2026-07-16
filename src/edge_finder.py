"""
Combines the Elo model + fighter finish-rate stats with sportsbook lines
to surface bets where the model disagrees most with the market.

This is a decision-SUPPORT tool, not a decision-maker. Edges are only as
good as the historical data feeding the model -- always sanity check
against recent form, injuries, weight cuts, camp changes, etc. that a
pure stats model can't see.
"""

import re

import pandas as pd

from .odds_utils import american_to_implied_prob, remove_vig_two_way, edge_percent, kelly_fraction, market_blended_prob
from .matchup_model import predict_matchup, compute_divisional_method_priors, blend_method_probability, _get


def compute_moneyline_edges(
    upcoming_df: pd.DataFrame, elo_ratings: dict[str, float], fighters_df: pd.DataFrame | None = None,
    fight_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    ml = upcoming_df[upcoming_df["market"] == "Moneyline"]

    for fight_id, group in ml.groupby("fight_id"):
        if len(group) != 2:
            print(f"[edge_finder] moneyline skip for fight_id={fight_id!r}: {len(group)} row(s) instead of 2 "
                  f"-- selections: {group['selection'].tolist()}")
            continue  # need both sides of the moneyline to devig

        a, b = group.iloc[0], group.iloc[1]

        matchup = None
        if fighters_df is not None:
            matchup = predict_matchup(a["selection"], b["selection"], fighters_df, elo_ratings, fight_history_df)

        if matchup:
            model_prob_a = matchup["prob_a"]
        else:
            # fallback: plain rating gap if we don't have style stats for these fighters
            elo_a = elo_ratings.get(a["selection"], 1500.0)
            elo_b = elo_ratings.get(b["selection"], 1500.0)
            model_prob_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
        model_prob_b = 1.0 - model_prob_a

        imp_a = american_to_implied_prob(a["odds_american"])
        imp_b = american_to_implied_prob(b["odds_american"])
        fair_a, fair_b = remove_vig_two_way(imp_a, imp_b)

        for fighter, opponent, model_p, fair_p, odds, token_id in [
            (a["selection"], b["selection"], model_prob_a, fair_a, a["odds_american"], a.get("clob_token_id")),
            (b["selection"], a["selection"], model_prob_b, fair_b, b["odds_american"], b.get("clob_token_id")),
        ]:
            rows.append({
                "fight_id": fight_id,
                "fighter": fighter,
                "opponent": opponent,
                "market": "Moneyline",
                "odds_american": odds,
                "model_prob": round(model_p, 3),
                "book_fair_prob": round(fair_p, 3),
                "edge_pct": round(edge_percent(model_p, fair_p), 2),
                "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, fair_p), odds) * 100, 2),
                "clob_token_id": token_id,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def compute_method_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Method-of-victory props (KO/TKO, Submission, Decision). Prior-informed
    blend: starts at the DIVISIONAL baseline rate for that method (a
    heavyweight fight has an inherently higher baseline KO/TKO rate than a
    strawweight fight, which leans toward decisions), then shifts toward
    the fighter's own career tendency (weighted by experience), then further
    incorporates how often THIS SPECIFIC opponent has actually lost that
    way before.
    """
    rows = []
    props = upcoming_df[upcoming_df["market"] == "Method"]
    divisional_priors = compute_divisional_method_priors(fighters_df)

    method_loss_col = {"KO/TKO": "ko_losses", "SUB": "sub_losses", "DEC": "dec_losses"}

    for _, row in props.iterrows():
        stats = fighters_df[fighters_df["name"] == row["selection"]]
        if stats.empty:
            continue
        f = stats.iloc[0]
        total_wins = max(int(f["wins"]), 1)

        rate_map = {
            "KO/TKO": _get(f, "ko_wins", 0) / total_wins,
            "SUB": _get(f, "sub_wins", 0) / total_wins,
            "DEC": _get(f, "dec_wins", 0) / total_wins,
        }
        own_rate = rate_map.get(row["selection_method"])
        if own_rate is None:
            continue

        # find the opponent to factor in their specific vulnerability
        opponent_name = row["fighter_b"] if row["selection"] == row["fighter_a"] else row["fighter_a"]
        opp_stats = fighters_df[fighters_df["name"] == opponent_name]

        divisional_prior = divisional_priors.get(f["weight_class"], {}).get(row["selection_method"], own_rate)

        opp_vulnerability = own_rate  # fallback if opponent data is missing
        if not opp_stats.empty:
            opp = opp_stats.iloc[0]
            opp_losses = max(int(opp["losses"]), 1) if opp["losses"] else 0
            if opp_losses:
                col = method_loss_col[row["selection_method"]]
                opp_vulnerability = opp[col] / opp_losses

        model_p = blend_method_probability(divisional_prior, own_rate, opp_vulnerability, total_wins)

        imp = american_to_implied_prob(row["odds_american"])

        rows.append({
            "fight_id": row["fight_id"],
            "fighter": row["selection"],
            "opponent": opponent_name,
            "market": f"Method: {row['selection_method']}",
            "odds_american": row["odds_american"],
            "model_prob": round(model_p, 3),
            "book_fair_prob": round(imp, 3),  # not devigged (single-sided prop)
            "edge_pct": round(edge_percent(model_p, imp), 2),
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, imp), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def _extract_round_line(selection: str) -> float | None:
    match = re.search(r"(\d+\.?\d*)", str(selection))
    return float(match.group(1)) if match else None


def compute_total_rounds_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Over/Under total rounds props. A fight card often offers multiple lines
    (1.5, 2.5, 3.5) for the same fight -- these are grouped separately so
    they don't collide.

    For the 1.5 line specifically ("does it end in round 1"), this uses each
    fighter's actual first_round_finish_pct directly -- a fighter like
    Terrance McKinney (16 of 17 wins finished in round 1, never gone to a
    decision) should swing this line hard, and a generic finish-rate proxy
    was missing that entirely. Other lines blend that same signal in rather
    than relying purely on a generic linear adjustment.
    """
    rows = []
    props = upcoming_df[upcoming_df["market"] == "TotalRounds"].copy()
    props["_line"] = props["selection"].apply(_extract_round_line)

    REFERENCE_LINE = 2.5
    ADJUSTMENT_PER_ROUND = 0.15

    for (fight_id, line), group in props.groupby(["fight_id", "_line"]):
        fighters_in_fight = group["fighter_a"].iloc[0], group["fighter_b"].iloc[0]
        finish_rates = []
        first_round_rates = []
        for name in fighters_in_fight:
            stats = fighters_df[fighters_df["name"] == name]
            if stats.empty:
                continue
            f = stats.iloc[0]
            total_wins = max(int(f["wins"]), 1)
            finish_rates.append((_get(f, "ko_wins", 0) + _get(f, "sub_wins", 0)) / total_wins)
            if "first_round_finish_pct" in f and pd.notna(f["first_round_finish_pct"]):
                first_round_rates.append(float(f["first_round_finish_pct"]))

        if not finish_rates:
            continue

        combined_finish_rate = sum(finish_rates) / len(finish_rates)
        combined_first_round_rate = sum(first_round_rates) / len(first_round_rates) if first_round_rates else None

        if line is not None and line <= 1.5 and combined_first_round_rate is not None:
            # the literal, most verifiable case: does it end in round 1
            model_prob_under = combined_first_round_rate
        elif line is not None:
            base = combined_finish_rate - (REFERENCE_LINE - line) * ADJUSTMENT_PER_ROUND
            # blend in the fast-finisher signal even for longer lines, rather
            # than only using it for the 1.5 boundary
            if combined_first_round_rate is not None:
                base = 0.7 * base + 0.3 * combined_first_round_rate
            model_prob_under = base
        else:
            model_prob_under = combined_finish_rate
        model_prob_under = min(0.95, max(0.05, model_prob_under))

        for _, row in group.iterrows():
            model_p = model_prob_under if "under" in row["selection"].lower() else (1 - model_prob_under)
            imp = american_to_implied_prob(row["odds_american"])
            rows.append({
                "fight_id": fight_id,
                "fighter": f"{fighters_in_fight[0]} vs {fighters_in_fight[1]}",
                "market": f"Total Rounds {row['selection']}",
                "odds_american": row["odds_american"],
                "model_prob": round(model_p, 3),
                "book_fair_prob": round(imp, 3),
                "edge_pct": round(edge_percent(model_p, imp), 2),
                "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, imp), row["odds_american"]) * 100, 2),
                "clob_token_id": row.get("clob_token_id"),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def compute_goes_the_distance_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    'Fight goes the distance' vs 'ends in a finish' -- derived the same way
    as method-of-victory (sum of both fighters' decision-win likelihood).
    """
    rows = []
    props = upcoming_df[upcoming_df["market"] == "GoesTheDistance"]

    for _, row in props.iterrows():
        f_a = fighters_df[fighters_df["name"] == row["fighter_a"]]
        f_b = fighters_df[fighters_df["name"] == row["fighter_b"]]
        if f_a.empty or f_b.empty:
            continue
        a, b = f_a.iloc[0], f_b.iloc[0]
        dec_rate_a = _get(a, "dec_wins", 0) / max(int(a["wins"]), 1)
        dec_rate_b = _get(b, "dec_wins", 0) / max(int(b["wins"]), 1)
        # rough proxy: average of both fighters' decision tendency as the
        # fight-level chance it goes the distance
        goes_distance_prob = (dec_rate_a + dec_rate_b) / 2

        model_p = goes_distance_prob if "distance" in row["selection"].lower() and "ends" not in row["selection"].lower() else (1 - goes_distance_prob)
        imp = american_to_implied_prob(row["odds_american"])
        rows.append({
            "fight_id": row["fight_id"],
            "fighter": f"{row['fighter_a']} vs {row['fighter_b']}",
            "market": f"Fight Outcome: {row['selection']}",
            "odds_american": row["odds_american"],
            "model_prob": round(model_p, 3),
            "book_fair_prob": round(imp, 3),
            "edge_pct": round(edge_percent(model_p, imp), 2),
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, imp), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def derive_card_label(row: pd.Series) -> str:
    """Prefer an explicit event name (e.g. 'UFC 329'); fall back to date-based grouping."""
    event_name = (row.get("event_name") or "").strip()
    if event_name:
        return event_name
    start_date = row.get("start_date")
    if pd.notna(start_date) and str(start_date).strip():
        date_part = str(start_date)[:10]
        return f"Fight Card — {date_part}"
    return "Upcoming Fights"


def build_fight_list(upcoming_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per fight (deduped), with card grouping metadata attached.
    Used to list every matchup on a card even if no prop cleared the edge
    threshold for that fight.
    """
    cols = [c for c in ["fight_id", "fighter_a", "fighter_b", "event_name",
                         "start_date", "weight_class", "card_position"] if c in upcoming_df.columns]
    fights = upcoming_df[cols].drop_duplicates(subset="fight_id").copy()
    fights["card_label"] = fights.apply(derive_card_label, axis=1)
    return fights.reset_index(drop=True)


def top_standout_props(edges_df: pd.DataFrame, n: int = 5, min_edge: float = 5.0) -> pd.DataFrame:
    """The headline shortlist: biggest model-vs-market disagreements, positive edge only."""
    if edges_df.empty:
        return edges_df
    standouts = edges_df[edges_df["edge_pct"] >= min_edge].copy()
    return standouts.sort_values("edge_pct", ascending=False).head(n).reset_index(drop=True)


def attach_fight_meta(edges_df: pd.DataFrame, fight_list_df: pd.DataFrame) -> pd.DataFrame:
    """Merges card_label/weight_class/opponent info into the edges dataframe for grouped display."""
    if edges_df.empty or fight_list_df.empty:
        return edges_df
    meta_cols = [c for c in ["fight_id", "card_label", "weight_class", "card_position",
                              "fighter_a", "fighter_b"] if c in fight_list_df.columns]
    return edges_df.merge(fight_list_df[meta_cols], on="fight_id", how="left")


def find_all_edges(
    upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame, elo_ratings: dict[str, float],
    fight_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = [
        compute_moneyline_edges(upcoming_df, elo_ratings, fighters_df, fight_history_df),
        compute_method_edges(upcoming_df, fighters_df),
        compute_total_rounds_edges(upcoming_df, fighters_df),
        compute_goes_the_distance_edges(upcoming_df, fighters_df),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("edge_pct", ascending=False).reset_index(drop=True)
