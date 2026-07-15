"""
Fixes a real gap in pure Elo: a fighter with zero recorded fights against
anyone already in fight_history.csv sits at the default rating forever,
so every matchup between two "isolated" fighters comes out exactly 50/50
no matter how different they actually are.

This blends Elo (when there's enough connected fight history to trust it)
with a stats-based power rating built from career record, finish rate,
and physical attributes -- so a fighter with a 26-7 record and a 54% KO
rate isn't rated identically to a 3-0 fighter just because neither has
fought anyone in our Elo graph yet.
"""

import pandas as pd

RATING_CENTER = 1500.0


def compute_stats_rating(row: pd.Series) -> float:
    """
    A rough power rating on the same numeric scale as Elo (centered at 1500),
    built purely from career stats. Not a substitute for real fight-by-fight
    history -- just a reasonable prior when that history doesn't exist yet.

    Every field read here is explicitly NaN-checked, not just defaulted via
    .get() -- .get(col, default) only falls back when the COLUMN is entirely
    absent from the row, not when it's present but holds NaN, which is the
    actual shape missing data takes in fighters.csv (the column always
    exists in the schema; individual fighters are just missing a value).
    Confirmed live: this was silently producing a NaN power rating for any
    fighter missing reach_in or a win-method breakdown, which then became
    their fallback Elo rating (since a new/obscure fighter typically has no
    connected fight_history.csv entries yet), corrupting every downstream
    prediction and ultimately crashing the parlay builder on a NaN price.
    """
    wins = row["wins"] if pd.notna(row.get("wins")) else 0
    losses = row["losses"] if pd.notna(row.get("losses")) else 0
    ko_wins = row["ko_wins"] if pd.notna(row.get("ko_wins")) else 0
    sub_wins = row["sub_wins"] if pd.notna(row.get("sub_wins")) else 0
    reach_in = row["reach_in"] if pd.notna(row.get("reach_in")) else 70

    total_fights = max(wins + losses, 1)
    win_pct = wins / total_fights
    finish_rate = (ko_wins + sub_wins) / max(wins, 1)

    # experience damps how much we trust a small sample (a 3-0 record
    # shouldn't swing as hard as a 26-7 record even at similar win%)
    experience_weight = min(1.0, total_fights / 15.0)

    rating = RATING_CENTER
    rating += 500.0 * (win_pct - 0.5) * experience_weight
    rating += 150.0 * (finish_rate - 0.4)
    rating += 4.0 * (reach_in - 70)

    return rating


def build_effective_ratings(
    fighters_df: pd.DataFrame,
    elo_ratings: dict[str, float],
    history_df: pd.DataFrame,
    min_fights_to_trust_elo: int = 4,
) -> dict[str, float]:
    """
    For each fighter: if they have enough *connected* fight history for Elo
    to mean something, blend toward Elo as that count grows. Otherwise, rely
    on the stats-based rating instead of the meaningless flat default.
    """
    fight_counts = pd.concat([
        history_df["fighter_a"] if "fighter_a" in history_df else pd.Series(dtype=str),
        history_df["fighter_b"] if "fighter_b" in history_df else pd.Series(dtype=str),
    ]).value_counts()

    effective = {}
    for _, row in fighters_df.iterrows():
        name = row["name"]
        stats_rating = compute_stats_rating(row)
        n_fights_tracked = int(fight_counts.get(name, 0))

        if n_fights_tracked == 0:
            effective[name] = stats_rating
        else:
            weight = min(1.0, n_fights_tracked / min_fights_to_trust_elo)
            elo_r = elo_ratings.get(name, RATING_CENTER)
            effective[name] = weight * elo_r + (1 - weight) * stats_rating

    return effective
