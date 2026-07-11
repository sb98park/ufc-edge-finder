"""
Walk-forward backtest of the Elo rating core.

How this differs from backtest_model.py (the leaky smell test):
here, fights are replayed in chronological order and each one is
predicted using ONLY the ratings as they stood the moment before that
fight happened -- no future information. This is the honest,
point-in-time evaluation of the rating engine.

What this validates: the Elo core -- ratings, K-factors, method
multipliers -- which drives the largest share of the final number.
What it does NOT validate: the style-adjustment layer (wrestling,
striking, layoff, stance, etc.), because those need career stats as
they stood at fight time, and we only have current-day snapshots.
Validating those honestly requires historical stat snapshots, which is
a future data project.

The calibration table is the most important output: when the engine
says 65%, does the favorite actually win ~65% of the time? That's what
determines whether a headline number like "87%" deserves trust.

Run: python3 walkforward_backtest.py
"""

import math

import pandas as pd

from src.elo import EloRatingSystem

FIGHT_HISTORY_PATH = "data/fight_history.csv"
MIN_PRIOR_FIGHTS = 3  # both fighters need this many tracked fights before the bout counts


def main():
    history = pd.read_csv(FIGHT_HISTORY_PATH)
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)

    elo = EloRatingSystem()
    fight_counts: dict[str, int] = {}
    rows = []

    for _, fight in history.iterrows():
        a, b, winner, method = fight["fighter_a"], fight["fighter_b"], fight["winner"], fight["method"]
        prior_a = fight_counts.get(a, 0)
        prior_b = fight_counts.get(b, 0)

        # Predict BEFORE updating -- this is the whole point.
        prob_a = elo.expected_score(elo.get_rating(a), elo.get_rating(b))

        if prior_a >= MIN_PRIOR_FIGHTS and prior_b >= MIN_PRIOR_FIGHTS:
            prob_winner = prob_a if winner == a else 1.0 - prob_a
            fav_prob = max(prob_a, 1.0 - prob_a)
            fav_won = (prob_a >= 0.5 and winner == a) or (prob_a < 0.5 and winner == b)
            rows.append({
                "date": fight["date"], "year": fight["date"].year,
                "prob_winner": prob_winner, "fav_prob": fav_prob, "fav_won": fav_won,
            })

        loser = b if winner == a else a
        elo.update_ratings(winner, loser, method=method)
        fight_counts[a] = prior_a + 1
        fight_counts[b] = prior_b + 1

    df = pd.DataFrame(rows)
    n = len(df)
    accuracy = df["fav_won"].mean()
    brier = ((df["prob_winner"] - 1.0) ** 2).mean()
    eps = 1e-9
    log_loss = -df["prob_winner"].clip(eps, 1 - eps).apply(math.log).mean()

    print("=" * 70)
    print("WALK-FORWARD BACKTEST (Elo core, point-in-time, no future information)")
    print("=" * 70)
    print(f"Fights evaluated (both fighters with {MIN_PRIOR_FIGHTS}+ prior tracked fights): {n}")
    print(f"Accuracy (higher-rated fighter won): {accuracy*100:.1f}%")
    print(f"Brier score (lower is better, 0.25 = coinflip): {brier:.3f}")
    print(f"Log-loss (lower is better, 0.693 = coinflip): {log_loss:.3f}")
    print()

    print("By era:")
    for label, lo, hi in (("pre-2016", 0, 2016), ("2016-2020", 2016, 2021), ("2021-2023", 2021, 2024), ("2024+", 2024, 9999)):
        era = df[(df["year"] >= lo) & (df["year"] < hi)]
        if len(era):
            print(f"  {label}: {era['fav_won'].mean()*100:.1f}% accuracy, Brier {((era['prob_winner']-1.0)**2).mean():.3f}, over {len(era)} fights")
    print()

    print("CALIBRATION -- when the engine says X%, how often does that favorite actually win?")
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.01)]
    for lo, hi in buckets:
        b = df[(df["fav_prob"] >= lo) & (df["fav_prob"] < hi)]
        if len(b):
            label = f"{lo*100:.0f}-{min(hi, 1.0)*100:.0f}%"
            print(f"  predicted {label:>8}: actual {b['fav_won'].mean()*100:5.1f}% over {len(b):4d} fights")


if __name__ == "__main__":
    main()
