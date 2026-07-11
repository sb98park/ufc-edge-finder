"""
Backtests predict_matchup() against data/fight_history.csv.

*** READ THIS BEFORE TRUSTING THE NUMBERS BELOW ***

This is NOT a rigorous backtest, and it can't be one with the data
currently available. A real backtest needs POINT-IN-TIME stats -- what
did fighters.csv look like the day BEFORE each historical fight, before
that fight's own result (and everything since) got folded into the
aggregate record. We only have fighters.csv as it looks TODAY. That
means every prediction below is made with knowledge that includes the
fight's own outcome and everything after it -- real data leakage, not a
hypothetical concern.

What this script is actually useful for: a rough sanity check on whether
the model's CURRENT predictions look reasonable against known outcomes,
and a way to eyeball whether the hand-picked scale constants
(WRESTLING_ADVANTAGE_SCALE, STRIKING_ADVANTAGE_SCALE, etc.) are producing
wildly miscalibrated results. It is NOT a substitute for tuning those
constants against genuine out-of-sample validation, which would require
building point-in-time snapshots of fighters.csv -- a real project, not
a quick add.

Run: python3 backtest_model.py
"""

import math

import pandas as pd

from src.matchup_model import predict_matchup
from src.power_rating import build_effective_ratings
from src.elo import EloRatingSystem

DATA_DIR = "data"


def main():
    fighters_df = pd.read_csv(f"{DATA_DIR}/fighters.csv")
    history_df = pd.read_csv(f"{DATA_DIR}/fight_history.csv")

    elo_system = EloRatingSystem()
    elo_ratings = elo_system.build_from_history(history_df)
    effective_ratings = build_effective_ratings(fighters_df, elo_ratings, history_df)

    results = []
    skipped = 0
    for _, fight in history_df.iterrows():
        fighter_a, fighter_b, winner = fight["fighter_a"], fight["fighter_b"], fight["winner"]
        matchup = predict_matchup(fighter_a, fighter_b, fighters_df, effective_ratings, history_df)
        if matchup is None:
            skipped += 1  # one or both fighters not in fighters.csv -- expected for most of the full graph
            continue

        predicted_prob_winner = matchup["prob_a"] if winner == fighter_a else matchup["prob_b"]
        correct = predicted_prob_winner > 0.5
        results.append({
            "date": fight.get("date", ""),
            "fighter_a": fighter_a, "fighter_b": fighter_b, "winner": winner,
            "predicted_prob_a": round(matchup["prob_a"], 3),
            "predicted_prob_winner": round(predicted_prob_winner, 3),
            "correct": correct,
        })

    if not results:
        print("No fights could be backtested -- check that fighters.csv covers fight_history.csv's names.")
        return

    df = pd.DataFrame(results)
    n = len(df)
    accuracy = df["correct"].mean()

    # Brier score: mean squared error between predicted probability and
    # actual outcome (0 or 1) for the ACTUAL winner. Lower is better; 0.25
    # is what a coin-flip model scores, 0 is a perfect (and suspicious) model.
    brier = ((df["predicted_prob_winner"] - 1.0) ** 2).mean()

    # Log-loss: penalizes confident wrong predictions much more heavily
    # than a plain accuracy number does. Lower is better.
    eps = 1e-9
    log_loss = -df["predicted_prob_winner"].clip(eps, 1 - eps).apply(math.log).mean()

    print(f"{'='*70}")
    print("BACKTEST RESULTS -- read the data-leakage caveat at the top of this file first")
    print(f"{'='*70}")
    print(f"Fights evaluated (both fighters in fighters.csv): {n} | skipped: {skipped}")
    print(f"Accuracy (model favored the actual winner): {accuracy*100:.1f}%")
    print(f"Brier score (lower is better, 0.25 = coinflip): {brier:.3f}")
    print(f"Log-loss (lower is better): {log_loss:.3f}")

    # Era breakdown: the data-leakage problem is worst for old fights
    # (predicting a 2015 fight with 2026 career stats), so recent-era
    # accuracy is the least-dishonest number here.
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    for label, lo in (("2024+", 2024), ("2020-2023", 2020), ("pre-2020", 0)):
        era = df[(df["year"] >= lo)] if lo else df[df["year"] < 2020]
        if lo == 2020:
            era = df[(df["year"] >= 2020) & (df["year"] < 2024)]
        if len(era):
            print(f"  {label}: {era['correct'].mean()*100:.1f}% over {len(era)} fights")

    print()
    print("Most recent 20 evaluated fights:")
    print(df.sort_values("date").tail(20).drop(columns=["year"]).to_string(index=False))


if __name__ == "__main__":
    main()
