"""
Chronological, out-of-sample backtest of the Elo rating backbone.

Walks fight_history.csv in strict date order. For every fight, the
prediction is made from ratings built ONLY on fights that happened
before it (predict first, then update) -- so every scored prediction
is genuinely out-of-sample, never retrodicted.

Scores three things:
  1. Accuracy vs. two baselines (coin flip; the trivial "pick whoever
     is currently rated higher", which is the same thing here but
     stated explicitly so the number is honest).
  2. Log loss and Brier score -- probability quality, not just
     pick correctness. A model that says 55% on everything can have
     decent accuracy and useless probabilities; these catch that.
  3. Calibration by predicted-probability bucket -- when the model
     says 70%, does that side actually win ~70% of the time? This is
     the number that matters most for edge-vs-market claims, since a
     claimed edge is only real if the model's probabilities mean what
     they say.

Warm-up handling: predictions are only SCORED when both fighters have
at least MIN_PRIOR_FIGHTS fights already in the replayed history.
Early-history predictions (everyone at 1500) are structurally
uninformative and would drag metrics toward the coin-flip baseline,
overstating how bad the informed predictions actually are -- and
excluding them mirrors production reality, where the effective-rating
blend already refuses to lean on Elo for thin-history fighters.
"""

import math

import pandas as pd

from src.elo import EloRatingSystem, METHOD_K_MULTIPLIER

MIN_PRIOR_FIGHTS = 4  # matches min_fights_to_trust_elo in power_rating.py


def run_backtest(history_path: str = "data/fight_history.csv") -> dict:
    df = pd.read_csv(history_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    elo = EloRatingSystem()
    fight_counts: dict[str, int] = {}

    records = []
    skipped_malformed = 0

    for _, fight in df.iterrows():
        a, b, winner = fight["fighter_a"], fight["fighter_b"], fight["winner"]
        if winner == a:
            loser = b
        elif winner == b:
            loser = a
        else:
            skipped_malformed += 1
            continue

        n_a = fight_counts.get(a, 0)
        n_b = fight_counts.get(b, 0)

        # Predict BEFORE updating -- this is what makes it out-of-sample.
        prob_a = EloRatingSystem.expected_score(elo.get_rating(a), elo.get_rating(b))

        if n_a >= MIN_PRIOR_FIGHTS and n_b >= MIN_PRIOR_FIGHTS:
            records.append({
                "date": fight["date"],
                "prob_winner": prob_a if winner == a else 1.0 - prob_a,
                "predicted_a": prob_a,
                "a_won": winner == a,
                "picked_correctly": (prob_a >= 0.5) == (winner == a),
                "confident_side_prob": max(prob_a, 1.0 - prob_a),
            })

        elo.update_ratings(winner, loser, method=fight.get("method", "DEC"))
        fight_counts[a] = n_a + 1
        fight_counts[b] = n_b + 1

    scored = pd.DataFrame(records)
    n = len(scored)
    if n == 0:
        return {"error": "no scoreable fights"}

    # Exact 50/50 predictions (identical ratings) count as coin flips for
    # accuracy; leave them in -- excluding them would flatter the number.
    accuracy = scored["picked_correctly"].mean()

    eps = 1e-12
    log_loss = -scored["prob_winner"].clip(eps, 1 - eps).apply(math.log).mean()
    brier = ((1.0 - scored["prob_winner"]) ** 2).mean()

    # Baselines for context: coin flip log loss = ln(2) ~ 0.6931, Brier = 0.25.
    baseline = {"coin_accuracy": 0.5, "coin_log_loss": math.log(2), "coin_brier": 0.25}

    # Calibration: bucket the CONFIDENT side's probability, compare claimed vs realized.
    scored["bucket"] = pd.cut(
        scored["confident_side_prob"],
        bins=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0],
        labels=["50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80+"],
        include_lowest=True,
    )
    calib = (
        scored.groupby("bucket", observed=True)
        .agg(n=("picked_correctly", "size"),
             claimed=("confident_side_prob", "mean"),
             realized=("picked_correctly", "mean"))
        .reset_index()
    )

    # Era split -- MMA data quality and depth changed a lot over 30 years;
    # a single blended number can hide era-specific behavior.
    scored["era"] = pd.cut(
        scored["date"].dt.year,
        bins=[1993, 2010, 2018, 2027],
        labels=["1994-2010", "2011-2018", "2019-2026"],
    )
    era = (
        scored.groupby("era", observed=True)
        .agg(n=("picked_correctly", "size"),
             accuracy=("picked_correctly", "mean"),
             brier_component=("prob_winner", lambda s: ((1 - s) ** 2).mean()))
        .reset_index()
    )

    return {
        "scored_fights": n,
        "total_history_rows": len(df),
        "skipped_malformed": skipped_malformed,
        "excluded_warmup": len(df) - skipped_malformed - n,
        "accuracy": accuracy,
        "log_loss": log_loss,
        "brier": brier,
        "baseline": baseline,
        "calibration": calib,
        "era_breakdown": era,
    }


if __name__ == "__main__":
    r = run_backtest()
    print(f"Scored fights (out-of-sample, both fighters >= {MIN_PRIOR_FIGHTS} prior fights): {r['scored_fights']}")
    print(f"Excluded as warm-up: {r['excluded_warmup']}   Malformed/draw rows skipped: {r['skipped_malformed']}")
    print()
    print(f"Accuracy:  {r['accuracy']*100:.1f}%   (coin flip: 50.0%)")
    print(f"Log loss:  {r['log_loss']:.4f}   (coin flip: {r['baseline']['coin_log_loss']:.4f} -- lower is better)")
    print(f"Brier:     {r['brier']:.4f}   (coin flip: {r['baseline']['coin_brier']:.4f} -- lower is better)")
    print()
    print("Calibration (claimed vs realized win rate for the confident side):")
    print(r["calibration"].to_string(index=False))
    print()
    print("Era breakdown:")
    print(r["era_breakdown"].to_string(index=False))
