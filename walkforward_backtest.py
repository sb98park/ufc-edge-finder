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
MIN_DURABILITY_SAMPLE = 3  # matches the >=3 prior losses guard already used in production rationale.py


def main():
    history = pd.read_csv(FIGHT_HISTORY_PATH)
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)

    elo = EloRatingSystem()
    fight_counts: dict[str, int] = {}
    # Point-in-time durability tracking, mirroring the production
    # finish_loss_rate formula exactly (ko_losses + sub_losses) / losses
    # -- but computed incrementally as fights are replayed, so a fight
    # from 2015 only ever sees that fighter's record AS OF 2015, never
    # their full career-to-date stats. Using today's snapshot on a
    # historical fight would leak future information into what's
    # supposed to be a point-in-time test.
    prior_losses: dict[str, int] = {}
    prior_finish_losses: dict[str, int] = {}
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
            fav_name = a if prob_a >= 0.5 else b

            fav_prior_losses = prior_losses.get(fav_name, 0)
            fav_durability = None
            if fav_prior_losses >= MIN_DURABILITY_SAMPLE:
                fav_durability = prior_finish_losses.get(fav_name, 0) / fav_prior_losses

            # When the favorite loses, "method" (the WINNER's method of
            # victory) is also the method by which the favorite was beaten
            # -- captured here so upsets can be broken down by how
            # decisively they happened, not just that they happened.
            rows.append({
                "date": fight["date"], "year": fight["date"].year,
                "prob_winner": prob_winner, "fav_prob": fav_prob, "fav_won": fav_won,
                "upset_method": (None if fav_won else method),
                "fav_durability": fav_durability,
            })

        loser = b if winner == a else a
        elo.update_ratings(winner, loser, method=method)
        prior_losses[loser] = prior_losses.get(loser, 0) + 1
        if method in ("KO/TKO", "SUB"):
            prior_finish_losses[loser] = prior_finish_losses.get(loser, 0) + 1
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
    print()

    print("=" * 70)
    print("UPSET ANALYSIS -- when a favorite loses, HOW does it happen?")
    print("=" * 70)
    print("Diagnostic only: this surfaces patterns, it doesn't automatically")
    print("change the model. A pattern here is a starting point for asking")
    print("why, not an instruction to reweight something.")
    print()
    for lo, hi in buckets:
        b = df[(df["fav_prob"] >= lo) & (df["fav_prob"] < hi)]
        upsets = b[~b["fav_won"]]
        if len(upsets) == 0:
            continue
        label = f"{lo*100:.0f}-{min(hi, 1.0)*100:.0f}%"
        method_counts = upsets["upset_method"].value_counts()
        n_upsets = len(upsets)
        method_str = ", ".join(
            f"{m}: {method_counts.get(m, 0)} ({100*method_counts.get(m, 0)/n_upsets:.0f}%)"
            for m in ["KO/TKO", "SUB", "DEC"]
        )
        print(f"  {label:>8} favorites: {n_upsets:4d} losses -- {method_str}")

    print()
    print("=" * 70)
    print("DURABILITY-SEGMENTED CALIBRATION -- the actual test")
    print("=" * 70)
    print("Question: among HIGH-CONFIDENCE favorites specifically, are the")
    print("ones with a worse own finish-loss history actually winning less")
    print("often than the model expects -- not just losing differently when")
    print("they do lose, but genuinely losing MORE than predicted?")
    print()
    print("Durability computed point-in-time (only prior fights as of that")
    print(f"date, same lookahead-safety as the Elo core itself). Requires")
    print(f"{MIN_DURABILITY_SAMPLE}+ prior losses to compute a rate at all -- fighters below")
    print("that are excluded from this specific comparison, not miscounted.")
    print()

    HIGH_CONF_THRESHOLD = 0.60
    high_conf = df[(df["fav_prob"] >= HIGH_CONF_THRESHOLD) & df["fav_durability"].notna()]
    n_excluded = len(df[df["fav_prob"] >= HIGH_CONF_THRESHOLD]) - len(high_conf)
    print(f"High-confidence fights (fav_prob >= {HIGH_CONF_THRESHOLD*100:.0f}%) with a usable durability reading: {len(high_conf)}")
    print(f"(excluded {n_excluded} where the favorite had under {MIN_DURABILITY_SAMPLE} prior losses)")
    print()

    if len(high_conf) < 20:
        print(f"Sample too small ({len(high_conf)} fights) to split further and trust the result.")
        print("Not drawing a conclusion from this -- reporting the limitation instead of a number.")
    else:
        median_durability = high_conf["fav_durability"].median()
        low_durability = high_conf[high_conf["fav_durability"] >= median_durability]  # WORSE chin = higher finish-loss rate
        high_durability = high_conf[high_conf["fav_durability"] < median_durability]

        for label, group in [("Worse durability (top half, higher finish-loss rate)", low_durability),
                              ("Better durability (bottom half, lower finish-loss rate)", high_durability)]:
            predicted = group["fav_prob"].mean()
            actual = group["fav_won"].mean()
            gap = predicted - actual
            print(f"  {label}:")
            print(f"    n={len(group)}, predicted win rate {predicted*100:.1f}%, actual {actual*100:.1f}%, gap {gap*100:+.1f}pp")
        print()
        print("A meaningfully larger positive gap in the 'worse durability' group")
        print("would be real evidence for a confidence-scaled durability penalty.")
        print("A similar or smaller gap would mean the current flat adjustment")
        print("is already doing its job and scaling it further isn't justified.")


if __name__ == "__main__":
    main()
