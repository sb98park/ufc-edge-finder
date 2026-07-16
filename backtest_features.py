"""
Feature experiments on top of the Elo backbone backtest.

Same methodology as backtest_elo.py (chronological replay, predict
before update, score only fights where both fighters have >= 4 prior
fights), extended to compute candidate adjustment-layer features from
PRIOR fights only at each prediction point:

  1. recent_form   -- sum over each fighter's last 3 results of
                      (+1 win / -1 loss) * linear decay over 2 years.
                      Production currently uses only the single most
                      recent fight; this tests whether looking deeper
                      helps and what scale the data supports.
  2. opp_quality   -- mean PRE-FIGHT Elo of the last 3 opponents faced.
                      Elo already bakes opponent strength into rating
                      updates, but slowly (K=32); this tests whether
                      recent schedule strength carries extra signal.
  3. layoff        -- years since last fight beyond a 1-year grace,
                      mirroring the production penalty's shape. The
                      production weight (60 pts/yr) has never been
                      validated; this tests it against 30 years of
                      real outcomes.

Each feature's scale is tuned on pre-2019 fights and evaluated on
held-out 2019+ fights (log loss). Scale 0 = feature off, so "0 wins"
is an honest possible outcome for any of them.
"""

import math

import numpy as np
import pandas as pd

METHOD_K = {"KO/TKO": 1.25, "SUB": 1.15, "DEC": 0.90, "DQ": 0.50}
MIN_PRIOR = 4
FORM_DECAY_YEARS = 2.0
LAYOFF_GRACE_YEARS = 1.0


def collect_features(history_path: str = "data/fight_history.csv") -> pd.DataFrame:
    df = pd.read_csv(history_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    # per fighter: list of (date, won: bool, opponent_prefight_elo)
    logs: dict[str, list] = {}
    records = []

    for _, fight in df.iterrows():
        a, b, winner = fight["fighter_a"], fight["fighter_b"], fight["winner"]
        if winner == a:
            loser = b
        elif winner == b:
            loser = a
        else:
            continue
        r_a, r_b = ratings.get(a, 1500.0), ratings.get(b, 1500.0)
        log_a, log_b = logs.get(a, []), logs.get(b, [])
        date = fight["date"]

        if len(log_a) >= MIN_PRIOR and len(log_b) >= MIN_PRIOR:
            def form(log):
                s = 0.0
                for d, won, _ in log[-3:]:
                    yrs = (date - d).days / 365.25
                    decay = max(0.0, 1.0 - yrs / FORM_DECAY_YEARS)
                    s += (1.0 if won else -1.0) * decay
                return s

            def opp_quality(log):
                return float(np.mean([o for _, _, o in log[-3:]]))

            def layoff(log):
                yrs = (date - log[-1][0]).days / 365.25
                return max(0.0, yrs - LAYOFF_GRACE_YEARS)

            records.append({
                "date": date,
                "gap": r_a - r_b,
                "a_won": winner == a,
                "form_diff": form(log_a) - form(log_b),
                "oppq_diff": (opp_quality(log_a) - opp_quality(log_b)) / 100.0,
                "layoff_diff": layoff(log_a) - layoff(log_b),  # positive = A rustier
            })

        # update Elo + logs AFTER the prediction snapshot
        r_w, r_l = ratings.get(winner, 1500.0), ratings.get(loser, 1500.0)
        exp_w = 1.0 / (1.0 + 10 ** ((r_l - r_w) / 400.0))
        k = 32.0 * METHOD_K.get(fight.get("method", "DEC"), 1.0)
        ratings[winner] = r_w + k * (1.0 - exp_w)
        ratings[loser] = r_l + k * (exp_w - 1.0)
        logs.setdefault(a, []).append((date, winner == a, r_b))
        logs.setdefault(b, []).append((date, winner == b, r_a))

    return pd.DataFrame(records)


def log_loss_vec(gap: np.ndarray, a_won: np.ndarray) -> float:
    p_a = 1.0 / (1.0 + 10 ** (-gap / 400.0))
    p_winner = np.where(a_won, p_a, 1.0 - p_a)
    return float(-np.log(np.clip(p_winner, 1e-12, 1 - 1e-12)).mean())


def tune_feature(train: pd.DataFrame, test: pd.DataFrame, col: str, scales: list, negate: bool = False) -> dict:
    """Tunes one feature's scale on train, reports train + held-out test log loss."""
    sign = -1.0 if negate else 1.0
    results = []
    for s in scales:
        adj_train = train["gap"].values + sign * s * train[col].values
        adj_test = test["gap"].values + sign * s * test[col].values
        results.append({
            "scale": s,
            "train_ll": log_loss_vec(adj_train, train["a_won"].values),
            "test_ll": log_loss_vec(adj_test, test["a_won"].values),
        })
    res = pd.DataFrame(results)
    best_train = res.loc[res["train_ll"].idxmin()]
    return {"table": res, "best_scale_by_train": best_train["scale"], "test_ll_at_best": best_train["test_ll"]}


if __name__ == "__main__":
    feats = collect_features()
    train = feats[feats["date"].dt.year < 2019]
    test = feats[feats["date"].dt.year >= 2019]
    baseline_test = log_loss_vec(test["gap"].values, test["a_won"].values)
    print(f"train: {len(train)}  test: {len(test)}  |  baseline (Elo only) test LL: {baseline_test:.4f}\n")

    print("=== recent_form (last 3, decayed; production uses last 1 @ 20 pts) ===")
    r = tune_feature(train, test, "form_diff", [0, 5, 10, 15, 20, 30, 40, 60])
    print(r["table"].to_string(index=False))
    print(f"best-by-train scale: {r['best_scale_by_train']}, held-out test LL: {r['test_ll_at_best']:.4f}\n")

    print("=== opp_quality (points per 100 Elo of recent-opponent quality) ===")
    r = tune_feature(train, test, "oppq_diff", [0, 5, 10, 20, 30, 50, 80])
    print(r["table"].to_string(index=False))
    print(f"best-by-train scale: {r['best_scale_by_train']}, held-out test LL: {r['test_ll_at_best']:.4f}\n")

    print("=== layoff (points per year beyond 1yr grace; production uses 60) ===")
    r = tune_feature(train, test, "layoff_diff", [0, 10, 20, 30, 40, 60, 80, 120], negate=True)
    print(r["table"].to_string(index=False))
    print(f"best-by-train scale: {r['best_scale_by_train']}, held-out test LL: {r['test_ll_at_best']:.4f}")
