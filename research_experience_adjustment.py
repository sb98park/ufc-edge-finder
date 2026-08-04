"""
Fit and validate an experience adjustment for the Elo rating.

THE FINDING IT ADDRESSES (research_experience_bias.py, n=5978 fights with a
closing line). The model's gap to the closing line moves monotonically with
UFC fight count:

      0-3 fights   -6.6%      9-15 fights  +1.5%
      4-8 fights   -4.7%      16+ fights   +9.8%

A 16-point swing, one direction, no reversals -- and it amplifies with an
active streak (-5.7% -> -8.3% -> -9.4% at streak 3 and 5). The mechanism is
structural: Elo accumulates over fights, so a rating reflects HOW MANY you
have won more than WHO you beat. It lags on the newly proven and
over-credits the long-tenured.

WHAT THIS DOES. Fits a correction as a function of fight count and streak,
then validates it the way recency weighting was validated -- on a FROZEN
holdout, on outcomes, not on the thing it was fit to.

THE TRAP THIS AVOIDS. Fitting to close the gap with the market and then
reporting that the gap closed is circular: of course it did, that was the
objective. The market is used to DISCOVER the bias and outcomes are used to
JUDGE the fix. If Brier and accuracy don't improve on held-out fights, the
adjustment is cosmetic and doesn't ship, however neat the gap looks.

Also swept against a control: the same adjustment with the sign FLIPPED. If
the flipped version scores about as well, the effect is noise dressed up as
structure.

Run: python3 research_experience_adjustment.py
"""

import numpy as np
import pandas as pd

import research_survival_model as R
from src.elo import EloRatingSystem

HOLDOUT_START = pd.Timestamp("2019-01-01")


def build():
    """Point-in-time rating, experience and streak for every fight."""
    fights = R.load_dated_fights()
    elo = EloRatingSystem()
    seen, streak, rows = {}, {}, []
    for f in fights.itertuples(index=False):
        a, b = f.fighter_1, f.fighter_2
        na, nb = seen.get(a, 0), seen.get(b, 0)
        sa, sb = streak.get(a, 0), streak.get(b, 0)
        rows.append({
            "date": f.date,
            "rating_gap": elo.get_rating(a) - elo.get_rating(b),
            "own_fights": na, "opp_fights": nb,
            "own_streak": sa, "opp_streak": sb,
            "won": 1 if f.winner == a else 0,
        })
        winner = f.winner
        loser = b if winner == a else a
        elo.update_ratings(winner, loser, method=str(f.method))
        seen[a], seen[b] = na + 1, nb + 1
        streak[winner] = streak.get(winner, 0) + 1
        streak[loser] = 0
    return pd.DataFrame(rows)


def prob(gap):
    return 1.0 / (1.0 + 10 ** (-gap / 400.0))


def adjustment(fights, streak, k_exp, k_streak, cap=8):
    """
    Rating points to ADD for a fighter.

    Positive for the inexperienced, negative for the long-tenured, scaled by
    an active streak. Capped so a debutant on a notional streak can't be
    handed an unbounded bonus -- the cohort evidence only covers the range
    actually observed.
    """
    exp_term = k_exp * (np.log1p(cap) - np.log1p(np.minimum(fights, 40)))
    streak_term = k_streak * np.minimum(streak, 6) * (fights <= 8)
    return exp_term + streak_term


def score(df, k_exp, k_streak):
    adj_a = adjustment(df["own_fights"].to_numpy(), df["own_streak"].to_numpy(), k_exp, k_streak)
    adj_b = adjustment(df["opp_fights"].to_numpy(), df["opp_streak"].to_numpy(), k_exp, k_streak)
    p = prob(df["rating_gap"].to_numpy() + adj_a - adj_b)
    y = df["won"].to_numpy()
    brier = np.mean((p - y) ** 2)
    acc = np.mean((p > 0.5) == (y == 1))
    ll = -np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
    return brier, acc, ll


def main():
    df = build()
    tr, te = df[df.date < HOLDOUT_START], df[df.date >= HOLDOUT_START]
    print(f"{len(df)} fights ({len(tr)} train / {len(te)} frozen holdout)\n")

    b0, a0, l0 = score(te, 0.0, 0.0)
    print(f"BASELINE (no adjustment), holdout")
    print(f"  Brier {b0:.4f}   accuracy {a0:.1%}   log-loss {l0:.4f}\n")

    # Swept on TRAIN only. The holdout is never used to choose parameters --
    # that would make its verdict meaningless.
    best, best_b = None, 1e9
    for k_exp in (0, 10, 20, 30, 40, 60, 80):
        for k_streak in (0, 5, 10, 15, 20):
            b, _, _ = score(tr, k_exp, k_streak)
            if b < best_b:
                best_b, best = b, (k_exp, k_streak)
    k_exp, k_streak = best
    print(f"swept on TRAIN -> k_exp={k_exp}, k_streak={k_streak} (train Brier {best_b:.4f})\n")

    b1, a1, l1 = score(te, k_exp, k_streak)
    print("ADJUSTED, same holdout")
    print(f"  Brier {b1:.4f} ({b1 - b0:+.4f})   accuracy {a1:.1%} ({a1 - a0:+.1%})   "
          f"log-loss {l1:.4f} ({l1 - l0:+.4f})\n")

    # CONTROL: the same magnitudes with the sign flipped. If this scores about
    # as well, the shape is noise rather than structure.
    b2, a2, l2 = score(te, -k_exp, -k_streak)
    print("CONTROL -- same size, opposite sign")
    print(f"  Brier {b2:.4f} ({b2 - b0:+.4f})   accuracy {a2:.1%}   log-loss {l2:.4f}\n")

    print("HOLDOUT BY COHORT (adjusted vs baseline Brier, lower is better)")
    for lo, hi, name in ((0, 3, "0-3 fights"), (4, 8, "4-8 fights"),
                         (9, 15, "9-15 fights"), (16, 99, "16+ fights")):
        m = (te.own_fights >= lo) & (te.own_fights <= hi)
        if m.sum() < 60:
            continue
        c0, _, _ = score(te[m], 0.0, 0.0)
        c1, _, _ = score(te[m], k_exp, k_streak)
        flag = "  better" if c1 < c0 else "  WORSE"
        print(f"  {name:14} n={m.sum():<5} {c0:.4f} -> {c1:.4f}  ({c1 - c0:+.4f}){flag}")

    print(f"\n{'='*62}")
    if b1 < b0 and a1 >= a0 - 0.002 and (b2 - b0) > (b0 - b1) * 0.5:
        print("PASS: better Brier on held-out fights, accuracy not worse, and the")
        print("flipped control is clearly worse. Ship it -- and note the site's")
        print(f"edges will SHRINK on low-experience fighters, which is the point.")
        print(f"\nEXPERIENCE_K = {k_exp}\nSTREAK_K = {k_streak}")
    elif b1 >= b0:
        print("FAIL: no Brier improvement out of sample. The gap to the market is")
        print("real but this parameterisation doesn't convert it into better")
        print("predictions -- don't ship a correction that only looks tidy.")
    else:
        print("WEAK: improves Brier but the control isn't clearly worse, or")
        print("accuracy slipped. Not enough separation to trust.")


if __name__ == "__main__":
    main()
