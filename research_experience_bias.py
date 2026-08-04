"""
Does the model underrate fast risers -- fighters with few UFC fights on a
streak?

THE HYPOTHESIS, and where it came from. Quillan Salkilld is 12-1 with five
straight UFC wins including a first-round KO of Beneil Dariush, a long-time
top-ten lightweight. The market makes him a -154 favourite; our model has him
at 36.3%, a 20-point disagreement on a liquid line.

The suspected mechanism is structural rather than a bug. Elo accumulates over
fights: a rating reflects HOW MANY you have won more than WHO you beat. A
fighter with five UFC fights is still climbing from a base no matter how good
those wins were, while an opponent with twenty-five sits high on history. The
market repriced Salkilld the night of that KO; a rating system needs several
more fights to catch up.

This tests it against every resolved pick rather than arguing about one fight.

WHY THE CLOSING LINE IS THE YARDSTICK. Outcomes on a few hundred picks are
too noisy to separate "underrated" from "got lucky". The closing line is the
market's final estimate and is well calibrated in aggregate, so a systematic
gap between our number and the close -- inside one cohort and not others --
is the signal. Outcome accuracy is reported alongside as a sanity check, not
as the primary test.

NOTE THE DIRECTION MATTERS. An earlier finding went the other way: on DEBUT
fights the model was too GENEROUS, assigning 48.2% where debutants won 43.3%.
If this comes back the same direction for experienced-but-few-fight risers,
the two findings conflict and neither should be acted on. If it comes back
opposite, the picture is coherent -- generous to the untested, slow to update
on the newly proven.

Run: python3 research_experience_bias.py
"""

import os
import sys

import numpy as np
import pandas as pd

import research_divergence_historical as D
import research_survival_model as R
from src.elo import EloRatingSystem

MIN_COHORT = 40


def build():
    """
    Every historical fight, with each side's UFC fight count and current win
    streak AS AT that fight, plus the model's probability and the closing line.
    """
    # load_market returns a DICT keyed by frozenset({folded_a, folded_b}),
    # each value carrying both fighters' folded names and their odds -- not a
    # DataFrame. Keying by an unordered pair is what makes it robust to the
    # two sources listing fighters in opposite order.
    market = D.load_market()
    if not market:
        print("No external odds -- drop the closing-line CSV at data/external_odds.csv.")
        return None

    fights = R.load_dated_fights()
    elo = EloRatingSystem()
    seen, streak, rows = {}, {}, []

    for f in fights.itertuples(index=False):
        a, b = f.fighter_1, f.fighter_2
        na, nb = seen.get(a, 0), seen.get(b, 0)
        sa, sb = streak.get(a, 0), streak.get(b, 0)

        # Point in time: rating and experience BEFORE this fight resolves.
        ra, rb = elo.get_rating(a), elo.get_rating(b)
        model_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

        rec = market.get(frozenset({D.fold(a), D.fold(b)}))
        close = None
        if rec:
            # Odds are stored per fighter, so pick the side matching THIS row
            # and devig against the other -- a raw implied probability carries
            # the book's margin and would bias every comparison the same way.
            odds_own = rec["odds_a"] if rec["a"] == D.fold(a) else rec["odds_b"]
            odds_opp = rec["odds_b"] if rec["a"] == D.fold(a) else rec["odds_a"]
            try:
                p_own, p_opp = D.implied(float(odds_own)), D.implied(float(odds_opp))
                if p_own and p_opp and (p_own + p_opp) > 0:
                    close = p_own / (p_own + p_opp)
            except (TypeError, ValueError):
                close = None
        if close is not None and na + nb > 0:
            rows.append({
                "date": f.date, "fighter": a, "opponent": b,
                "model_prob": model_a, "close_prob": close,
                "won": 1 if f.winner == a else 0,
                "own_fights": na, "opp_fights": nb,
                "own_streak": sa, "opp_streak": sb,
            })

        winner = f.winner
        loser = b if winner == a else a
        elo.update_ratings(winner, loser, method=str(f.method))
        seen[a] = na + 1
        seen[b] = nb + 1
        streak[winner] = streak.get(winner, 0) + 1
        streak[loser] = 0
    return pd.DataFrame(rows)


def report(df, mask, label):
    sub = df[mask]
    if len(sub) < MIN_COHORT:
        print(f"  {label:34} n={len(sub):<5} too few to judge")
        return None
    gap = (sub["model_prob"] - sub["close_prob"]).mean()
    acc = (sub["won"] == (sub["model_prob"] > 0.5)).mean()
    actual = sub["won"].mean()
    predicted = sub["model_prob"].mean()
    flag = "  <-- MODEL LOW" if gap < -0.03 else ("  <-- MODEL HIGH" if gap > 0.03 else "")
    print(f"  {label:34} n={len(sub):<5} model-close {gap:+.1%}   "
          f"predicted {predicted:.1%} actual {actual:.1%}   acc {acc:.1%}{flag}")
    return gap


def main():
    df = build()
    if df is None or df.empty:
        return
    print(f"{len(df)} fights with a closing line\n")

    print("BY THE FIGHTER'S OWN UFC FIGHT COUNT")
    print("  (negative model-close = our number is BELOW the market's)")
    for lo, hi, name in ((0, 3, "0-3 fights"), (4, 8, "4-8 fights"),
                         (9, 15, "9-15 fights"), (16, 99, "16+ fights")):
        report(df, (df.own_fights >= lo) & (df.own_fights <= hi), name)

    print("\nTHE HYPOTHESIS: few fights AND on a streak")
    for lo, hi, name in ((0, 8, "<=8 fights"), (0, 8, "<=8 fights, streak >=3"),
                         (0, 8, "<=8 fights, streak >=5")):
        if "streak >=3" in name:
            m = (df.own_fights <= hi) & (df.own_streak >= 3)
        elif "streak >=5" in name:
            m = (df.own_fights <= hi) & (df.own_streak >= 5)
        else:
            m = (df.own_fights <= hi)
        report(df, m, name)

    print("\nTHE MIRROR: the same fighters' OPPONENTS")
    print("  (if we underrate risers, we should overrate whoever faces them)")
    report(df, (df.opp_fights <= 8) & (df.opp_streak >= 3), "facing a <=8-fight riser on a streak")

    print("\nCONTROL: experienced fighters on a streak")
    report(df, (df.own_fights >= 16) & (df.own_streak >= 3), "16+ fights, streak >=3")

    riser = df[(df.own_fights <= 8) & (df.own_streak >= 3)]
    rest = df[~((df.own_fights <= 8) & (df.own_streak >= 3))]
    print(f"\n{'='*66}")
    if len(riser) < MIN_COHORT:
        print("Not enough riser fights to conclude anything.")
        return
    g_r = (riser["model_prob"] - riser["close_prob"]).mean()
    g_o = (rest["model_prob"] - rest["close_prob"]).mean()
    print(f"risers   model-close {g_r:+.2%}  (n={len(riser)})")
    print(f"everyone else        {g_o:+.2%}  (n={len(rest)})")
    print(f"difference           {g_r - g_o:+.2%}")
    print()
    if g_r - g_o < -0.03:
        print("CONFIRMED: the model sits meaningfully below the market on fast")
        print("risers specifically. That is a correctable bias -- and it means")
        print("fading your own model in this spot, which is the opposite of")
        print("what the raw edge suggests.")
    elif abs(g_r - g_o) <= 0.03:
        print("NOT CONFIRMED: risers look like everyone else. The Salkilld")
        print("disagreement is then a one-off, not a pattern -- which makes it")
        print("weaker evidence either way, not stronger.")
    else:
        print("OPPOSITE: the model reads HIGHER than the market on risers.")
        print("That contradicts the hypothesis; don't act on either.")


if __name__ == "__main__":
    main()
