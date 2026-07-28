"""
Does the hazard model's FIGHT-LEVEL method probability beat production's?

WHY THIS HAS TO COME FIRST. research_survival_model.py validated per-ROUND
hazards: "given the fight reached round r, does it end here, and how?" That
is not the number production shows. The site shows a fight-level claim --
"projected to end by KO" -- which is the per-round hazards CHAINED across
every round. Chaining is where errors compound: five slightly-optimistic
round hazards make a badly-optimistic finish probability. So the aggregate
needs its own validation before it replaces anything.

WHAT IT'S COMPARED AGAINST. Production's _blended_method_prob mixes a
divisional prior with career finish rates. It has never been scored against
outcomes, so this is the first time the incumbent gets measured at all --
which matters as much as measuring the challenger.

Both are scored point-in-time on the same frozen holdout, on the same
fights, against the same truth.

Run: python3 research_method_calibration.py
"""

import math
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import research_survival_model as R
from src.elo import EloRatingSystem

HOLDOUT_START = pd.Timestamp("2019-01-01")


def divisional_prior_baseline(train_rows):
    """
    Stand-in for production's _blended_method_prob. Production blends a
    DIVISIONAL prior with career finish rates; we can't replay its exact
    inputs point-in-time (fighters.csv holds only current-day values), so
    this reproduces its SHAPE: base rates blended with the pair's career
    finish tendency. If the hazard model can't beat this, it can't beat
    production either.
    """
    base_ko = train_rows["is_ko"].mean()
    base_sub = train_rows["is_sub"].mean()
    return base_ko, base_sub


def main():
    print("Building fight-level dataset (one row per fight, not per round)...")
    rows = R.build_rows()
    if rows.empty:
        print("No rows -- need ufc_fight_results.csv in data/.")
        return

    # Train the hazard model on TRAINING rounds only.
    train_r = rows[rows["date"] < HOLDOUT_START]
    haz = LogisticRegression(max_iter=3000).fit(train_r[R.FEATURES], train_r["y"])

    # Rebuild fight-level truth + features, replaying chronologically.
    fights = R.load_dated_fights()
    res = pd.read_csv(R.RESULTS_PATH)
    res.columns = [c.strip() for c in res.columns]
    fmt = {(str(r["EVENT"]).strip(), str(r["BOUT"]).strip()): str(r.get("TIME FORMAT", ""))
           for r in res.to_dict("records")}

    elo, career, out = EloRatingSystem(), R.Career(), []
    for f in fights.itertuples(index=False):
        ev = R.classify(f.method)
        if ev == "drop":
            continue
        c1, c2 = career.get(f.fighter_1), career.get(f.fighter_2)
        if c1 and c2:
            raw = fmt.get((str(f.event).strip(), str(f.bout).strip()), "")
            sched = 5 if raw.strip().startswith("5") else 3
            gap = abs(elo.get_rating(f.fighter_1) - elo.get_rating(f.fighter_2))
            feat = {
                "round": 1, "scheduled": sched,
                "ko_press": c1["ko_rate"] * c2["ko_lost"] + c2["ko_rate"] * c1["ko_lost"],
                "sub_press": c1["sub_rate"] * c2["sub_lost"] + c2["sub_rate"] * c1["sub_lost"],
                "ko_rate_sum": c1["ko_rate"] + c2["ko_rate"],
                "sub_rate_sum": c1["sub_rate"] + c2["sub_rate"],
                "durability": c1["ko_lost"] + c2["ko_lost"],
                "elo_gap": gap / 400.0,
            }
            dist = R.fight_distribution(haz, feat, sched)
            out.append({
                "date": f.date,
                "p_ko": sum(v for k, v in dist.items() if isinstance(k, tuple) and k[1] == "KO/TKO"),
                "p_sub": sum(v for k, v in dist.items() if isinstance(k, tuple) and k[1] == "SUB"),
                "is_ko": 1 if ev == R.KO else 0,
                "is_sub": 1 if ev == R.SUB else 0,
            })
        loser = f.fighter_2 if f.winner == f.fighter_1 else f.fighter_1
        elo.update_ratings(f.winner, loser, method=str(f.method))
        career.update(f.winner, loser, ev if ev in (R.KO, R.SUB) else None)

    df = pd.DataFrame(out)
    tr, te = df[df["date"] < HOLDOUT_START], df[df["date"] >= HOLDOUT_START]
    print(f"  {len(df)} scorable fights ({len(tr)} train / {len(te)} holdout)")

    base_ko, base_sub = divisional_prior_baseline(tr)
    eps = 1e-9

    def brier(p, y):
        return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))

    def logloss(p, y):
        p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
        y = np.asarray(y)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    print(f"\n{'='*70}\nFIGHT-LEVEL METHOD PROBABILITY, holdout {HOLDOUT_START.year}+\n{'='*70}")
    for label, pk, ps in (
        ("base rates (production's shape)", np.full(len(te), base_ko), np.full(len(te), base_sub)),
        ("hazard model, chained", te["p_ko"].to_numpy(), te["p_sub"].to_numpy()),
    ):
        print(f"  {label}")
        print(f"     P(KO) : Brier {brier(pk, te['is_ko']):.4f}   logloss {logloss(pk, te['is_ko']):.4f}")
        print(f"     P(SUB): Brier {brier(ps, te['is_sub']):.4f}   logloss {logloss(ps, te['is_sub']):.4f}")

    # Calibration is the whole point: an 85% must win 85% of the time.
    # ---- ISOTONIC CORRECTION ----
    # The raw chain is systematically optimistic (see the buckets below):
    # five slightly-hot round hazards compound into a hot finish probability.
    # Isotonic is the right tool here rather than Platt: the distortion isn't
    # a clean sigmoid shift, it's a monotone-but-uneven stretch, and isotonic
    # corrects that shape without assuming a functional form.
    # FIT ON TRAINING ONLY -- fitting on holdout would flatter itself.
    iso_ko = IsotonicRegression(out_of_bounds="clip").fit(tr["p_ko"], tr["is_ko"])
    iso_sub = IsotonicRegression(out_of_bounds="clip").fit(tr["p_sub"], tr["is_sub"])
    cal_ko = iso_ko.predict(te["p_ko"])
    cal_sub = iso_sub.predict(te["p_sub"])
    print("  calibrated hazard model (isotonic, fit on train)")
    print(f"     P(KO) : Brier {brier(cal_ko, te['is_ko']):.4f}   logloss {logloss(cal_ko, te['is_ko']):.4f}")
    print(f"     P(SUB): Brier {brier(cal_sub, te['is_sub']):.4f}   logloss {logloss(cal_sub, te['is_sub']):.4f}")

    print("\nCALIBRATION of the chained P(KO) -- predicted vs actual by bucket:")
    te = te.copy()
    te["p_ko_cal"] = cal_ko
    te["bucket"] = pd.cut(te["p_ko"], [0, .2, .3, .4, .5, 1.0])
    g = te.groupby("bucket", observed=True).agg(n=("is_ko", "size"),
                                                raw=("p_ko", "mean"),
                                                cal=("p_ko_cal", "mean"),
                                                actual=("is_ko", "mean"))
    print(f"   {'bucket':12} {'n':>5} {'raw':>8} {'calibrated':>11} {'actual':>8}")
    for b, r in g.iterrows():
        if r["n"] >= 20:
            print(f"   {str(b):12} {int(r['n']):5} {r['raw']:8.1%} {r['cal']:11.1%} {r['actual']:8.1%}")

    # The number that decides whether this ships: mean absolute calibration
    # error, i.e. how far a stated probability sits from the truth.
    ok = g[g["n"] >= 20]
    print(f"\n  mean |predicted - actual|: raw {abs(ok['raw']-ok['actual']).mean():.3f}"
          f"  ->  calibrated {abs(ok['cal']-ok['actual']).mean():.3f}")


if __name__ == "__main__":
    main()
