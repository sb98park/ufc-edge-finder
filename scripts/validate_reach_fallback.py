"""
POINT-IN-TIME validation of what to substitute when a fighter's reach is
unknown.

THE CHANGE UNDER TEST. compute_stats_rating adds 4.0 * (reach_in - 70) and
falls back to 70 when reach is missing -- which is exactly the term's centring
constant, so a missing reach contributes precisely zero. That is not a neutral
choice. The roster mean is 72.1 and division means run 64.1 (strawweight) to
77.8 (heavyweight), so the flat fallback quietly under-rates every heavyweight
with no reach on file and over-credits every strawweight.

TWO MEASUREMENTS, AND THE FIRST IS THE STRONGER ONE.

(1) HOW WRONG IS THE ESTIMATE. Leave-one-out absolute error against the 353
    roster fighters whose reach we do hold. This is direct, has real sample
    size, and needs no model at all. Reported by --estimators.

(2) DOES A BETTER ESTIMATE PREDICT BETTER. The paired-arm run below. It is
    UNDERPOWERED BY CONSTRUCTION and the number to keep in view is 93, not the
    p-value -- see the population note.

WHY THE POPULATION IS SO SMALL, and why that is a fact about the model rather
than a flaw in the harness. build_effective_ratings blends Elo against
compute_stats_rating with weight = min(1, n_prior/4), so at four or more
connected bouts the stats rating -- and with it the whole reach term -- is
multiplied by ZERO. Reach can only move a prediction for a fighter with three
or fewer connected fights. That is the same regime trap the finish-rate term
died in, and it is worth stating plainly: this parameter is a low-experience
prior and nothing else.

Of 11,925 spine rows, 589 have both corners on the current roster with a reach
and height we can ablate against, and 93 of those have a corner where the term
still carries weight. Nothing can raise that; it is the intersection of "we
know the truth" and "the truth matters".

THE ARMS. One corner per fight is blinded -- the one with fewer connected
bouts, i.e. where the term has the most weight -- and each arm substitutes a
different value for it. All estimates are LEAVE-ONE-OUT: the blinded fighter
is excluded from the fit, matching production, where a fighter with no reach
is by definition absent from the fitting set.

    control   70          the shipped fallback
    roster    72.1        roster mean; isolates "70 is simply too low"
    division  by weight class
    height    -3.25 + 1.073 * height_in   (corr 0.910, R2 0.828)
    oracle    the true reach

ORACLE IS THE POINT OF THE DESIGN. It is the ceiling: the best any fallback
could do is match it. If oracle does not beat control, reach is not carrying
signal in this population and no fallback can be justified on outcomes.

RESULT, 2026-08-31. The two measurements disagree about how much can be
claimed, and the honest reading is the weaker one.

  leave-one-out estimator error, n=353
    control (70)   MAE 4.08 in   16.3 rating pts   worst 14.00 in
    roster (72.1)      3.78      15.1                   13.09
    division           2.48       9.9                   12.40
    height             1.54       6.2                    5.07

  point-in-time paired arms, 92 fights scored, 493 ineligible
    arm          acc     brier   logloss    d.brier      p
    control   0.5761   0.24425   0.68296        --     --
    roster    0.5652   0.24394   0.68186  -0.00031  0.704
    division  0.5652   0.24358   0.68078  -0.00067  0.687
    height    0.5652   0.24250   0.67849  -0.00175  0.306
    oracle    0.5761   0.24119   0.67557  -0.00306  0.095

NOTHING HERE IS SIGNIFICANT, INCLUDING THE ORACLE. At 92 fights, even
knowing every fighter's true reach does not separate from the flat 70
(p=0.095). So this harness CANNOT support a claim that a better fallback
predicts better, and no such claim should be made from it. The accuracy
column moves by one fight and means nothing.

What it does show is that the four arms rank in exactly the order their
estimation error predicts, monotonically, on both Brier and logloss. That is
corroboration of direction, not evidence of effect.

The case for changing the fallback therefore rests on the ESTIMATOR table,
which needs no outcome data: 70 is the reach term's own centring constant and
is wrong by 4.08 inches on average, where a two-parameter fit on height --
already on file for 15 of the 16 affected fighters -- is wrong by 1.54, with a
worst case of 5.07 inches against 14.00. That is a measurement-quality
argument about an input the model has already decided to use, not a tuning
argument, and it should be made and judged as one.

Run:  python3 scripts/validate_reach_fallback.py --estimators
      python3 scripts/validate_reach_fallback.py
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem, ufc_only                          # noqa: E402
from src.matchup_model import predict_matchup                          # noqa: E402
from src.power_rating import (RATING_CENTER, DEBUT_RATING_SHRINK,      # noqa: E402
                              _streak_bonus, compute_stats_rating)
from scripts.pit_roster import build_fight_index, roster_as_of         # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"

# Below this many connected bouts the reach term still carries weight.
TERM_ALIVE_BELOW = 4
# A division needs this many known reaches before its own mean beats the
# roster mean as an estimate.
MIN_DIVISION = 8


def _fold(n) -> str:
    return str(n).strip().lower()


def build_estimators(fighters: pd.DataFrame) -> dict:
    """Leave-one-out fallback estimates, per fighter, for each arm."""
    k = fighters.dropna(subset=["reach_in", "height_in"]).copy()
    k["weight_class"] = k["weight_class"].fillna("Unknown")
    k = k.reset_index(drop=True)
    gsum = k.groupby("weight_class")["reach_in"].sum()
    gcnt = k.groupby("weight_class")["reach_in"].count()
    tot, n = k["reach_in"].sum(), len(k)

    out = {}
    for i, row in k.iterrows():
        t, wc, h = row["reach_in"], row["weight_class"], row["height_in"]
        o = k.drop(i)
        b, a = np.polyfit(o["height_in"], o["reach_in"], 1)
        div = (gsum[wc] - t) / (gcnt[wc] - 1) if gcnt[wc] > MIN_DIVISION else (tot - t) / (n - 1)
        out[_fold(row["name"])] = {
            "control": None,                    # -> compute_stats_rating's own 70
            "roster": (tot - t) / (n - 1),
            "division": div,
            "height": a + b * h,
            "oracle": t,
        }
    return out


def report_estimators(fighters: pd.DataFrame) -> None:
    est = build_estimators(fighters)
    k = fighters.dropna(subset=["reach_in", "height_in"])
    truth = {_fold(r["name"]): r["reach_in"] for _, r in k.iterrows()}
    print(f"LEAVE-ONE-OUT absolute error against a known reach, n={len(truth)}\n")
    print(f"  {'arm':10s} {'MAE in':>8s} {'MAE pts':>9s} {'p90 in':>8s} {'max in':>8s}")
    for arm in ("control", "roster", "division", "height", "oracle"):
        e = [abs((70.0 if est[f][arm] is None else est[f][arm]) - t)
             for f, t in truth.items()]
        e = np.array(e)
        print(f"  {arm:10s} {e.mean():8.2f} {e.mean()*4:9.1f} "
              f"{np.percentile(e,90):8.2f} {e.max():8.2f}")


def _effective(row: dict, n_prior: int, elo_r: float, streak: int) -> float:
    sr = compute_stats_rating(pd.Series(row))
    if n_prior == 0:
        eff = RATING_CENTER + (sr - RATING_CENTER) * DEBUT_RATING_SHRINK
    else:
        w = min(1.0, n_prior / 4.0)
        eff = w * elo_r + (1 - w) * sr
    return eff + _streak_bonus(n_prior, streak)


def _score(pairs):
    n = len(pairs)
    acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in pairs) / n
    return n, acc, brier, ll


def _paired_test(rows, arm, base, n_boot=4000, seed=12345):
    deltas = [(pr[arm] - y) ** 2 - (pr[base] - y) ** 2 for y, pr in rows]
    if not deltas:
        return 0.0, 1.0
    obs = sum(deltas) / len(deltas)
    rnd = random.Random(seed)
    hits = 0
    for _ in range(n_boot):
        s = sum(d if rnd.random() < 0.5 else -d for d in deltas)
        if abs(s / len(deltas)) >= abs(obs):
            hits += 1
    return obs, hits / n_boot


ARMS = ("control", "roster", "division", "height", "oracle")


def run():
    fighters = pd.read_csv(FIGHTERS)
    est = build_estimators(fighters)
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}

    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date", kind="stable")
    # Elo and the connected-bout count must agree; both are UFC-only.
    history = ufc_only(history)
    fight_index = build_fight_index(history)
    wc = pd.read_csv(WC_HISTORY) if os.path.exists(WC_HISTORY) else None

    elo = EloRatingSystem()
    counts = defaultdict(int)
    streaks = defaultdict(int)
    rows_out, skipped, ineligible = [], 0, 0

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()

        if winner in (a, b):
            if fa in est and fb in est:
                # Blind the corner with fewer connected bouts -- where the
                # reach term retains the most weight. Deterministic.
                if counts[fa] != counts[fb]:
                    blind = fa if counts[fa] < counts[fb] else fb
                else:
                    blind = min(fa, fb)
                if counts[blind] >= TERM_ALIVE_BELOW:
                    ineligible += 1
                else:
                    ra = roster_as_of(a, when, fight_index, static_rows, today=when)
                    rb = roster_as_of(b, when, fight_index, static_rows, today=when)
                    past = history[history["date"] < f["date"]]
                    y = 1.0 if winner == a else 0.0
                    probs = {}
                    for arm in ARMS:
                        rows2 = []
                        for name, fold_, r in ((a, fa, ra), (b, fb, rb)):
                            rr = dict(r)
                            if fold_ == blind:
                                v = est[fold_][arm]
                                rr["reach_in"] = None if v is None else float(v)
                            rows2.append(rr)
                        eff = {
                            a: _effective(rows2[0], counts[fa], elo.get_rating(a), streaks[fa]),
                            b: _effective(rows2[1], counts[fb], elo.get_rating(b), streaks[fb]),
                        }
                        try:
                            res = predict_matchup(a, b, pd.DataFrame(rows2), eff, past,
                                                  wc, None, reference_date=when.date())
                        except Exception:
                            res = None
                        p = (res or {}).get("prob_a")
                        if p is not None and not math.isnan(p):
                            probs[arm] = p
                    if len(probs) == len(ARMS):
                        rows_out.append((y, probs))
                    else:
                        skipped += 1

            loser = b if winner == a else a
            elo.update_ratings(winner, loser, method=method)
            streaks[_fold(winner)] += 1
            streaks[_fold(loser)] = 0
        counts[fa] += 1
        counts[fb] += 1

    return rows_out, skipped, ineligible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimators", action="store_true",
                    help="report leave-one-out estimator error only (n=353) "
                         "and skip the point-in-time run")
    args = ap.parse_args()

    fighters = pd.read_csv(FIGHTERS)
    report_estimators(fighters)
    if args.estimators:
        return 0

    rows, skipped, ineligible = run()
    print(f"\n\nPOINT-IN-TIME paired arms, one corner blinded per fight\n")
    print(f"  {len(rows)} fights scored, {skipped} skipped (an arm could not predict), "
          f"{ineligible} ineligible (blinded corner already at "
          f"{TERM_ALIVE_BELOW}+ connected bouts, where the reach term is "
          f"multiplied by zero)\n")
    if not rows:
        print("  Nothing to score.")
        return 0

    print(f"  {'arm':10s} {'acc':>8s} {'brier':>9s} {'logloss':>9s} {'d.brier':>10s} {'p':>7s}")
    print("  " + "-" * 58)
    for arm in ARMS:
        n, acc, brier, ll = _score([(pr[arm], y) for y, pr in rows])
        if arm == "control":
            print(f"  {arm:10s} {acc:8.4f} {brier:9.5f} {ll:9.5f} {'--':>10s} {'--':>7s}")
        else:
            d, p = _paired_test(rows, arm, "control")
            print(f"  {arm:10s} {acc:8.4f} {brier:9.5f} {ll:9.5f} {d:+10.5f} {p:7.3f}")
    print("\n  d.brier is the arm MINUS the control, so negative is better.")
    print("  ORACLE IS THE CEILING: no fallback can beat knowing the real reach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
