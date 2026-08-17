"""
POINT-IN-TIME validation of CHIN_SCALE -- a measured chin term.

THE CHANGE UNDER TEST. The model's only durability signal is the method split
of a fighter's LOSSES:

    finish_loss_rate = (ko_losses + sub_losses) / losses      [Beta-shrunk]

That has two structural blind spots. It sees nothing in a fight the fighter
SURVIVED, and its denominator is the loss count -- frequently 1 or 2, which is
why it needed a Beta shrink before it was usable at all.

data/pit_stats.csv carries kd_against and fight_seconds on 17,524 fighter-bout
rows and the model reads neither. Knockdowns absorbed per 15 minutes is a
direct measurement of the same underlying quantity, denominated in cage time
rather than in defeats, so it accumulates from every minute fought.

WHY THIS IS AN ADDITION AND NOT A REPLACEMENT. Measured on 4,525 losses from
2012 on, the two are only 0.345 correlated. Each predicts a KO loss about
equally well on its own and neither is close to the pair:

    predictor            5-fold CV log loss   (base rate 0.311)
    finish_loss_rate           0.61301
    kd_against_per_15          0.61334
    BOTH                       0.60985

A swap would have thrown away half the information. The quintile gradient on
the new one is monotone -- 24.9% / 31.9% / 34.7% / 40.2% chance a loss came by
KO/TKO (point-biserial r = 0.122, p = 2e-17).

WHAT REMAINS UNKNOWN, and is what this harness is for: whether a signal that
predicts the METHOD of a loss also improves the probability of WINNING, which
is the quantity the site actually publishes. Those are different questions and
the first does not imply the second -- a fighter with a suspect chin who wins
by decision anyway costs the term nothing in the analysis above and everything
here.

Arms sweep CHIN_SCALE with 0.0 as the control, which is byte-identical to the
shipped model. Corners randomised, both rosters rebuilt as of fight night,
point-in-time rate stats, paired sign-flip bootstrap clustered by card.

Usage:  python3 scripts/validate_chin.py
        python3 scripts/validate_chin.py --offset 2500
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem  # noqa: E402
from src import matchup_model  # noqa: E402
from src.matchup_model import predict_matchup  # noqa: E402
from src.power_rating import compute_stats_rating, _streak_bonus  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of  # noqa: E402
from scripts.build_pit_stats import load_pit_stats, stats_as_of  # noqa: E402
from scripts.harness_stats import (  # noqa: E402
    paired_signflip, randomize_corner, score as _score, trivial_baseline)

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"


def _fold(n) -> str:
    return str(n).strip().lower()


def _paired(rows, arm, base):
    deltas = [(r[2][arm] - r[1]) ** 2 - (r[2][base] - r[1]) ** 2 for r in rows]
    return paired_signflip(deltas, clusters=[r[3] for r in rows])


def run(arms, limit, offset=0):
    fighters = pd.read_csv(FIGHTERS)
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}
    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    fight_index = build_fight_index(history)
    wc = pd.read_csv(WC_HISTORY) if os.path.exists(WC_HISTORY) else None
    pit = load_pit_stats()

    elo = EloRatingSystem()
    counts, streaks = defaultdict(int), defaultdict(int)
    records = []
    trimmed = history.iloc[:-offset] if offset else history
    rows = trimmed.tail(limit) if limit else trimmed
    cutoff = rows.iloc[0]["date"] if len(rows) else None
    ceiling = rows.iloc[-1]["date"] if len(rows) else None

    saved = matchup_model.CHIN_SCALE
    try:
        for _, f in history.iterrows():
            a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
            fa, fb = _fold(a), _fold(b)
            when = f["date"].to_pydatetime()
            na, nb = counts[fa], counts[fb]

            in_window = ((cutoff is None or f["date"] >= cutoff)
                         and (ceiling is None or f["date"] <= ceiling))
            if in_window and na > 0 and nb > 0 and winner in (a, b):
                ra = roster_as_of(a, when, fight_index, static_rows, today=when)
                rb = roster_as_of(b, when, fight_index, static_rows, today=when)
                past = history[history["date"] < f["date"]]
                for row, fold in ((ra, fa), (rb, fb)):
                    row.update(stats_as_of(pit.get(fold, []), when.date()))
                frame = pd.DataFrame([ra, rb])
                # CHIN_SCALE lives downstream in predict_matchup, so unlike the
                # streak sweep the effective rating is arm-independent and can
                # be hoisted out of the loop.
                eff = {}
                for name, row, prior, fold in ((a, ra, na, fa), (b, rb, nb, fb)):
                    sr = compute_stats_rating(pd.Series(row))
                    w = min(1.0, prior / 4.0)
                    eff[name] = w * elo.get_rating(name) + (1 - w) * sr + _streak_bonus(prior, streaks[fold])

                y = 1.0 if winner == a else 0.0
                probs = {}
                for k in arms:
                    matchup_model.CHIN_SCALE = k
                    try:
                        res = predict_matchup(a, b, frame, eff, past, wc,
                                              f.get("weight_class"),
                                              reference_date=when.date())
                    except Exception:
                        res = None
                    p = (res or {}).get("prob_a")
                    if p is not None and not math.isnan(p):
                        probs[k] = p
                if len(probs) == len(arms):
                    probs = {k: randomize_corner(v, y, a, b, when)[0] for k, v in probs.items()}
                    y = randomize_corner(0.5, y, a, b, when)[1]
                    # The differential the term actually sees. Fights where it
                    # is zero are ones the term cannot have moved, and pooling
                    # them in hides the effect in a mass of untouched rows.
                    ka, kb = ra.get("kd_against_per_15"), rb.get("kd_against_per_15")
                    gap = (abs(float(ka) - float(kb))
                           if ka is not None and kb is not None else None)
                    records.append((gap, y, probs, when.date()))

            # Advance the running state on EVERY fight in history, not just
            # the scored window -- the ratings a windowed fight is judged
            # against have to reflect everything that came before it.
            loser = b if winner == a else a
            if winner in (a, b):
                elo.update_ratings(winner, loser, method=method)
                streaks[_fold(winner)] += 1
                streaks[_fold(loser)] = 0
            counts[fa] += 1
            counts[fb] += 1
    finally:
        matchup_model.CHIN_SCALE = saved

    return records


def report(records, arms, base=0.0):
    n = len(records)
    if not n:
        print("no scored fights")
        return
    fired = sum(1 for r in records if r[0] is not None)
    print(f"n = {n} scored fights   trivial baseline "
          f"{trivial_baseline([(0.5, r[1]) for r in records]):.1%}")
    print(f"term has data on {fired} of {n} ({fired / n:.1%})\n")
    print(f"{'CHIN_SCALE':>11} {'Brier':>9} {'d vs 0':>10} {'p':>8} {'deff':>6} "
          f"{'log loss':>10} {'acc':>7}")
    for k in arms:
        pairs = [(r[2][k], r[1]) for r in records]
        _, acc, brier, ll = _score(pairs)
        if k == base:
            print(f"{k:>11} {brier:>9.5f} {'--':>10} {'--':>8} {'--':>6} {ll:>10.5f} {acc:>7.2%}")
            continue
        d, p, deff = _paired(records, k, base)
        print(f"{k:>11} {brier:>9.5f} {d:>+10.5f} {p:>8.4f} {deff:>6.2f} {ll:>10.5f} {acc:>7.2%}")

    # ON THE FIGHTS IT CAN ACTUALLY MOVE. A term with data on two thirds of
    # fights is diluted by a third of rows where every arm is identical, and
    # the wide-gap subset is where a real effect has to show up if it exists.
    subsets = [("has data", lambda g: g is not None),
               ("gap > 0.25", lambda g: g is not None and g > 0.25),
               ("gap > 0.50", lambda g: g is not None and g > 0.50)]
    print("\nby chin differential:")
    for lbl, keep in subsets:
        sub = [r for r in records if keep(r[0])]
        if len(sub) < 100:
            continue
        line = f"  {lbl:>11} (n={len(sub):5d})  "
        for k in arms:
            _, _, brier, _ = _score([(r[2][k], r[1]) for r in sub])
            line += f"{k}: {brier:.5f}  "
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=float, nargs="+", default=[0.0, 15.0, 30.0, 60.0])
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    label = f"last {a.limit}" + (f" skipping {a.offset}" if a.offset else "")
    print(f"CHIN_SCALE sweep, window: {label}\n")
    report(run(a.arms, a.limit, a.offset), a.arms)
