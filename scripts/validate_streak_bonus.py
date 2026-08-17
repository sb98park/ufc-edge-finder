"""
POINT-IN-TIME re-validation of STREAK_K.

THE CHANGE UNDER TEST. power_rating._streak_bonus adds STREAK_K rating points
per win to a fighter with fewer than nine tracked fights on an active streak,
capped at six wins. At the shipped STREAK_K = 20 that is up to 120 points --
and it is added to the EFFECTIVE RATING, outside ADJUSTMENT_TOTAL_CAP, so
unlike every term in the style layer nothing bounds its contribution against
the others.

WHY RE-MEASURE SOMETHING THAT ALREADY HAS A VALIDATION TRAIL. The trail on
STREAK_K (power_rating.py) reports Brier 0.2432 -> 0.2402 on a frozen 2019+
holdout, and it was honest when written. It is also the OLDEST surviving
validation in the model, and the baseline underneath it has moved four times
since:

    - point-in-time wrestling and striking rates from data/pit_stats.csv,
      which is where a rising fighter's improvement would actually show up
    - the divisional method priors, rebuilt from real UFC bouts
    - the durability Beta shrink at k = 2
    - reference_date threaded into the style layer, enabling the recency term

Every one of those adds information about recent form -- which is precisely
what a streak bonus is a proxy FOR. A crude proxy earns its keep only while
nothing better is measuring the same thing. This harness asks whether it
still does.

ARMS. STREAK_K in {0, 10, 20, 30}, where 0 is the term switched off and 20 is
shipped. The bonus is folded into the effective rating INSIDE the arm loop,
so each arm gets its own blend rather than sharing one -- the durability
harness could hoist that computation because its parameter lived downstream in
predict_matchup, and this one cannot.

Everything else matches the shared harness conventions: corners randomised so
the trivial baseline sits at ~50%, both rosters rebuilt as they stood that
night, point-in-time rate stats, and a paired sign-flip bootstrap clustered by
card.

Usage:  python3 scripts/validate_streak_bonus.py
        python3 scripts/validate_streak_bonus.py --offset 2500
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem  # noqa: E402
from src import power_rating  # noqa: E402
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

    saved = power_rating.STREAK_K
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

                # The stats rating and the Elo lookup do not depend on the arm,
                # so they are computed once; only the bonus varies.
                base_parts = {}
                for name, row, prior, fold in ((a, ra, na, fa), (b, rb, nb, fb)):
                    sr = compute_stats_rating(pd.Series(row))
                    w = min(1.0, prior / 4.0)
                    base_parts[name] = (w * elo.get_rating(name) + (1 - w) * sr, prior, fold)

                y = 1.0 if winner == a else 0.0
                probs = {}
                for k in arms:
                    power_rating.STREAK_K = k
                    eff = {n: bl + _streak_bonus(pr, streaks[fo])
                           for n, (bl, pr, fo) in base_parts.items()}
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
                    # The cohort the term actually touches: the LOWER of the
                    # two fight counts, since a bonus only lands on a fighter
                    # under STREAK_APPLIES_UNDER.
                    thin = min(na, nb)
                    records.append((thin, y, probs, when.date()))

            loser = b if winner == a else a
            if winner in (a, b):
                elo.update_ratings(winner, loser, method=method)
                streaks[_fold(winner)] += 1
                streaks[_fold(loser)] = 0
            counts[fa] += 1
            counts[fb] += 1
    finally:
        power_rating.STREAK_K = saved

    return records


def report(records, arms, base=20):
    n = len(records)
    if not n:
        print("no scored fights")
        return
    print(f"n = {n} scored fights   trivial baseline "
          f"{trivial_baseline([(0.5, r[1]) for r in records]):.1%}\n")
    print(f"{'STREAK_K':>9} {'Brier':>9} {'d vs base':>10} {'p':>8} {'deff':>6} "
          f"{'log loss':>10} {'acc':>7}")
    for k in arms:
        pairs = [(r[2][k], r[1]) for r in records]
        _, acc, brier, ll = _score(pairs)
        if k == base:
            print(f"{k:>9} {brier:>9.5f} {'--':>10} {'--':>8} {'--':>6} {ll:>10.5f} {acc:>7.2%}")
            continue
        d, p, deff = _paired(records, k, base)
        print(f"{k:>9} {brier:>9.5f} {d:>+10.5f} {p:>8.4f} {deff:>6.2f} {ll:>10.5f} {acc:>7.2%}")

    # THE COHORT THE TERM EXISTS FOR. A bonus restricted to fighters under
    # nine fights cannot show its effect in a pooled average dominated by
    # veterans, and a term that helps nowhere it applies is not a term.
    print("\nby thinner corner's fight count:")
    for lo, hi, lbl in ((0, 3, "0-3"), (4, 8, "4-8"), (9, 999, "9+")):
        sub = [r for r in records if lo <= r[0] <= hi]
        if len(sub) < 50:
            continue
        line = f"  {lbl:>5} (n={len(sub):5d})  "
        for k in arms:
            _, _, brier, _ = _score([(r[2][k], r[1]) for r in sub])
            line += f"k={k}: {brier:.5f}   "
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=int, nargs="+", default=[0, 10, 20, 30])
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--offset", type=int, default=0)
    # Which arm the deltas are measured AGAINST. Defaults to the shipped
    # value; set --base 0 to ask the different question of whether the term
    # earns its place at all, rather than whether its size is right.
    ap.add_argument("--base", type=int, default=20)
    a = ap.parse_args()
    label = f"last {a.limit}" + (f" skipping {a.offset}" if a.offset else "")
    print(f"STREAK_K sweep, window: {label}   base = {a.base}\n")
    report(run(a.arms, a.limit, a.offset), a.arms, base=a.base)
