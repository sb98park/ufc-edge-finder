"""
Is the model's confidence earned, and can temperature scaling fix it?

THE FINDING THAT PROMPTED THIS. validate_debutant_shrink.py --control
bucketed 7,059 NON-debut fights against their own hit rate and found the
model overstates almost everywhere, worst around 75-85%:

    says 72.4% -> wins 64.8%   (n=792)
    says 77.2% -> wins 65.6%   (n=512)
    says 82.5% -> wins 71.6%   (n=310)

That is ordinary fights, not debuts, and it sits where the site stakes 5 and
10 units. If real it matters more than anything else measured this week.

THE CONFOUND, WHICH IS THE POINT OF THIS SCRIPT. That number came from a
backtest, and a backtest is NOT production. Records and dates are
reconstructed point-in-time by pit_roster, but fighters who have since left
the roster carry no height, reach, age or stat columns at all, so many style
terms gate off and the model runs on thinner input than it ever does live.
A model fed less than it was tuned for would look overconfident even if the
production one is fine. So every table below is also cut by whether BOTH
corners are on the current roster -- the subset that actually resembles a
live prediction -- and by era, since coverage improves sharply over time.

Only if the overconfidence survives those cuts is it worth correcting.

THE CORRECTION UNDER TEST. Temperature scaling, the standard one-parameter
fix for a model whose ordering is good but whose probabilities are too
extreme:

    logit(p') = logit(p) / T          T > 1 shrinks toward 0.5

It cannot change a single pick -- p and p' sit on the same side of 0.5 -- so
accuracy is invariant by construction and only Brier and log loss can move.
That is the right shape here: the complaint is not that the picks are wrong,
it is that they are sold too loudly.

NO FITTING ON THE TEST SET. T is fit by minimising log loss on fights
strictly BEFORE a cutoff and scored only on fights after it. T is also fit
separately on each half and reported, because a temperature that swings
between eras is describing that era's data coverage rather than the model.

RE-VALIDATED AFTER THE RECENCY FIX. This harness originally scored a model
with recent_form silenced (see audit_term_coverage.py), which matters here
more than anywhere: the subject is whether the probabilities are too
extreme, and recency moves them. Now threaded. Accuracy 62.5% -> 62.9%, and
the modern-era calibration TIGHTENED rather than moved the other way:

    bucket      before        after
    72.4%       -2.6pp        -1.3pp
    77.3%       -3.5pp        -1.7pp
    fitted T     1.097         1.131

The conclusion is unchanged and better supported. T=2.066 fit on pre-2021
is still significantly worse after it (Brier +0.0036, p=0.009); T=1.15 is
still a wash (p=0.193). The 82.3% bucket reads -7.8pp on n=98 against
+5.2pp at 87.1% on n=78 -- noise in the tail, not a trend.

THE ANSWER: NOTHING TO CORRECT. The overconfidence is an artifact of old
fights, and it has already gone. Fitting T separately per era:

    1993-2010   T = 2.858     says 77.5% -> wins 47.2%
    2010-2016   T = 2.330     says 77.1% -> wins 63.9%
    2016-2021   T = 1.697     says 77.3% -> wins 63.9%
    2021-2030   T = 1.097     says 77.2% -> wins 73.6%

A monotone fall toward 1.0 is the signature of a coverage problem curing
itself, not of a miscalibrated model. Modern fights have the stat columns
the style layer needs; 2011 fights do not.

On the held-out modern era (n=2577, never used for fitting) every bucket
sits within 1-4pp, and the top two are UNDER-confident by +4.1 and +3.4. The
model is, if anything, slightly conservative where it is most sure.

The held-out test is unambiguous. T=2.042, fit honestly on everything before
2021, is significantly WORSE when applied after it -- Brier +0.0038,
p=0.004 -- and turns a 72% bucket that wins 70% into one that claims 72%
and wins 92.5%. Overcorrecting is not a safe direction; it just moves the
error to the other side. T=1.15 is a wash (Brier -0.0003, p=0.396), which
is what "already calibrated" looks like.

So no temperature ships, and predict_matchup is unchanged. The earlier
-11.6pp figure was real arithmetic over a population that is 63% pre-2021,
and it does not describe the model that runs today.

WHY THE PRODUCTION-LIKE CUT COULD NOT SETTLE THIS ON ITS OWN. Only 395 of
7,059 non-debut fights have both corners on the current roster -- fighters.csv
covers who fights NOW, so any historical bout has a retired corner more
often than not. That subset has no coverage at all above 75% and fits
T=1.85 on n=395 with no room left to hold anything out. It is reported
above for completeness and should not be read as evidence; the era trend on
the full population is what carries the argument.

Usage:  python3 scripts/validate_probability_calibration.py
        python3 scripts/validate_probability_calibration.py --cutoff 2021-01-01
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem  # noqa: E402
from src.power_rating import RATING_CENTER, compute_stats_rating, _streak_bonus  # noqa: E402
from src.matchup_model import predict_matchup  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of  # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"


def _fold(n) -> str:
    return str(n).strip().lower()


def _effective(row, n_prior, elo_r, streak):
    stats_rating = compute_stats_rating(pd.Series(row))
    if n_prior == 0:
        eff = stats_rating
    else:
        weight = min(1.0, n_prior / 4.0)
        eff = weight * elo_r + (1 - weight) * stats_rating
    return eff + _streak_bonus(n_prior, streak)


def temper(p, t):
    """logit(p)/T back through the sigmoid. T=1 is the identity."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    if t == 1.0:
        return p
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) / t)))


def fit_temperature(rows, lo=0.5, hi=5.0, iters=60):
    """
    Golden-section-free ternary search on log loss. The objective is convex
    in T for a fixed set of logits, so a bracketing search is enough and
    there is no need to pull in scipy for one parameter.
    """
    def loss(t):
        s = 0.0
        for p, y in rows:
            q = min(max(temper(p, t), 1e-9), 1 - 1e-9)
            s -= y * math.log(q) + (1 - y) * math.log(1 - q)
        return s / max(len(rows), 1)
    for _ in range(iters):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if loss(m1) < loss(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def _score(pairs):
    n = len(pairs)
    acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in pairs) / n
    return n, acc, brier, ll


def _paired(rows, t, n_boot=4000, seed=12345):
    """Paired sign-flip bootstrap on the per-fight change in squared error."""
    deltas = [(temper(p, t) - y) ** 2 - (p - y) ** 2 for p, y in rows]
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


def calib(rows, title):
    print(f"\n{title}  (n={len(rows)})")
    if not rows:
        return
    print(f"  {'model says':<14}{'actually wins':<16}{'gap':<10}{'n'}")
    print("  " + "-" * 46)
    for lo, hi in [(.5, .6), (.6, .7), (.7, .75), (.75, .8), (.8, .85), (.85, .9), (.9, 1.01)]:
        g = [(p, y) for p, y in rows if lo <= max(p, 1 - p) < hi]
        if len(g) >= 25:
            said = sum(max(p, 1 - p) for p, _ in g) / len(g)
            hit = sum(1 for p, y in g if (p >= .5) == (y == 1.)) / len(g)
            print(f"  {said:<14.1%}{hit:<16.1%}{hit-said:<+10.1%}{len(g)}")
    print(f"  fitted T on this subset: {fit_temperature(rows):.3f}")


def run():
    fighters = pd.read_csv(FIGHTERS)
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}

    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    fight_index = build_fight_index(history)
    wc = pd.read_csv(WC_HISTORY) if os.path.exists(WC_HISTORY) else None

    elo = EloRatingSystem()
    counts, streaks = defaultdict(int), defaultdict(int)
    out = []

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()
        na, nb = counts[fa], counts[fb]

        # Non-debut only: the debut population has its own harness and its
        # own, different miscalibration, and mixing them would let one mask
        # the other.
        if na > 0 and nb > 0 and winner in (a, b):
            ra = roster_as_of(a, when, fight_index, static_rows, today=when)
            rb = roster_as_of(b, when, fight_index, static_rows, today=when)
            eff = {a: _effective(ra, na, elo.get_rating(a), streaks[fa]),
                   b: _effective(rb, nb, elo.get_rating(b), streaks[fb])}
            try:
                # POINT-IN-TIME CONTEXT -- see audit_term_coverage.py. The
                # four-argument form silenced recent_form on every scored
                # fight, and this harness's whole subject is whether the
                # model's probabilities are too extreme, which the recency
                # term moves. Only fights strictly before this one are visible.
                past = history[history["date"] < f["date"]]
                res = predict_matchup(a, b, pd.DataFrame([ra, rb]), eff,
                                      past, wc, f.get("weight_class"),
                                      reference_date=when.date())
            except Exception:
                res = None
            p = (res or {}).get("prob_a")
            if p is not None and not math.isnan(p):
                out.append({
                    "date": when, "p": p, "y": 1.0 if winner == a else 0.0,
                    # Both corners on the CURRENT roster means physicals and
                    # stat columns were available -- the production-like cut.
                    "full": fa in static_rows and fb in static_rows,
                })

        loser = b if winner == a else a
        if winner in (a, b):
            elo.update_ratings(winner, loser, method=method)
            streaks[_fold(winner)] += 1
            streaks[_fold(loser)] = 0
        counts[fa] += 1
        counts[fb] += 1

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2021-01-01",
                    help="fights before this train T; fights after test it")
    args = ap.parse_args()

    if not os.path.exists(HISTORY):
        print(f"No {HISTORY}.")
        sys.exit(1)

    data = run()
    if not data:
        print("No scorable fights.")
        sys.exit(1)

    pairs = [(d["p"], d["y"]) for d in data]
    full = [(d["p"], d["y"]) for d in data if d["full"]]
    print(f"scored {len(data)} non-debut fights; {len(full)} with both corners "
          f"on the current roster")

    calib(pairs, "ALL NON-DEBUT FIGHTS")
    calib(full, "BOTH CORNERS FULLY POPULATED  (the production-like cut)")

    # ERA. Data coverage improves over time, so a temperature that is really
    # a coverage artifact should fall as the eras get more recent.
    for lo, hi in [("1993", "2010"), ("2010", "2016"), ("2016", "2021"), ("2021", "2030")]:
        era = [(d["p"], d["y"]) for d in data
               if lo <= d["date"].strftime("%Y") < hi]
        if len(era) >= 200:
            calib(era, f"BY ERA, {lo}-{hi}")

    # HELD OUT. T fit strictly before the cutoff, scored strictly after.
    # HELD OUT ON THE BROAD POPULATION. The production-like cut is far too
    # small to split (fighters.csv only covers the CURRENT roster, so
    # historical fights overwhelmingly have a retired corner with no
    # physicals). This tests whether temperature scaling generalises AT ALL
    # across eras; whether the T it finds transfers to production is a
    # separate question this data cannot answer, and the caller is told so.
    cut = pd.Timestamp(args.cutoff)
    train = [(d["p"], d["y"]) for d in data if d["date"] < cut]
    test = [(d["p"], d["y"]) for d in data if d["date"] >= cut]
    if len(train) < 200 or len(test) < 200:
        print(f"\nNot enough data either side of {args.cutoff} to hold out.")
        return

    t_hat = fit_temperature(train)
    print(f"\n\nHELD-OUT TEST  (train n={len(train)} before {args.cutoff}, "
          f"test n={len(test)} after)")
    print(f"  T fitted on TRAIN only: {t_hat:.3f}")
    print(f"\n  {'T':<22}{'accuracy':>10}{'Brier':>10}{'log loss':>11}")
    print("  " + "-" * 53)
    base = None
    for t, lab in [(1.0, "1.000  (control)"), (t_hat, f"{t_hat:.3f}  (fitted)"),
                   (1.15, "1.150"), (1.30, "1.300"), (1.50, "1.500")]:
        _, acc, brier, ll = _score([(temper(p, t), y) for p, y in test])
        print(f"  {lab:<22}{acc:>9.1%}{brier:>10.4f}{ll:>11.4f}")
        if t == 1.0:
            base = (acc, brier, ll)
    print()
    for t, lab in [(t_hat, "fitted"), (1.15, "1.150"), (1.30, "1.300"), (1.50, "1.500")]:
        _, acc, brier, ll = _score([(temper(p, t), y) for p, y in test])
        _, p_val = _paired(test, t)
        verdict = "BETTER" if brier < base[1] else "WORSE"
        print(f"    {lab:<10} acc {acc-base[0]:+.2%}  Brier {brier-base[1]:+.4f}  "
              f"log loss {ll-base[2]:+.4f}  -> {verdict:6}  [p={p_val:.3f}]")

    calib(test, "HELD-OUT TEST SET, UNCORRECTED")
    calib([(temper(p, t_hat), y) for p, y in test],
          f"HELD-OUT TEST SET, T={t_hat:.3f} APPLIED")


if __name__ == "__main__":
    main()
