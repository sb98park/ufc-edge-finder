"""
POINT-IN-TIME validation of DURABILITY_SHRINK_K.

THE CHANGE UNDER TEST. durability_adjustment is the largest single term in
the style layer -- DURABILITY_SCALE is 120, against an ADJUSTMENT_TOTAL_CAP
of 150 -- and it is built from an UNSHRUNK ratio:

    finish_loss_rate = (ko_losses + sub_losses) / losses

whose denominator is frequently 1. One stoppage on a 12-1 record reads as a
100% finish-loss rate, a fighter who has never survived being hit, and
drives the term to its full magnitude. The 0-loss case is already guarded.
The 1- and 2-loss cases are not, and they are common.

DURABILITY_SHRINK_K adds k pseudo-observations at the DIVISION'S own finish
rate, from real UFC results:

    (finish_losses + k * base) / (losses + k)

A 1-1 record moves most of the way to the base; a 20-8 record barely moves.
K = 0 is the shipped behaviour and is byte-identical to it.

WHY THE BASE IS DIVISIONAL. Heavyweight bouts end in a finish 67% of the
time and Women's Strawweight 33%. Shrinking every fighter toward one global
number would import a heavyweight's chin expectations into a strawweight's
rating, which is the same category error the flat method priors made.

THIS TERM IS ACTUALLY MEASURABLE, which is why it is worth testing at all.
audit_term_coverage.py shows durability firing on 84% of backtested fights
against 89% live -- unlike age, quick-return, wrestling or striking, all of
which are dark in replay. A verdict here means something.

The harness passes full point-in-time context (history, weight-class table,
booked division, reference date) so the scored model has its recency term,
and both corners are rebuilt by pit_roster as they stood that night.

THE RESULT: SHIPPED AT k = 2.0. Two disjoint windows, both significant:

    window                    k=2 Brier    p        accuracy
    recent 2,500 (n=1834)      -0.0030   0.011       +0.22%
    prior  2,500 (n=1663)      -0.0052   0.000       +1.44%

Every arm improved Brier and log loss in every subset -- 16 of 16 in the
same direction in the first window alone -- and the effect is 3-5x anything
else measured on this model recently. Replication on an independent window
is what carries it past the multiplicity objection a 4-arm x 4-subset sweep
would otherwise deserve.

The response curve is FLAT in k, which is the diagnosis rather than a
disappointment: the damage is the 0-or-1 ratio at a denominator of 1, and
any pseudo-count fixes it. k = 2 is the least intervention that captures the
effect and was significant in both windows.

Accuracy improves in BOTH windows, so this is not a calibration-for-accuracy
trade. An earlier run measured -0.0021 / -0.0027 with accuracy -0.22% /
+0.30%; that was taken while _shrunk_finish_loss_rate still returned 0.0 for
a 0-loss fighter, pinning every undefeated corner below a value the shrunk
estimator could otherwise never produce. Removing that discontinuity roughly
doubled the gain. A shrink measured against an artifact of its own
introduction understates itself.

Usage:  python3 scripts/validate_durability_shrink.py
        python3 scripts/validate_durability_shrink.py --sweep
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
from src import matchup_model  # noqa: E402
from src.matchup_model import predict_matchup  # noqa: E402
from src.power_rating import compute_stats_rating, _streak_bonus  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of  # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"


def _fold(n) -> str:
    return str(n).strip().lower()


def _score(pairs):
    n = len(pairs)
    acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in pairs) / n
    return n, acc, brier, ll


def _paired(rows, arm, base, n_boot=4000, seed=12345):
    """Paired sign-flip bootstrap on the per-fight change in squared error."""
    deltas = [(pr[arm] - y) ** 2 - (pr[base] - y) ** 2 for _, y, pr in rows]
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


def run(arms, limit, offset=0):
    fighters = pd.read_csv(FIGHTERS)
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}
    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    fight_index = build_fight_index(history)
    wc = pd.read_csv(WC_HISTORY) if os.path.exists(WC_HISTORY) else None

    elo = EloRatingSystem()
    counts, streaks = defaultdict(int), defaultdict(int)
    records = []
    # WINDOW. offset skips the most recent N rows, so --offset 2500 --limit
    # 2500 scores a period that is DISJOINT from the default run. Replication
    # on independent fights is what separates a real effect from one arm of a
    # sweep getting lucky on one era.
    trimmed = history.iloc[:-offset] if offset else history
    rows = trimmed.tail(limit) if limit else trimmed
    cutoff = rows.iloc[0]["date"] if len(rows) else None
    ceiling = rows.iloc[-1]["date"] if len(rows) else None

    saved = matchup_model.DURABILITY_SHRINK_K
    try:
        for _, f in history.iterrows():
            a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
            fa, fb = _fold(a), _fold(b)
            when = f["date"].to_pydatetime()
            na, nb = counts[fa], counts[fb]

            in_window = (cutoff is None or f["date"] >= cutoff) and (ceiling is None or f["date"] <= ceiling)
            if in_window and na > 0 and nb > 0 and winner in (a, b):
                ra = roster_as_of(a, when, fight_index, static_rows, today=when)
                rb = roster_as_of(b, when, fight_index, static_rows, today=when)
                frame = pd.DataFrame([ra, rb])
                past = history[history["date"] < f["date"]]
                eff = {}
                for name, row, prior, fold in ((a, ra, na, fa), (b, rb, nb, fb)):
                    sr = compute_stats_rating(pd.Series(row))
                    w = min(1.0, prior / 4.0)
                    eff[name] = w * elo.get_rating(name) + (1 - w) * sr + _streak_bonus(prior, streaks[fold])

                y = 1.0 if winner == a else 0.0
                probs = {}
                for k in arms:
                    matchup_model.DURABILITY_SHRINK_K = k
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
                    # thinner LOSS count -- the sample size the shrink is about
                    thin = min(float(ra.get("losses") or 0), float(rb.get("losses") or 0))
                    records.append((thin, y, probs))

            loser = b if winner == a else a
            if winner in (a, b):
                elo.update_ratings(winner, loser, method=method)
                streaks[_fold(winner)] += 1
                streaks[_fold(loser)] = 0
            counts[fa] += 1
            counts[fb] += 1
    finally:
        matchup_model.DURABILITY_SHRINK_K = saved

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the most recent N rows, for a disjoint window")
    ap.add_argument("--limit", type=int, default=3000,
                    help="score only the most recent N history rows (0 = all)")
    args = ap.parse_args()

    if not os.path.exists(HISTORY):
        print(f"No {HISTORY}.")
        sys.exit(1)

    arms = [0.0, 2.0, 5.0, 10.0, 20.0] if args.sweep else [0.0, 5.0]
    records = run(arms, args.limit, args.offset)
    if not records:
        print("No scorable fights.")
        sys.exit(1)

    def table(rows, title):
        print(f"\n{title}  (n={len(rows)})")
        print(f"  {'DURABILITY_SHRINK_K':<26}{'accuracy':>10}{'Brier':>10}{'log loss':>11}")
        print("  " + "-" * 57)
        base = None
        for k in arms:
            _, acc, brier, ll = _score([(pr[k], y) for _, y, pr in rows])
            label = f"{k:g}" + ("  (control, current)" if k == 0.0 else "")
            print(f"  {label:<26}{acc:>9.1%}{brier:>10.4f}{ll:>11.4f}")
            if k == 0.0:
                base = (acc, brier, ll)
        if not base:
            return
        print()
        for k in arms:
            if k == 0.0:
                continue
            _, acc, brier, ll = _score([(pr[k], y) for _, y, pr in rows])
            verdict = "BETTER" if brier < base[1] else ("no change" if brier == base[1] else "WORSE")
            _, p = _paired(rows, k, 0.0)
            sig = "significant" if p < 0.05 else "NOT significant"
            print(f"    k={k:<5g} acc {acc-base[0]:+.2%}  Brier {brier-base[1]:+.4f}  "
                  f"log loss {ll-base[2]:+.4f}  -> {verdict:9}  [p={p:.3f}, {sig}]")

    table(records, "ALL SCORED FIGHTS")

    # THE SUBSET THE CHANGE IS ABOUT. Where both corners have deep loss
    # records the raw ratio is already stable and every arm nearly agrees, so
    # including them dilutes a real effect toward zero.
    for cut in (2, 3, 5):
        thin = [r for r in records if r[0] <= cut]
        if len(thin) >= 100:
            table(thin, f"ONLY FIGHTS WITH A CORNER AT {cut} LOSSES OR FEWER")


if __name__ == "__main__":
    main()
