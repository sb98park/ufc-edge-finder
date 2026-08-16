"""
POINT-IN-TIME sweep of DURABILITY_SCALE.

THE CONSTANT UNDER TEST. durability_adjustment is the largest single term in
the style layer at 120 rating points, against an ADJUSTMENT_TOTAL_CAP of 150,
and it fires on 96% of scored fights. No harness has ever swept it --
validate_durability_shrink sweeps k, the shrink pseudo-count, not the scale.

WHY IT IS BEING SWEPT NOW, AND NOT EARLIER. A five-agent audit had all four
technical reviewers independently conclude 120 is too large, by four
different methods. They then proposed three different replacements -- 0, 90
and 94 -- each fitted on a different baseline, and every one of those
baselines was measured BEFORE the defects fixed today:

  - _shrunk_finish_loss_rate pinned 0-loss fighters at 0.0, handing every
    undefeated fighter a better chin than anyone who had ever lost
  - style_matchup_adjustment took no reference_date, so layoff fired on 70%
    of backtested fights against 13% live
  - pit_roster's fall-through guard was dead code, so 0.7% of corners had
    ko_losses + sub_losses > losses and durability could reach +/-240

Sweeping a term while its own estimator is discontinuous measures the
discontinuity. The loss-minimising response to an unstable term is to shrink
it toward zero, which is exactly what the pre-fix sweeps found -- and is why
"delete the term" was one of the three answers.

This runs on the corrected baseline, with corners randomised and the
bootstrap clustered by card.

THE RESULT: NO CHANGE. DURABILITY_SCALE stays at 120.

    scale     window A Brier    p        window B Brier    p
    0 (del)      -0.0007      0.384        -0.0009      0.365
    30           -0.0008      0.199        -0.0009      0.220
    60           -0.0007      0.086        -0.0007      0.138
    90           -0.0004      0.030        -0.0005      0.062
    150          +0.0006      0.001        +0.0006      0.007   WORSE

The audit's direction survives and its magnitude does not. Every arm below
120 improves Brier in both windows -- 8 of 8 in the same direction, which is
not nothing -- but the improvements are -0.0004 to -0.0009, roughly a
quarter of what the pre-fix sweeps reported, and no value clears this
project's bar of significance on two disjoint windows. 90 comes closest at
0.030 and 0.062, which does not clear it, and it is one arm of a five-arm
sweep.

The one decisively replicated result is the falsification: 150 is
significantly worse in both windows. The term is real and it is not
underweighted.

THE MORE USEFUL READING is that the curve is FLAT from 0 to 90. A term whose
scale can be moved from 120 to zero -- deleting it outright -- without a
measurable change in Brier is a term carrying very little information, and no
amount of tuning that constant will change this model's accuracy. That is an
argument for spending the effort on the two terms that fire on nothing
(wrestling and striking, blocked on an ETL over the now-tracked
ufc_fight_stats.csv), not on this one.

It also vindicates the sequencing. The three replacement values the audit
proposed -- 0, 90 and 94 -- were each fitted on a baseline containing the
0-loss discontinuity, the undated layoff, or the dead pit_roster guard.
Sweeping a term while its own estimator is discontinuous measures the
discontinuity, and the loss-minimising response to an unstable term is to
shrink it toward zero. Fixing the estimator first cut the apparent effect by
four and dissolved the case for every one of those values.

Usage:  python3 scripts/validate_durability_scale.py --sweep
        python3 scripts/validate_durability_scale.py --sweep --offset 2500
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
from scripts.build_pit_stats import load_pit_stats, stats_as_of  # noqa: E402
from scripts.harness_stats import (  # noqa: E402
    paired_signflip, randomize_corner, score as _score_pairs, trivial_baseline)

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"


def _fold(n) -> str:
    return str(n).strip().lower()


_score = _score_pairs


def _paired(rows, arm, base):
    """
    Paired sign-flip bootstrap, CLUSTERED BY CARD. See scripts/harness_stats.
    Row shape is (thin, y, probs, card_key).
    """
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
    # Point-in-time rate stats, so the scored model has its wrestling and
    # striking terms -- see scripts/build_pit_stats.
    pit = load_pit_stats()

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

    saved = matchup_model.DURABILITY_SCALE
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
                for row, fold in ((ra, fa), (rb, fb)):
                    row.update(stats_as_of(pit.get(fold, []), when.date()))
                frame = pd.DataFrame([ra, rb])
                eff = {}
                for name, row, prior, fold in ((a, ra, na, fa), (b, rb, nb, fb)):
                    sr = compute_stats_rating(pd.Series(row))
                    w = min(1.0, prior / 4.0)
                    eff[name] = w * elo.get_rating(name) + (1 - w) * sr + _streak_bonus(prior, streaks[fold])

                y = 1.0 if winner == a else 0.0
                probs = {}
                for k in arms:
                    matchup_model.DURABILITY_SCALE = k
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
                    # CORNERS RANDOMISED, identically across every arm, so the
                    # arms stay paired on the same fights while the trivial
                    # always-pick-A baseline falls to ~50%. Brier and log loss
                    # are invariant under the flip; accuracy is not, and that
                    # is the number the flip exists to make meaningful.
                    probs = {k: randomize_corner(v, y, a, b, when)[0]
                             for k, v in probs.items()}
                    y = randomize_corner(0.5, y, a, b, when)[1]
                    # thinner LOSS count -- the sample size the shrink is about
                    thin = min(float(ra.get("losses") or 0), float(rb.get("losses") or 0))
                    records.append((thin, y, probs, when.date()))

            loser = b if winner == a else a
            if winner in (a, b):
                elo.update_ratings(winner, loser, method=method)
                streaks[_fold(winner)] += 1
                streaks[_fold(loser)] = 0
            counts[fa] += 1
            counts[fb] += 1
    finally:
        matchup_model.DURABILITY_SCALE = saved

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

    arms = [120.0, 0.0, 30.0, 60.0, 90.0, 150.0] if args.sweep else [120.0, 90.0]
    records = run(arms, args.limit, args.offset)
    if not records:
        print("No scorable fights.")
        sys.exit(1)

    def table(rows, title):
        base_rate = trivial_baseline([(r[2][arms[0]], r[1]) for r in rows])
        print(f"\n{title}  (n={len(rows)}, always-pick-A baseline {base_rate:.1%})")
        print(f"  {'DURABILITY_SCALE':<26}{'accuracy':>10}{'Brier':>10}{'log loss':>11}")
        print("  " + "-" * 57)
        base = None
        for k in arms:
            _, acc, brier, ll = _score([(r[2][k], r[1]) for r in rows])
            label = f"{k:g}" + ("  (control, current)" if k == 120.0 else
                                "  (term deleted)" if k == 0.0 else "")
            print(f"  {label:<26}{acc:>9.1%}{brier:>10.4f}{ll:>11.4f}")
            if k == 120.0:
                base = (acc, brier, ll)
        if not base:
            return
        print()
        for k in arms:
            if k == 120.0:
                continue
            _, acc, brier, ll = _score([(r[2][k], r[1]) for r in rows])
            verdict = "BETTER" if brier < base[1] else ("no change" if brier == base[1] else "WORSE")
            _, p, deff = _paired(rows, k, 120.0)
            sig = "significant" if p < 0.05 else "NOT significant"
            print(f"    k={k:<5g} acc {acc-base[0]:+.2%}  Brier {brier-base[1]:+.4f}  "
                  f"log loss {ll-base[2]:+.4f}  -> {verdict:9}  [p={p:.3f} clustered, "
                  f"{sig}, deff={deff:.2f}]")

    table(records, "ALL SCORED FIGHTS")

    # THE SUBSET THE CHANGE IS ABOUT. Where both corners have deep loss
    # records the raw ratio is already stable and every arm nearly agrees, so
    # including them dilutes a real effect toward zero.
    for cut in (2, 5):
        thin = [r for r in records if r[0] <= cut]
        if len(thin) >= 100:
            table(thin, f"ONLY FIGHTS WITH A CORNER AT {cut} LOSSES OR FEWER")


if __name__ == "__main__":
    main()
