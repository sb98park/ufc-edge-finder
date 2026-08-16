"""
POINT-IN-TIME validation of DEBUT_RATING_SHRINK.

THE CHANGE UNDER TEST. build_effective_ratings has a branch for a fighter
with zero connected fight history: it uses compute_stats_rating -- their
career W-L -- at full strength. DEBUT_RATING_SHRINK scales that rating's
distance from 1500, so 0.5 halves a debutant's edge and 0.0 puts every
debutant at the neutral centre. 1.0 is exactly the shipped behaviour.

WHY IT WAS PROPOSED. research_debutant_prior.py reconstructed the pre-UFC
record and debut outcome for 263 fighters and found the prior is not merely
noisy but INVERTED at the top: debutants the curve rates at 82% win 45.5%.
Anthony Wint sits there -- 7-0, rated 1781, an 85% pick.

WHY THAT IS NOT ENOUGH TO SHIP ON. That measurement compares a debutant to
a NEUTRAL 1500 opponent, which is a calibration check on the rating curve,
not a prediction. Real debut fights are usually debutant vs. a rated
veteran, so the gap that actually drives the pick is a different number, and
the style layer, age, layoff and physical terms all move it further. A
curve that is miscalibrated in isolation can still be the better input to
the full model. Only a head-to-head backtest settles it.

THE HARNESS. Walk fight_history forward. Before each bout, rebuild BOTH
corners as they stood that night (scripts/pit_roster: record, splits,
layoff, age -- never today's roster, which contains the outcome being
predicted), rate them off Elo built only from earlier fights, and predict.
Every arm sees identical inputs and differs ONLY in DEBUT_RATING_SHRINK, so
the arms are paired on the same fights and the delta is the change alone.

SCORED POPULATION. Only fights with at least one true debutant -- a corner
with zero prior fights in history. Everywhere else the branch never fires
and all arms are byte-identical, so including them would dilute a real
effect toward zero. The `both` cut is reported separately because two
debutants is the Wint-vs-Chatman case that started this.

THE RESULT: NO RATING CHANGE SHIPPED. DEBUT_RATING_SHRINK stays at 1.0.

Shrinking helps where the debutant faces a RATED opponent and hurts where
both corners are debuting, and the two cancel:

  debutant vs established (n=3332)   0.75  Brier -0.0004  p=0.015
                                     0.5   Brier -0.0007  p=0.040
                                     0.25  Brier -0.0008  p=0.108
  both debuting           (n=634)    0.5   Brier +0.0007  p=0.457
                                     0.0   Brier +0.0066  p=0.009  WORSE
  all debut fights        (n=3966)   0.5   Brier -0.0004  p=0.162  n.s.

The mechanism reads clearly. Against a rated opponent the debutant's
stats-curve rating is being compared to an Elo on a different scale, and
shrinking narrows the mismatch. When BOTH corners come off the stats curve
they already share a scale, so shrinking only compresses a gap that was
measured correctly -- at 0.0 both sit at exactly 1500, every fight is
50/50, and accuracy drops 71.8% -> 70.0%.

A `vs-rated-only` arm confirms it: it captures the full gain (-0.0007,
p=0.040) and is a bit-exact no-op on the both-debuting cut. But it cannot
be built. build_effective_ratings assigns ONE rating per fighter for the
whole roster and never sees a matchup, so a shrink conditional on the
opponent has nowhere to live without restructuring it. The variant that IS
implementable -- unconditional 0.5 -- is not significant (p=0.162). Nothing
here justifies that refactor: the honest effect is under 0.001 Brier, from
an eight-arm sweep, which does not survive any correction for multiplicity.

WHAT THE RUN ACTUALLY FOUND, which is much larger than the arm it was
testing. Bucketing the CURRENT model against its own hit rate:

  debutant vs established        says      wins      gap        n
                                 55.0%     51.6%     -3.3%    1284
                                 64.6%     51.6%    -13.0%    1008
                                 74.8%     59.1%    -15.6%     749
                                 90.9%     87.6%     -3.3%     291

A 60-80% pick in a debut fight is worth about 55%. That is 1,757 fights, a
13-16pp overstatement, and it sits exactly where betting edges are computed
-- an order of magnitude larger than any shrink arm above. The fix belongs
in the reporting layer, alongside MIN_RECORD_FOR_HIGH_CONFIDENCE, not in
the ratings.

The both-debuting cut runs the other way: 51.1% claimed, 71.9% actual. The
model barely leans, and the lean is right 72% of the time. Note the limit
before reading that as Wint vs Chatman -- 606 of those 634 land in one
50-60% bucket because early-era history gives both corners a thin or empty
record, so both ratings collapse toward 1500. It does NOT test a 7-0 versus
a 5-1, where the curve produces a 168-point gap.

WHAT SHIPPED, from the calibration tables rather than any arm above. See
model_preview.DEBUT_MEDIUM_CEILING: a 60-75% pick on a debut fight is
demoted from Medium Confidence to Low. Against the --control population the
debut-specific damage is ~8-9pp on n=1381, and it has all but vanished by
75%. High Confidence and Lock of the Week are deliberately left alone --
debut picks at the Lock floor go 252/252-scale 89.3% against a ~85-90%
claim, and are among the best-calibrated the model makes.

Usage:  python3 scripts/validate_debutant_shrink.py
        python3 scripts/validate_debutant_shrink.py --sweep
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
from src import power_rating  # noqa: E402
from src.power_rating import RATING_CENTER, compute_stats_rating, _streak_bonus  # noqa: E402
from src.matchup_model import predict_matchup  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of  # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"


def _fold(n) -> str:
    return str(n).strip().lower()


def _effective(row: dict, n_prior: int, elo_r: float, streak: int, shrink: float) -> float:
    """
    build_effective_ratings for ONE fighter, at one point in time.

    Deliberately mirrors the production function rather than calling it:
    that one takes a whole DataFrame and derives fight counts by
    value_counts() over all of history, which would count fights that have
    not happened yet. The arithmetic below is line-for-line the same.
    """
    stats_rating = compute_stats_rating(pd.Series(row))
    if n_prior == 0:
        eff = RATING_CENTER + (stats_rating - RATING_CENTER) * shrink
    else:
        weight = min(1.0, n_prior / 4.0)
        eff = weight * elo_r + (1 - weight) * stats_rating
    return eff + _streak_bonus(n_prior, streak)


def _score(pairs):
    n = len(pairs)
    acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in pairs) / n
    return n, acc, brier, ll


def _paired_test(rows, arm, base="1.0  (control, current)", n_boot=4000, seed=12345):
    """
    Paired sign-flip bootstrap on the per-fight change in squared error.

    Both arms score the SAME fights, so an unpaired test would throw away
    the pairing and badly overstate the noise. Null hypothesis: the change
    had no effect, so each fight's delta was equally likely to carry the
    opposite sign. Deterministic seed -- a validation number that moves
    between runs is not a validation number.
    """
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


def run(arms, control=False):
    fighters = pd.read_csv(FIGHTERS)
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}

    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    fight_index = build_fight_index(history)

    elo = EloRatingSystem()
    counts = defaultdict(int)
    streaks = defaultdict(int)
    records = []
    skipped = 0

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()
        na, nb = counts[fa], counts[fb]

        # At least one true debutant, or the branch under test never fires
        # and every arm returns the identical number. --control inverts the
        # filter to score the OPPOSITE population, which is what makes the
        # calibration gaps attributable to the debut rather than to the
        # model being loose in that probability band generally.
        want = (na > 0 and nb > 0) if control else (na == 0 or nb == 0)
        if want and winner in (a, b):
            ra = roster_as_of(a, when, fight_index, static_rows, today=when)
            rb = roster_as_of(b, when, fight_index, static_rows, today=when)
            frame = pd.DataFrame([ra, rb])
            y = 1.0 if winner == a else 0.0

            probs = {}
            for label, (shrink, conditional) in arms.items():
                # CONDITIONAL arms shrink a debutant only when the OTHER
                # corner has real tracked history. When both corners are on
                # the stats curve their ratings are already on a common
                # scale, and shrinking both toward 1500 only compresses a
                # gap that was measured correctly.
                sa = 1.0 if (conditional and nb == 0) else shrink
                sb = 1.0 if (conditional and na == 0) else shrink
                eff = {
                    a: _effective(ra, na, elo.get_rating(a), streaks[fa], sa),
                    b: _effective(rb, nb, elo.get_rating(b), streaks[fb], sb),
                }
                try:
                    res = predict_matchup(a, b, frame, eff)
                except Exception:
                    res = None
                p = (res or {}).get("prob_a")
                if p is not None and not math.isnan(p):
                    probs[label] = p
            # Only fights EVERY arm could predict. An arm scored on a
            # different population is not a comparison.
            if len(probs) == len(arms):
                records.append(((na == 0) + (nb == 0), y, probs))
            else:
                skipped += 1

        loser = b if winner == a else a
        if winner in (a, b):
            elo.update_ratings(winner, loser, method=method)
            streaks[_fold(winner)] += 1
            streaks[_fold(loser)] = 0
        counts[fa] += 1
        counts[fb] += 1

    return records, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="calibrate on NON-debut fights, as a baseline for the "
                         "debut gaps -- no arm differs on these, so only the "
                         "calibration tables are meaningful")
    args = ap.parse_args()

    if not os.path.exists(HISTORY):
        print(f"No {HISTORY}.")
        sys.exit(1)

    # label -> (shrink factor, conditional-on-opponent-having-history)
    arms = {"1.0  (control, current)": (1.0, False)}
    for s_ in ([0.75, 0.5, 0.25, 0.0] if args.sweep else [0.5, 0.0]):
        arms[f"{s_:g}"] = (s_, False)
    for s_ in ([0.75, 0.5, 0.25, 0.0] if args.sweep else [0.5]):
        arms[f"{s_:g}  vs-rated-only"] = (s_, True)
    saved = power_rating.DEBUT_RATING_SHRINK
    try:
        records, skipped = run(arms, control=args.control)
    finally:
        power_rating.DEBUT_RATING_SHRINK = saved

    if not records:
        print("No scorable debut fights.")
        sys.exit(1)
    if skipped:
        print(f"({skipped} debut fights dropped -- not every arm produced a probability)")

    def table(rows, title):
        print(f"\n{title}  (n={len(rows)})")
        print(f"  {'DEBUT_RATING_SHRINK':<26}{'accuracy':>10}{'Brier':>10}{'log loss':>11}")
        print("  " + "-" * 57)
        ctl = "1.0  (control, current)"
        base = None
        for label in arms:
            _, acc, brier, ll = _score([(pr[label], y) for _, y, pr in rows])
            print(f"  {label:<26}{acc:>9.1%}{brier:>10.4f}{ll:>11.4f}")
            if label == ctl:
                base = (acc, brier, ll)
        if not base:
            return
        print()
        for label in arms:
            if label == ctl:
                continue
            _, acc, brier, ll = _score([(pr[label], y) for _, y, pr in rows])
            verdict = "BETTER" if brier < base[1] else ("no change" if brier == base[1] else "WORSE")
            _, p = _paired_test(rows, label)
            sig = "significant" if p < 0.05 else "NOT significant"
            print(f"    {label:<22} acc {acc-base[0]:+.2%}  Brier {brier-base[1]:+.4f}  "
                  f"log loss {ll-base[2]:+.4f}   -> {verdict:9}  [p={p:.3f}, {sig}]")

    def calib(rows, title, arm="1.0  (control, current)"):
        """
        Is the CONTROL model's confidence earned, inside this population?

        Brier and accuracy answer "is arm X better than arm Y". They do not
        answer the question the Wint pick actually raises, which is whether
        an 85% out of this branch means 85%. That needs the predictions
        bucketed against their own hit rate.
        """
        print(f"\n{title}  (n={len(rows)})")
        print(f"  {'model says':<14}{'actually wins':<16}{'gap':<10}{'n'}")
        print("  " + "-" * 46)
        for lo, hi in [(.5, .6), (.6, .7), (.7, .75), (.75, .8), (.8, .85),
                       (.85, .9), (.9, 1.01)]:
            g = [(pr[arm], y) for _, y, pr in rows
                 if lo <= max(pr[arm], 1 - pr[arm]) < hi]
            if len(g) >= 15:
                # Scored from the FAVOURITE's side, so a 0.2 and a 0.8 are
                # the same 80% claim rather than cancelling each other out.
                said = sum(max(p, 1 - p) for p, _ in g) / len(g)
                hit = sum(1 for p, y in g if (p >= .5) == (y == 1.)) / len(g)
                print(f"  {said:<14.1%}{hit:<16.1%}{hit-said:<+10.1%}{len(g)}")
        # COVERAGE, printed even where a bucket was too small to score.
        # "No bucket met the print threshold" and "the model never makes a
        # pick that confident here" look identical in the table above and
        # mean opposite things -- one is missing evidence, the other is
        # evidence of absence.
        for cut, lab in ((.75, "High Confidence floor"), (.82, "Lock of the Week floor")):
            g = [(pr[arm], y) for _, y, pr in rows if max(pr[arm], 1 - pr[arm]) >= cut]
            if g:
                hit = sum(1 for p, y in g if (p >= .5) == (y == 1.)) / len(g)
                print(f"  >= {cut:.0%} ({lab}): n={len(g)}, hit {hit:.1%}")
            else:
                print(f"  >= {cut:.0%} ({lab}): n=0 -- NO historical coverage")

    if args.control:
        calib(records, "CALIBRATION, NEITHER CORNER DEBUTING -- current model")
        return

    table(records, "ALL FIGHTS WITH AT LEAST ONE DEBUTANT")
    one = [r for r in records if r[0] == 1]
    both = [r for r in records if r[0] == 2]
    if one:
        table(one, "DEBUTANT vs ESTABLISHED FIGHTER")
    if both:
        table(both, "BOTH CORNERS DEBUTING  (the Wint vs Chatman case)")
        calib(both, "CALIBRATION, BOTH DEBUTING -- current model")
    if one:
        calib(one, "CALIBRATION, DEBUTANT vs ESTABLISHED -- current model")


if __name__ == "__main__":
    main()
