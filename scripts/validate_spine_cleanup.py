"""
POINT-IN-TIME check on whether the spine cleanup improved PREDICTIONS, not
just correctness.

WHAT WAS DONE, AND WHAT WAS NEVER CHECKED. Over 2026-08-31 and 09-01 the
fight-history spine lost 231 duplicate bouts and gained a promotion label on
2,469 rows, which src/elo.ufc_only then excludes from the rating graph. Every
change was argued from correctness -- a bout written twice is written twice, a
KSW fight is not a UFC fight -- and I still believe those arguments. But
"individual fights moved" was the only thing measured, and moving is not
improving. This repo grades its own parameter changes with point-in-time
paired arms; the data changes went in without one.

THE ARMS DIFFER ONLY IN WHAT BUILT THE RATINGS.

    before  ratings replayed from the pre-cleanup file: duplicates included,
            regional bouts inside the graph
    after   ratings replayed from the current file, through ufc_only

Both are scored on THE SAME fights -- the deduped, UFC-only set -- because
scoring each arm on its own population would compare two different questions
and flatter whichever had the easier one.

STRICTLY POINT IN TIME. The two files are merged into one timeline in date
order. A fight is predicted by both arms from the ratings as they stood before
it, and only then does each arm absorb the rows it owns: the before-arm takes
every row in its file (including a duplicate, twice -- that is the defect
under test), the after-arm takes only what survives ufc_only.

RESULT, 2026-09-01, 9,198 UFC bouts scored point-in-time:

    arm             acc     brier   logloss    d.brier       p   vs
    orig         0.6171   0.23373   0.65959         --      --
    dedup        0.6190   0.23361   0.65936   -0.00012   0.294   orig
    ufconly      0.6018   0.23945   0.67158   +0.00584   0.000   dedup

THE DEDUPE IS FINE. Neutral to slightly positive, and a bout written twice is
wrong regardless of what Brier says about it.

EXCLUDING REGIONAL BOUTS FROM THE GRAPH IS NOT. It costs +0.0058 Brier at
p=0.000, and the argument it shipped on was mine: "a regional opponent with no
other results sits at the 1500 default, so beating them scores like beating an
average UFC fighter." That is true as a statement about BIAS and wrong as a
decision, because the alternative is not an unbiased estimate -- it is no
estimate. A fighter with ten regional wins really is better than the 1500
default, and dropping those bouts throws that away.

Ruled out, because a finding against your own work deserves more scrutiny, not
less:

  PARTIAL LABELLING. Only 2,559 athletes were crawled, so the exclusion is
  uneven and a half-excluded graph could plausibly be worse than either
  consistent choice. It is not: the harm is LARGER where both corners are
  classified (+0.00936, n=758) than where one is not (+0.00539, n=8,440).

  A THRESHOLD EFFECT. The harm is flat across experience -- +0.0059 for a
  debut corner, +0.0066 at 1-2 prior bouts, +0.0058 at 3-5, +0.0060 at 6-10 --
  and only vanishes past 11 UFC bouts (-0.0004), exactly where a fighter's UFC
  record has made the regional one redundant.

  A BETTER MIDDLE. Weighting regional bouts instead of excluding them is
  monotonic in the weight, so there is no sweet spot to find:
  w=1.00 -> 0.23346, w=0.75 -> 0.23423, w=0.50 -> 0.23551,
  w=0.25 -> 0.23757, w=0.00 -> 0.24079.

WHAT THIS DOES NOT SAY. It measures the Elo core, which is what
walkforward_backtest also measures and which drives the largest share of the
final number -- not the adjustment layer. And it says nothing about fun_facts,
where the same filter answers a different question: a published "12-fight win
streak" that is eight regional bouts is misleading on a UFC site whatever
Brier thinks, and comparability across fighters is the point there.

Run: python3 scripts/validate_spine_cleanup.py
     python3 scripts/validate_spine_cleanup.py --since 2024-01-01
"""

import argparse
import math
import os
import random
import subprocess
import sys
from io import StringIO

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem, ufc_only                      # noqa: E402
from src.names import fight_key                                    # noqa: E402

HISTORY = "data/fight_history.csv"
BEFORE_REV = "43dd8d54~1"          # the commit that removed the duplicates


def _load_before() -> pd.DataFrame:
    out = subprocess.run(["git", "show", f"{BEFORE_REV}:{HISTORY}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"could not read the pre-cleanup spine: {out.stderr[:200]}")
    return pd.read_csv(StringIO(out.stdout))


def _score(pairs):
    n = len(pairs)
    acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in pairs) / n
    return n, acc, brier, ll


def _paired(rows, i, j, n_boot=4000, seed=12345):
    """Squared-error delta of arm i minus arm j, sign-flipped."""
    deltas = [(r[i] - r[0]) ** 2 - (r[j] - r[0]) ** 2 for r in rows]
    obs = sum(deltas) / len(deltas)
    rnd = random.Random(seed)
    hits = sum(1 for _ in range(n_boot)
               if abs(sum(d if rnd.random() < 0.5 else -d for d in deltas) / len(deltas)) >= abs(obs))
    return obs, hits / n_boot


def _calibration(rows, idx):
    bins = {}
    for r in rows:
        y, p = r[0], r[idx]
        lo = min(int(p * 10) / 10, 0.9)
        b = bins.setdefault(lo, [0, 0.0, 0])
        b[0] += 1
        b[1] += p
        b[2] += int(y == 1.0)
    return bins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="only score fights on/after this date")
    args = ap.parse_args()

    before_df = _load_before()
    after_df = pd.read_csv(HISTORY)
    print(f"spine rows   before {len(before_df)}   after {len(after_df)}")
    print(f"in the graph before {len(before_df)}   after {len(ufc_only(after_df))}")

    # Rows the after-arm feeds its ratings from.
    after_graph = set()
    for r in ufc_only(after_df).itertuples(index=False):
        after_graph.add(fight_key(r.fighter_a, r.fighter_b, r.date))

    # The scored set: decided UFC bouts that survived the cleanup.
    scored = set()
    for r in ufc_only(after_df).itertuples(index=False):
        w = getattr(r, "winner", None)
        if pd.notna(w) and str(w).strip():
            scored.add(fight_key(r.fighter_a, r.fighter_b, r.date))

    # One timeline. Each row carries which arms consume it.
    timeline = []
    for tag, df in (("orig", before_df), ("dedup", after_df), ("ufconly", ufc_only(after_df))):
        for r in df.itertuples(index=False):
            timeline.append((str(r.date)[:10], tag, r))
    timeline.sort(key=lambda t: t[0])

    # THREE ARMS, because the two interventions must be attributed separately.
    #   orig     the file as it was: 231 duplicates, regional bouts in the graph
    #   dedup    duplicates removed, regional bouts STILL in the graph
    #   ufconly  duplicates removed AND regional excluded -- what ships today
    # Comparing only the ends would blame or credit whichever change happened
    # to travel with the other.
    elo_o, elo_d, elo_u = EloRatingSystem(), EloRatingSystem(), EloRatingSystem()
    rows, seen = [], set()
    for date, tag, r in timeline:
        w = getattr(r, "winner", None)
        decided = pd.notna(w) and str(w).strip() and w in (r.fighter_a, r.fighter_b)
        k = fight_key(r.fighter_a, r.fighter_b, r.date)
        # Predict once per scored fight, the first time the timeline reaches it.
        if decided and k in scored and k not in seen and (not args.since or date >= args.since):
            seen.add(k)
            rows.append((1.0 if w == r.fighter_a else 0.0,
                         EloRatingSystem.expected_score(elo_o.get_rating(r.fighter_a),
                                                        elo_o.get_rating(r.fighter_b)),
                         EloRatingSystem.expected_score(elo_d.get_rating(r.fighter_a),
                                                        elo_d.get_rating(r.fighter_b)),
                         EloRatingSystem.expected_score(elo_u.get_rating(r.fighter_a),
                                                        elo_u.get_rating(r.fighter_b))))
        if not decided:
            continue
        loser = r.fighter_b if w == r.fighter_a else r.fighter_a
        method = getattr(r, "method", "DEC")
        method = method if isinstance(method, str) and method.strip() else "DEC"
        {"orig": elo_o, "dedup": elo_d, "ufconly": elo_u}[tag].update_ratings(
            w, loser, method=method)

    if not rows:
        print("nothing scored")
        return 0

    print(f"\nfights scored (identical set for every arm): {len(rows)}\n")
    print(f"  {'arm':10s} {'acc':>8s} {'brier':>9s} {'logloss':>9s} {'d.brier':>10s} {'p':>7s}   vs")
    print("  " + "-" * 64)
    ARMS = [("orig", 1, None), ("dedup", 2, 1), ("ufconly", 3, 2), ("ufconly", 3, 1)]
    for lbl, i, base in ARMS:
        n, acc, brier, ll = _score([(r[i], r[0]) for r in rows])
        if base is None:
            print(f"  {lbl:10s} {acc:8.4f} {brier:9.5f} {ll:9.5f} {'--':>10s} {'--':>7s}")
        else:
            d, p = _paired(rows, i, base)
            base_lbl = {1: "orig", 2: "dedup"}[base]
            print(f"  {lbl:10s} {acc:8.4f} {brier:9.5f} {ll:9.5f} {d:+10.5f} {p:7.3f}   {base_lbl}")
    print("\n  d.brier is the arm MINUS its comparison, so negative means better.")

    print(f"\n  calibration -- does 65% win 65%?")
    print(f"    {'bucket':>8s} {'n':>6s} {'orig':>7s} {'ufconly':>8s} {'won':>7s}")
    co, cu = _calibration(rows, 1), _calibration(rows, 3)
    for lo in sorted(cu):
        n, psum, wins = cu[lo]
        no, pso, _ = co.get(lo, (0, 0.0, 0))
        print(f"    {lo:>6.0%}+ {n:6d} {(pso / no if no else 0):7.1%} {psum / n:8.1%} {wins / n:7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
