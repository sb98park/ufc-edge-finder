"""
POINT-IN-TIME re-sweep of min_fights_to_trust_elo, and why
DEBUT_RATING_SHRINK is not swept alongside it.

WHAT IS BEING RE-TUNED. build_effective_ratings blends a fighter's Elo
against compute_stats_rating with weight = min(1, n_prior / K), K = 4. Below
four connected bouts the career record carries part of the rating; at four and
above the stats rating is multiplied by zero and the fighter is pure Elo.

WHY NOW. K was chosen when compute_stats_rating still contained a finish-rate
term, 150.0 * (finish_rate - 0.4). That term has since been deleted on its own
evidence, so the function K balances against is no longer the function K was
balanced against. What remains is a win-percentage term damped by experience
plus a reach term. A parameter tuned against a deleted term is not a tuned
parameter, it is an inherited one.

SCORED POPULATION. Fights where a corner sits below the largest K under test,
which is the only place any arm can differ from any other. 417 of 597 fights
with both corners on the roster. The --control cut takes the complement, where
every arm is saturated and identical by construction; any movement there means
the harness is wrong.

WHY DEBUT_RATING_SHRINK IS REPORTED, NOT SWEPT. It only acts when a fighter
has ZERO connected bouts, and there are 42 such fights in the entire
scoreable set. That is smaller than the reach sweep that already failed to
separate an ORACLE arm, so a sweep here could only produce a number with no
power behind it. It is also currently 1.0, which applies no shrink whatever:
the debutant branch is exactly the stats rating. --shrink runs it anyway and
prints the population next to the result, so the size is impossible to quote
without the caveat attached.

RESULT, 2026-08-31: BOTH PARAMETERS SURVIVE AT THEIR CURRENT VALUES.

  min_fights_to_trust_elo, 417 fights with a corner below 16 bouts
     arm      acc     brier   logloss    d.brier       p  n.diff    d|diff  p|diff
       2   0.5755   0.24237   0.67880   +0.00030   0.786      71  +0.00178   0.781
       4   0.5683   0.24206   0.67801         --      --      --        --      --
       6   0.5755   0.24114   0.67593   -0.00093   0.215     135  -0.00286   0.221
       8   0.5779   0.24090   0.67547   -0.00116   0.338     191  -0.00253   0.339
      12   0.5779   0.24130   0.67637   -0.00076   0.677     323  -0.00098   0.672
      16   0.5803   0.24206   0.67798   -0.00001   0.999     411  -0.00001   0.997

  falsification, 180 saturated fights: every arm identical, +0.00000, p=1.000

  DEBUT_RATING_SHRINK, 42 fights with a debuting corner
     1.0   0.6190   0.23540   0.66676         --      --
    0.75   0.6905   0.23714   0.67119   +0.00174   0.758
     0.5   0.6429   0.24251   0.68261   +0.00711   0.533
    0.25   0.5952   0.25122   0.70100   +0.01582   0.364

NEITHER MOVES. K shows a shape rather than a result: worse at 2, better at
6-12, back to baseline at 16, consistent on Brier and logloss, which is what
a real but small optimum near 6-8 would look like. It is also what noise
looks like at p=0.215, and 0.215 is the best of five arms, which is worse
than it reads. Restricting each arm to the fights it actually changes barely
moves the p-values (0.215 -> 0.221, 0.338 -> 0.339), so the dilution
hypothesis is dead: the power is not hiding anywhere, it is absent.

Changing a live parameter that prices real money on p=0.22 is exactly the
move CLAUDE.md s7 exists to prevent, so K stays at 4 -- not because 4 is
demonstrably best, but because nothing here demonstrates otherwise.

The shrink arms at least point one way: every level of shrinking is worse
than none, monotonically in the amount shrunk. At 42 fights that is direction
and not evidence, but there is no case for introducing a shrink, and the
inert 1.0 stands.

WHAT WOULD ACTUALLY SETTLE THIS. Not more arms and not a finer grid -- the
binding constraint is that outcome scoring needs both corners in
fighters.csv, which caps the whole question at 597 fights however it is cut.
Retired fighters never get a roster row. Any future parameter that only acts
below a few connected bouts faces the same ceiling, and that is worth knowing
before commissioning the next sweep rather than after.

Run:  python3 scripts/validate_elo_trust_ramp.py
      python3 scripts/validate_elo_trust_ramp.py --control
      python3 scripts/validate_elo_trust_ramp.py --shrink
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem, ufc_only                          # noqa: E402
from src.matchup_model import predict_matchup                          # noqa: E402
from src.power_rating import (RATING_CENTER, _streak_bonus,            # noqa: E402
                              attach_imputed_reach, compute_stats_rating)
from scripts.pit_roster import build_fight_index, roster_as_of         # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"

K_ARMS = (2, 4, 6, 8, 12, 16)          # 4 is shipped
SHRINK_ARMS = (1.0, 0.75, 0.5, 0.25)   # 1.0 is shipped (i.e. no shrink)
SHIPPED_K = 4
SHIPPED_SHRINK = 1.0


def _fold(n) -> str:
    return str(n).strip().lower()


def _effective(row, n_prior, elo_r, streak, k, shrink):
    sr = compute_stats_rating(pd.Series(row))
    if n_prior == 0:
        eff = RATING_CENTER + (sr - RATING_CENTER) * shrink
    else:
        w = min(1.0, n_prior / k)
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
    hits = sum(1 for _ in range(n_boot)
               if abs(sum(d if rnd.random() < 0.5 else -d for d in deltas) / len(deltas)) >= abs(obs))
    return obs, hits / n_boot


def run(arms, shrink_mode, control):
    fighters = attach_imputed_reach(pd.read_csv(FIGHTERS))
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}

    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = ufc_only(history.dropna(subset=["date"])).sort_values("date", kind="stable")
    fight_index = build_fight_index(history)
    wc = pd.read_csv(WC_HISTORY) if os.path.exists(WC_HISTORY) else None

    elo = EloRatingSystem()
    counts, streaks = defaultdict(int), defaultdict(int)
    rows_out, skipped = [], 0
    widest = max(K_ARMS)

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()

        if pd.notna(winner) and winner in (a, b):
            if fa in static_rows and fb in static_rows:
                low = min(counts[fa], counts[fb])
                if shrink_mode:
                    want = (low > 0) if control else (low == 0)
                else:
                    want = (low >= widest) if control else (low < widest)
                if want:
                    ra = roster_as_of(a, when, fight_index, static_rows, today=when)
                    rb = roster_as_of(b, when, fight_index, static_rows, today=when)
                    past = history[history["date"] < f["date"]]
                    y = 1.0 if winner == a else 0.0
                    probs = {}
                    for arm in arms:
                        k = SHIPPED_K if shrink_mode else arm
                        sh = arm if shrink_mode else SHIPPED_SHRINK
                        eff = {
                            a: _effective(ra, counts[fa], elo.get_rating(a), streaks[fa], k, sh),
                            b: _effective(rb, counts[fb], elo.get_rating(b), streaks[fb], k, sh),
                        }
                        try:
                            res = predict_matchup(a, b, pd.DataFrame([ra, rb]), eff, past,
                                                  wc, None, reference_date=when.date())
                        except Exception:
                            res = None
                        p = (res or {}).get("prob_a")
                        if p is not None and not math.isnan(p):
                            probs[arm] = p
                    if len(probs) == len(arms):
                        rows_out.append((y, probs))
                    else:
                        skipped += 1

            loser = b if winner == a else a
            elo.update_ratings(winner, loser, method=method)
            streaks[_fold(winner)] += 1
            streaks[_fold(loser)] = 0
        counts[fa] += 1
        counts[fb] += 1

    return rows_out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="score the saturated complement, where every arm is "
                         "identical by construction")
    ap.add_argument("--shrink", action="store_true",
                    help="sweep DEBUT_RATING_SHRINK instead; underpowered by "
                         "construction, see the module docstring")
    args = ap.parse_args()

    arms = SHRINK_ARMS if args.shrink else K_ARMS
    shipped = SHIPPED_SHRINK if args.shrink else SHIPPED_K
    label = "DEBUT_RATING_SHRINK" if args.shrink else "min_fights_to_trust_elo"

    rows, skipped = run(arms, args.shrink, args.control)
    cut = ("saturated complement (every arm identical by construction)"
           if args.control else
           ("fights with a debuting corner" if args.shrink
            else f"fights with a corner below {max(K_ARMS)} connected bouts"))
    print(f"\nSweeping {label}\n")
    print(f"  population: {cut}")
    print(f"  {len(rows)} fights scored, {skipped} skipped (an arm could not predict)\n")
    if not rows:
        print("  Nothing to score.")
        return 0
    if len(rows) < 100:
        print(f"  *** {len(rows)} fights is too few to separate anything. Read the")
        print("      p-values as 'not measured', not as 'no effect'. ***\n")

    print(f"  {'arm':>8s} {'acc':>8s} {'brier':>9s} {'logloss':>9s} {'d.brier':>10s} {'p':>7s}"
          f" {'n.diff':>7s} {'d|diff':>9s} {'p|diff':>7s}")
    print("  " + "-" * 82)
    for arm in arms:
        n, acc, brier, ll = _score([(pr[arm], y) for y, pr in rows])
        tag = "  <- shipped" if arm == shipped else ""
        if arm == shipped:
            print(f"  {arm:>8} {acc:8.4f} {brier:9.5f} {ll:9.5f} {'--':>10s} {'--':>7s}"
                  f" {'--':>7s} {'--':>9s} {'--':>7s}{tag}")
            continue
        d, p = _paired_test(rows, arm, shipped)
        # THE PER-ARM POPULATION. A fight where this arm and the shipped value
        # produce the same probability contributes an exact zero to the paired
        # delta -- it cannot carry evidence either way, and averaging it in
        # only dilutes the estimate. Restricting to the fights an arm actually
        # changes is the correct paired population for that arm, decided by
        # the arm's own definition rather than by looking at the outcome.
        sub = [(y, pr) for y, pr in rows if abs(pr[arm] - pr[shipped]) > 1e-9]
        if sub:
            d2, p2 = _paired_test(sub, arm, shipped)
            extra = f" {len(sub):7d} {d2:+9.5f} {p2:7.3f}"
        else:
            extra = f" {0:7d} {'--':>9s} {'--':>7s}"
        print(f"  {arm:>8} {acc:8.4f} {brier:9.5f} {ll:9.5f} {d:+10.5f} {p:7.3f}{extra}")
    print("\n  d.brier is the arm MINUS the shipped value, so negative is better.")
    print("  n.diff / d|diff / p|diff restrict to the fights the arm actually changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
