"""
Head-to-head: production's adjustment formula vs a simpler symmetric one.

WHY THIS EXISTS. The RAPM research (research_opponent_adjusted_stats.py)
carried a control arm built the lazy way -- plain differences of three
career rates -- purely so the opponent-adjusted arm had something to beat.
It scored 58.6% on the holdout. validate_adjustment_layer.py's more
elaborate formulation, which mirrors production's actual signals, scored
57.7% on essentially the same split. The simple one won by ~0.9pp, which
was never the point of that experiment and so was never tested properly.

This tests it properly. The two formulations differ in one real way:

  PRODUCTION-SPIRIT (asymmetric wrestling):
      striking : (SLpM - SApM) differential
      wrestling: max(0, A's TD accuracy - B's TD defense) in each
                 direction -- a one-sided "can he get you down" term that
                 contributes nothing when accuracy is below defense
      control  : control-share differential

  SIMPLE (symmetric everywhere):
      striking : strikes landed per minute differential
      wrestling: takedowns landed per 15 min differential
      control  : control-share differential

The asymmetric version encodes a real intuition -- being hard to take
down should not itself be an offensive weapon -- but max(0, ...) discards
sign information, and that may cost more than the intuition earns.

FAIRNESS. One shared accumulator feeds both (so neither sees different
data), both are scored on an IDENTICAL fight set, and each gets its OWN
weight tuned on the pre-2019 split alone. Holdout is evaluated once.

Run: python3 head_to_head_adjustment.py
"""

import math

import pandas as pd

from src.elo import EloRatingSystem
from validate_adjustment_layer import load_per_fight_stats, load_dated_fights

HOLDOUT_START = pd.Timestamp("2019-01-01")
MIN_PRIOR_FIGHTS = 3
CAP = 80.0
WEIGHT_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


class Accumulator:
    """
    Career-to-date totals rich enough to feed BOTH formulations, so the
    comparison is about the formulas and not about which one happened to
    get better inputs.
    """

    def __init__(self):
        self.t = {}

    def get(self, f):
        t = self.t.get(f)
        if not t or t["fights"] < MIN_PRIOR_FIGHTS or t["seconds"] <= 0:
            return None
        mins = t["seconds"] / 60.0
        return {
            # shared
            "ctrl_share": t["ctrl"] / t["seconds"],
            # simple arm
            "sig_per_min": t["sig"] / mins,
            "td_per_15": t["td_landed"] / mins * 15.0,
            # production-spirit arm
            "strike_acc": t["sig"] / t["sig_att"] if t["sig_att"] else None,
            "td_def": 1.0 - (t["td_absorbed"] / t["td_faced"]) if t["td_faced"] else None,
            "slpm": t["sig"] / mins,
            "sapm": t["sig_absorbed"] / mins,
            "td_acc": t["td_landed"] / t["td_att"] if t["td_att"] else None,
        }

    def update(self, f, own, opp, dur):
        t = self.t.setdefault(f, {"fights": 0, "seconds": 0.0, "sig": 0, "sig_absorbed": 0,
                                  "sig_att": 0, "td_landed": 0, "td_att": 0, "td_absorbed": 0, "td_faced": 0,
                                  "ctrl": 0.0})
        t["fights"] += 1
        t["seconds"] += dur
        t["sig"] += own["sig_landed"]
        t["sig_att"] += own["sig_attempted"]
        t["sig_absorbed"] += opp["sig_landed"]
        t["td_landed"] += own["td_landed"]
        t["td_att"] += own["td_attempted"]
        t["td_absorbed"] += opp["td_landed"]
        t["td_faced"] += opp["td_attempted"]
        t["ctrl"] += own["ctrl_sec"]


def adj_production(a, b):
    """validate_adjustment_layer.py's formulation -- mirrors production's signals."""
    adj = (a["slpm"] - a["sapm"] - (b["slpm"] - b["sapm"])) * 12.0
    if a["td_acc"] is not None and b["td_def"] is not None:
        adj += max(0.0, a["td_acc"] - b["td_def"]) * 60.0
    if b["td_acc"] is not None and a["td_def"] is not None:
        adj -= max(0.0, b["td_acc"] - a["td_def"]) * 60.0
    adj += (a["ctrl_share"] - b["ctrl_share"]) * 50.0
    return max(-CAP, min(CAP, adj))


def adj_simple(a, b):
    """Plain symmetric differentials -- the arm that unexpectedly won."""
    adj = ((a["sig_per_min"] - b["sig_per_min"]) * 12.0
           + (a["td_per_15"] - b["td_per_15"]) * 10.0
           + (a["ctrl_share"] - b["ctrl_share"]) * 50.0)
    return max(-CAP, min(CAP, adj))


def adj_hybrid(a, b):
    """
    The obvious third option nobody has tried: keep production's
    absorbed-strikes term (net output, which the simple arm throws away by
    ignoring SApM) but make wrestling SYMMETRIC (a signed takedown-rate
    differential rather than a clipped one). If the asymmetry is what
    hurts, this should match or beat both.
    """
    adj = (a["slpm"] - a["sapm"] - (b["slpm"] - b["sapm"])) * 12.0
    adj += (a["td_per_15"] - b["td_per_15"]) * 10.0
    adj += (a["ctrl_share"] - b["ctrl_share"]) * 50.0
    return max(-CAP, min(CAP, adj))


def adj_unclipped(a, b):
    """
    The MINIMAL change to production: identical to adj_production except the
    wrestling term drops its max(0, ...) clipping, so a takedown-accuracy
    deficit counts against a fighter instead of silently reading as zero.

    This is the arm that actually answers "should production remove the
    clipping", because it holds every other term fixed. adj_simple/adj_hybrid
    change the wrestling INPUT as well (rates instead of accuracy-vs-defense),
    so they can't isolate the clipping on their own.
    """
    adj = (a["slpm"] - a["sapm"] - (b["slpm"] - b["sapm"])) * 12.0
    if a["td_acc"] is not None and b["td_def"] is not None:
        adj += (a["td_acc"] - b["td_def"]) * 60.0
    if b["td_acc"] is not None and a["td_def"] is not None:
        adj -= (b["td_acc"] - a["td_def"]) * 60.0
    adj += (a["ctrl_share"] - b["ctrl_share"]) * 50.0
    return max(-CAP, min(CAP, adj))


def adj_prodshape_clipped(a, b):
    """Production's SHAPE: accuracy-based striking + clipped wrestling."""
    if a["strike_acc"] is None or b["strike_acc"] is None:
        return 0.0
    adj = (a["strike_acc"] - b["strike_acc"]) * 150.0
    if a["td_acc"] is not None and b["td_def"] is not None:
        adj += max(0.0, a["td_acc"] - b["td_def"]) * 300.0
    if b["td_acc"] is not None and a["td_def"] is not None:
        adj -= max(0.0, b["td_acc"] - a["td_def"]) * 300.0
    return max(-CAP, min(CAP, adj))


def adj_prodshape_tdrate(a, b):
    """
    Production's SHAPE with ONLY the wrestling term swapped for a symmetric
    takedown-rate differential. This is the exact change under consideration,
    with everything else held fixed.
    """
    if a["strike_acc"] is None or b["strike_acc"] is None:
        return 0.0
    adj = (a["strike_acc"] - b["strike_acc"]) * 150.0
    adj += (a["td_per_15"] - b["td_per_15"]) * 10.0
    return max(-CAP, min(CAP, adj))


ARMS = {"production (asymmetric)": adj_production,
        "PRODSHAPE clipped wrestling": adj_prodshape_clipped,
        "PRODSHAPE td-rate wrestling": adj_prodshape_tdrate,
        "production MINUS clipping": adj_unclipped,
        "simple (symmetric)": adj_simple,
        "hybrid (net strikes + symmetric TD)": adj_hybrid}


def score(rows):
    n = len(rows)
    if n == 0:
        return None
    acc = sum(1 for r in rows if r["hit"]) / n
    brier = sum((r["p"] - r["y"]) ** 2 for r in rows) / n
    eps = 1e-12
    ll = -sum(r["y"] * math.log(max(r["p"], eps)) + (1 - r["y"]) * math.log(max(1 - r["p"], eps))
              for r in rows) / n
    return n, acc, brier, ll


def predict(gap, y):
    p = 1.0 / (1.0 + 10 ** (-gap / 400.0))
    return {"p": p, "y": y, "hit": (y == 1.0) == (p >= 0.5)}


def main():
    per_fight = load_per_fight_stats()
    fights = load_dated_fights()
    lookup = {(r["event"], r["bout"], r["fighter"]): r for r in per_fight.to_dict("records")}

    elo, acc, base = EloRatingSystem(), Accumulator(), []
    for f in fights.itertuples(index=False):
        f1, f2 = f.fighter_1, f.fighter_2
        s1, s2 = lookup.get((f.event, f.bout, f1)), lookup.get((f.event, f.bout, f2))
        a1, a2 = acc.get(f1), acc.get(f2)
        if a1 and a2:
            base.append({
                "date": f.date, "y": 1.0 if f.winner == f1 else 0.0,
                "base_gap": elo.get_rating(f1) - elo.get_rating(f2),
                "adj": {name: fn(a1, a2) for name, fn in ARMS.items()},
            })
        loser = f2 if f.winner == f1 else f1
        elo.update_ratings(f.winner, loser, method=str(f.method))
        if s1 and s2:
            acc.update(f1, s1, s2, f.duration_sec)
            acc.update(f2, s2, s1, f.duration_sec)

    tune = [b for b in base if b["date"] < HOLDOUT_START]
    hold = [b for b in base if b["date"] >= HOLDOUT_START]
    print(f"{len(base)} scorable fights -> {len(tune)} tuning / {len(hold)} holdout\n")

    print("Tuning each arm's own weight on the pre-2019 split only:")
    best_w = {}
    for name in ARMS:
        bw, bb = None, float("inf")
        for w in WEIGHT_GRID:
            res = score([predict(b["base_gap"] + w * b["adj"][name], b["y"]) for b in tune])
            if res and res[2] < bb:
                bb, bw = res[2], w
        best_w[name] = bw
        print(f"  {name:38} weight {bw}  (tuning Brier {bb:.4f})")

    print(f"\n{'='*84}\nHOLDOUT (2019+) -- frozen, identical fights\n{'='*84}")
    ctrl = score([predict(b["base_gap"], b["y"]) for b in hold])
    print(f"  {'Elo only (control)':40} n={ctrl[0]}  acc {ctrl[1]:.1%}  Brier {ctrl[2]:.4f}  logloss {ctrl[3]:.4f}")
    for name in ARMS:
        w = best_w[name]
        res = score([predict(b["base_gap"] + w * b["adj"][name], b["y"]) for b in hold])
        print(f"  {name + f' (w={w})':40} n={res[0]}  acc {res[1]:.1%}  Brier {res[2]:.4f}  logloss {res[3]:.4f}")


if __name__ == "__main__":
    main()
