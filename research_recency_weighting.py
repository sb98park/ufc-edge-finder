"""
Do RECENCY-WEIGHTED career stats beat all-time averages?

THE PREMISE, MEASURED FIRST. Across 1,061 fighters with 6+ tracked fights,
the median within-career drift in significant strikes per minute is 0.62 of
the between-fighter standard deviation, and ~32% drift by more than a full
sd. So an all-time average often blends two materially different fighters,
and the model can't tell which one turns up.

WHAT'S BEING CHANGED. Nothing is added to the model -- the same features
feed it. Only the ACCUMULATION changes: instead of a flat career mean, each
past fight is weighted by exp(-age / tau), so recent form dominates and old
form fades. That sharpens the existing signals rather than inventing new
ones, which is the opposite of the round-shape experiment (new covariates
that turned out to be redundant).

HONEST DESIGN. The half-life is SWEPT ON THE TUNING SPLIT ONLY, then the
winner is scored once on a frozen holdout against the all-time baseline on
an IDENTICAL fight set. An infinite half-life is included in the sweep as a
control -- it reproduces the current model exactly, so if recency doesn't
help, the sweep should pick it.

Run: python3 research_recency_weighting.py
"""

import math

import numpy as np
import pandas as pd

from src.elo import EloRatingSystem
from validate_adjustment_layer import load_per_fight_stats, load_dated_fights

HOLDOUT_START = pd.Timestamp("2019-01-01")
TUNE_START = pd.Timestamp("2015-01-01")
MIN_PRIOR = 3
CAP = 80.0
HALF_LIVES = [12, 18, 24, 36, 60, None]   # months; None = all-time (control)
WEIGHT_GRID = [1.0, 1.5, 2.0, 2.5]


class DecayAccumulator:
    """
    Career rates with an exponential half-life. tau=None reproduces a flat
    all-time mean exactly, so the control shares every other code path and
    any difference is attributable to weighting alone.
    """

    def __init__(self, half_life_months):
        self.tau = None if half_life_months is None else half_life_months * 30.44
        self.t = {}

    def _w(self, days_ago):
        if self.tau is None:
            return 1.0
        return 0.5 ** (days_ago / self.tau)

    def get(self, f, today):
        e = self.t.get(f)
        if not e or e["n"] < MIN_PRIOR:
            return None
        wsig = wsigabs = wtd = wtdatt = wtdabs = wtdfaced = wsec = wtot = 0.0
        for (d, sig, sig_abs, td, td_att, td_abs, td_faced, sec) in e["rows"]:
            w = self._w(max(0.0, (today - d).days))
            wtot += w
            wsig += w * sig; wsigabs += w * sig_abs
            wtd += w * td; wtdatt += w * td_att
            wtdabs += w * td_abs; wtdfaced += w * td_faced
            wsec += w * sec
        if wtot <= 0 or wsec <= 0:
            return None
        mins = wsec / 60.0
        return {
            "slpm": wsig / mins,
            "sapm": wsigabs / mins,
            "td_per_15": wtd / mins * 15.0,
            "td_acc": (wtd / wtdatt) if wtdatt else None,
            "td_def": (1.0 - wtdabs / wtdfaced) if wtdfaced else None,
        }

    def update(self, f, date, own, opp, dur):
        e = self.t.setdefault(f, {"n": 0, "rows": []})
        e["n"] += 1
        e["rows"].append((date, own["sig_landed"], opp["sig_landed"],
                          own["td_landed"], own["td_attempted"],
                          opp["td_landed"], opp["td_attempted"], dur))


def adjustment(a, b):
    """Production's current shape: net strikes + symmetric TD rate."""
    adj = (a["slpm"] - a["sapm"] - (b["slpm"] - b["sapm"])) * 12.0
    adj += (a["td_per_15"] - b["td_per_15"]) * 10.0
    return max(-CAP, min(CAP, adj))


def run(half_life):
    per_fight = load_per_fight_stats()
    fights = load_dated_fights()
    look = {(r["event"], r["bout"], r["fighter"]): r for r in per_fight.to_dict("records")}
    elo, acc, rows = EloRatingSystem(), DecayAccumulator(half_life), []
    for f in fights.itertuples(index=False):
        f1, f2 = f.fighter_1, f.fighter_2
        s1, s2 = look.get((f.event, f.bout, f1)), look.get((f.event, f.bout, f2))
        a1, a2 = acc.get(f1, f.date), acc.get(f2, f.date)
        if a1 and a2:
            rows.append({"date": f.date, "y": 1.0 if f.winner == f1 else 0.0,
                         "base": elo.get_rating(f1) - elo.get_rating(f2),
                         "adj": adjustment(a1, a2)})
        loser = f2 if f.winner == f1 else f1
        elo.update_ratings(f.winner, loser, method=str(f.method))
        if s1 and s2:
            acc.update(f1, f.date, s1, s2, f.duration_sec)
            acc.update(f2, f.date, s2, s1, f.duration_sec)
    return pd.DataFrame(rows)


def score(df, w):
    p = 1.0 / (1.0 + 10 ** (-(df["base"] + w * df["adj"]) / 400.0))
    y = df["y"].to_numpy()
    eps = 1e-12
    return {
        "n": len(df),
        "acc": float((((p >= 0.5).astype(float)) == y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ll": float(-np.mean(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))),
    }


def main():
    print("Running each half-life over the full point-in-time replay...\n")
    frames = {}
    for hl in HALF_LIVES:
        frames[hl] = run(hl)
        print(f"  half-life {str(hl) + ' mo' if hl else 'all-time (control)':22} "
              f"{len(frames[hl])} scorable fights")

    # ---- tune on the pre-holdout window only ----
    print(f"\n{'='*72}\nTUNING (pre-{HOLDOUT_START.year}) -- half-life and blend weight chosen here\n{'='*72}")
    best = (None, None, 9e9)
    for hl, df in frames.items():
        tune = df[(df.date >= TUNE_START) & (df.date < HOLDOUT_START)]
        for w in WEIGHT_GRID:
            r = score(tune, w)
            if r["brier"] < best[2]:
                best = (hl, w, r["brier"])
        r = score(tune, 1.5)
        print(f"  {str(hl) + ' mo' if hl else 'all-time':14} w=1.5  Brier {r['brier']:.4f}")
    hl_star, w_star, _ = best
    print(f"\n  selected: half-life={hl_star if hl_star else 'all-time'}  weight={w_star}")

    # ---- frozen holdout, identical fight set ----
    print(f"\n{'='*72}\nHOLDOUT ({HOLDOUT_START.year}+) -- scored once, identical fights\n{'='*72}")
    ctrl = frames[None]
    keys = set(map(tuple, ctrl[ctrl.date >= HOLDOUT_START][["date", "base"]].round(6).to_numpy()))

    for label, hl, w in (("all-time averages (current model)", None, w_star),
                         (f"recency-weighted ({hl_star} mo)", hl_star, w_star)):
        df = frames[hl]
        hold = df[df.date >= HOLDOUT_START]
        r = score(hold, w)
        print(f"  {label:36} n={r['n']:5}  acc {r['acc']:.1%}  Brier {r['brier']:.4f}  logloss {r['ll']:.4f}")

    a = score(frames[None][frames[None].date >= HOLDOUT_START], w_star)
    b = score(frames[hl_star][frames[hl_star].date >= HOLDOUT_START], w_star)
    d_acc = (b["acc"] - a["acc"]) * 100
    d_br = a["brier"] - b["brier"]
    print(f"\n  delta: accuracy {d_acc:+.2f} pp | Brier {d_br:+.4f} "
          f"({'RECENCY WINS' if d_br > 0 else 'no gain -- all-time averages are fine'})")
    if hl_star is None:
        print("  NOTE: the sweep selected the all-time control, i.e. recency weighting")
        print("        did not help even on the tuning split.")


if __name__ == "__main__":
    main()
