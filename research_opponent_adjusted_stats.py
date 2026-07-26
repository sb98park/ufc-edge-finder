"""
Opponent-adjusted fighter stats -- MMA's analogue of regularized adjusted
plus-minus (RAPM), the technique basketball uses to separate a player's
production from the quality of who they played.

THE PROBLEM WITH RAW RATES. The current adjustment layer (validated at
+1.8pp in validate_adjustment_layer.py) feeds on raw career rates: strikes
landed per minute, takedown accuracy, control share. A raw rate conflates
skill with schedule -- 5.5 significant strikes per minute against elite
defensive strikers is a categorically different accomplishment from 5.5
against overmatched opposition, and raw rates cannot tell them apart.

THE MODEL. Treat each fight as two observations, one per fighter, and fit:

    rate(A against B)  =  mu + off[A] + def[B]  +  error

off[A] is A's ability to generate that stat above league average; def[B] is
how much B tends to ALLOW above average (so a strong defensive fighter has
a NEGATIVE def). Both are latent and solved simultaneously across every
fighter, which is what makes the schedule wash out: off[A] is estimated
holding the defensive quality of each opponent faced constant.

RIDGE, NOT PLAIN LEAST SQUARES, for two reasons. (1) Identifiability: off
and def are pinned down only up to a constant shift (add c to every off,
subtract c from every def, predictions unchanged); the L2 penalty resolves
this by pulling both toward zero. (2) Sparse fighters: someone with two
UFC fights would otherwise get a wild coefficient fit to noise, and
shrinking them toward league average is the honest prior for a fighter
we've barely observed.

POINT-IN-TIME DISCIPLINE. Ratings are refit on a fixed cadence using only
fights STRICTLY BEFORE each refit date, and a fight is always scored with
the most recent fit that could not have seen it.

FAIRNESS NOTE -- a real bug caught in the first version of this script.
The raw-rate accumulator updates after every single fight, so it is never
stale. With ridge ratings refit only yearly, a late-in-year fight was
scored against ratings up to 12 months old (median staleness measured at
183 days). That is not "raw vs adjusted", it is "fresh vs stale", and the
first run lost on exactly that artifact. Refit cadence is therefore a
swept parameter, chosen on the tuning split like any other.

WHAT THIS ANSWERS. Three arms, same walk-forward split as the existing
harness, all scored on an IDENTICAL set of fights:
    A. Elo only                       (control)
    B. Elo + RAW stat adjustments     (what production does today)
    C. Elo + OPPONENT-ADJUSTED stats  (this technique)
Cadence, regularization and blend weight are all selected on pre-2019 only;
the holdout is evaluated once, frozen.

Run: python3 research_opponent_adjusted_stats.py
"""

import math

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from src.elo import EloRatingSystem
from validate_adjustment_layer import load_per_fight_stats, load_dated_fights

HOLDOUT_START = pd.Timestamp("2019-01-01")
MIN_PRIOR_FIGHTS = 3
ADJ_CAP = 80.0
STATS = ["sig_per_min", "td_per_15", "ctrl_share"]

# Swept on the tuning split only -- never on the holdout.
REFIT_MONTHS_GRID = [3, 6, 12]
ALPHA_GRID = [5.0, 25.0, 100.0]
WEIGHT_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

# Per-stat Elo-point scaling. Only the RELATIVE sizes matter -- the tuned
# blend weight scales the whole adjustment afterward.
STAT_SCALE = {"sig_per_min": 12.0, "td_per_15": 10.0, "ctrl_share": 50.0}


def build_observations(per_fight, fights):
    """One row per (fight, fighter): that fighter's rates, opponent, and date."""
    lookup = {(r["event"], r["bout"], r["fighter"]): r for r in per_fight.to_dict("records")}
    rows = []
    for f in fights.itertuples(index=False):
        minutes = f.duration_sec / 60.0
        if minutes <= 0:
            continue
        for me, opp in ((f.fighter_1, f.fighter_2), (f.fighter_2, f.fighter_1)):
            s = lookup.get((f.event, f.bout, me))
            if s is None:
                continue
            rows.append({
                "date": f.date, "fighter": me, "opponent": opp,
                "sig_per_min": s["sig_landed"] / minutes,
                "td_per_15": s["td_landed"] / minutes * 15.0,
                "ctrl_share": s["ctrl_sec"] / f.duration_sec,
            })
    return pd.DataFrame(rows)


def fit_opponent_adjusted(obs, alpha):
    """
    Solve rate = mu + off[fighter] + def[opponent] for all fighters at once,
    per stat, by ridge on a sparse design matrix (two non-zeros per row, so
    a dense matrix would be gigabytes for no reason).
    """
    fighters = sorted(set(obs["fighter"]) | set(obs["opponent"]))
    idx = {f: i for i, f in enumerate(fighters)}
    n_f, n_obs = len(fighters), len(obs)

    rows = np.repeat(np.arange(n_obs), 2)
    cols = np.empty(n_obs * 2, dtype=np.int64)
    cols[0::2] = [idx[f] for f in obs["fighter"]]
    cols[1::2] = [idx[o] + n_f for o in obs["opponent"]]
    X = sparse.csr_matrix((np.ones(n_obs * 2), (rows, cols)), shape=(n_obs, 2 * n_f))

    out = {}
    for stat in STATS:
        y = obs[stat].to_numpy(dtype=np.float64)
        model = Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg", max_iter=5000)
        model.fit(X, y)
        c = model.coef_
        out[stat] = {
            "off": {f: float(c[idx[f]]) for f in fighters},
            "def": {f: float(c[idx[f] + n_f]) for f in fighters},
        }
    return out


def build_pit_ratings(obs, refit_months, alpha):
    """(effective_from, ratings) snapshots, each fit only on strictly earlier fights."""
    obs = obs.sort_values("date")
    first, last = pd.Timestamp(obs["date"].min()), pd.Timestamp(obs["date"].max())
    snapshots, cursor = [], first + pd.DateOffset(years=3)
    while cursor <= last + pd.DateOffset(months=refit_months):
        train = obs[obs["date"] < cursor]
        if len(train) >= 500:
            snapshots.append((cursor, fit_opponent_adjusted(train, alpha)))
        cursor += pd.DateOffset(months=refit_months)
    return snapshots


def ratings_asof(snapshots, when):
    chosen = None
    for eff, rat in snapshots:
        if eff <= when:
            chosen = rat
        else:
            break
    return chosen


def adjusted_edge(ratings, a, b):
    """A's expected per-stat differential vs B. mu cancels; None if either is unseen."""
    edges = {}
    for stat, r in ratings.items():
        if a not in r["off"] or b not in r["off"]:
            return None
        edges[stat] = (r["off"][a] - r["off"][b]) + (r["def"][b] - r["def"][a])
    return edges


def edge_to_elo(edges):
    return max(-ADJ_CAP, min(ADJ_CAP, sum(edges[s] * STAT_SCALE[s] for s in edges)))


class RawAccumulator:
    def __init__(self):
        self.t = {}

    def get(self, f):
        t = self.t.get(f)
        if not t or t["fights"] < MIN_PRIOR_FIGHTS or t["seconds"] <= 0:
            return None
        mins = t["seconds"] / 60.0
        return {"sig_per_min": t["sig"] / mins, "td_per_15": t["td"] / mins * 15.0,
                "ctrl_share": t["ctrl"] / t["seconds"]}

    def update(self, f, s, dur):
        t = self.t.setdefault(f, {"fights": 0, "seconds": 0.0, "sig": 0, "td": 0, "ctrl": 0.0})
        t["fights"] += 1; t["seconds"] += dur
        t["sig"] += s["sig_landed"]; t["td"] += s["td_landed"]; t["ctrl"] += s["ctrl_sec"]


def replay_base(fights, per_fight):
    """
    One chronological pass recording, per scorable fight, the base Elo gap,
    the outcome, and the RAW edge. Done once and reused across every
    (cadence, alpha) config -- only the ridge part varies, so re-replaying
    Elo for each would be pure waste.
    """
    lookup = {(r["event"], r["bout"], r["fighter"]): r for r in per_fight.to_dict("records")}
    elo, acc, out = EloRatingSystem(), RawAccumulator(), []
    for f in fights.itertuples(index=False):
        f1, f2 = f.fighter_1, f.fighter_2
        s1, s2 = lookup.get((f.event, f.bout, f1)), lookup.get((f.event, f.bout, f2))
        a1, a2 = acc.get(f1), acc.get(f2)
        if a1 and a2:
            out.append({
                "date": f.date, "f1": f1, "f2": f2,
                "base_gap": elo.get_rating(f1) - elo.get_rating(f2),
                "y": 1.0 if f.winner == f1 else 0.0,
                "raw_edge": {s: a1[s] - a2[s] for s in STATS},
            })
        loser = f2 if f.winner == f1 else f1
        elo.update_ratings(f.winner, loser, method=str(f.method))
        if s1 and s2:
            acc.update(f1, s1, f.duration_sec); acc.update(f2, s2, f.duration_sec)
    return out


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


def predict(base_gap, delta, w, y):
    gap = base_gap + w * delta
    p = 1.0 / (1.0 + 10 ** (-gap / 400.0))
    return {"p": p, "y": y, "hit": (y == 1.0) == (p >= 0.5)}


def main():
    print("Loading per-fight stats and dating fights...")
    per_fight = load_per_fight_stats()
    fights = load_dated_fights()
    obs = build_observations(per_fight, fights)
    base = replay_base(fights, per_fight)
    tune = [b for b in base if b["date"] < HOLDOUT_START]
    print(f"  {len(fights)} dated fights -> {len(obs)} observations, {len(base)} scorable")
    print(f"  tuning split: {len(tune)} fights (pre-{HOLDOUT_START.year})\n")

    print("Sweeping refit cadence x regularization on the TUNING SPLIT ONLY...")
    best, cache = None, {}
    for months in REFIT_MONTHS_GRID:
        for alpha in ALPHA_GRID:
            snaps = build_pit_ratings(obs, months, alpha)
            cache[(months, alpha)] = snaps
            edges = [adjusted_edge(ratings_asof(snaps, b["date"]), b["f1"], b["f2"])
                     if ratings_asof(snaps, b["date"]) else None for b in tune]
            for w in WEIGHT_GRID:
                rows = [predict(b["base_gap"], edge_to_elo(e), w, b["y"])
                        for b, e in zip(tune, edges) if e is not None]
                res = score(rows)
                if res and (best is None or res[2] < best[0]):
                    best = (res[2], months, alpha, w)
            print(f"  refit={months:>2}mo alpha={alpha:>6} -> best so far Brier {best[0]:.4f} "
                  f"(cadence {best[1]}mo, alpha {best[2]}, w {best[3]})")

    _, months, alpha, w_adj = best
    print(f"\nSelected on tuning split: refit every {months} months, alpha {alpha}, weight {w_adj}")

    w_raw, best_raw = 0.0, float("inf")
    for w in WEIGHT_GRID:
        res = score([predict(b["base_gap"], edge_to_elo(b["raw_edge"]), w, b["y"]) for b in tune])
        if res and res[2] < best_raw:
            best_raw, w_raw = res[2], w
    print(f"Raw arm's tuned weight: {w_raw}")

    snaps = cache[(months, alpha)]
    hold = [b for b in base if b["date"] >= HOLDOUT_START]
    rows_a, rows_b, rows_c = [], [], []
    for b in hold:
        r = ratings_asof(snaps, b["date"])
        e = adjusted_edge(r, b["f1"], b["f2"]) if r else None
        if e is None:
            continue  # keep every arm on the same fights
        rows_a.append(predict(b["base_gap"], 0.0, 0.0, b["y"]))
        rows_b.append(predict(b["base_gap"], edge_to_elo(b["raw_edge"]), w_raw, b["y"]))
        rows_c.append(predict(b["base_gap"], edge_to_elo(e), w_adj, b["y"]))

    print(f"\n{'='*78}\nHOLDOUT ({HOLDOUT_START.year}+) -- frozen, identical fight set across arms\n{'='*78}")
    for label, rows in (("A. Elo only (control)", rows_a),
                        (f"B. Elo + RAW stats (w={w_raw})", rows_b),
                        (f"C. Elo + OPPONENT-ADJUSTED (w={w_adj})", rows_c)):
        res = score(rows)
        if res:
            n, a, br, ll = res
            print(f"  {label:46} n={n}  acc {a:.1%}  Brier {br:.4f}  logloss {ll:.4f}")


if __name__ == "__main__":
    main()
