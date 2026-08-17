"""
POINT-IN-TIME validation of a DIVISION-CONDITIONED round-of-finish curve.

THE CHANGE UNDER TEST. method_model._ROUND_FINISH_SHARE allocates a fight's
finishes across rounds, and every Under X.5 round prop is derived from it. It
currently holds ONE curve per fight length, applied to every division:

    3: [0.547, 0.308, 0.144]

Measured on 3,945 dated three-round finishes, divisions do not share that
shape. Light Heavyweight ends in round one 62.0% of the time and Women's
Bantamweight 41.2% -- a 21-point spread on the single number that drives
Under 1.5. A chi-square on the division x round table gives p = 0.0002, and
the R1 share is almost perfectly ordered by division weight (Spearman rho =
0.909, p = 0.0001). Heavier fighters end fights sooner; that is not a
surprise, and it is not in the model.

WHY THIS NEEDS A HARNESS AT ALL. The gradient being real does not mean
conditioning on it predicts better. Eleven divisions cut from 3,945 finishes
leaves the women's divisions on 80-110 observations each, and an unshrunk
per-division curve would happily fit their noise. The question is not "do
divisions differ" -- they do -- but "does a division-conditioned curve beat
the pooled one on fights it has never seen".

ARMS. Every arm is refit at each scored fight from finishes STRICTLY BEFORE
that date, so nothing scores on its own data:

    shipped    the hardcoded constant, frozen
    global     pooled curve, refit point-in-time
    div_kN     per-division, shrunk to the pooled curve with N pseudo-counts
    weight_kN  per-division, shrunk to a WEIGHT-LINEAR fit instead of to the
               pooled mean -- the thin divisions borrow strength from the
               ordering rather than being flattened toward the middle

The weight arm exists because of that rho = 0.909. Shrinking Women's
Strawweight toward the pooled mean discards the very fact the ordering
establishes: that a 115lb division should sit BELOW the middle, not at it.
A structured prior keeps that; a flat one throws it away.

SCORING. Multiclass log loss and Brier over the realised round, which is the
quantity a round prop is actually graded on. Bootstrap is the shared paired
sign-flip, CLUSTERED BY EVENT -- one card's finishes share a referee whose
stoppage threshold moves rounds directly.

Usage:  python3 scripts/validate_divisional_finish_curve.py
        python3 scripts/validate_divisional_finish_curve.py --window 1200
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matchup_model import _division_from_bout_label  # noqa: E402
from scripts.harness_stats import paired_signflip  # noqa: E402

RESULTS = "data/ufc_fight_results.csv"
PIT = "data/pit_stats.csv"

SHIPPED_3 = [0.547, 0.308, 0.144]

# Nominal division limits. Used only as a REGRESSOR for the structured prior,
# never as a lookup -- so a catchweight or an unlisted division simply falls
# out of the fit rather than needing a value invented for it.
DIVISION_LBS = {
    "Women's Strawweight": 115, "Women's Flyweight": 125, "Women's Bantamweight": 135,
    "Women's Featherweight": 145,
    "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145, "Lightweight": 155,
    "Welterweight": 170, "Middleweight": 185, "Light Heavyweight": 205, "Heavyweight": 265,
}


def load() -> pd.DataFrame:
    """Dated three-round finishes, one row per bout."""
    r = pd.read_csv(RESULTS)
    p = pd.read_csv(PIT)
    # The two files disagree on trailing whitespace in EVENT, which silently
    # produced a zero-row join before this strip. 781 events on both sides.
    for df, cols in ((r, ("EVENT", "BOUT")), (p, ("event", "bout"))):
        for c in cols:
            df[c] = df[c].astype(str).str.strip()
    key = p.drop_duplicates(subset=["event", "bout"])[["event", "bout", "date"]]
    m = r.merge(key, left_on=["EVENT", "BOUT"], right_on=["event", "bout"], how="left")

    m = m[m["TIME FORMAT"] == "3 Rnd (5-5-5)"]
    m = m[~m["METHOD"].str.contains("Decision", case=False, na=False)]
    m["ROUND"] = pd.to_numeric(m["ROUND"], errors="coerce")
    m["div"] = m["WEIGHTCLASS"].map(_division_from_bout_label)
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["ROUND", "date", "div"])
    m = m[m["ROUND"].between(1, 3)]
    return m.sort_values("date").reset_index(drop=True)


def _norm(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1 / 3] * 3


def _weight_prior(counts, pooled):
    """
    Per-division curve predicted by a WEIGHTED LEAST-SQUARES line in division
    weight, fit across divisions on the rounds-1 and rounds-2 shares.

    Divisions enter the fit weighted by their own finish count, so Heavyweight
    informs the slope more than Women's Featherweight does. Round 3 is the
    remainder, which keeps the vector summing to one without a third fit.

    Falls back to the pooled curve whenever fewer than four divisions have
    enough data to define a line -- early in the point-in-time sweep that is
    most of the sample.
    """
    pts = [(DIVISION_LBS[d], sum(c), _norm(c)) for d, c in counts.items()
           if d in DIVISION_LBS and sum(c) >= 20]
    if len(pts) < 4:
        return {}
    out = {}
    coefs = []
    for idx in (0, 1):
        sw = sum(w for _, w, _ in pts)
        mx = sum(w * x for x, w, _ in pts) / sw
        my = sum(w * s[idx] for x, w, s in pts) / sw
        num = sum(w * (x - mx) * (s[idx] - my) for x, w, s in pts)
        den = sum(w * (x - mx) ** 2 for x, w, _ in pts)
        b = num / den if den > 0 else 0.0
        coefs.append((my - b * mx, b))
    for d, lbs in DIVISION_LBS.items():
        r1 = coefs[0][0] + coefs[0][1] * lbs
        r2 = coefs[1][0] + coefs[1][1] * lbs
        r1 = min(max(r1, 0.05), 0.90)
        r2 = min(max(r2, 0.05), 0.90)
        r3 = max(1.0 - r1 - r2, 0.02)
        out[d] = _norm([r1, r2, r3])
    return out or {k: pooled for k in DIVISION_LBS}


def curves(counts, k, structured):
    """
    Per-division shrunk curves plus the pooled fallback.

    counts:     division -> [n_r1, n_r2, n_r3] observed so far
    k:          pseudo-observations drawn from the prior
    structured: shrink toward the weight-linear prior rather than the pooled
                curve
    """
    tot = [sum(counts[d][i] for d in counts) for i in range(3)]
    pooled = _norm(tot) if sum(tot) else list(SHIPPED_3)
    prior_by_div = _weight_prior(counts, pooled) if structured else {}
    out = {}
    for d, c in counts.items():
        prior = prior_by_div.get(d, pooled)
        out[d] = _norm([c[i] + k * prior[i] for i in range(3)])
    return out, pooled


def run(window, ks):
    df = load()
    n_total = len(df)
    start = max(0, n_total - window)
    print(f"{n_total} dated three-round finishes; scoring the last {n_total - start} "
          f"({df.iloc[start]['date'].date()} -> {df.iloc[-1]['date'].date()})\n")

    arms = ["shipped", "global"] + [f"div_k{k}" for k in ks] + [f"weight_k{k}" for k in ks]
    ll = defaultdict(list)
    br = defaultdict(list)
    clusters = []

    counts = defaultdict(lambda: [0, 0, 0])
    for i, row in enumerate(df.itertuples()):
        rnd = int(row.ROUND)
        div = row.div
        if i >= start:
            preds = {"shipped": SHIPPED_3}
            tot = [sum(counts[d][j] for d in counts) for j in range(3)]
            preds["global"] = _norm(tot) if sum(tot) else list(SHIPPED_3)
            for k in ks:
                for tag, structured in (("div", False), ("weight", True)):
                    cv, pooled = curves(counts, k, structured)
                    preds[f"{tag}_k{k}"] = cv.get(div, pooled)
            for a in arms:
                p = preds[a]
                ll[a].append(-math.log(max(p[rnd - 1], 1e-9)))
                br[a].append(sum((p[j] - (1.0 if j == rnd - 1 else 0.0)) ** 2 for j in range(3)))
            clusters.append(row.EVENT)
        counts[div][rnd - 1] += 1

    n = len(clusters)
    base = "global"
    print(f"{'arm':<12} {'log loss':>10} {'d vs global':>12} {'p':>8} {'deff':>6} {'Brier':>9}")
    for a in arms:
        m_ll = sum(ll[a]) / n
        m_br = sum(br[a]) / n
        if a == base:
            print(f"{a:<12} {m_ll:>10.5f} {'--':>12} {'--':>8} {'--':>6} {m_br:>9.5f}")
            continue
        deltas = [ll[a][i] - ll[base][i] for i in range(n)]
        d, p, deff = paired_signflip(deltas, clusters=clusters)
        print(f"{a:<12} {m_ll:>10.5f} {d:>+12.5f} {p:>8.4f} {deff:>6.2f} {m_br:>9.5f}")
    print(f"\nn = {n} scored finishes across {len(set(clusters))} events")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=1500)
    ap.add_argument("--ks", type=int, nargs="+", default=[30, 80, 200])
    a = ap.parse_args()
    run(a.window, a.ks)
