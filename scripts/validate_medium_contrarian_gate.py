"""
PRE-REGISTERED test of a gate on Medium-tier picks that disagree with the market.

THE OBSERVATION. Splitting the 102 graded picks by tier and by whether the
model agreed with the de-vigged market price:

    Medium, AGAINST market      n= 8    1-7   hit 0.125   claimed 0.660
    Medium, agreed with market  n=16   12-4   hit 0.750   claimed 0.678

The Medium BAND is not broken -- 39 picks, hit 0.590 against a claimed 0.667,
a gap of exactly 1.0 SD at p = 0.312. Neither is any other band: High p =
0.156, Low p = 0.064. The entire defect is the eight fights in that first
cell, where a Medium pick took the underdog and lost seven times against a
claimed 66%.

WHY THIS FILE IS CALLED PRE-REGISTERED, AND WHAT THAT OBLIGES.
That cell was chosen AFTER looking at the data, out of 3 tiers x 2 market
positions = 6 cells, and it is the best of the six. Its p = 0.0029 is
therefore worth about 0.017 after Bonferroni, and eight fights landing in a
post-hoc cell is the exact shape of the "5-sigma method bias" CLAUDE.md
records as an argmax artefact.

So the numbers below are EXPLORATORY and are not a licence to ship. The point
of writing the rule down now, with its threshold fixed, is that the NEXT
check is confirmatory: the cell grows about one fight per card, and a rule
declared in advance cannot be fitted to the data that later tests it.

    PRE-REGISTERED CLAIM: Medium-tier picks whose de-vigged market
    probability is <= 0.5 win less often than the tier claims.
    DECISION THRESHOLD: revisit when that cell reaches n >= 25. Ship the gate
    only if the hit rate is still below the claimed probability with a
    one-sided binomial p < 0.05 computed on the fights accumulated AFTER
    2026-09-03, scored independently of the eight that motivated it.

THE RULE UNDER TEST HAS ZERO FREE PARAMETERS, which is the whole reason it is
shaped this way. There is no weight to tune and no threshold to sweep, so it
cannot be overfitted to 8 observations -- it can only be right or wrong. Two
variants:

    demote     such a pick publishes at Low Confidence instead of Medium
    suppress   such a pick is not published as a rated call at all

NO MODEL IS RE-RUN, and that is a feature rather than a shortcut. Every input
the gate needs -- favorite_prob, pick_odds, opponent_odds, confidence_label --
is frozen at publication by CLAUDE.md section 1, so this scores the policy
against exactly the numbers a reader saw on the day. There is no window for
the leakage that broke two harnesses this week.

THE LIMIT THAT MATTERS MOST. Only 59 of 102 graded picks carry BOTH prices, so
the gate is silent on the other 43 -- it can never fire on a fight whose
market was never logged. Any effect measured here is therefore an effect on
the priced subset, and the site publishes plenty of unpriced picks.

Nothing here restates the published record: it reads predictions_log and
fight results and reports what a policy WOULD have produced. It writes
nothing.

Usage:  python3 scripts/validate_medium_contrarian_gate.py
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.track_record import compute_track_record  # noqa: E402

BOOTSTRAP = 5000
PREREG_N = 25
PREREG_DATE = "2026-09-03"


def imp(o):
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    if o != o or o == 0:
        return None
    return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)


def load():
    r = compute_track_record()
    m = pd.DataFrame(r["results"])
    m["p"] = m.favorite_prob.astype(float)
    m["won"] = m.correct.astype(bool)
    log = pd.read_csv("data/predictions_log.csv").set_index(
        ["event_name", "fighter_a", "fighter_b"])
    mkt, odds = [], []
    for _, x in m.iterrows():
        try:
            row = log.loc[(x.event_name, x.fighter_a, x.fighter_b)]
        except Exception:
            mkt.append(None); odds.append(None); continue
        a, b = imp(row.get("pick_odds")), imp(row.get("opponent_odds"))
        mkt.append(a / (a + b) if a is not None and b is not None else None)
        odds.append(row.get("pick_odds"))
    m["mkt"] = mkt
    m["pick_odds"] = odds
    return r, m


def units_for(tier, won, odds):
    """Notional ladder return. Medium and Low are PUBLISHED but not staked."""
    from src.plays import TIER_CAP_UNITS
    u = TIER_CAP_UNITS.get(tier)
    if u is None or odds is None or odds != odds:
        return 0.0
    o = float(odds)
    return u * ((o / 100) if o > 0 else (100 / -o)) if won else -u


def main() -> int:
    r, m = load()

    # FIDELITY ANCHOR. If this does not match the live record, the frame has
    # been reshaped somewhere and nothing below is about the published site.
    us = r["units_stats"]
    print(f"anchor: {r['correct']}/{r['total']} picks, {us['total_units']} units, "
          f"{us['by_tier']['Lock of the Week']['count']} locks")
    print("        (must read 72/102 and 64.87 unless a card has graded since)\n")

    priced = m[m.mkt.notna()]
    print(f"graded picks {len(m)}; with BOTH prices logged {len(priced)} "
          f"({len(priced)/len(m):.0%}) -- the gate is silent on the rest\n")

    cell = priced[(priced.confidence_label == "Medium Confidence") & (priced.mkt <= 0.5)]
    rest = priced[~priced.index.isin(cell.index)]
    print(f"{'cohort':34s} {'n':>4} {'record':>8} {'hit':>6} {'claimed':>8}")
    print(f"{'Medium x against market (the cell)':34s} {len(cell):4d} "
          f"{int(cell.won.sum())}-{len(cell)-int(cell.won.sum()):<3d} "
          f"{cell.won.mean():6.3f} {cell.p.mean():8.3f}")
    print(f"{'everything else priced':34s} {len(rest):4d} "
          f"{int(rest.won.sum())}-{len(rest)-int(rest.won.sum()):<3d} "
          f"{rest.won.mean():6.3f} {rest.p.mean():8.3f}")

    from scipy.stats import binomtest
    if len(cell):
        pv = binomtest(int(cell.won.sum()), len(cell), cell.p.mean(), "less").pvalue
        print(f"\n   one-sided binomial on the cell: p = {pv:.4f}")
        print(f"   after Bonferroni over the 6 tier x market cells: p ~ {min(1.0, pv*6):.4f}")
        print(f"   n = {len(cell)}. This is exploratory. See the docstring.")

    # WHAT THE GATE WOULD HAVE DONE, both variants.
    print(f"\n{'variant':12s} {'demoted':>8} {'tier hit rates after the gate'}")
    for variant in ("shipped", "demote", "suppress"):
        mm = m.copy()
        hit = mm.mkt.notna() & (mm.confidence_label == "Medium Confidence") & (mm.mkt <= 0.5)
        if variant == "demote":
            mm.loc[hit, "confidence_label"] = "Low Confidence"
        elif variant == "suppress":
            mm = mm[~hit]
        parts = []
        for t in ("High Confidence", "Medium Confidence", "Low Confidence"):
            g = mm[mm.confidence_label == t]
            if len(g):
                parts.append(f"{t.split()[0]} {int(g.won.sum())}-{len(g)-int(g.won.sum())} "
                             f"({g.won.mean():.2f} vs {g.p.mean():.2f})")
        print(f"{variant:12s} {int(hit.sum()) if variant!='shipped' else 0:8d} {'  '.join(parts)}")

    # Notional units, since Medium and Low are published but never staked --
    # this is the headline number on the site, not money at risk.
    print(f"\n{'variant':12s} {'notional units on the priced subset':>38}")
    for variant in ("shipped", "demote", "suppress"):
        mm = m.copy()
        hit = mm.mkt.notna() & (mm.confidence_label == "Medium Confidence") & (mm.mkt <= 0.5)
        if variant == "demote":
            mm.loc[hit, "confidence_label"] = "Low Confidence"
        elif variant == "suppress":
            mm = mm[~hit]
        tot = sum(units_for(x.confidence_label, x.won, x.pick_odds)
                  for _, x in mm[mm.mkt.notna()].iterrows())
        print(f"{variant:12s} {tot:38.2f}")

    # CLUSTERED BY CARD, because 8 fights sitting on 5 cards are not 8
    # independent draws and the whole session has been strict about this.
    if len(cell) >= 3:
        per = defaultdict(list)
        for _, x in cell.iterrows():
            per[x.event_name].append(float(x.won) - x.p)
        diffs = np.array([np.mean(v) for v in per.values()])
        rng = np.random.RandomState(0)
        null = np.array([np.mean(diffs * rng.choice([-1, 1], len(diffs)))
                         for _ in range(BOOTSTRAP)])
        pv = float(np.mean(np.abs(null) >= abs(diffs.mean())))
        print(f"\n   card-clustered sign-flip on the cell: {len(per)} clusters, "
              f"mean (won - claimed) {diffs.mean():+.3f}, p = {pv:.3f}")

    print(f"\nPRE-REGISTERED, fixed today and not to be edited to fit later data:")
    print(f"   revisit when the cell reaches n >= {PREREG_N}")
    print(f"   ship only if, on fights graded AFTER {PREREG_DATE} alone, the hit rate is")
    print(f"   still below the claimed probability at one-sided binomial p < 0.05")
    return 0


if __name__ == "__main__":
    sys.exit(main())
