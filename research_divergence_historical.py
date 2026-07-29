"""
RETROACTIVE divergence analysis: replay the model over history, join real
closing odds, and ask whether its disagreements with the market paid.

WHY THIS EXISTS. research_divergence.py answers the same question on live
picks, but there are only ~50 of them -- each divergence band holds 5-8
fights, which is suggestive and nothing more. This runs the identical
analysis against ~9 years of published closing odds, taking those bands from
single digits into the hundreds.

WHAT IT NEEDS. A public odds dataset, dropped in data/ as
external_odds.csv. Recommended:
    https://github.com/shortlikeafox/ultimate_ufc_dataset  (ufc-master.csv,
    Apache-2.0, actively maintained, odds + results + fighter stats)
Column names differ between these datasets and change over time, so nothing
is hard-coded -- the loader SNIFFS for date, fighter and odds columns and
reports exactly what it matched. A silent mis-detection would be worse than
a crash.

WHAT IT MEASURES, per divergence band (model probability minus the
price-implied probability):
    hit rate       did the model's pick win?
    units          realised profit at the closing price
Units is the honest column: bigger divergences mean underdog picks, which
lose more often by construction, so hit rate alone would condemn them
unfairly.

CONTEXT WORTH KNOWING BEFORE READING THE OUTPUT. Two independent analyses
using this same odds data found the UFC moneyline market close to efficient
-- vig-removed book probabilities sitting almost exactly on the 45-degree
calibration line, and ML models topping out around the accuracy of simply
always backing the favourite. The live-pick analysis pointed the same way:
+0.24 units per pick where the model AGREED with the market, +0.04 to +0.07
where it strongly disagreed. So the honest prior is that large divergences
are the model being wrong rather than the market. This script exists to test
that at a sample size where the answer means something.

Run: python3 research_divergence_historical.py
"""

import os
import sys
import unicodedata

import numpy as np
import pandas as pd

from src.elo import EloRatingSystem
from validate_adjustment_layer import load_per_fight_stats, load_dated_fights
import head_to_head_adjustment as H

ODDS_PATH = next((p for p in ("data/external_odds.csv", "data/ufc-master.csv",
                              "/mnt/user-data/uploads/ufc-master.csv") if os.path.exists(p)), None)
ADJ_WEIGHT = 2.0
MIN_BAND = 40


def fold(t):
    return "".join(c for c in unicodedata.normalize("NFKD", str(t).lower())
                   if not unicodedata.combining(c)).strip()


def implied(american):
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return (-a / (-a + 100.0)) if a < 0 else (100.0 / (a + 100.0))


def sniff(cols, *keywords):
    """Find the first column whose name contains all of a keyword group."""
    for kw in keywords:
        for c in cols:
            lc = c.lower()
            if all(k in lc for k in kw):
                return c
    return None


def load_market():
    df = pd.read_csv(ODDS_PATH, low_memory=False)
    cols = list(df.columns)
    date_c = sniff(cols, ("date",))
    r_c = sniff(cols, ("r_fighter",), ("red",), ("fighter1",), ("f1",))
    b_c = sniff(cols, ("b_fighter",), ("blue",), ("fighter2",), ("f2",))
    r_odds = sniff(cols, ("r", "odds"), ("red", "odds"), ("fighter1", "odds"))
    b_odds = sniff(cols, ("b", "odds"), ("blue", "odds"), ("fighter2", "odds"))
    win_c = sniff(cols, ("winner",), ("outcome",), ("result",))

    print("column detection:")
    for label, c in (("date", date_c), ("fighter A", r_c), ("fighter B", b_c),
                     ("A odds", r_odds), ("B odds", b_odds), ("winner", win_c)):
        print(f"   {label:10} -> {c}")
    if not all([date_c, r_c, b_c, r_odds, b_odds]):
        print("\nCould not identify the needed columns. Paste the header and I'll map it:")
        print("   ", cols[:25])
        sys.exit(1)

    out = {}
    for x in df.to_dict("records"):
        key = frozenset({fold(x[r_c]), fold(x[b_c])})
        out[key] = {"a": fold(x[r_c]), "b": fold(x[b_c]),
                    "odds_a": x.get(r_odds), "odds_b": x.get(b_odds),
                    "winner": fold(x[win_c]) if win_c else None}
    print(f"\nmarket rows loaded: {len(out)}")
    return out


def main():
    if not ODDS_PATH:
        print(__doc__)
        print("MISSING: put the odds CSV at data/external_odds.csv and re-run.")
        return
    market = load_market()

    # Replay the model point-in-time -- identical construction to every
    # validated harness in this repo, so the probabilities are the ones the
    # model would actually have produced before each fight.
    pf = load_per_fight_stats()
    fights = load_dated_fights()
    look = {(r["event"], r["bout"], r["fighter"]): r for r in pf.to_dict("records")}
    elo, acc, rows = EloRatingSystem(), H.Accumulator(), []
    matched = unmatched = 0
    for f in fights.itertuples(index=False):
        a1, a2 = acc.get(f.fighter_1), acc.get(f.fighter_2)
        s1, s2 = look.get((f.event, f.bout, f.fighter_1)), look.get((f.event, f.bout, f.fighter_2))
        if a1 and a2:
            gap = (elo.get_rating(f.fighter_1) - elo.get_rating(f.fighter_2)) \
                  + ADJ_WEIGHT * H.adj_production(a1, a2)
            p1 = 1.0 / (1.0 + 10 ** (-gap / 400.0))
            m = market.get(frozenset({fold(f.fighter_1), fold(f.fighter_2)}))
            if m:
                matched += 1
                # Orient the market price to OUR fighter_1.
                same = m["a"] == fold(f.fighter_1)
                imp1 = implied(m["odds_a"] if same else m["odds_b"])
                if imp1 is not None:
                    pick_is_f1 = p1 >= 0.5
                    model_p = p1 if pick_is_f1 else 1 - p1
                    mkt_p = imp1 if pick_is_f1 else 1 - imp1
                    odds = (m["odds_a"] if same else m["odds_b"]) if pick_is_f1 \
                        else (m["odds_b"] if same else m["odds_a"])
                    won = (f.winner == f.fighter_1) == pick_is_f1
                    try:
                        o = float(odds)
                        u = (o / 100.0 if o > 0 else 100.0 / -o) if won else -1.0
                    except (TypeError, ValueError):
                        u = np.nan
                    rows.append({"date": f.date, "model_p": model_p, "mkt_p": mkt_p,
                                 "divergence": model_p - mkt_p, "won": won, "units": u})
            else:
                unmatched += 1
        loser = f.fighter_2 if f.winner == f.fighter_1 else f.fighter_1
        elo.update_ratings(f.winner, loser, method=str(f.method))
        if s1 and s2:
            acc.update(f.fighter_1, s1, s2, f.duration_sec)
            acc.update(f.fighter_2, s2, s1, f.duration_sec)

    d = pd.DataFrame(rows).dropna(subset=["units"])
    print(f"name-matched to the odds file: {matched} | unmatched: {unmatched} "
          f"({matched/(matched+unmatched):.0%} matched)" if matched + unmatched else "")
    print(f"scorable fights with a usable price: {len(d)}\n")
    if len(d) < MIN_BAND:
        print("Too few matched fights to analyse -- check the column detection above.")
        return

    print(f"{'divergence':>18}{'n':>7}{'hit rate':>10}{'units':>10}{'u/pick':>9}")
    for lo, hi, lab in [(-1, 0.05, "model <= market"), (0.05, 0.15, "+5 to +15pp"),
                        (0.15, 0.30, "+15 to +30pp"), (0.30, 1.0, "+30pp or more")]:
        s = d[(d.divergence > lo) & (d.divergence <= hi)]
        if s.empty:
            continue
        flag = "" if len(s) >= MIN_BAND else "  <- thin"
        print(f"{lab:>18}{len(s):7}{s.won.mean():10.1%}{s.units.sum():10.1f}"
              f"{s.units.mean():9.3f}{flag}")

    print(f"\noverall: {len(d)} picks, {d.won.mean():.1%} hit rate, "
          f"{d.units.sum():+.1f} units ({d.units.mean():+.3f} per pick)")
    print("\nA flat-stake return near or below zero is the EXPECTED result against an")
    print("efficient market -- it is evidence about where to look next, not a failure.")


if __name__ == "__main__":
    main()
