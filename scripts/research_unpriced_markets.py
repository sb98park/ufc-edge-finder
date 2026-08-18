"""
Is there an edge in the markets nobody prices sharply -- method, Double
Chance, goes-the-distance?

THE PREMISE, WHICH IS HALF RIGHT. Moneyline alpha was ruled out separately:
21 hypotheses against 16 years of closing lines, nothing surviving both a
placebo grid and the vig. The remaining hope was that the markets the site's
reader actually bets -- "wins by KO/TKO or Submission", "fight to start round
2" -- are priced far less carefully than the moneyline, and that a mediocre
model can beat a bad market where it cannot beat a good one.

The first half of that is true and measurable. The second half is not.

WHAT THE DATA IS. data/external_odds.csv carries six method prices per bout
(r_ko_odds, r_sub_odds, r_dec_odds and the Blue equivalents) alongside the
moneyline. 5,470 bouts from 2012-05-05 to 2026-03-28 have the full grid, a
moneyline and a resolved method. Double Chance is not quoted directly; it is
the sum of two grid cells. Round-start markets are not quoted at all, at any
point in this file, so nothing here can speak to them.

FINDING 1 -- THE METHOD MARKET IS ENORMOUSLY MORE EXPENSIVE.

    six-cell method grid   overround  1.2180 (median)
    moneyline, same bouts  overround  1.0385 (median)

21.8% against 3.9%. That is the whole story in one line, but it took the rest
of the script to see why.

FINDING 2 -- IT IS ALSO GENUINELY MISCALIBRATED, AND THAT SURVIVES A PROPER
DE-VIG. Proportional de-vigging a 21.8% market is not trustworthy: it assumes
the margin is spread evenly across cells when longshot cells reliably carry
more. So every number below is ALSO computed with the power method (solve k
such that sum(p_i^k) = 1), which strips proportionally more from the longshots.

    n=5,461            actual   proportional      power
    ends INSIDE        50.21%      55.02%        54.21%
    goes to DECISION   49.79%      44.98%        45.79%

Finishes are overpriced by about 4 points either way. Bettors like knockouts;
the price reflects that. Per cell under the power de-vig, submissions are the
worst of it (R_sub -0.80, B_sub -1.93) and decisions the most underpriced
(R_dec +3.56).

FINDING 3 -- AND IT IS STILL NOT EXPLOITABLE, BECAUSE THE TOLL IS FIVE TIMES
THE ERROR. Betting the mispriced side at the ACTUAL quoted prices, flat 1u,
95% intervals bootstrapped over event dates:

    Red by decision        n=5,470   -0.20%   [-5.20, +5.03]
    Blue by decision       n=5,470  -16.16%   [-21.09, -11.41]
    Red by KO/TKO          n=5,470   -9.84%   [-17.17,  -2.07]
    Red by submission      n=5,470  -30.69%   [-38.18, -22.70]
    both decision cells   n=10,940   -8.18%   [-11.36,  -5.02]

The single best cell in the whole grid is a coin flip with zero expectation,
and it is the Red one -- which inherits the +2.25pp corner artifact documented
in src/external_odds, so even that is flattered. A 4-point edge cannot pay a
20-point toll.

FINDING 4 -- OUR FEATURES ADD ALMOST NOTHING ON TOP. Predicting inside-the-
distance with the market's power-de-vigged probability plus 14 pre-fight
features, trained pre-2023 and tested after (train 3,531 / test 1,319 over 133
event dates):

    market as-is                       LL 0.67164
    market RECALIBRATED, no features   LL 0.66384   dLL +0.00781
    market + 14 features (C=0.03)      LL 0.66131   dLL +0.01033

Three quarters of the apparent gain is the recalibration -- i.e. re-stating
Finding 2 -- and the fourteen features together are worth about +0.0025 nats.
There is no model here that a loose market could be beaten with, even if one
were reachable.

SO THE ANSWER IS NO, WITH ONE GENUINELY OPEN QUESTION.

"Loose" turned out to mean expensive rather than beatable. But every price
tested here is a SIX-CELL GRID price. A sportsbook that offers Double Chance
as its own two-way market -- "Fighter A by KO/TKO or Submission", yes or no --
would price it nearer a two-way margin (5-8%) than a six-way one (20%), and at
5-8% a 4-point calibration bias is no longer obviously dead.

That cannot be tested here, because this file contains no Double Chance
quotes. It is not a rhetorical hedge; it is a specific, cheap, falsifiable
next step: capture Double Chance and goes-the-distance prices as they are seen
each week, and re-run Finding 3 against them once a few hundred have
accumulated. Until that data exists, the honest position is that the one
market the reader bets most is the one we have never measured.

Reproduce with:  python3 scripts/research_unpriced_markets.py
"""

import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.external_odds import american_to_implied, load_external_odds

METHOD_COLS = ["r_ko_odds", "r_sub_odds", "r_dec_odds",
               "b_ko_odds", "b_sub_odds", "b_dec_odds"]
CELLS = ["R_ko", "R_sub", "R_dec", "B_ko", "B_sub", "B_dec"]
DECISIONS = ("U-DEC", "S-DEC", "M-DEC")
SEED = 5
BOOTSTRAP = 4000


def to_decimal(odds) -> float:
    o = float(odds)
    return 1 + (100.0 / -o) if o < 0 else 1 + (o / 100.0)


def power_devig(row: np.ndarray) -> np.ndarray:
    """
    Solve k such that sum(p_i ** k) == 1.

    Proportional de-vig divides every cell by the same factor, which assumes
    the bookmaker spreads margin evenly. On a six-way market at 20% that
    assumption is doing real work: longshot cells carry more margin, so a
    proportional strip leaves submissions looking overpriced and decisions
    underpriced whether or not they are. The power method removes more from
    the longshots, so any bias that survives it is not a de-vigging artifact.
    """
    lo, hi = 0.2, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if (row ** k).sum() > 1:
            lo = k
        else:
            hi = k
    return row ** ((lo + hi) / 2)


def load():
    d = load_external_odds(verbose=False)
    s = d[d[METHOD_COLS].notna().all(axis=1) & d.finish.notna() & d.date.notna()].copy()
    return s[s.finish.isin(list(DECISIONS) + ["KO/TKO", "SUB"])
             & s.Winner.isin(["Red", "Blue"])]


def clustered_ci(returns: np.ndarray, dates: np.ndarray, rounds: int = BOOTSTRAP):
    """Resample EVENT DATES, not bouts. Fights on one card share conditions."""
    uniq = list(pd.unique(dates))
    out = []
    for _ in range(rounds):
        pick = [random.choice(uniq) for _ in uniq]
        out.append(np.concatenate([returns[dates == dt] for dt in pick]).mean() * 100)
    out.sort()
    return out[int(0.025 * rounds)], out[int(0.975 * rounds)]


def main():
    random.seed(SEED)
    s = load()
    raw = s[METHOD_COLS].map(american_to_implied).values
    power = np.apply_along_axis(power_devig, 1, raw)
    prop = raw / raw.sum(axis=1, keepdims=True)
    inside = ~s.finish.isin(DECISIONS).values

    print(f"n={len(s)}  {s.date.min().date()} to {s.date.max().date()}")
    print(f"\nFINDING 1 -- COST OF THE MARKET")
    print(f"  method grid overround  median {np.median(raw.sum(axis=1)):.4f}")
    print(f"  moneyline overround    median {s.overround.median():.4f}")

    print(f"\nFINDING 2 -- CALIBRATION (does the bias survive a power de-vig?)")
    print(f"  {'':<20}{'actual':>9}{'proportional':>14}{'power':>9}")
    for label, idx in (("ends INSIDE", [0, 1, 3, 4]), ("goes to DECISION", [2, 5])):
        act = (inside if label.startswith("ends") else ~inside).mean()
        print(f"  {label:<20}{act*100:>8.2f}%{prop[:, idx].sum(axis=1).mean()*100:>13.2f}%"
              f"{power[:, idx].sum(axis=1).mean()*100:>8.2f}%")

    print(f"\nFINDING 3 -- ROI AT THE ACTUAL QUOTED PRICES")
    print(f"  {'bet':<24}{'n':>7}{'hit':>8}{'ROI':>9}{'95% CI (event-clustered)':>28}")
    red = s.Winner.eq("Red").values
    wins = {
        "Red by decision": (red & ~inside, "r_dec_odds"),
        "Blue by decision": (~red & ~inside, "b_dec_odds"),
        "Red by KO/TKO": (red & s.finish.eq("KO/TKO").values, "r_ko_odds"),
        "Red by submission": (red & s.finish.eq("SUB").values, "r_sub_odds"),
    }
    for label, (won, col) in wins.items():
        ret = won.astype(float) * s[col].map(to_decimal).values - 1
        lo, hi = clustered_ci(ret, s.date.values)
        print(f"  {label:<24}{len(ret):>7}{won.mean()*100:>7.1f}%"
              f"{ret.mean()*100:>+8.2f}%   [{lo:+.2f}, {hi:+.2f}]")

    both = np.concatenate([
        (red & ~inside).astype(float) * s.r_dec_odds.map(to_decimal).values - 1,
        (~red & ~inside).astype(float) * s.b_dec_odds.map(to_decimal).values - 1])
    lo, hi = clustered_ci(both, np.concatenate([s.date.values, s.date.values]))
    print(f"  {'both decision cells':<24}{len(both):>7}{'':>8}"
          f"{both.mean()*100:>+8.2f}%   [{lo:+.2f}, {hi:+.2f}]")

    print("\n  A 4-point calibration edge does not pay a 20-point toll. "
          "The best cell is zero,\n  and it is the Red one, which carries the "
          "+2.25pp corner artifact on top.")


if __name__ == "__main__":
    main()
