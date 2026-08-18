"""
Should the market blend weight depend on whether the model agrees with the
market?

THE OBSERVATION. Split by agreement, the model's calibration error is large
and points in opposite directions -- underconfident by ~19 points where it
agrees with the market, overconfident by ~26 where it takes the underdog, both
surviving a card-clustered bootstrap. market_blended_prob applies ONE fixed
0.30 weight to both, and that constant's own docstring asks for exactly this:
"revisit this weight once enough graded picks exist to fit it out-of-sample."

WHY THE OBVIOUS READING IS NOT ENOUGH. It is tempting to go straight from
"the model is anti-informative when it disagrees" to "set that weight to
zero". But the sample is 84 picks across SEVEN cards, and the effect was
discovered in the same data any weight would be fitted on. Fitting and
evaluating on one sample of seven clusters will report an improvement whether
or not one exists.

So every number here is leave-one-CARD-out. The weight is fitted on six cards
and scored on the seventh, rotated, and only the held-out scores are reported.
Cards, not picks, because thirteen fights on one night share a slate, a
market regime and one set of late-money conditions.

THE GRID SPANS NEGATIVE WEIGHTS DELIBERATELY. w*model + (1-w)*market with
w < 0 pushes the estimate PAST the market, away from the model. That is not a
curiosity here: where the two agree, the observed win rate sits above both
estimates, so no convex mix of them can reach it and the correction that
cohort actually needs is extremising rather than shrinking.

Reports Brier and log loss, and a card-clustered bootstrap on the DIFFERENCE
against the current fixed weight -- a difference interval that straddles zero
means this is not worth shipping no matter how good the point estimate looks.

RESULT, 2026-08-18, 84 settled picks across 7 cards. NOTHING SHIPPED.

    pooled out-of-sample Brier      vs current fixed 0.30
    cohort-fitted      0.1730       -0.0095  CI [-0.0194, -0.0006]  WORSE
    pre-specified rule 0.1621       +0.0013  CI [-0.0080, +0.0114]  noise
    pure market        0.1624       +0.0011  CI [-0.0059, +0.0099]  noise
    pure model         0.1905       -0.0270  CI [-0.0516, -0.0077]  WORSE
    current            0.1635

THE FITTED VERSION IS RELIABLY WORSE, which is the more useful half of this.
Its per-fold weights swing from -0.20 to +0.35 on the disagree cohort across
seven folds -- there is not enough data to estimate two weights, so it fits
each fold's noise and carries it into the next. The calibration split that
motivated all this is real and survives its own bootstrap; being real is not
the same as being estimable.

THE PRE-SPECIFIED RULE -- keep 0.30 where the model agrees with the market,
drop to 0.00 where it disagrees -- has the best point estimate of the four and
wins 4 of 7 held-out cards, and it still does not clear the bar. Restricting
the comparison to the 23 picks it actually changes (pooling over 84 dilutes an
effect that touches 23) does not rescue it: +0.0049 Brier, CI [-0.0332,
+0.0387].

AND THAT LAST INTERVAL IS THE REASON TO STOP RATHER THAN WAIT. The effect is
around +0.005 Brier against noise of +-0.036. Interval width falls with the
square root of the number of CARDS, so separating an effect this small from
zero needs on the order of fifty times the current sample -- hundreds of
cards, years of them. This is not a "revisit when more data arrives" item at
its current effect size; it is a dead end unless the underlying calibration
gap widens a lot.

WHAT THE NUMBERS DO SAY, and it is worth more than the change would have
been: pure market ties the current blend (0.1624 vs 0.1635, interval across
zero). The model's probability is contributing essentially nothing beyond the
de-vigged market price. The 0.30 weight is not hurting either -- but its value
is not in the number it produces.

Re-run this after a stretch of new cards. It is a pre-registered test now, so
the answer will not depend on who asks.
"""

import csv
import random
import sys
import unicodedata

import pandas as pd

sys.path.insert(0, ".")

from src.odds_utils import MARKET_BLEND_MODEL_WEIGHT
from src.track_record import _backfill_legacy_fair_probs

RESULTS = "data/fight_results.csv"
LOG = "data/predictions_log.csv"
GRID = [round(x * 0.05, 2) for x in range(-20, 21)]      # -1.00 .. +1.00
BOOTSTRAP_ROUNDS = 4000
SEED = 20260818


def _fold(s) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def load() -> pd.DataFrame:
    res = {}
    for _, x in pd.read_csv(RESULTS).iterrows():
        res[frozenset({_fold(x.fighter_a), _fold(x.fighter_b)})] = _fold(x.winner)

    rows = []
    with open(LOG, newline="") as fh:
        for raw in csv.DictReader(fh):
            # Same migration production applies on load, so this harness sees
            # exactly the probabilities the site sees.
            p = _backfill_legacy_fair_probs(dict(raw))
            key = frozenset({_fold(p["fighter_a"]), _fold(p["fighter_b"])})
            if key not in res:
                continue
            try:
                model = float(p["favorite_prob"])
                market = float(p["pick_fair_prob"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (0.0 < model < 1.0 and 0.0 < market < 1.0):
                continue
            rows.append({
                "card": p["event_name"],
                "model": model,
                "market": market,
                "won": 1.0 if res[key] == _fold(p["favorite"]) else 0.0,
                # The model and the market naming the same fighter. Computed
                # from the market side because pick_fair_prob is the price of
                # whoever the MODEL picked: above 0.5 means the market made
                # that same fighter its favourite.
                "agree": market > 0.5,
            })
    return pd.DataFrame(rows)


def blend(model, market, w):
    p = w * model + (1.0 - w) * market
    # A weight outside [0,1] can push past 0 or 1. Clamped well short of the
    # boundary: log loss is unbounded at a confident miss, and one clipped
    # certainty would otherwise decide the whole comparison.
    return min(max(p, 0.02), 0.98)


def brier(p, y):
    return ((p - y) ** 2).mean()


def logloss(p, y):
    import math
    return -sum(yi * math.log(pi) + (1 - yi) * math.log(1 - pi)
                for pi, yi in zip(p, y)) / len(p)


def score(df, w_agree, w_dis):
    p = pd.Series([blend(r.model, r.market, w_agree if r.agree else w_dis)
                   for r in df.itertuples()], index=df.index)
    return brier(p, df.won), logloss(p.tolist(), df.won.tolist()), p


def fit_on(df):
    """Best (w_agree, w_dis) by Brier on the given rows. Each cohort is
    independent, so they are searched separately rather than over a 41x41
    product."""
    best = {}
    for cohort, sub in (("agree", df[df.agree]), ("dis", df[~df.agree])):
        if sub.empty:
            best[cohort] = MARKET_BLEND_MODEL_WEIGHT
            continue
        best[cohort] = min(
            GRID, key=lambda w: brier(pd.Series([blend(r.model, r.market, w)
                                                 for r in sub.itertuples()],
                                                index=sub.index), sub.won))
    return best["agree"], best["dis"]


def main():
    random.seed(SEED)
    df = load()
    cards = sorted(df.card.unique())
    print(f"{len(df)} settled picks across {len(cards)} cards "
          f"({int(df.agree.sum())} agree / {int((~df.agree).sum())} disagree)\n")

    # --- In-sample fit, for reference only. Never the basis for a decision.
    ia, idis = fit_on(df)
    print(f"in-sample best weights: agree {ia:+.2f}  disagree {idis:+.2f}  "
          f"(current: {MARKET_BLEND_MODEL_WEIGHT:.2f} for both)")

    # --- Leave-one-card-out. This is the number that counts.
    held = []
    for card in cards:
        train, test = df[df.card != card], df[df.card == card]
        if train.empty or test.empty:
            continue
        wa, wd = fit_on(train)
        held.append({"card": card, "n": len(test), "wa": wa, "wd": wd,
                     "fitted": score(test, wa, wd)[0],
                     "current": score(test, MARKET_BLEND_MODEL_WEIGHT,
                                      MARKET_BLEND_MODEL_WEIGHT)[0],
                     "market": score(test, 0.0, 0.0)[0],
                     "model": score(test, 1.0, 1.0)[0],
                     # PRE-SPECIFIED, not fitted. Keep the current weight
                     # where the model agrees with the market and drop it to
                     # zero where it disagrees -- the rule the calibration
                     # split implies, with no free parameters to overfit.
                     # This is the honest test of the IDEA, separate from the
                     # test of whether the weights can be estimated at all.
                     "rule": score(test, MARKET_BLEND_MODEL_WEIGHT, 0.0)[0]})
    h = pd.DataFrame(held)
    print("\nLEAVE-ONE-CARD-OUT (weights fitted on the other six, scored here)")
    print(f"{'held-out card':<44}{'n':>4}{'w_ag':>7}{'w_dis':>7}"
          f"{'fitted':>9}{'rule':>7}{'current':>9}{'market':>9}{'model':>9}")
    for r in h.itertuples():
        print(f"{r.card[:42]:<44}{r.n:>4}{r.wa:>+7.2f}{r.wd:>+7.2f}"
              f"{r.fitted:>9.4f}{r.rule:>7.4f}{r.current:>9.4f}{r.market:>9.4f}{r.model:>9.4f}")

    # Pooled over held-out picks, weighted by card size.
    def pooled(col):
        return (h[col] * h.n).sum() / h.n.sum()
    print(f"\n{'POOLED out-of-sample Brier':<44}{h.n.sum():>4}"
          f"{'':>14}{pooled('fitted'):>9.4f}{pooled('rule'):>7.4f}{pooled('current'):>9.4f}"
          f"{pooled('market'):>9.4f}{pooled('model'):>9.4f}")

    # --- Does the improvement survive resampling CARDS?
    def boot(col_a, col_b):
        out = []
        for _ in range(BOOTSTRAP_ROUNDS):
            pick = [random.choice(cards) for _ in cards]
            sub = h[h.card.isin(pick)]
            if sub.empty:
                continue
            rows = pd.concat([h[h.card == c] for c in pick])
            out.append(((rows[col_b] * rows.n).sum() / rows.n.sum())
                       - ((rows[col_a] * rows.n).sum() / rows.n.sum()))
        out.sort()
        return out[int(0.025 * len(out))], out[int(0.975 * len(out))]

    print("\nIMPROVEMENT IN BRIER vs the current fixed weight "
          "(positive = better; 95% CI over resampled cards)")
    for label, col in (("cohort-fitted", "fitted"), ("pre-specified rule", "rule"),
                       ("pure market", "market"), ("pure model", "model")):
        gain = pooled("current") - pooled(col)
        lo, hi = boot(col, "current")
        # THREE OUTCOMES, NOT TWO. An interval entirely BELOW zero is not
        # "inconclusive", it is a reliable regression -- and calling it noise
        # would let a change that measurably makes things worse look merely
        # unproven.
        if lo > 0:
            verdict = "BETTER -- ships"
        elif hi < 0:
            verdict = "WORSE -- reliably, do not ship"
        else:
            verdict = "not distinguishable from noise"
        print(f"  {label:<16}{gain:>+9.4f}   CI [{lo:+.4f}, {hi:+.4f}]   {verdict}")


if __name__ == "__main__":
    main()
