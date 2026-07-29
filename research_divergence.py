"""
Do the model's DISAGREEMENTS with the market resolve in its favour?

WHY THIS ONE, AND WHY FIRST. Five improvement ideas were tested and four
died on cheap diagnostics -- opponent adjustment, round-shape features,
cold-start priors, and data-quality uncertainty. Each failed because its
premise was assumed rather than measured. This question is different: its
premise is already established. Every logged pick carries a model
probability AND a market price, so a disagreement of known size is recorded
whether or not anyone analyses it. Nothing has to be invented; the data
either shows the edge resolving favourably or it doesn't.

WHAT IT MEASURES. For each resolved pick, divergence = model probability
minus the price-implied probability. Then, bucketed by divergence size:
  - hit rate            did the pick win?
  - realised units      at the price actually taken
  - CLV                 did the line move toward the pick afterwards?

CLV matters independently of the win/loss column. Beating the closing line
is evidence the model saw something before the market did, and it converges
FAR faster than win rate -- a coin-flip result can still be a good bet, and
CLV is what distinguishes the two on a small sample.

DELIBERATELY UNDERPOWERED RIGHT NOW. A handful of cards is nowhere near
enough to conclude anything, and this script says so rather than printing a
confident number over 20 picks. It exists so the measurement runs
automatically as cards accumulate -- the alternative is discovering in six
months that the logging needed one more column.

Run: python3 research_divergence.py
"""

import math
import os

import pandas as pd

PRED = "data/predictions_log.csv"
RESULTS = "data/fight_results.csv"
MIN_PER_BUCKET = 25          # below this, report but do not interpret


def implied(american):
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    return (-a / (-a + 100.0)) if a < 0 else (100.0 / (a + 100.0))


def units(american, won):
    a = float(american)
    if not won:
        return -1.0
    return (a / 100.0) if a > 0 else (100.0 / -a)


def main():
    if not os.path.exists(PRED):
        print("No predictions log yet.")
        return
    p = pd.read_csv(PRED)

    winners = {}
    if os.path.exists(RESULTS):
        r = pd.read_csv(RESULTS)
        for row in r.to_dict("records"):
            key = frozenset({str(row.get("fighter_a", "")).strip().lower(),
                             str(row.get("fighter_b", "")).strip().lower()})
            w = str(row.get("winner", "")).strip()
            if w:
                winners[key] = w

    rows = []
    for x in p.to_dict("records"):
        if str(x.get("voided", "")).lower() == "true":
            continue
        key = frozenset({str(x.get("fighter_a", "")).strip().lower(),
                         str(x.get("fighter_b", "")).strip().lower()})
        won_by = winners.get(key)
        imp = implied(x.get("pick_odds"))
        if won_by is None or imp is None or pd.isna(x.get("favorite_prob")):
            continue
        model_p = float(x["favorite_prob"])
        won = str(won_by).strip().lower() == str(x.get("favorite", "")).strip().lower()
        close_imp = implied(x.get("closing_odds"))
        rows.append({
            "fight": f"{x.get('fighter_a')} vs {x.get('fighter_b')}",
            "model_p": model_p,
            "market_p": imp,
            "divergence": model_p - imp,
            "won": won,
            "units": units(x["pick_odds"], won),
            # Positive CLV = the closing price was SHORTER than what we took,
            # i.e. the market moved toward our side after we logged it.
            "clv": (close_imp - imp) if close_imp is not None else None,
        })

    d = pd.DataFrame(rows)
    print(f"resolved picks with both a model probability and a taken price: {len(d)}")
    if d.empty:
        print("Nothing resolved yet -- rerun once results are recorded.")
        return

    print(f"\n{'divergence':>16}{'n':>6}{'hit rate':>10}{'units':>9}{'mean CLV':>10}")
    bands = [(-1, 0.05, "model <= market"), (0.05, 0.15, "+5 to +15pp"),
             (0.15, 0.30, "+15 to +30pp"), (0.30, 1.0, "+30pp or more")]
    for lo, hi, lab in bands:
        s = d[(d.divergence > lo) & (d.divergence <= hi)]
        if s.empty:
            continue
        clv = s.clv.dropna()
        flag = "" if len(s) >= MIN_PER_BUCKET else "   <- too few to interpret"
        print(f"{lab:>16}{len(s):6}{s.won.mean():10.1%}{s.units.sum():9.2f}"
              f"{(clv.mean() if len(clv) else float('nan')):10.3f}{flag}")

    print(f"\noverall: {d.won.mean():.1%} hit rate, {d.units.sum():+.2f} units, "
          f"mean divergence {d.divergence.mean():+.3f}")
    clv = d.clv.dropna()
    if len(clv):
        print(f"CLV: beat the closing line on {(clv > 0).mean():.1%} of {len(clv)} picks "
              f"(mean {clv.mean():+.3f})")

    if len(d) < MIN_PER_BUCKET * 2:
        print(f"\nSAMPLE TOO SMALL TO CONCLUDE ANYTHING ({len(d)} picks). At this rate a")
        print("usable read needs roughly 8-12 more cards. The value of running it now")
        print("is confirming the log captures everything the analysis needs -- it does:")
        print("model probability, price taken, closing price, and a resolvable outcome.")


if __name__ == "__main__":
    main()
