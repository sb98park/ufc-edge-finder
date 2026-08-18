"""
What WOULD the published parlays have returned?

READ THE LIMITATION FIRST -- IT IS THE WHOLE RESULT.

No parlay has ever been persisted. build_bankroll_builder_parlays and its two
siblings run at render time, from live prop odds, and nothing writes the
output anywhere. data/predictions_log.csv logs fight-level MONEYLINE picks
only. So the honest answer to "how have the published parlays done" is: THAT
IS NOT RECOVERABLE, and no amount of analysis here changes it.

What this script can do is a RULE REPLAY. It takes the settled moneyline
picks that were logged, applies the same selection rules the real builders
use -- leg counts, payout bands, minimum model probability, one leg per
fight -- and grades the result against data/fight_results.csv.

HOW THIS DIFFERS FROM THE REAL SLIPS, precisely:

  1. MONEYLINE LEGS ONLY. Real slips draw on Total Rounds, Method and Fight
     Outcome markets too, and combine winner+length legs from the SAME fight
     into one piece. Those prices are not logged anywhere, so they cannot be
     replayed at all.
  2. Fewer candidate pieces therefore fewer combinations, so the search finds
     different -- generally worse-priced -- slips than the real builder had
     available.
  3. pick_odds is the price when the pick was FIRST logged. The real builder
     used whatever the price was at render time, which drifts.

So treat every number below as an ORDER OF MAGNITUDE on the strategy, not as
a track record. It is the difference between "would this shape of bet have
worked" and "here is what we published", and only the first is answerable.

Usage:  python3 scripts/replay_parlays.py
"""

import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.odds_utils import american_to_decimal, decimal_to_american  # noqa: E402

PREDICTIONS = "data/predictions_log.csv"
RESULTS = "data/fight_results.csv"

# Mirrors the three real builders in src/parlay_builder.
TIERS = [
    ("Bankroll", (2, 3), 100, 320, 0.50, 3),
    ("Lotto", (2, 3, 4, 5), 1000, None, 0.15, 3),
    ("Moonshot", (2, 3, 4, 5, 6, 7, 8), 5000, None, 0.05, 3),
]


def _fold(s) -> str:
    return str(s).strip().lower()


def load():
    preds = pd.read_csv(PREDICTIONS)
    res = pd.read_csv(RESULTS)
    winners = {}
    for r in res.itertuples():
        key = frozenset((_fold(r.fighter_a), _fold(r.fighter_b)))
        winners[key] = _fold(r.winner)

    rows = []
    for p in preds.itertuples():
        # `voided` marks a pick pulled before it could settle -- a cancelled
        # bout or a replaced fighter. Those never reached a slip and must not
        # be graded as anything.
        if str(getattr(p, "voided", "")).strip().lower() == "true":
            continue
        key = frozenset((_fold(p.fighter_a), _fold(p.fighter_b)))
        if key not in winners:
            continue                       # not settled yet
        if pd.isna(p.pick_odds) or pd.isna(p.favorite_prob):
            continue
        rows.append({
            "event": p.event_name,
            "fight": key,
            "pick": p.favorite,
            "prob": float(p.favorite_prob),
            "odds": float(p.pick_odds),
            "decimal": american_to_decimal(float(p.pick_odds)),
            "won": winners[key] == _fold(p.favorite),
        })
    return rows


def find_slips(pool, leg_counts, lo, hi, min_prob, max_results):
    """The same search the real builder runs: best combined model probability
    among combinations that clear the payout band, one leg per fight."""
    eligible = [p for p in pool if p["prob"] >= min_prob]
    found = []
    for n in leg_counts:
        for combo in itertools.combinations(eligible, n):
            if len({c["fight"] for c in combo}) != n:
                continue               # one leg per fight, same as production
            dec = 1.0
            prob = 1.0
            for c in combo:
                dec *= c["decimal"]
                prob *= c["prob"]
            am = decimal_to_american(dec)
            if am < lo or (hi is not None and am > hi):
                continue
            found.append({"legs": combo, "american": am, "decimal": dec, "prob": prob})
    found.sort(key=lambda s: s["prob"], reverse=True)
    # Dedupe on leg identity so near-identical slips don't fill the slate.
    out, seen = [], set()
    for s in found:
        k = frozenset(id(l) for l in s["legs"])
        names = frozenset(l["pick"] for l in s["legs"])
        if names in seen:
            continue
        seen.add(names)
        out.append(s)
        if len(out) >= max_results:
            break
    return out


def main():
    rows = load()
    if not rows:
        print("no settled picks to replay")
        return
    by_event = {}
    for r in rows:
        by_event.setdefault(r["event"], []).append(r)

    print(f"{len(rows)} settled moneyline picks across {len(by_event)} events\n")

    grand = {}
    for name, leg_counts, lo, hi, min_prob, max_results in TIERS:
        staked = 0.0
        returned = 0.0
        hits = 0
        total = 0
        per_event = []
        for event, pool in by_event.items():
            slips = find_slips(pool, leg_counts, lo, hi, min_prob, max_results)
            ev_ret = 0.0
            for s in slips:
                total += 1
                staked += 1.0                      # 1 unit flat per slip
                if all(l["won"] for l in s["legs"]):
                    hits += 1
                    returned += s["decimal"]
                    ev_ret += s["decimal"] - 1.0
                else:
                    ev_ret -= 1.0
            if slips:
                per_event.append((event, len(slips), ev_ret))
        if not total:
            print(f"{name}: no qualifying slips\n")
            continue
        profit = returned - staked
        exp_hits = sum(1 for _ in range(0))  # placeholder, computed below
        print(f"--- {name} ---")
        print(f"  slips {total}   hit {hits} ({hits/total:.1%})   "
              f"staked {staked:.0f}u   returned {returned:.2f}u   "
              f"P/L {profit:+.2f}u   ROI {profit/staked:+.1%}")
        for ev, n, pl in sorted(per_event, key=lambda t: t[2]):
            print(f"      {ev[:52]:<52} {n} slips  {pl:+7.2f}u")
        grand[name] = (total, hits, profit, staked)
        print()

    print("=" * 64)
    t_slips = sum(g[0] for g in grand.values())
    t_profit = sum(g[2] for g in grand.values())
    t_staked = sum(g[3] for g in grand.values())
    if t_staked:
        print(f"ALL TIERS: {t_slips} slips, {t_profit:+.2f}u on {t_staked:.0f}u staked "
              f"({t_profit/t_staked:+.1%})")
    print("\nThis is a RULE REPLAY on moneyline legs only -- see the module\n"
          "docstring. It is not a record of what was published.")


if __name__ == "__main__":
    main()
