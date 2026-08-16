"""
Does blending the model with the closing market beat either alone?

THE QUESTION, AND WHY IT IS THE RIGHT ONE. Across seven logged events the
model picks the market favourite well (40/47 = 85%) and picks against the
market badly (9/21 = 43%). Disagreement is the entire product -- agreeing
with the market has no edge in it -- so a sub-coinflip rate there is the
finding that matters, and UFC 330's 0/3 is consistent with it rather than
exceptional.

A 50/50 blend beat both inputs on the 68 logged fights that carry closing
odds (Brier 0.1800 vs 0.1946 model-alone, 0.1989 market-alone). At n=68
that was p=0.36 -- suggestive, not decisive. data/external_odds.csv carries
7,177 fights with both sides' prices and outcomes back to 2010, 6,542 of
which match fight_history, so the same test can run at roughly fifty times
the sample.

READ THIS BEFORE QUOTING THE NUMBERS. This harness does NOT score the
production model. It scores a near-Elo-only one, and the difference is the
whole interpretation.

fighters.csv holds 293 CURRENT fighters. fight_history holds 10,692 bouts,
of which only 401 -- 4% -- have both corners on that roster. Everyone else
falls to `base = {"name": x}`: no age, height, reach, stance, record,
recency stats or control time. Every style term that needs those gates
itself off, exactly as designed. So ~96% of the fights below are predicted
by Elo plus whatever point-in-time striking/takedown stats the cache
supplies, and nothing else.

That is why this reports the model at ~54.5% while the logged production
predictions run 74.4% over 86 fights. Those two numbers describe different
models, and neither invalidates the other.

WHAT THIS HARNESS CAN AND CANNOT SETTLE:
  CAN  -- whether Elo+stats beats the closing line (it does not, decisively)
  CAN  -- the DIRECTION of the disagreement problem, at real sample size
  CANNOT -- whether the production model beats the market, because the
            production model is not what ran here

Making it fair needs point-in-time reconstruction of age, record and
physicals, which walkforward_backtest.py's docstring already calls "a
future data project". The ESPN cache can supply record and dates; age needs
birthdates. Until then, treat every model-vs-market figure here as a floor
on the production model, not an estimate of it.

NO LOOK-AHEAD, same discipline as validate_pointintime_stats.py:
  - Elo is read BEFORE the fight updates it.
  - Striking/takedown stats are summed only from bouts strictly earlier.
  - Odds are the closing line for that specific bout, matched on the fighter
    pair AND the date, so a rematch cannot borrow its predecessor's price.

WHAT A POSITIVE RESULT WOULD ACTUALLY MEAN. A blend improves forecast
quality but SHRINKS every edge, since an edge is model-minus-market by
definition. Fewer, truer edges rather than more, phantom ones. Given
disagreements currently run 43%, that is probably the right trade -- but it
is a change to the product's premise, not a parameter, and this script only
measures it.

Usage:  python3 scripts/validate_market_blend.py
        python3 scripts/validate_market_blend.py --min-prior-fights 5
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem  # noqa: E402
from src.matchup_model import predict_matchup  # noqa: E402
from scripts.validate_pointintime_stats import (  # noqa: E402
    _fold, build_timelines, stats_as_of, CACHE_DIR, ID_MAP, FIGHTERS, HISTORY,
)

EXTERNAL_ODDS = "data/external_odds.csv"


def _american_to_implied(o):
    """
    NaN MUST BE REJECTED EXPLICITLY. 253 of the 7,177 rows have a missing
    price on one side, and float('nan') neither raises nor fails a
    truthiness or a <= 0 check -- so a first version let them through and
    every Brier downstream came out NaN while accuracy silently scored those
    fights as picking the second corner. The comparison looked complete and
    was not.
    """
    try:
        o = float(o)
    except (TypeError, ValueError):
        return None
    if math.isnan(o) or o == 0:
        return None
    return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)


def load_market():
    """
    {(pair_key, date_str): {name_lower: devigged_prob}}

    De-vigged by normalising the two implied probabilities to sum to 1. That
    removes the bookmaker's margin, which would otherwise make the market
    look systematically overconfident on BOTH fighters and hand the model an
    edge that is really just the vig.

    Keyed on pair AND date: several of these fighters met twice, and letting
    a rematch inherit the earlier bout's closing line would quietly score a
    2019 price against a 2023 result.
    """
    df = pd.read_csv(EXTERNAL_ODDS, low_memory=False)
    out = {}
    for r in df.itertuples(index=False):
        a, b = str(r.R_fighter).strip().lower(), str(r.B_fighter).strip().lower()
        ia, ib = _american_to_implied(r.R_odds), _american_to_implied(r.B_odds)
        if ia is None or ib is None:
            continue
        tot = ia + ib
        if not (tot > 0):          # also false for NaN, unlike `tot <= 0`
            continue
        date = str(r.date)[:10]
        out[(frozenset({a, b}), date)] = {a: ia / tot, b: ib / tot}
    return out


def _score(pairs):
    n = len(pairs)
    acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in pairs) / n
    return n, acc, brier, ll


def _paired_p(deltas, n_boot=20000, seed=99):
    """Sign-flip bootstrap on per-fight squared-error deltas. Fixed seed."""
    if not deltas:
        return 1.0
    obs = sum(deltas) / len(deltas)
    rnd = random.Random(seed)
    hits = 0
    for _ in range(n_boot):
        s = sum(d if rnd.random() < 0.5 else -d for d in deltas)
        if abs(s / len(deltas)) >= abs(obs):
            hits += 1
    return hits / n_boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prior-fights", type=int, default=3)
    args = ap.parse_args()

    if not os.path.isdir(CACHE_DIR):
        print(f"No {CACHE_DIR}. Run scripts/backfill_espn_fight_stats.py first.")
        sys.exit(1)

    market = load_market()
    print(f"market prices loaded: {len(market)} bouts")

    ids = {_fold(r["name"]): str(r["espn_id"]) for _, r in pd.read_csv(ID_MAP).iterrows()}
    print("building point-in-time stat timelines...")
    timelines = build_timelines(ids)
    print(f"  {len(timelines)} usable timelines\n")

    fighters = pd.read_csv(FIGHTERS)
    rows_by_name = {_fold(r["name"]): r for _, r in fighters.iterrows()}
    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")

    elo = EloRatingSystem()
    counts = defaultdict(int)
    recs = []          # (model_p_on_a, market_p_on_a, y)
    missing_price = 0

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()
        date_s = when.strftime("%Y-%m-%d")

        if (counts[fa] >= args.min_prior_fights and counts[fb] >= args.min_prior_fights
                and fa in timelines and fb in timelines):
            key = (frozenset({str(a).strip().lower(), str(b).strip().lower()}), date_s)
            prices = market.get(key)
            if prices is None:
                missing_price += 1
            else:
                mkt_a = prices.get(str(a).strip().lower())
                if mkt_a is not None:
                    base_a = rows_by_name[fa].to_dict() if fa in rows_by_name else {"name": a}
                    base_b = rows_by_name[fb].to_dict() if fb in rows_by_name else {"name": b}
                    eff = {a: elo.get_rating(a), b: elo.get_rating(b)}
                    sa = stats_as_of(timelines[fa], when)
                    sb = stats_as_of(timelines[fb], when)
                    ra, rb = dict(base_a), dict(base_b)
                    for col in ("strike_accuracy_pct", "td_accuracy_pct", "td_defense_pct"):
                        ra[col] = None
                        rb[col] = None
                    ra.update(sa)
                    rb.update(sb)
                    try:
                        res = predict_matchup(a, b, pd.DataFrame([ra, rb]), eff)
                    except Exception:
                        res = None
                    p = (res or {}).get("prob_a")
                    if p is not None:
                        recs.append((p, mkt_a, 1.0 if winner == a else 0.0))

        loser = b if winner == a else a
        elo.update_ratings(winner, loser, method=method)
        counts[fa] += 1
        counts[fb] += 1

    if not recs:
        print("No fights scored with both a model prediction and a market price.")
        sys.exit(1)

    print(f"scored {len(recs)} fights with BOTH a point-in-time model prediction "
          f"and a matched closing line")
    print(f"  (skipped {missing_price} scorable fights with no matched price)\n")

    model = [(m, y) for m, _, y in recs]
    mkt = [(k, y) for _, k, y in recs]
    print(f"{'source':<28}{'accuracy':>10}{'Brier':>10}{'log loss':>11}")
    print("-" * 59)
    for label, pairs in (("model alone", model), ("market alone (closing)", mkt)):
        n, acc, brier, ll = _score(pairs)
        print(f"{label:<28}{acc:>9.1%}{brier:>10.4f}{ll:>11.4f}")

    print(f"\nBLEND   p = w*model + (1-w)*market")
    base_model = _score(model)[2]
    best = None
    for i in range(11):
        w = i / 10.0
        bl = [(w * m + (1 - w) * k, y) for m, k, y in recs]
        n, acc, brier, ll = _score(bl)
        deltas = [(w * m + (1 - w) * k - y) ** 2 - (m - y) ** 2 for m, k, y in recs]
        p = _paired_p(deltas)
        tag = "  <- market alone" if w == 0 else ("  <- model alone" if w == 1 else "")
        sig = "" if w == 1 else f"   vs model: {brier-base_model:+.4f}  p={p:.4f}{'  SIGNIFICANT' if p < 0.05 else ''}"
        print(f"  w={w:>3.1f}  acc {acc:>5.1%}  Brier {brier:.4f}  ll {ll:.4f}{tag}{sig}")
        if best is None or brier < best[1]:
            best = (w, brier, p)

    print(f"\nbest blend: w={best[0]:.1f}   Brier {best[1]:.4f}   p={best[2]:.4f}")
    print(f"  model alone Brier {base_model:.4f}   market alone Brier {_score(mkt)[2]:.4f}")

    # THE SUBSET THE PRODUCT LIVES ON. Blending is uninteresting where model
    # and market already agree; the question is whether it rescues the fights
    # where they diverge, which is exactly where the logged record is 43%.
    dis = [(m, k, y) for m, k, y in recs if (m >= 0.5) != (k >= 0.5)]
    if dis:
        print(f"\nDISAGREEMENTS ONLY (model and market pick different fighters), n={len(dis)}")
        dm = [(m, y) for m, _, y in dis]
        dk = [(k, y) for _, k, y in dis]
        print(f"  model alone : acc {_score(dm)[1]:.1%}  Brier {_score(dm)[2]:.4f}")
        print(f"  market alone: acc {_score(dk)[1]:.1%}  Brier {_score(dk)[2]:.4f}")
        print(f"  => when they disagree, {'the MARKET is more often right' if _score(dk)[1] > _score(dm)[1] else 'the MODEL is more often right'}")


if __name__ == "__main__":
    main()
