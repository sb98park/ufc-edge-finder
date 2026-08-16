"""
Should a contrarian pick have to clear a bar?

THE FINDING THIS COMES FROM. On the 82 logged production picks, flat
1-unit stakes at the price actually available:

    the model                 +14.03u   17.1% ROI
    always back the favourite +12.44u   15.2% ROI
      model's favourite picks +13.51u   22.5%  (n=60)
      model's underdog picks   +0.52u    2.4%  (n=22)

So the model is profitable, but essentially all of it comes from WHICH
favourites it backs. The underdog picks roughly break even and cost 7.3
points of accuracy. The obvious question is whether they are worth making
at all, or only worth making when the model disagrees with the market by a
lot rather than a little.

WHAT IS SWEPT. A contrarian pick is only taken when

    model_prob - market_prob >= threshold

on the fighter the model likes. Below the bar, two policies:

    DEFER -- back the market favourite instead (what a site that must
             publish a pick for every fight would do)
    SKIP  -- make no bet (what a bettor would do)

ROI USES THE RAW AMERICAN PRICE, not the de-vigged probability. De-vigging
is right for measuring forecast quality and wrong for measuring money: you
are paid the posted number, vig included. Getting this backwards would
inflate every ROI in the table by roughly the overround.

THE USUAL CAVEAT, which has bitten twice already tonight. This harness
scores a model weaker than production -- ESPN per-fight stats reach only
901 of 4,376 fighters, and the recency-weighted columns cannot be
reconstructed at all. Treat the LEVEL of every ROI here as a floor and the
SHAPE across thresholds as the finding.

Usage:  python3 scripts/validate_underdog_threshold.py
"""

import argparse
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
from scripts.pit_roster import build_fight_index, roster_as_of  # noqa: E402
from scripts.validate_market_blend import _american_to_implied, EXTERNAL_ODDS  # noqa: E402


def load_market_with_prices():
    """{(pair, date): {name: (devigged_prob, american_odds)}}"""
    df = pd.read_csv(EXTERNAL_ODDS, low_memory=False)
    out = {}
    for r in df.itertuples(index=False):
        a, b = str(r.R_fighter).strip().lower(), str(r.B_fighter).strip().lower()
        ia, ib = _american_to_implied(r.R_odds), _american_to_implied(r.B_odds)
        if ia is None or ib is None:
            continue
        tot = ia + ib
        if not (tot > 0):
            continue
        out[(frozenset({a, b}), str(r.date)[:10])] = {
            a: (ia / tot, float(r.R_odds)),
            b: (ib / tot, float(r.B_odds)),
        }
    return out


def payout(american, won) -> float:
    """Profit on a 1-unit flat stake."""
    if not won:
        return -1.0
    o = float(american)
    return o / 100.0 if o > 0 else 100.0 / (-o)


def build_records(min_prior_fights=3):
    market = load_market_with_prices()
    ids = {_fold(r["name"]): str(r["espn_id"]) for _, r in pd.read_csv(ID_MAP).iterrows()}
    print(f"market prices: {len(market)} bouts")
    print("building point-in-time stat timelines...")
    timelines = build_timelines(ids)
    fighters = pd.read_csv(FIGHTERS)
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}
    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    fight_index = build_fight_index(history)
    print(f"  {len(timelines)} stat timelines, {len(fight_index)} roster timelines\n")

    elo = EloRatingSystem()
    counts = defaultdict(int)
    recs = []
    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()
        if (counts[fa] >= min_prior_fights and counts[fb] >= min_prior_fights
                and fa in timelines and fb in timelines):
            prices = market.get((frozenset({str(a).strip().lower(), str(b).strip().lower()}),
                                 when.strftime("%Y-%m-%d")))
            if prices:
                pa = prices.get(str(a).strip().lower())
                pb = prices.get(str(b).strip().lower())
                if pa and pb:
                    ra = roster_as_of(a, when, fight_index, static_rows, timelines)
                    rb = roster_as_of(b, when, fight_index, static_rows, timelines)
                    ra.update(stats_as_of(timelines[fa], when))
                    rb.update(stats_as_of(timelines[fb], when))
                    eff = {a: elo.get_rating(a), b: elo.get_rating(b)}
                    try:
                        res = predict_matchup(a, b, pd.DataFrame([ra, rb]), eff)
                    except Exception:
                        res = None
                    p = (res or {}).get("prob_a")
                    if p is not None:
                        recs.append({
                            "model_a": p, "mkt_a": pa[0],
                            "odds_a": pa[1], "odds_b": pb[1],
                            "a_won": 1.0 if winner == a else 0.0,
                        })
        loser = b if winner == a else a
        elo.update_ratings(winner, loser, method=method)
        counts[fa] += 1
        counts[fb] += 1
    return recs


def evaluate(recs, threshold, policy):
    """Returns (bets, hits, units) under one contrarian bar."""
    bets = hits = 0
    units = 0.0
    for r in recs:
        m_a, k_a = r["model_a"], r["mkt_a"]
        model_likes_a = m_a >= 0.5
        market_likes_a = k_a >= 0.5
        # Model's own probability on its pick, and the market's on the same
        # fighter -- the gap is the claim being made.
        m_pick = m_a if model_likes_a else 1 - m_a
        k_pick = k_a if model_likes_a else 1 - k_a
        contrarian = model_likes_a != market_likes_a

        if contrarian and (m_pick - k_pick) < threshold:
            if policy == "skip":
                continue
            pick_a = market_likes_a          # defer to the market
        else:
            pick_a = model_likes_a

        won = (r["a_won"] == 1.0) if pick_a else (r["a_won"] == 0.0)
        odds = r["odds_a"] if pick_a else r["odds_b"]
        units += payout(odds, won)
        bets += 1
        hits += 1 if won else 0
    return bets, hits, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prior-fights", type=int, default=3)
    args = ap.parse_args()
    if not os.path.isdir(CACHE_DIR):
        print(f"No {CACHE_DIR}. Run scripts/backfill_espn_fight_stats.py first.")
        sys.exit(1)

    recs = build_records(args.min_prior_fights)
    if not recs:
        print("no scorable fights")
        sys.exit(1)
    n_contra = sum(1 for r in recs
                   if (r["model_a"] >= 0.5) != (r["mkt_a"] >= 0.5))
    print(f"scored {len(recs)} fights; the model disagrees with the market on "
          f"{n_contra} ({n_contra/len(recs):.0%})\n")

    # Reference points that do not depend on the threshold.
    always_fav = sum(payout(r["odds_a"] if r["mkt_a"] >= 0.5 else r["odds_b"],
                            (r["a_won"] == 1.0) if r["mkt_a"] >= 0.5 else (r["a_won"] == 0.0))
                     for r in recs)
    print(f"{'baseline':<34}{'bets':>7}{'hits':>7}{'units':>10}{'ROI':>9}")
    print("-" * 67)
    fav_hits = sum(1 for r in recs
                   if ((r["a_won"] == 1.0) if r["mkt_a"] >= 0.5 else (r["a_won"] == 0.0)))
    print(f"{'always back the favourite':<34}{len(recs):>7}{fav_hits:>7}{always_fav:>10.1f}{always_fav/len(recs):>8.1%}")
    b, h, u = evaluate(recs, -1.0, "defer")     # no bar at all == today's behaviour
    print(f"{'model, no bar (current)':<34}{b:>7}{h:>7}{u:>10.1f}{u/b:>8.1%}")

    for policy in ("defer", "skip"):
        print(f"\nCONTRARIAN BAR -- policy: {policy.upper()}"
              f"{'  (below the bar, back the favourite)' if policy=='defer' else '  (below the bar, no bet)'}")
        print(f"  {'threshold':<14}{'bets':>7}{'hits':>7}{'acc':>8}{'units':>10}{'ROI':>9}")
        print("  " + "-" * 55)
        for t in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 1.01):
            b, h, u = evaluate(recs, t, policy)
            if b == 0:
                continue
            lbl = f"{t:.2f}" + ("  (never)" if t > 1 else "")
            print(f"  {lbl:<14}{b:>7}{h:>7}{h/b:>7.1%}{u:>10.1f}{u/b:>8.1%}")


if __name__ == "__main__":
    main()
