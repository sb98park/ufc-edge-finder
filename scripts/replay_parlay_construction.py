"""
Replay the REAL parlay builders over real historical cards, and grade them.

WHAT THIS ANSWERS THAT NOTHING ELSE CAN. The site publishes ~9 slips a week
and, until data/parlay_ledger.jsonl, recorded none of them. Waiting for
forward data to say anything about the construction is hopeless at that rate:
detecting a 10-point ROI difference between two constructions needs on the
order of 3,000 slips per arm, which is about twenty years.

data/external_odds.csv carries 7,177 real UFC bouts across 616 event dates
with BOTH moneylines, the full six-cell method grid, and exact settlement
(winner, method, finish round, fight duration). Every leg the builder can
produce is either priced there or derivable from it, and every leg is exactly
settleable. That is ~500 usable cards -- roughly 500x the forward rate,
available today.

Critically, this drives the SHIPPED functions -- build_bankroll_builder_parlays,
build_lotto_parlays, build_moonshot_parlays -- rather than a reimplementation.
What is under test is the construction, and the construction is held fixed
while the probability source is swapped.

THE ARMS. A "model" here is the de-vigged market perturbed in logit space:

    logit(p_model) = logit(p_market) + N(0, sigma)

    sigma = 0     the model IS the market. A pure mechanics control.
    sigma = 0.83  measured on 104 real production picks -- the sd of
                  logit(model_prob) - logit(market_fair) in predictions_log.

THE sigma = 0 ARM IS THE POINT. If the construction is sound, a perfectly
calibrated model must produce slips whose realised hit rate matches their
published one. Any gap at sigma = 0 is the construction's own bias --
dependence between legs, band artifacts, selection over the payout grid.
Any ADDITIONAL gap at sigma > 0 is model error being selected for, which is a
property of the objective function rather than of the model.

That decomposition is the whole reason to build this: it separates "the
search is broken" from "the model is noisy", and those need different fixes.

WHY ROI IS REPORTED BUT NOT TRUSTED. Per-slip return sd is enormous (a lotto
slip returns -1 or +15). At n = 400 the ROI standard error is around 19
points, so two arms can differ by 40 points of ROI and mean nothing. The
calibration ratio -- published hit rate over realised -- has orders of
magnitude more power on the same sample. Read the ratio column; treat ROI as
decoration.

Usage:
    python3 scripts/replay_parlay_construction.py
    python3 scripts/replay_parlay_construction.py --sigmas 0 0.42 0.83 --events 200
"""

import argparse
import contextlib
import io
import math
import os
import random
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parlay_builder import (  # noqa: E402
    build_bankroll_builder_parlays, build_lotto_parlays, build_moonshot_parlays)
from src.odds_utils import decimal_to_american, implied_prob_to_american  # noqa: E402
from src.method_model import finish_share_before  # noqa: E402

ODDS = "data/external_odds.csv"

# finish -> the ESPN-style slug the leg conditions grade against.
_FINISH_SLUG = {
    "KO/TKO": "kotko", "SUB": "submission",
    "U-DEC": "decision---unanimous", "S-DEC": "decision---split",
    "M-DEC": "decision---majority",
}


def _logit(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def _devig_two_way(a_am, b_am):
    """De-vigged pair from two American prices."""
    def imp(am):
        return (-am / (-am + 100.0)) if am < 0 else (100.0 / (am + 100.0))
    ia, ib = imp(a_am), imp(b_am)
    tot = ia + ib
    if tot <= 0:
        return None
    return ia / tot, ib / tot


def build_card(rows, sigma, rnd):
    """
    Turn one event's bouts into the edge-row shape the builders consume.

    Prices are the REAL book prices. Only model_prob is synthetic, and it is
    the de-vigged market pushed around in logit space by `sigma`.
    """
    edges = []
    truth = {}
    for r in rows:
        fid = f"{r['R_fighter']}|{r['B_fighter']}"
        pair = _devig_two_way(r["R_odds"], r["B_odds"])
        if not pair:
            continue
        pr, pb = pair

        # Settlement for every condition kind this card can produce.
        slug = _FINISH_SLUG.get(str(r.get("finish")))
        if slug is None:
            continue                       # DQ / CNC / overturned -- ungradeable
        if str(r["Winner"]) not in ("Red", "Blue"):
            continue                       # draws void, out of scope for a replay
        secs = r.get("total_fight_time_secs")
        if secs is None or secs != secs:
            continue
        went_distance = slug.startswith("decision")
        truth[fid] = {
            "winner": r["R_fighter"] if r["Winner"] == "Red" else r["B_fighter"],
            "slug": slug,
            "went_distance": went_distance,
            "rounds_elapsed": float(secs) / 300.0,
        }

        def perturbed(p):
            return _sigmoid(_logit(p) + (rnd.gauss(0, sigma) if sigma else 0.0))

        for name, opp, p_fair, am in ((r["R_fighter"], r["B_fighter"], pr, r["R_odds"]),
                                      (r["B_fighter"], r["R_fighter"], pb, r["B_odds"])):
            edges.append({"fight_id": fid, "fight_key": fid, "fighter": name, "opponent": opp,
                          "market": "Moneyline", "odds_american": float(am),
                          "model_prob": perturbed(p_fair), "book_fair_prob": p_fair})

        # Method legs, from the real six-cell grid, normalised to sum to 1.
        cells = [("R", "KO/TKO", r.get("r_ko_odds")), ("R", "SUB", r.get("r_sub_odds")),
                 ("R", "DEC", r.get("r_dec_odds")), ("B", "KO/TKO", r.get("b_ko_odds")),
                 ("B", "SUB", r.get("b_sub_odds")), ("B", "DEC", r.get("b_dec_odds"))]
        def imp(am):
            return (-am / (-am + 100.0)) if am < 0 else (100.0 / (am + 100.0))
        vals = [(c, m, am, imp(am)) for c, m, am in cells if am == am and am is not None]
        tot = sum(v[3] for v in vals)
        if len(vals) == 6 and tot > 0:
            for corner, meth, am, raw in vals:
                p_fair = raw / tot
                who = r["R_fighter"] if corner == "R" else r["B_fighter"]
                edges.append({"fight_id": fid, "fight_key": fid, "fighter": who,
                              "opponent": r["B_fighter"] if corner == "R" else r["R_fighter"],
                              "market": f"Method: {meth}", "odds_american": float(am),
                              "model_prob": perturbed(p_fair), "book_fair_prob": p_fair})

        # Total Rounds, derived the way the site derives it: P(finish) times
        # the share of finishes landing before the line. Priced with a 5%
        # two-way margin, which is the overround odds_utils already assumes
        # for a two-way prop.
        if len(vals) == 6 and tot > 0:
            p_dec = sum(v[3] for v in vals if v[1] == "DEC") / tot
            sched = int(r.get("no_of_rounds") or 3)
            for line in ((1.5, 2.5) if sched == 3 else (3.5, 4.5)):
                p_under = (1 - p_dec) * finish_share_before(line, sched,
                                                            r.get("weight_class"))
                p_under = min(max(p_under, 0.02), 0.98)
                for side, p_fair in (("Under", p_under), ("Over", 1 - p_under)):
                    priced = min(max(p_fair * 1.05, 0.02), 0.98)
                    edges.append({
                        "fight_id": fid, "fight_key": fid,
                        "fighter": f"{r['R_fighter']} vs {r['B_fighter']}",
                        "market": f"Total Rounds {side} {line}",
                        "odds_american": float(implied_prob_to_american(priced)),
                        "model_prob": perturbed(p_fair), "book_fair_prob": p_fair})
    return edges, truth


def grade(slip, truth):
    """True if every leg's every condition holds. None if anything is unknown."""
    for leg in slip["legs"]:
        conds = leg.get("conditions") or []
        if not conds:
            return None
        t = truth.get(leg.get("fight_key"))
        if not t:
            return None
        for c in conds:
            k = c["kind"]
            if k == "winner":
                if str(c.get("fighter")).strip().lower() != t["winner"].strip().lower():
                    return False
            elif k == "method":
                if not any(t["slug"].startswith(s) for s in c.get("any_of", [])):
                    return False
            elif k == "rounds":
                if t["went_distance"]:
                    ok = (c["op"] == "over")
                else:
                    ok = (t["rounds_elapsed"] < c["line"]) if c["op"] == "under" \
                        else (t["rounds_elapsed"] > c["line"])
                if not ok:
                    return False
            elif k == "distance":
                if bool(t["went_distance"]) != bool(c["value"]):
                    return False
            else:
                return None
    return True


TIERS = (("bankroll", build_bankroll_builder_parlays),
         ("lotto", build_lotto_parlays),
         ("moonshot", build_moonshot_parlays))


def run(sigma, events, tiers, seed=7):
    df = pd.read_csv(ODDS)
    df = df.dropna(subset=["R_odds", "B_odds", "Winner", "finish", "total_fight_time_secs"])
    by_date = defaultdict(list)
    for r in df.to_dict("records"):
        by_date[r["date"]].append(r)
    dates = [d for d, rs in by_date.items() if len(rs) >= 6]
    dates.sort()
    dates = dates[-events:] if events else dates

    rnd = random.Random(seed)
    stats = defaultdict(lambda: {"n": 0, "pub": 0.0, "hit": 0, "staked": 0.0, "ret": 0.0})
    for d in dates:
        edges, truth = build_card(by_date[d], sigma, rnd)
        if len(truth) < 4:
            continue
        for tier, fn in TIERS:
            if tier not in tiers:
                continue
            try:
                # The builders print a diagnostic when a card yields no
                # qualifying slip. That is useful during a render and pure
                # noise across hundreds of replayed cards, where an empty
                # lotto slate is the ordinary case.
                with contextlib.redirect_stdout(io.StringIO()):
                    slips = fn(edges, None)
            except Exception:
                continue
            for s in slips:
                g = grade(s, truth)
                if g is None:
                    continue
                st = stats[tier]
                st["n"] += 1
                st["pub"] += s["combined_prob"]
                st["staked"] += 1.0
                if g:
                    st["hit"] += 1
                    st["ret"] += s["combined_decimal"]
    return stats, len(dates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.42, 0.83])
    ap.add_argument("--events", type=int, default=180)
    # MOONSHOT IS OFF BY DEFAULT, and the reason is a finding rather than a
    # convenience. At a full card the piece pool hits MAX_POOL_SIZE = 30, and
    # 2-to-8-leg combinations over 30 pieces is 8,656,906 slips PER CARD. A
    # 60-card replay is half a billion combinations, which does not finish.
    # The tier cannot be validated at any useful scale -- not because the data
    # is missing, but because its own search space forbids it. A product whose
    # quality is unmeasurable in principle is hard to defend on any other
    # ground, and this is the same 8.7M that costs ~62s of every 300s rebuild.
    ap.add_argument("--tiers", nargs="+", default=["bankroll", "lotto"],
                    choices=["bankroll", "lotto", "moonshot"])
    a = ap.parse_args()

    print(f"{'sigma':>6} {'tier':<10} {'slips':>6} {'published':>10} {'realised':>9} "
          f"{'ratio':>7} {'ROI':>8}   (ROI is noise -- read the ratio)")
    for sigma in a.sigmas:
        stats, n_dates = run(sigma, a.events, set(a.tiers))
        for tier in a.tiers:
            st = stats.get(tier)
            if not st or not st["n"]:
                continue
            pub = st["pub"] / st["n"]
            real = st["hit"] / st["n"]
            ratio = (pub / real) if real > 0 else float("inf")
            roi = (st["ret"] - st["staked"]) / st["staked"] if st["staked"] else 0.0
            print(f"{sigma:>6.2f} {tier:<10} {st['n']:>6} {pub:>10.1%} {real:>9.1%} "
                  f"{ratio:>7.2f} {roi:>+8.1%}")
        print()
    print(f"({a.events} most recent event dates with 6+ gradeable bouts)")
    print("A ratio near 1.00 means the published hit rate is honest.\n"
          "The sigma = 0 row is the construction's own bias, with no model error in play.")


if __name__ == "__main__":
    main()
