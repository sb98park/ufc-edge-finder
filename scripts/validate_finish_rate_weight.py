"""
POINT-IN-TIME validation of weighting the finish-rate term by wins.

THE CHANGE UNDER TEST. compute_stats_rating builds a fighter's record-based
rating from three terms:

    rating  = 1500
            + 500 * (win_pct - 0.5) * min(1, total_fights / 15)     <- weighted
            + 150 * (finish_rate - 0.4)                             <- NOT weighted
            + 4   * (reach_in - 70)

finish_rate is (ko_wins + sub_wins) / max(wins, 1), so it is estimated from
WINS, and a fighter with none has a finish rate of 0 by construction. The term
carries no experience weighting, so that fighter takes a flat -60 no matter how
little is known about them.

WHY IT WAS PROPOSED. power_rating already refuses to read an empty record as a
bad one -- wins + losses == 0 returns the neutral prior, added after a debutant
with no findable record was scored near a 0-5 fighter and made Lock of the Week.
But 0-1 misses that guard:

    0-1 recorded    win_pct -16.7   finish_rate -60.0   = 1423
    0-0 recorded    neutral prior                       = 1500

An almost-unknown fighter is rated 77 points BELOW a completely unknown one,
which is backwards. Found on 2026-08-30 via Michael Aljarouj, carried in the
roster as 0-1 when he is really 13-3: the bad record inflated his opponent's
published probability by 23.6 points, and only the thin-record label cap stopped
it shipping as Lock of the Week.

THE ARMS. finish_weight = min(1, wins / K). The control is the shipped
behaviour, weight fixed at 1.0. Every arm sees identical inputs and differs ONLY
in K, so arms are paired on the same fights and the delta is the change alone.

SCORED POPULATION. Fights where at least one corner has FEW WINS on the night --
that is where an unweighted finish rate is doing the most damage and where the
arms actually differ. --control inverts the filter to score the well-evidenced
fights, where every arm is near-identical, as a check that the change is not
quietly moving numbers it has no business moving.

Run:  python3 scripts/validate_finish_rate_weight.py            (default sweep)
      python3 scripts/validate_finish_rate_weight.py --control  (the other cut)
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem                                    # noqa: E402
from src.matchup_model import predict_matchup                          # noqa: E402
from src.power_rating import RATING_CENTER, DEBUT_RATING_SHRINK, _streak_bonus  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of         # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"

# "Few wins" for the scored cut. Chosen as the point below which the finish
# rate rests on a handful of fights; reported so the cut is visible rather than
# implied.
FEW_WINS = 4


def _fold(n) -> str:
    return str(n).strip().lower()


def _num(row, key, default=0.0):
    v = row.get(key)
    try:
        if v is None or (isinstance(v, float) and v != v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _stats_rating(row: dict, k: float | None) -> float:
    """
    compute_stats_rating with the finish term weighted by wins / k.

    Mirrors production line-for-line rather than calling it, so an arm can vary
    k without monkeypatching a module every other harness also imports. k=None
    is the shipped behaviour: the finish term at full strength.
    """
    wins = _num(row, "wins")
    losses = _num(row, "losses")
    ko = _num(row, "ko_wins")
    sub = _num(row, "sub_wins")
    reach = _num(row, "reach_in", 70.0) or 70.0

    if wins + losses == 0:
        return RATING_CENTER + 4.0 * (reach - 70.0)

    total = wins + losses
    win_pct = wins / total
    finish_rate = (ko + sub) / max(wins, 1.0)
    experience = min(1.0, total / 15.0)
    # k == 0 means drop the term entirely, which is the limit of turning it
    # down. Guarded before the division rather than after.
    if k is None:
        finish_w = 1.0
    elif k == 0:
        finish_w = 0.0
    else:
        finish_w = min(1.0, wins / k)

    rating = RATING_CENTER
    rating += 500.0 * (win_pct - 0.5) * experience
    rating += 150.0 * (finish_rate - 0.4) * finish_w
    rating += 4.0 * (reach - 70.0)
    return rating


def _effective(row: dict, n_prior: int, elo_r: float, streak: int, k) -> float:
    sr = _stats_rating(row, k)
    if n_prior == 0:
        eff = RATING_CENTER + (sr - RATING_CENTER) * DEBUT_RATING_SHRINK
    else:
        w = min(1.0, n_prior / 4.0)
        eff = w * elo_r + (1 - w) * sr
    return eff + _streak_bonus(n_prior, streak)


def _score(pairs):
    n = len(pairs)
    acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in pairs) / n
    return n, acc, brier, ll


def _paired_test(rows, arm, base, n_boot=4000, seed=12345):
    """Paired sign-flip bootstrap on per-fight change in squared error."""
    deltas = [(pr[arm] - y) ** 2 - (pr[base] - y) ** 2 for _, y, pr in rows]
    if not deltas:
        return 0.0, 1.0
    obs = sum(deltas) / len(deltas)
    rnd = random.Random(seed)
    hits = 0
    for _ in range(n_boot):
        s = sum(d if rnd.random() < 0.5 else -d for d in deltas)
        if abs(s / len(deltas)) >= abs(obs):
            hits += 1
    return obs, hits / n_boot


def run(arms, control=False):
    fighters = pd.read_csv(FIGHTERS)
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}

    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    fight_index = build_fight_index(history)
    wc = pd.read_csv(WC_HISTORY) if os.path.exists(WC_HISTORY) else None

    elo = EloRatingSystem()
    counts = defaultdict(int)
    wins_so_far = defaultdict(int)
    streaks = defaultdict(int)
    records, skipped = [], 0

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()
        na, nb = counts[fa], counts[fb]
        wa, wb = wins_so_far[fa], wins_so_far[fb]

        thin = (wa < FEW_WINS) or (wb < FEW_WINS)
        want = (not thin) if control else thin
        if want and winner in (a, b):
            past = history[history["date"] < f["date"]]
            ra = roster_as_of(a, when, fight_index, static_rows, today=when)
            rb = roster_as_of(b, when, fight_index, static_rows, today=when)
            frame = pd.DataFrame([ra, rb])
            y = 1.0 if winner == a else 0.0

            probs = {}
            for label, k in arms.items():
                eff = {
                    a: _effective(ra, na, elo.get_rating(a), streaks[fa], k),
                    b: _effective(rb, nb, elo.get_rating(b), streaks[fb], k),
                }
                try:
                    res = predict_matchup(a, b, frame, eff, past, wc,
                                          f.get("weight_class"),
                                          reference_date=when.date())
                except Exception:
                    res = None
                p = (res or {}).get("prob_a")
                if p is not None and not math.isnan(p):
                    probs[label] = p
            if len(probs) == len(arms):
                records.append((min(wa, wb), y, probs))
            else:
                skipped += 1

        loser = b if winner == a else a
        if winner in (a, b):
            elo.update_ratings(winner, loser, method=method)
            streaks[_fold(winner)] += 1
            streaks[_fold(loser)] = 0
            wins_so_far[_fold(winner)] += 1
        counts[fa] += 1
        counts[fb] += 1

    return records, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="score the WELL-EVIDENCED fights instead, where the arms "
                         "should be near-identical")
    args = ap.parse_args()

    BASE = "control (unweighted, shipped)"
    # k=15 matches the denominator win_pct already uses, so it is the natural
    # stopping point -- but Brier improved monotonically up to it on the first
    # sweep, and a parameter that keeps getting better as you turn it down is
    # usually telling you the term itself is the problem. k=30 and the
    # finish-term-removed arm are here to find out which.
    arms = {BASE: None, "k=8": 8.0, "k=15": 15.0, "k=30": 30.0,
            "finish term REMOVED": 0.0}

    cut = "well-evidenced (both corners >= %d wins)" % FEW_WINS if args.control \
        else "thin (a corner with < %d wins on the night)" % FEW_WINS
    print(f"Scored population: {cut}\n")

    rows, skipped = run(arms, control=args.control)
    if not rows:
        print("no scorable fights")
        return 1
    print(f"  {len(rows)} fights scored, {skipped} skipped (an arm could not predict)\n")
    print(f"  {'arm':<32}{'acc':>8}{'brier':>10}{'logloss':>10}{'d.brier':>11}{'p':>8}")
    print("  " + "-" * 79)
    base_pairs = [(pr[BASE], y) for _, y, pr in rows]
    _, bacc, bbri, bll = _score(base_pairs)
    for label in arms:
        pairs = [(pr[label], y) for _, y, pr in rows]
        n, acc, bri, ll = _score(pairs)
        if label == BASE:
            print(f"  {label:<32}{acc:>8.4f}{bri:>10.5f}{ll:>10.5f}{'--':>11}{'--':>8}")
        else:
            d, p = _paired_test(rows, label, BASE)
            star = "  *" if p < 0.05 else ""
            print(f"  {label:<32}{acc:>8.4f}{bri:>10.5f}{ll:>10.5f}{d:>+11.5f}{p:>8.3f}{star}")
    print("\n  d.brier is arm minus control; NEGATIVE is better. p from a paired")
    print("  sign-flip bootstrap over per-fight squared error, 4000 resamples.")

    # SECOND TABLE, AGAINST THE BEST ARM. Every arm beating a common control
    # does NOT establish that they differ from each other, and the choice of k
    # is exactly that question. Re-based on removal, which won both proper
    # scoring rules on the first pass.
    REMOVED = "finish term REMOVED"
    if REMOVED in arms:
        print(f"\n  Pairwise, re-based on {REMOVED!r}:")
        print(f"  {'arm':<32}{'d.brier':>11}{'p':>8}")
        print("  " + "-" * 51)
        for label in arms:
            if label == REMOVED:
                continue
            d, p = _paired_test(rows, label, REMOVED)
            star = "  *" if p < 0.05 else "   (indistinguishable)"
            print(f"  {label:<32}{d:>+11.5f}{p:>8.3f}{star}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
