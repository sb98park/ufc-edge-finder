"""
POINT-IN-TIME validation of suppressing the layoff penalty on a history we
know is incomplete.

THE CHANGE UNDER TEST. layoff_penalty charges -20 rating points per year past
a one-year grace, capped at -300, computed from last_fight_date. That date is
the newest bout WE HOLD. When our history of a fighter is partial, the date is
a lower bound on their real activity -- the true last fight can only be more
recent, never older.

The penalty is one-directional. So a partial history can only ever MANUFACTURE
ring rust; it can never remove any. This is not a tuning question, it is an
asymmetry.

WHAT PROMPTED IT. Michael Aljarouj, priced for 2026-09-05. fighters.csv has
him 13-3; fight_history.csv has one of those sixteen bouts, from 2021. The
model read a 5.47-year layoff and charged -89.3 points. Tapology shows him
fighting in Nov 2024 and again on 2025-04-12 -- a real layoff of 1.40 years,
worth -8.0. Eighty-one points of the gap in the card's second-largest
model-vs-market disagreement were an artefact of our own coverage.

MEASURING COVERAGE POINT-IN-TIME. record_as_of returns record_source
"subtracted" when a fighter's career totals in fighters.csv can be reconciled
against their known subsequent bouts -- that is a claim about how many times
they had fought by that date, sourced independently of the history table.
Coverage is (bouts we hold before this date) / (that claim). Only corners with
a subtracted record can be scored; an accumulated record is counted FROM the
history table, so it would read 100% by construction and say nothing.

THE ARMS. Identical inputs; the guarded arms blank last_fight_date on a corner
whose coverage is below the threshold, which is what layoff_years already
treats as "we do not know" (it returns None, and both penalties fall to zero).

SCORED POPULATION. Fights where at least one corner is below the floor. The
--control cut inverts it: fights where both corners are fully covered, where
every arm is identical by construction and any movement would mean the harness
is wrong.

Run:  python3 scripts/validate_layoff_coverage_guard.py
      python3 scripts/validate_layoff_coverage_guard.py --control
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
from src.power_rating import (RATING_CENTER, DEBUT_RATING_SHRINK,      # noqa: E402
                              _streak_bonus, compute_stats_rating)
from scripts.pit_roster import build_fight_index, roster_as_of         # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"

# A record this short cannot support a coverage ratio worth reading.
MIN_CLAIMED = 3
# The cut used to select the scored population. Reported, not implied.
SCORE_FLOOR = 0.60


def _fold(n) -> str:
    return str(n).strip().lower()


def _effective(row: dict, n_prior: int, elo_r: float, streak: int) -> float:
    sr = compute_stats_rating(pd.Series(row))
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
    held = defaultdict(int)          # bouts of theirs we hold BEFORE this fight
    streaks = defaultdict(int)
    records, skipped, no_cov = [], 0, 0

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()

        if winner in (a, b):
            ra = roster_as_of(a, when, fight_index, static_rows, today=when)
            rb = roster_as_of(b, when, fight_index, static_rows, today=when)

            cov = {}
            for name, fold, r in ((a, fa, ra), (b, fb, rb)):
                claimed = (r.get("wins") or 0) + (r.get("losses") or 0)
                if r.get("record_source") != "subtracted" or claimed < MIN_CLAIMED:
                    cov[name] = None          # not measurable, not scored
                else:
                    cov[name] = held[fold] / claimed

            measurable = [c for c in cov.values() if c is not None]
            if not measurable:
                no_cov += 1
            else:
                thin = any(c < SCORE_FLOOR for c in measurable)
                want = (not thin) if control else thin
                if want:
                    past = history[history["date"] < f["date"]]
                    y = 1.0 if winner == a else 0.0
                    na, nb = counts[fa], counts[fb]
                    eff = {
                        a: _effective(ra, na, elo.get_rating(a), streaks[fa]),
                        b: _effective(rb, nb, elo.get_rating(b), streaks[fb]),
                    }
                    probs = {}
                    for label, floor in arms.items():
                        # BLANKING last_fight_date IS the guard. layoff_years
                        # already reads a missing date as "we do not know" and
                        # returns None, which zeroes both layoff_penalty and
                        # quick_return_penalty -- so the arm needs no
                        # monkeypatch of a module every other harness imports.
                        rows2 = []
                        for name, r in ((a, ra), (b, rb)):
                            rr = dict(r)
                            c = cov[name]
                            if floor is not None and c is not None and c < floor:
                                rr["last_fight_date"] = None
                            rows2.append(rr)
                        frame = pd.DataFrame(rows2)
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
                        records.append((min(measurable), y, probs))
                    else:
                        skipped += 1

            loser = b if winner == a else a
            elo.update_ratings(winner, loser, method=method)
            streaks[_fold(winner)] += 1
            streaks[_fold(loser)] = 0
        counts[fa] += 1
        counts[fb] += 1
        held[fa] += 1
        held[fb] += 1

    return records, skipped, no_cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="score the FULLY COVERED fights instead, where every "
                         "arm is identical by construction")
    args = ap.parse_args()

    BASE = "control (penalty always fires, shipped)"
    arms = {BASE: None, "guard < 0.40": 0.40, "guard < 0.60": 0.60,
            "guard < 0.80": 0.80, "guard < 1.00 (any gap at all)": 1.00}

    cut = ("fully covered (every measurable corner at 100%)" if args.control
           else f"thin (a measurable corner below {SCORE_FLOOR:.0%} coverage)")
    print(f"Scored population: {cut}\n")

    rows, skipped, no_cov = run(arms, control=args.control)
    if not rows:
        print("no scorable fights")
        return 1
    print(f"  {len(rows)} fights scored, {skipped} skipped (an arm could not "
          f"predict), {no_cov} with no measurable coverage on either corner\n")
    print(f"  {'arm':<34}{'acc':>8}{'brier':>10}{'logloss':>10}{'d.brier':>11}{'p':>8}")
    print("  " + "-" * 81)
    for label in arms:
        pairs = [(pr[label], y) for _, y, pr in rows]
        n, acc, bri, ll = _score(pairs)
        if label == BASE:
            print(f"  {label:<34}{acc:>8.4f}{bri:>10.5f}{ll:>10.5f}{'--':>11}{'--':>8}")
        else:
            d, p = _paired_test(rows, label, BASE)
            print(f"  {label:<34}{acc:>8.4f}{bri:>10.5f}{ll:>10.5f}{d:>+11.5f}{p:>8.3f}")
    print("\n  d.brier is the arm MINUS the control, so negative is better.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
