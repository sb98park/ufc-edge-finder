"""
POINT-IN-TIME validation of a FIVE-ROUND interaction.

THE GAP. predict_matchup has no scheduled_rounds parameter. A main event and a
prelim are scored by the identical function, while the site's own copy asserts
that championship rounds matter. Two mechanisms are usually claimed, and they
point in the same direction:

    - more time for the better fighter to express an edge, so P(favourite)
      should be FURTHER from 0.5 than a three-round model says
    - cardio and durability weigh more heavily in rounds four and five

Both predict that a three-round-calibrated probability is UNDERCONFIDENT on a
five-round bout.

HOW THIS IS TESTED WITHOUT TOUCHING PRODUCTION. The interaction is applied
post-hoc as a temperature on the log-odds of five-round fights only:

    p' = sigmoid(logit(p) * s)

s = 1.0 is the shipped model exactly. s > 1 sharpens (more confident), s < 1
flattens. Applying this after predict_matchup is arithmetically identical to
applying it as a final transform inside it, so nothing needs threading through
the model until there is a verdict worth threading.

IDENTIFYING FIVE-ROUND BOUTS. fight_history.csv carries no round format. It is
recovered by joining ufc_fight_results.csv (TIME FORMAT) to pit_stats.csv for
a date, then matching on (date, unordered fighter pair). That resolves 75.2%
of history and 755 five-round bouts; the rest are scored as three-round, which
is the correct default and is what the site already assumes.

Note the sample this can ever have: five-round fights are ~7% of the record
and are concentrated in main events between ranked fighters. A null result
here is a statement about achievable power as much as about the effect.

Usage:  python3 scripts/validate_five_round.py
        python3 scripts/validate_five_round.py --offset 2500
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem  # noqa: E402
from src.matchup_model import predict_matchup  # noqa: E402
from src.power_rating import compute_stats_rating, _streak_bonus  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of  # noqa: E402
from scripts.build_pit_stats import load_pit_stats, stats_as_of  # noqa: E402
from scripts.harness_stats import (  # noqa: E402
    paired_signflip, randomize_corner, score as _score, trivial_baseline)

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"
RESULTS = "data/ufc_fight_results.csv"
PIT = "data/pit_stats.csv"


def _fold(n) -> str:
    return str(n).strip().lower()


def round_format_lookup() -> dict:
    """(date, frozen fighter pair) -> 3 or 5."""
    r = pd.read_csv(RESULTS)
    p = pd.read_csv(PIT)
    for df, cols in ((r, ("EVENT", "BOUT")), (p, ("event", "bout"))):
        for c in cols:
            df[c] = df[c].astype(str).str.strip()
    key = p.drop_duplicates(subset=["event", "bout"])[["event", "bout", "date"]]
    m = r.merge(key, left_on=["EVENT", "BOUT"], right_on=["event", "bout"], how="left")
    m = m.dropna(subset=["date"])

    out = {}
    for bout, fmt, date in zip(m["BOUT"], m["TIME FORMAT"], m["date"]):
        s = str(fmt)
        rounds = 5 if s.startswith("5 Rnd") else (3 if s.startswith("3 Rnd") else None)
        if rounds is None:
            continue
        parts = re.split(r"\s+vs\.?\s+", str(bout))
        if len(parts) != 2:
            continue
        out[(str(date)[:10], frozenset(x.strip().lower() for x in parts))] = rounds
    return out


def _sharpen(p: float, s: float) -> float:
    if s == 1.0:
        return p
    q = min(max(p, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + math.exp(-(math.log(q / (1 - q)) * s)))


def run(arms, limit, offset=0):
    rounds_by = round_format_lookup()
    fighters = pd.read_csv(FIGHTERS)
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}
    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    fight_index = build_fight_index(history)
    wc = pd.read_csv(WC_HISTORY) if os.path.exists(WC_HISTORY) else None
    pit = load_pit_stats()

    elo = EloRatingSystem()
    counts, streaks = defaultdict(int), defaultdict(int)
    records = []
    trimmed = history.iloc[:-offset] if offset else history
    rows = trimmed.tail(limit) if limit else trimmed
    cutoff = rows.iloc[0]["date"] if len(rows) else None
    ceiling = rows.iloc[-1]["date"] if len(rows) else None

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()
        na, nb = counts[fa], counts[fb]

        in_window = ((cutoff is None or f["date"] >= cutoff)
                     and (ceiling is None or f["date"] <= ceiling))
        if in_window and na > 0 and nb > 0 and winner in (a, b):
            ra = roster_as_of(a, when, fight_index, static_rows, today=when)
            rb = roster_as_of(b, when, fight_index, static_rows, today=when)
            past = history[history["date"] < f["date"]]
            for row, fold in ((ra, fa), (rb, fb)):
                row.update(stats_as_of(pit.get(fold, []), when.date()))
            frame = pd.DataFrame([ra, rb])
            eff = {}
            for name, row, prior, fold in ((a, ra, na, fa), (b, rb, nb, fb)):
                sr = compute_stats_rating(pd.Series(row))
                w = min(1.0, prior / 4.0)
                eff[name] = w * elo.get_rating(name) + (1 - w) * sr + _streak_bonus(prior, streaks[fold])

            try:
                res = predict_matchup(a, b, frame, eff, past, wc,
                                      f.get("weight_class"), reference_date=when.date())
            except Exception:
                res = None
            p = (res or {}).get("prob_a")
            if p is not None and not math.isnan(p):
                y = 1.0 if winner == a else 0.0
                p, y = randomize_corner(p, y, a, b, when)
                sched = rounds_by.get((when.strftime("%Y-%m-%d"), frozenset((fa, fb))), 3)
                # The sharpen applies to five-round bouts ONLY; three-round
                # fights are byte-identical across every arm, which is what
                # makes the paired test measure the interaction rather than a
                # global recalibration.
                probs = {s: (_sharpen(p, s) if sched == 5 else p) for s in arms}
                records.append((sched, y, probs, when.date()))

        loser = b if winner == a else a
        if winner in (a, b):
            elo.update_ratings(winner, loser, method=method)
            streaks[_fold(winner)] += 1
            streaks[_fold(loser)] = 0
        counts[fa] += 1
        counts[fb] += 1

    return records


def report(records, arms, base=1.0):
    n = len(records)
    if not n:
        print("no scored fights")
        return
    five = [r for r in records if r[0] == 5]
    print(f"n = {n} scored fights   trivial baseline "
          f"{trivial_baseline([(0.5, r[1]) for r in records]):.1%}")
    print(f"five-round bouts: {len(five)} ({len(five) / n:.1%})\n")

    # IS THERE ANYTHING TO CORRECT? Mean predicted vs observed on each subset,
    # at the shipped model. If five-round favourites win MORE often than
    # predicted, the underconfidence story holds and sharpening should help.
    for lbl, sub in (("three-round", [r for r in records if r[0] == 3]),
                     ("five-round", five)):
        if not sub:
            continue
        conf = [max(r[2][base], 1 - r[2][base]) for r in sub]
        hit = [1.0 if (r[2][base] >= 0.5) == (r[1] == 1.0) else 0.0 for r in sub]
        _, acc, brier, ll = _score([(r[2][base], r[1]) for r in sub])
        print(f"  {lbl:<12} n={len(sub):5d}  mean confidence {sum(conf)/len(conf):.4f}  "
              f"observed {sum(hit)/len(hit):.4f}  gap {sum(hit)/len(hit) - sum(conf)/len(conf):+.4f}"
              f"   Brier {brier:.5f}")

    if len(five) < 30:
        print("\ntoo few five-round bouts to sweep")
        return
    print(f"\nsharpen sweep, scored on the {len(five)} five-round bouts only:")
    print(f"{'s':>6} {'Brier':>9} {'d vs 1.0':>10} {'p':>8} {'deff':>6} {'log loss':>10} {'acc':>7}")
    for s in arms:
        pairs = [(r[2][s], r[1]) for r in five]
        _, acc, brier, ll = _score(pairs)
        if s == base:
            print(f"{s:>6} {brier:>9.5f} {'--':>10} {'--':>8} {'--':>6} {ll:>10.5f} {acc:>7.2%}")
            continue
        deltas = [(r[2][s] - r[1]) ** 2 - (r[2][base] - r[1]) ** 2 for r in five]
        d, pv, deff = paired_signflip(deltas, clusters=[r[3] for r in five])
        print(f"{s:>6} {brier:>9.5f} {d:>+10.5f} {pv:>8.4f} {deff:>6.2f} {ll:>10.5f} {acc:>7.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=float, nargs="+", default=[0.85, 1.0, 1.15, 1.3])
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    label = f"last {a.limit}" + (f" skipping {a.offset}" if a.offset else "")
    print(f"five-round interaction, window: {label}\n")
    report(run(a.arms, a.limit, a.offset), a.arms)
