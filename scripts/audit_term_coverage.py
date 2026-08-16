"""
Which model terms actually fire in a backtest, and which only fire live?

WHY THIS EXISTS. Every validation harness in scripts/ reports a Brier delta
and a p-value, and every one of them is silent about a much more basic
question: was the model it scored the model that runs in production? A term
that never fires in the scored population contributes nothing to the
measured difference, so a change to it -- or to anything competing with it
-- gets judged against a model that is missing it.

That is not hypothetical. predict_matchup takes fight_history_df and
weight_class_history_df as OPTIONAL arguments, and the harnesses call it
without them:

    predict_matchup(a, b, frame, eff)          # what the harnesses do
    predict_matchup(a, b, df, eff, history, wc_history, booked_class)

Called the first way, recent_form_adjustment returns 0.0 for every fight and
weight_class_change_penalty has no table to read. Both terms are then
structurally absent from every backtest verdict this project has published,
including the two model changes it recently REJECTED. A confident "not
significant" from a model missing half its style layer is not the same
finding as one from the production model, and until now nothing in the repo
distinguished them.

WHAT IT PRINTS. Firing rate per term, side by side:

    LIVE   -- the booked card, through the production call path
    BACKTEST -- historical fights, through the harness call path

and then flags every term whose backtest rate is zero while its live rate is
not. That flag list IS the output; a clean run means the harnesses are
scoring something close to production, and any entry on it is a term whose
validation verdicts should not be trusted.

RESULT ON FIRST RUN: five dark terms, one of them a real call-path bug.

    term                       LIVE    OLD CALL   THREADED
    recent_form                100%      0%         100%    <- fixed
    wrestling                   22%      0%           0%
    striking                    82%      0%           0%
    quick_return                 7%      0%           0%
    age_cliff                   13%      0%           0%

recent_form was dark for a fixable reason and is now measurable. It fired on
EVERY live prediction and none of 401 backtested ones, so every verdict this
project has published -- including two rejected model changes -- was computed
against a model missing its only recency signal. Two things had to change:
predict_matchup accepted a reference_date it never forwarded, and
recent_form_adjustment never filtered history by date, so passing full
history read FUTURE fights at full weight (a negative years_ago clamps the
decay to 1.0). The second is why omitting history was the safe choice, and
why the term stayed dark.

THE OTHER FOUR ARE DATA LIMITS, NOT BUGS, and no amount of threading fixes
them. Stated here so nobody reads a persistent flag as a defect:

  wrestling / striking  need strike_accuracy_pct and td_accuracy_pct on BOTH
      corners, which pit_roster deliberately does not reconstruct (only the
      current fighters.csv has them, and that is contaminated by the future).
      validate_pointintime_stats.py rebuilds them from the cached ESPN
      timelines; a harness that needs these terms must do the same.

  age_cliff  needs age, and _age_as_of can only derive it for fighters on the
      current roster -- 310 of the thousands in history. No DOB source exists
      for the rest.

  quick_return  needs a SHORT gap since the last fight, and the reconstructed
      gap is biased long because fight_history is a subset of each fighter's
      career. Measured over 723 reconstructed corners: median 182 days and 23%
      under 120, against 99 days and 55% on the live roster. The same bias
      inflates layoff, which fires on 70% of backtested fights and 13% of live
      ones -- the one term whose backtest rate being HIGHER than live is
      itself the symptom.

Deliberately no pass/fail exit code. This is a measuring instrument, and the
right response to a flagged term is to fix the call path or say why it cannot
be fixed -- not to make a script go green.

Usage:  python3 scripts/audit_term_coverage.py
        python3 scripts/audit_term_coverage.py --limit 400
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem  # noqa: E402
from src.power_rating import build_effective_ratings, compute_stats_rating, _streak_bonus, RATING_CENTER  # noqa: E402
from src.matchup_model import predict_matchup  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of  # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
FUTURE = "data/future_cards.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"

# Every term the waterfall can attribute a probability move to. Kept in the
# same order as matchup_model.build_probability_waterfall's FACTORS so the
# two can be eyeballed against each other.
TERMS = [
    "wrestling_adjustment", "striking_adjustment", "durability_adjustment",
    "recent_form_adjustment", "submission_threat_adjustment", "stance_adjustment",
    "height_adjustment", "layoff_adjustment", "quick_return_adjustment",
    "age_cliff_adjustment", "missed_weight_adjustment",
    "weight_class_change_adjustment", "short_notice_adjustment",
]

FIRING_EPS = 1e-9      # a term "fires" when it moves the rating at all


def _fold(n) -> str:
    return str(n).strip().lower()


def _fired(res: dict) -> dict:
    out = {}
    for t in TERMS:
        v = res.get(t)
        try:
            out[t] = v is not None and not math.isnan(float(v)) and abs(float(v)) > FIRING_EPS
        except (TypeError, ValueError):
            out[t] = False
    return out


def live_rates():
    """The booked card, through the production call path."""
    fighters = pd.read_csv(FIGHTERS)
    history = pd.read_csv(HISTORY)
    wc = pd.read_csv(WC_HISTORY) if os.path.exists(WC_HISTORY) else None
    elo = EloRatingSystem()
    elo.build_from_history(history)
    eff = build_effective_ratings(fighters, elo.ratings, history)

    counts = defaultdict(int)
    hits, n = defaultdict(int), 0
    if not os.path.exists(FUTURE):
        return {}, 0
    for _, r in pd.read_csv(FUTURE).iterrows():
        a, b = r.get("fighter_a"), r.get("fighter_b")
        try:
            res = predict_matchup(a, b, fighters, eff, history, wc, r.get("weight_class"))
        except Exception:
            res = None
        if not res:
            continue
        n += 1
        for t, f in _fired(res).items():
            hits[t] += int(f)
    return ({t: hits[t] / n for t in TERMS} if n else {}), n


def backtest_rates(limit, full=False):
    """
    Historical fights through the harness call path.

    Pass full=False for the four-argument form the validators used before
    this audit existed -- that is what their published verdicts were computed
    on, and it is the baseline the fix is measured against. full=True adds
    the point-in-time context (history, weight-class table, booked division,
    reference date) that predict_matchup accepts.
    """
    fighters = pd.read_csv(FIGHTERS)
    static_rows = {_fold(r["name"]): r.to_dict() for _, r in fighters.iterrows()}
    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    fight_index = build_fight_index(history)
    wc = pd.read_csv(WC_HISTORY) if (full and os.path.exists(WC_HISTORY)) else None

    elo = EloRatingSystem()
    counts, streaks = defaultdict(int), defaultdict(int)
    hits, n = defaultdict(int), 0

    rows = history.tail(limit) if limit else history
    cutoff = rows.iloc[0]["date"] if len(rows) else None

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()
        na, nb = counts[fa], counts[fb]

        if (cutoff is None or f["date"] >= cutoff) and na > 0 and nb > 0 and winner in (a, b):
            ra = roster_as_of(a, when, fight_index, static_rows, today=when)
            rb = roster_as_of(b, when, fight_index, static_rows, today=when)
            eff = {}
            for name, row, prior, fold in ((a, ra, na, fa), (b, rb, nb, fb)):
                sr = compute_stats_rating(pd.Series(row))
                w = min(1.0, prior / 4.0)
                eff[name] = w * elo.get_rating(name) + (1 - w) * sr + _streak_bonus(prior, streaks[fold])
            try:
                if full:
                    # Only fights strictly before this one are visible. The
                    # date filter inside recent_form_adjustment is a second
                    # guard, not a substitute for this one.
                    past = history[history["date"] < f["date"]]
                    res = predict_matchup(a, b, pd.DataFrame([ra, rb]), eff,
                                          past, wc, f.get("weight_class"),
                                          reference_date=when.date())
                else:
                    res = predict_matchup(a, b, pd.DataFrame([ra, rb]), eff)
            except Exception:
                res = None
            if res:
                n += 1
                for t, fired in _fired(res).items():
                    hits[t] += int(fired)

        loser = b if winner == a else a
        if winner in (a, b):
            elo.update_ratings(winner, loser, method=method)
            streaks[_fold(winner)] += 1
            streaks[_fold(loser)] = 0
        counts[fa] += 1
        counts[fb] += 1

    return ({t: hits[t] / n for t in TERMS} if n else {}), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600,
                    help="score only the most recent N history rows (0 = all)")
    args = ap.parse_args()

    live, n_live = live_rates()
    print(f"live card: {n_live} predicted fights")
    old, n_back = backtest_rates(args.limit, full=False)
    new, _ = backtest_rates(args.limit, full=True)
    print(f"backtest:  {n_back} scored fights\n")

    print(f"  {'term':<34}{'LIVE':>9}{'OLD CALL':>11}{'THREADED':>11}")
    print("  " + "-" * 65)
    dark = []
    for t in TERMS:
        lv, bk, nb = live.get(t), old.get(t), new.get(t)
        fmt = lambda v: f"{v:.0%}" if v is not None else "  -"
        flag = ""
        if lv and nb == 0.0:
            flag = "   <- STILL DARK"
            dark.append(t)
        elif lv and bk == 0.0 and nb:
            flag = "   <- fixed"
        print(f"  {t:<34}{fmt(lv):>9}{fmt(bk):>11}{fmt(nb):>11}{flag}")

    print()
    if dark:
        print(f"{len(dark)} term(s) contribute to every live prediction and NOTHING to any")
        print("backtest verdict. Any validation result that competes with these terms")
        print("was measured against a model that does not have them:")
        for t in dark:
            print(f"  - {t}")
    else:
        print("No term is dark. Backtest verdicts are being computed on a model that")
        print("at least exercises every term the live card does.")


if __name__ == "__main__":
    main()
