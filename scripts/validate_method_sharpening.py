"""
POINT-IN-TIME validation of sharpening the fight-level method distribution.

THE OBSERVATION THAT PROMPTED THIS. The owner: "for most of the model picks,
it usually picks [fighter] to win by decision. I feel like there has been a
ton of finishes." Both halves check out and neither means the model is wrong:

    UFC all-time, 8,741 fights   KO 33.2%  SUB 19.7%  DEC 47.1%
    the model's mean prediction  KO 36.2%  SUB 16.8%  DEC 47.1%

Its mean sits on the base rate, and "by Decision" dominating the headline is
an ARGMAX artefact -- Decision is the single biggest of six outcomes in a
sport that goes to the judges 47% of the time. CLAUDE.md records a previous
"5-sigma method bias" that was exactly this, so that is not what is tested
here.

WHAT IS TESTED IS THE SPREAD, NOT THE MEAN. On the 102 graded fights the
model's P(finish) never left 0.339-0.720, and sorted into quartiles it ran

    predicted 0.427 0.495 0.559 0.633
    observed  0.480 0.654 0.720 0.923      AUC 0.720

Good ordering, and every quartile short -- worst at the top, where 0.633
predicted 0.923. A model whose ranking is right and whose extremes are too
timid is the textbook case for a one-parameter temperature, and this asks
whether that survives out of sample.

    p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))     T < 1 SHARPENS

It cannot reorder a single fight, so AUC is invariant by construction and only
Brier and log loss can move -- the same shape as validate_probability_
calibration, which tested the opposite direction on win probability.

WHY 102 FIGHTS COULD NOT ANSWER IT. That window ran 69.6% finishes against a
52.9% base rate, and it survives clustering by card (t = 3.24, p = 0.0071).
But UFC-wide 2026 is 54.6% over 337 fights, +0.6 SD from all-time. The year is
ordinary; eight cards ran hot inside it. Fitting a constant to them would
repeat the "4.2x worse than baseline" error that came from one narrow card.
So this scores thousands of fights, point-in-time.

THE CONFOUND THIS HARNESS MUST CARRY, borrowed from validate_probability_
calibration, which found the model "overconfident" in a backtest and traced it
to the backtest itself: fighters who have since left the roster carry no
height, reach or stat columns, so style terms gate off and the model runs on
thinner input than it ever does live. Every table is therefore also cut to
fights where BOTH corners are on the current roster -- the subset that
resembles a live prediction. A temperature that only helps on the thin subset
is fixing the harness, not the model.

NO FITTING ON THE TEST SET. T is fit by minimising log loss strictly BEFORE a
cutoff and scored strictly after, and fit separately on each half so a
temperature that swings between them is visible rather than averaged away.
Bootstrap is the paired sign-flip CLUSTERED BY EVENT: one card's finishes
share a referee whose stoppage threshold moves every fight on it.

THE ANSWER SO FAR IS "THIS HARNESS IS NOT READY", AND THAT IS THE FINDING.

Run it two ways and it reverses, each time at p = 0.000 over 103 event
clusters:

    method rates from          mean P(finish)   best T      says
    rates_or_prior (today's)        0.519        0.55       SHARPEN hard
    raw point-in-time counts        0.699        1.30       FLATTEN hard

Production sits at 0.529 against a 0.529 base rate, so NEITHER arm reproduces
it, and both optima sit on a grid edge -- the sign of a transform absorbing a
defect rather than finding a setting.

    Run 1 LEAKS. rates_or_prior reads today's UFC method rates, which include
    the scored fight and everything after it, so a man who later became a
    finisher carries a high KO rate at every bout of his career. It scored
    AUC 0.804 against the production model's measured 0.720. A backtest
    beating the live model is not a good result; it is a receipt for leakage,
    and the sharpening looked wonderful because what it sharpened was partly
    the answer.

    Run 2 DISCARDS THE SHRINKAGE. Point-in-time counts are honest but raw, and
    rates_or_prior does not serve raw rates -- it blends toward a divisional
    prior. A 2-0 fighter with two knockouts reads a 100% KO rate, ko_press
    goes through the roof (the [method_model] clipping warnings are that
    happening thousands of times), and the model over-predicts finishes by 17
    points. Flattening then wins for the same reason sharpening won before.

WHAT IT NEEDS BEFORE IT CAN ANSWER ANYTHING: point-in-time method rates WITH
production's shrinkage -- a rates_or_prior that takes counts as of a date
instead of reading the current file. Until then the sweep measures the
harness. It is committed unfinished, with the numbers above, so the next
attempt starts from the trap rather than walking into it.

The roster-coverage cut -- the confound that sank the win-probability
temperature in validate_probability_calibration -- could not even be run: only
497 of 8,494 scored fights have both corners on the current roster, and the
held-out split leaves 186/311. So the one check that would distinguish "the
model is miscalibrated" from "the backtest is malnourished" has no sample.

NOTHING SHIPPED. No temperature is applied anywhere; method_model is
unchanged.

Usage:  python3 scripts/validate_method_sharpening.py
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matchup_model import predict_matchup, compute_divisional_method_priors  # noqa: E402
from src.method_model import method_probabilities  # noqa: E402
from src.ufc_method_rates import rates_or_prior  # noqa: E402
from src.names import _normalize_name  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of  # noqa: E402

CUTOFF = "2024-01-01"
GRID = [0.55, 0.65, 0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30]
BOOTSTRAP = 3000
# UFC all-time, 8,741 fights: the fallback when a corner has no wins or no
# losses on record yet.
BASE_KO, BASE_SUB = 0.332, 0.197


def _pit_rates(row):
    """(own KO rate, own SUB rate, KO-lost rate, SUB-lost rate) as of the date.

    Denominators are wins for the win-side rates and losses for the loss-side
    ones, matching how rates_or_prior defines them. A fighter with no wins (or
    no losses) yet falls back to the pooled base rate rather than to zero --
    zero is a claim, and "we have not seen one" is not.
    """
    w = max(int(row.get("wins") or 0), 0)
    l = max(int(row.get("losses") or 0), 0)
    ko_w, sub_w = float(row.get("ko_wins") or 0), float(row.get("sub_wins") or 0)
    ko_l, sub_l = float(row.get("ko_losses") or 0), float(row.get("sub_losses") or 0)
    return (ko_w / w if w else BASE_KO,
            sub_w / w if w else BASE_SUB,
            ko_l / l if l else BASE_KO,
            sub_l / l if l else BASE_SUB)


def sharpen(p, T):
    """Temperature on a binary. T<1 pushes toward 0/1, T>1 toward 0.5."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if abs(T - 1.0) < 1e-12:
        return p
    a, b = p ** (1.0 / T), (1.0 - p) ** (1.0 / T)
    return a / (a + b)


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _bucket(m):
    m = str(m).lower()
    if "ko" in m or "tko" in m:
        return 1
    if "sub" in m:
        return 1
    if "dec" in m:
        return 0
    return None            # DQ, NC, overturned: not a method the model predicts


def main() -> int:
    fighters = pd.read_csv("data/fighters.csv")
    history = pd.read_csv("data/fight_history.csv")
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    static_rows = {_normalize_name(r["name"]): r.to_dict() for _, r in fighters.iterrows()}
    fight_index = build_fight_index(history)
    priors = compute_divisional_method_priors(fighters)

    # Elo replayed forward so each fight is scored on ratings that existed
    # before it. Rebuilding the whole ladder per fight would be O(n^2).
    from src.elo import EloRatingSystem
    elo = EloRatingSystem()

    rows = []
    for r in history.itertuples(index=False):
        y = _bucket(getattr(r, "method", None))
        when = r.date
        a, b = str(r.fighter_a).strip(), str(r.fighter_b).strip()
        if y is not None:
            fa, fb = _normalize_name(a), _normalize_name(b)
            try:
                ra = roster_as_of(a, when, fight_index, static_rows, today=when)
                rb = roster_as_of(b, when, fight_index, static_rows, today=when)
                eff = elo.ratings
                m = predict_matchup(a, b, pd.DataFrame([ra, rb]), eff,
                                    reference_date=when.date())
            except Exception:
                m = None
            if m:
                # RATES FROM THE POINT-IN-TIME RECORD, NOT rates_or_prior.
                #
                # This is the whole difference between a measurement and a
                # leak. rates_or_prior reads TODAY's UFC method rates, which
                # include the fight being scored and every fight after it, so
                # a man who went on to become a finisher carries a high KO
                # rate at every historical bout he ever had. The first run of
                # this harness did exactly that and scored AUC 0.804 against
                # the production model's measured 0.720 -- a backtest beating
                # the live model is not a good result, it is a receipt for
                # leakage, and sharpening looked wonderful because what it
                # sharpened was partly the answer.
                #
                # record_as_of already accumulates the method splits from
                # bouts strictly before the date, which is the same quantity
                # without the future in it.
                ko_a, sub_a, kl_a, sl_a = _pit_rates(ra)
                ko_b, sub_b, kl_b, sl_b = _pit_rates(rb)
                gap = abs(eff.get(a, 1500) - eff.get(b, 1500)) / 400.0
                md = method_probabilities(
                    ko_press=ko_a * kl_b + ko_b * kl_a,
                    sub_press=sub_a * sl_b + sub_b * sl_a,
                    ko_rate_sum=ko_a + ko_b, sub_rate_sum=sub_a + sub_b,
                    durability=kl_a + kl_b, elo_gap=gap)
                rows.append({
                    "date": when, "event": f"{when:%Y-%m-%d}",
                    "p": float(md.get("ko", 0.0)) + float(md.get("sub", 0.0)),
                    "y": y,
                    "full": fa in static_rows and fb in static_rows,
                })
        # advance elo AFTER scoring, so nothing sees its own result
        w = str(r.winner).strip()
        if w and w.lower() in (a.lower(), b.lower()):
            loser = b if w.lower() == a.lower() else a
            try:
                elo.update_ratings(w, loser, str(getattr(r, "method", "") or "DEC"))
            except Exception:
                pass

    d = pd.DataFrame(rows)
    if d.empty:
        print("[method-sharpen] nothing scored")
        return 0
    print(f"scored {len(d)} decided fights, {int(d.full.sum())} with both corners "
          f"on the current roster\n")
    print(f"   model mean P(finish) {d.p.mean():.3f}   observed {d.y.mean():.3f}")
    print(f"   spread: min {d.p.min():.3f}  p25 {d.p.quantile(.25):.3f}  "
          f"p75 {d.p.quantile(.75):.3f}  max {d.p.max():.3f}")

    for label, sub in (("ALL", d), ("BOTH CORNERS ON ROSTER", d[d.full])):
        if len(sub) < 200:
            print(f"\n{label}: only {len(sub)} fights, skipping")
            continue
        tr = sub[sub.date < CUTOFF]
        te = sub[sub.date >= CUTOFF]
        print(f"\n=== {label} ===   fit on {len(tr)} before {CUTOFF}, score {len(te)} after")
        if len(tr) < 200 or len(te) < 200:
            print("   too thin to split"); continue
        best = min(GRID, key=lambda T: logloss(sharpen(tr.p, T), tr.y))
        print(f"   T minimising log loss on the FIT half: {best}")
        print(f"\n   {'T':>6} {'logloss':>9} {'brier':>9}   (scored half)")
        for T in GRID:
            ps = sharpen(te.p, T)
            mark = "  <- fitted" if T == best else ("  <- shipped" if T == 1.0 else "")
            print(f"   {T:6.2f} {logloss(ps, te.y):9.5f} {brier(ps, te.y):9.5f}{mark}")
        base, tuned = sharpen(te.p, 1.0), sharpen(te.p, best)
        dll = logloss(tuned, te.y) - logloss(base, te.y)
        # PAIRED SIGN-FLIP, CLUSTERED BY EVENT.
        per = defaultdict(list)
        for ev, pb, pt, yy in zip(te.event, base, tuned, te.y):
            lb = -(yy * np.log(max(pb, 1e-9)) + (1 - yy) * np.log(max(1 - pb, 1e-9)))
            lt = -(yy * np.log(max(pt, 1e-9)) + (1 - yy) * np.log(max(1 - pt, 1e-9)))
            per[ev].append(lt - lb)
        diffs = np.array([np.mean(v) for v in per.values()])
        rng = np.random.RandomState(0)
        null = np.array([np.mean(diffs * rng.choice([-1, 1], len(diffs)))
                         for _ in range(BOOTSTRAP)])
        p_val = float(np.mean(np.abs(null) >= abs(diffs.mean())))
        print(f"\n   d.logloss at T={best}: {dll:+.5f}   ({len(per)} event clusters, "
              f"sign-flip p = {p_val:.3f})")
        print(f"   {'lower is better; negative d means the sharpened arm won'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
