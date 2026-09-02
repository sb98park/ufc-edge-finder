"""
POINT-IN-TIME validation of a CAREER-RATE fallback for fighters with no UFC data.

THE GAP. ufc_method_rates serves UFC-only rates and returns None below three
UFC bouts; rates_or_prior then hands back divisional_fallback_rates -- half the
divisional prior on each side, identical for both corners. That is the right
answer for a true unknown and the wrong one for Salahdine Parnasse, who
headlines 2026-09-05 with 24 bouts in our spine and a 23-2 record carrying
8 KO, 7 SUB and 8 DEC wins. The model reads him as the average lightweight:
30/22/48. The market has him -620 with KO at -125.

Measured across bouts since 2015, one or both corners fall back on 64.8% of
them, so this is not an edge case.

WHY NOT JUST SERVE CAREER RATES -- and this is the whole difficulty. That is
what the code did before, and src/ufc_method_rates' docstring records what it
cost: career totals made every fighter look 1.49x more finish-prone on KO and
1.61x on submissions, because a prospect builds a regional record finishing
weak opposition and gets signed off it. The site then underpriced decisions by
4.95pp -- reproducing the market's own measured 4.03pp error, and so blinding
itself to the single edge this project has found. A fallback that quietly
reintroduces that is worse than reading Parnasse as generic.

SO THE ARM UNDER TEST DEBIASES BEFORE IT SHRINKS. Career rates are treated as
a biased estimator of UFC rates whose bias is itself measurable: on fighters
who have BOTH, fitted STRICTLY BEFORE the cutoff, the ratio of UFC rate to
career rate is estimated per component and applied to fighters who have only
the career side. Then shrunk toward the divisional prior by record depth,
n/(n+k), so a 3-fight regional record barely moves and a 24-bout one moves a
lot.

    shipped     half the divisional prior, identical for both corners
    raw_kN      career rates, shrunk, NOT debiased -- the old mistake, kept as
                an arm so the cost of skipping the correction is visible
    debias_kN   career rates, debiased then shrunk

Nothing is fitted on the scored half: the debias ratios and the choice of k
both come from fights before CUTOFF.

THE ANSWER IS NO. Every arm LOSES, and loses hardest exactly where it fires:

    FALLBACK FIRES (n=530, observed finish rate 0.528)
      arm          logloss   d.logloss     p
      shipped      0.67948    +0.00000
      debias_k12   0.68257    +0.00309   0.098
      debias_k6    0.68437    +0.00490   0.076
      debias_k3    0.68595    +0.00647   0.068
      raw_k3       0.68652    +0.00704   0.065

Positive d means worse than what ships. The ordering is the tell: the MORE
weight the arm puts on the career record (smaller k), the worse it gets, and
every debias arm beats its raw twin by a hair -- so the correction helps, and
what it is correcting is not worth using. Half the divisional prior beats a
fighter's own 24-bout record at predicting how his next fight ends.

Individually the p-values sit either side of 0.05 and none would survive being
called significant on their own. The monotone gradient across seven arms is
what makes this a result rather than noise: a useless signal would scatter.

WHY, MOST LIKELY. The prior is a well-calibrated "I do not know" -- symmetric,
near the base rate, and identical for both corners, so it adds no false
asymmetry. A regional finishing record is a real number about a different
sport: knocking out weak opposition predicts little about a UFC opponent, and
the model was trained on UFC-only profiles, so career rates are out of
distribution even after debiasing.

FITTED, NOT BORROWED. src/ufc_method_rates' docstring quotes career-to-UFC
ratios of 1.49x on KO and 1.61x on submissions. Fitted here on 4,320
pre-cutoff bouts the ratios are 1.10 and 1.11, because those figures compare
differently-defined quantities -- this file puts every rate over FIGHTS, the
same denominator ufc_method_rates uses, which is what makes a ratio between
them mean anything.

SO PARNASSE STAYS GENERIC, and that is the honest outcome rather than the
satisfying one. The model reading a 23-2 KSW champion as the average
lightweight looks wrong and is, on this evidence, still better than the
alternative.

THE OBVIOUS NEXT IDEA IS OUT OF REACH, and this is written down so nobody
spends a day rediscovering it. The natural refinement is to stop treating
"not UFC" as one bucket and give the major promotions -- Bellator, KSW, PFL,
ONE, RIZIN, Cage Warriors, LFA, Invicta, Brave, ACB -- their own tier, on the
theory that a KSW title run transfers where a small-hall record does not.
Counted before building it:

    named non-UFC bouts in the spine                2,497
    classifiable as a major promotion                 727  (29%)
    fallback corners on UFC bouts since 2015         4,886
    ...with 3+ prior MAJOR bouts                       260  (5.3%)
    UFC bouts such a tier could move, 2015+            203
    of those HELD OUT after the cutoff                  59

Fifty-nine fights. The effect this file measured on the career arms is about
0.005 in log loss; separating it from noise needs several hundred. The idea is
not refuted -- it is unmeasurable on the data we hold, which is a different
thing and must not be reported as the first.

It becomes testable only if the promotion field gets much denser: it is
populated on 2,497 of 11,709 spine rows, and blank means UFC by convention
(see elo.ufc_only), so "we did not record it" and "it was a UFC fight" are the
same value. Checked while here, and that conflation is NOT currently biting --
Parnasse, Andrusca and Aljarouj carry named promotions on every one of their
bouts. The 622 rows ufc_only calls UFC that miss a UFCStats match are the
UFCStats file lagging (its latest event is 2026-08-15, the spine runs to
08-29), not mislabelling.

WHAT THIS DOES NOT COVER. Scoring is the fight-level finish probability, not
the three-way split or the per-fighter grid. And the roster-coverage confound
from validate_probability_calibration applies here too: departed fighters
carry no stat columns, so the backtest runs thinner than production.

Usage:  python3 scripts/validate_career_method_fallback.py
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matchup_model import predict_matchup, compute_divisional_method_priors  # noqa: E402
from src.method_model import method_probabilities  # noqa: E402
from src.ufc_method_rates import (  # noqa: E402
    ufc_method_rates_as_of, load_dated_ufc_bouts, divisional_fallback_rates,
)
from src.names import _normalize_name  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of, record_as_of  # noqa: E402

CUTOFF = "2024-01-01"
START = "2015-01-01"
KS = (3, 6, 12)
BOOTSTRAP = 3000


def career_rates_as_of(fight_index, name, when):
    """(ko, sub, ko_lost, sub_lost) over ALL bouts before `when`.

    Same denominator convention as ufc_method_rates -- every rate is over
    FIGHTS, not over wins or losses -- so the two are directly comparable and
    a ratio between them means something.
    """
    rec = record_as_of(fight_index, name, when)
    if not rec:
        return None, 0
    n = int(rec.get("wins") or 0) + int(rec.get("losses") or 0)
    if n <= 0:
        return None, 0
    return (float(rec.get("ko_wins") or 0) / n, float(rec.get("sub_wins") or 0) / n,
            float(rec.get("ko_losses") or 0) / n, float(rec.get("sub_losses") or 0) / n), n


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _bucket(m):
    m = str(m).lower()
    if "ko" in m or "tko" in m or "sub" in m:
        return 1
    return 0 if "dec" in m else None


def main() -> int:
    fighters = pd.read_csv("data/fighters.csv")
    history = pd.read_csv("data/fight_history.csv")
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")
    static_rows = {_normalize_name(r["name"]): r.to_dict() for _, r in fighters.iterrows()}
    fight_index = build_fight_index(history)
    priors = compute_divisional_method_priors(fighters)
    ufc_tbl = load_dated_ufc_bouts()

    from src.elo import EloRatingSystem
    elo = EloRatingSystem()

    rows = []
    for r in history.itertuples(index=False):
        when = r.date
        a, b = str(r.fighter_a).strip(), str(r.fighter_b).strip()
        y = _bucket(getattr(r, "method", None))
        if y is not None and str(when)[:10] >= START:
            try:
                ra = roster_as_of(a, when, fight_index, static_rows, today=when)
                rb = roster_as_of(b, when, fight_index, static_rows, today=when)
                m = predict_matchup(a, b, pd.DataFrame([ra, rb]), elo.ratings,
                                    reference_date=when.date())
            except Exception:
                m = None
            if m:
                def _wc(row):
                    v = row.get("weight_class")
                    return None if v is None or (isinstance(v, float) and v != v) else str(v).strip()
                rec = {}
                for tag, name, rr in (("a", a, ra), ("b", b, rb)):
                    rec[f"ufc_{tag}"] = ufc_method_rates_as_of(name, when, ufc_tbl)
                    rec[f"car_{tag}"], rec[f"n_{tag}"] = career_rates_as_of(fight_index, name, when)
                    rec[f"fb_{tag}"] = divisional_fallback_rates(priors, _wc(rr))
                # THE REAL ELO GAP, captured HERE rather than defaulted to
                # zero at scoring time. method_probabilities takes it and it
                # interacts with the rate terms, so zeroing it would compare
                # the arms fairly but on a model production never runs.
                rec["gap"] = abs(elo.ratings.get(a, 1500) - elo.ratings.get(b, 1500)) / 400.0
                rows.append({"date": when, "event": f"{when:%Y-%m-%d}", "y": y, **rec})
        w = str(r.winner).strip()
        if w and w.lower() in (a.lower(), b.lower()):
            try:
                elo.update_ratings(w, b if w.lower() == a.lower() else a,
                                   str(getattr(r, "method", "") or "DEC"))
            except Exception:
                pass

    d = pd.DataFrame(rows)
    if d.empty:
        print("[career-fallback] nothing scored")
        return 0
    falls = d.apply(lambda r: r["ufc_a"] is None or r["ufc_b"] is None, axis=1)
    print(f"scored {len(d)} decided bouts since {START}")
    print(f"   at least one corner falls back: {int(falls.sum())} ({falls.mean():.1%})")

    # ---- DEBIAS FITTED STRICTLY BEFORE THE CUTOFF -------------------------
    tr = d[d.date < CUTOFF]
    num = np.zeros(4)
    den = np.zeros(4)
    for _, r in tr.iterrows():
        for tag in ("a", "b"):
            u, c = r[f"ufc_{tag}"], r[f"car_{tag}"]
            if u is None or c is None:
                continue
            for i in range(4):
                num[i] += u[i]
                den[i] += c[i]
    ratio = np.array([num[i] / den[i] if den[i] > 1e-9 else 1.0 for i in range(4)])
    print(f"\n   UFC-to-career ratio, fitted on {len(tr)} pre-cutoff bouts:")
    for lab, x in zip(("own ko", "own sub", "opp ko lost", "opp sub lost"), ratio):
        print(f"      {lab:14s} {x:.3f}")

    def rates_for(r, tag, arm, k):
        u = r[f"ufc_{tag}"]
        if u is not None:
            return u                       # measured UFC rates always win
        fb = r[f"fb_{tag}"]
        c, n = r[f"car_{tag}"], r[f"n_{tag}"]
        if arm == "shipped" or c is None or n <= 0:
            return fb
        base = np.array(c) * (ratio if arm == "debias" else 1.0)
        w = n / (n + k)
        return tuple(w * base[i] + (1 - w) * fb[i] for i in range(4))

    def predict(r, arm, k):
        ka, sa, la, xa = rates_for(r, "a", arm, k)
        kb, sb, lb, xb = rates_for(r, "b", arm, k)
        md = method_probabilities(ko_press=ka * lb + kb * la, sub_press=sa * xb + sb * xa,
                                  ko_rate_sum=ka + kb, sub_rate_sum=sa + sb,
                                  durability=la + lb, elo_gap=float(r.get("gap") or 0.0))
        return float(md.get("ko", 0.0)) + float(md.get("sub", 0.0))

    te = d[d.date >= CUTOFF]
    te_falls = te.apply(lambda r: r["ufc_a"] is None or r["ufc_b"] is None, axis=1)
    arms = [("shipped", None)] + [(a, k) for a in ("raw", "debias") for k in KS]
    preds = {name if k is None else f"{name}_k{k}":
             te.apply(lambda r, a=name, kk=k: predict(r, a, kk or 0), axis=1).to_numpy()
             for name, k in arms}

    for label, mask in (("ALL SCORED", np.ones(len(te), bool)), ("FALLBACK FIRES", te_falls.to_numpy())):
        yy = te["y"].to_numpy()[mask]
        if mask.sum() < 200:
            print(f"\n=== {label} === only {int(mask.sum())} bouts, skipping")
            continue
        print(f"\n=== {label} ===  n={int(mask.sum())}, observed finish rate {yy.mean():.3f}")
        print(f"   {'arm':12s} {'logloss':>9} {'brier':>9} {'mean p':>8} {'d.logloss':>10} {'p':>7}")
        base = preds["shipped"][mask]
        for name in preds:
            p = preds[name][mask]
            dll = logloss(p, yy) - logloss(base, yy)
            pv = ""
            if name != "shipped":
                per = defaultdict(list)
                for ev, pb, pt, t in zip(te["event"].to_numpy()[mask], base, p, yy):
                    lb = -(t * np.log(max(pb, 1e-9)) + (1 - t) * np.log(max(1 - pb, 1e-9)))
                    lt = -(t * np.log(max(pt, 1e-9)) + (1 - t) * np.log(max(1 - pt, 1e-9)))
                    per[ev].append(lt - lb)
                diffs = np.array([np.mean(v) for v in per.values()])
                rng = np.random.RandomState(0)
                null = np.array([np.mean(diffs * rng.choice([-1, 1], len(diffs)))
                                 for _ in range(BOOTSTRAP)])
                pv = f"{float(np.mean(np.abs(null) >= abs(diffs.mean()))):.3f}"
            print(f"   {name:12s} {logloss(p, yy):9.5f} {brier(p, yy):9.5f} {p.mean():8.3f} "
                  f"{dll:+10.5f} {pv:>7}")
    print("\n   negative d.logloss means the arm beat the shipped fallback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
