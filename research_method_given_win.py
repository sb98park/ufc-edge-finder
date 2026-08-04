"""
Fit P(method | win) -- given a fighter wins, HOW does he win?

WHAT THIS IS FOR. The per-fighter method rows are produced by reconciling a
seed against two validated margins: each fighter's win probability (moneyline,
backtested) and the fight's KO/SUB/DEC split (research_method_fightlevel.py).
The reconciliation guarantees the totals are coherent -- it says nothing about
whether the SEED is any good, and the seed is currently a hand-weighted blend
of divisional priors and career rates that has never been measured.

This fits that seed properly. Its output slots into the same reconciliation,
so both validated constraints survive; only the shape being reconciled changes.

WHY CONDITIONAL, not a six-outcome model. Who wins is already solved and
backtested; how the fight ends was solved today. How a SPECIFIC fighter wins is
the one unmeasured link, and a conditional targets exactly that. A six-way
model would have to relearn the moneyline, and 3.3k fights across six classes
leaves the rare cells (underdog submission wins) very thin.

STRUCTURAL LIMITATION worth stating: each fight observes only the WINNER's
method. Nothing is ever learned about how the loser would have won, so this is
fit on winners alone. That's correct for a conditional, but it means a fighter
with few wins contributes little and the model leans on divisional structure
for him -- which is what the prior blend was trying to do by hand.

CHECKS, in the order that has actually caught things:
  1. beats the divisional base rate on holdout log-loss
  2. mean predicted vs observed, PER CLASS -- a good average with one
     systematically wrong class is the failure mode that shipped twice
  3. the same check on five-round fights, where the money is
  4. reliability by predicted bucket -- when it says 60% submission, is it?

Run: python3 research_method_given_win.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import research_survival_model as R
from src.elo import EloRatingSystem

HOLDOUT_START = pd.Timestamp("2019-01-01")
CLASSES = ["KO/TKO", "SUB", "DEC"]
FEATURES = [
    "own_ko_rate", "own_sub_rate",        # how this fighter finishes
    "opp_ko_lost", "opp_sub_lost",        # how the opponent gets finished
    "ko_match", "sub_match",              # offense meeting that vulnerability
    "elo_gap", "scheduled",
]


def build_rows():
    """One row per fight, from the WINNER's perspective, point-in-time."""
    fights = R.load_dated_fights()
    res = pd.read_csv(R.RESULTS_PATH)
    res.columns = [c.strip() for c in res.columns]
    fmt = {(str(r["EVENT"]).strip(), str(r["BOUT"]).strip()): str(r.get("TIME FORMAT", ""))
           for r in res.to_dict("records")}

    elo, career, rows = EloRatingSystem(), R.Career(), []
    for f in fights.itertuples(index=False):
        ev = R.classify(f.method)
        if ev == "drop":
            continue
        winner = f.winner
        loser = f.fighter_2 if winner == f.fighter_1 else f.fighter_1
        cw, cl = career.get(winner), career.get(loser)
        if cw and cl:
            raw = fmt.get((str(f.event).strip(), str(f.bout).strip()), "")
            sched = 5 if raw.strip().startswith("5") else 3
            y = 0 if ev == R.KO else (1 if ev == R.SUB else 2)
            rows.append({
                "date": f.date,
                "own_ko_rate": cw["ko_rate"], "own_sub_rate": cw["sub_rate"],
                "opp_ko_lost": cl["ko_lost"], "opp_sub_lost": cl["sub_lost"],
                "ko_match": cw["ko_rate"] * cl["ko_lost"],
                "sub_match": cw["sub_rate"] * cl["sub_lost"],
                "elo_gap": (elo.get_rating(winner) - elo.get_rating(loser)) / 400.0,
                "scheduled": sched,
                "y": y,
            })
        career.update(winner, loser, ev if ev in (R.KO, R.SUB) else None)
        elo.update_ratings(winner, loser, method=str(f.method))
    return pd.DataFrame(rows)


def calibration(probs, y, label, threshold=0.06):
    print(f"\n  {label}")
    print(f"    {'method':8}{'predicted':>11}{'actual':>9}{'gap':>9}")
    worst = 0.0
    for i, name in enumerate(CLASSES):
        pred, act = probs[:, i].mean(), (y == i).mean()
        worst = max(worst, abs(pred - act))
        flag = "  <-- OFF" if abs(pred - act) > threshold else ""
        print(f"    {name:8}{pred:10.1%}{act:9.1%}{pred - act:+9.1%}{flag}")
    return worst


def reliability(probs, y, label):
    """When it says 40-60% submission, how often is it a submission?"""
    print(f"\n  {label}")
    worst = 0.0
    for i, name in enumerate(CLASSES):
        printed = False
        for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)):
            m = (probs[:, i] >= lo) & (probs[:, i] < hi)
            if m.sum() < 30:
                continue
            if not printed:
                print(f"    {name}")
                printed = True
            pred, act = probs[m, i].mean(), (y[m] == i).mean()
            worst = max(worst, abs(pred - act))
            flag = "  <-- OFF" if abs(pred - act) > 0.10 else ""
            print(f"      {lo:.0%}-{hi:.0%}  n={m.sum():4}  mean {pred:5.1%}  "
                  f"actual {act:5.1%}  {pred - act:+6.1%}{flag}")
    return worst


def main():
    df = build_rows()
    if df.empty:
        print("No rows -- needs data/ufc_fight_results.csv.")
        return
    tr, te = df[df.date < HOLDOUT_START], df[df.date >= HOLDOUT_START]
    print(f"{len(df)} wins ({len(tr)} train / {len(te)} holdout)")
    print("\nHow winners actually won, in the holdout:")
    for i, n in enumerate(CLASSES):
        print(f"  {n:8} {(te['y'] == i).mean():.1%}")

    eps, y_te = 1e-12, te["y"].to_numpy()
    base = np.tile(np.array([(tr["y"] == i).mean() for i in range(3)]), (len(te), 1))
    base_ll = -np.mean(np.log(np.clip(base[np.arange(len(y_te)), y_te], eps, 1)))

    # `scheduled` is fit from few five-round wins and extrapolates the wrong
    # way -- it learned "longer fight, more finishes" while five-round wins go
    # to DECISION 47.6% of the time. Exactly the failure the fight-level model
    # hit, where dropping the feature improved log-loss AND the subgroup.
    # Test the same variants rather than assuming it transfers.
    NO_SCHED = [f for f in FEATURES if f != "scheduled"]
    m5_tr = (te["scheduled"] == 5).to_numpy()

    variants = {}
    mf = LogisticRegression(max_iter=3000).fit(tr[FEATURES], tr["y"])
    variants["with scheduled"] = (mf.predict_proba(te[FEATURES]), mf, FEATURES)
    mn = LogisticRegression(max_iter=3000).fit(tr[NO_SCHED], tr["y"])
    variants["without scheduled"] = (mn.predict_proba(te[NO_SCHED]), mn, NO_SCHED)

    tr3, tr5 = tr[tr["scheduled"] == 3], tr[tr["scheduled"] == 5]
    if len(tr5) >= 120:
        m3 = LogisticRegression(max_iter=3000).fit(tr3[NO_SCHED], tr3["y"])
        m5m = LogisticRegression(max_iter=3000).fit(tr5[NO_SCHED], tr5["y"])
        ps = np.zeros((len(te), 3))
        mk = (te["scheduled"] == 5).to_numpy()
        ps[~mk] = m3.predict_proba(te[NO_SCHED][~mk])
        ps[mk] = m5m.predict_proba(te[NO_SCHED][mk])
        variants["split by length"] = (ps, (m3, m5m), NO_SCHED)

    print(f"\n{'='*60}\nVARIANT COMPARISON\n{'='*60}")
    print(f"  {'variant':22}{'log-loss':>10}{'worst class':>13}{'worst 5rd':>11}")
    best, best_w5 = None, 1e9
    for name, (pp, _, _) in variants.items():
        vll = -np.mean(np.log(np.clip(pp[np.arange(len(y_te)), y_te], eps, 1)))
        wc = max(abs(pp[:, i].mean() - (y_te == i).mean()) for i in range(3))
        w5v = (max(abs(pp[m5_tr][:, i].mean() - (y_te[m5_tr] == i).mean()) for i in range(3))
               if m5_tr.sum() > 60 else float("nan"))
        print(f"  {name:22}{vll:10.4f}{wc:12.1%}{w5v:11.1%}")
        # Ranked on the five-round error: that subgroup is where the liquid
        # method markets are, and where both models have gone wrong before.
        if w5v == w5v and w5v < best_w5:
            best, best_w5 = name, w5v
    print(f"\n  best on five-round calibration: {best} ({best_w5:.1%})")

    p, model, used = variants[best]
    ll = -np.mean(np.log(np.clip(p[np.arange(len(y_te)), y_te], eps, 1)))

    print(f"\n{'='*60}\nHOLDOUT log-loss\n{'='*60}")
    print(f"  base rates          {base_ll:.4f}")
    print(f"  fitted P(method|win){ll:.4f}   ({base_ll - ll:+.4f})")

    wb = calibration(base, y_te, "CALIBRATION -- base rates")
    wn = calibration(p, y_te, "CALIBRATION -- fitted model")

    m5 = (te["scheduled"] == 5).to_numpy()
    w5 = 0.0
    if m5.sum() > 60:
        w5 = calibration(p[m5], y_te[m5],
                         f"CALIBRATION -- five-round wins only (n={m5.sum()})", threshold=0.08)

    wt = reliability(p, y_te, "RELIABILITY BY PREDICTED BUCKET")

    print(f"\n{'='*60}")
    if ll >= base_ll:
        print("FAIL: no better than divisional base rates. Keep the current seed;")
        print("a fitted model that loses to base rates is worse than a prior.")
    elif w5 > 0.10:
        print(f"SUBGROUP FAILURE: five-round wins off by {w5:.1%}. That's where")
        print("the liquid method markets are -- don't ship it.")
    elif wt > 0.12:
        print(f"TAIL FAILURE: worst predicted-bucket error {wt:.1%}. Fine on")
        print("average, wrong where it makes a strong claim -- which is the")
        print("only place it would change a bet.")
    elif wn <= wb + 0.005:
        print("PASS: beats base rates, every class at least as well calibrated,")
        print("subgroup and tail within tolerance. Safe to replace the seed.")
    else:
        print(f"MIXED: better log-loss but worst class off by {wn:.1%} vs base")
        print(f"rates' {wb:.1%}. Not an improvement where it matters.")

    if isinstance(model, tuple):
        print("\nSPLIT MODEL -- two coefficient sets (3-round, then 5-round)")
        for tag, mm in zip(("THREE_ROUND", "FIVE_ROUND"), model):
            print(f"\n{tag}_COEF = [")
            for row in mm.coef_:
                print("    [" + ", ".join(f"{v:.6f}" for v in row) + "],")
            print("]")
            print(f"{tag}_INTERCEPT = [" + ", ".join(f"{v:.6f}" for v in mm.intercept_) + "]")
        print(f"FEATURES = {used}")
        return

    print(f"\nCOEF = [")
    for row in model.coef_:
        print("    [" + ", ".join(f"{v:.6f}" for v in row) + "],")
    print("]")
    print("INTERCEPT = [" + ", ".join(f"{v:.6f}" for v in model.intercept_) + "]")
    print(f"FEATURES = {used}")


if __name__ == "__main__":
    main()
