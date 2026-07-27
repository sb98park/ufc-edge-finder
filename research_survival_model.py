"""
Time-to-finish: a discrete-time competing-risks hazard model.

WHAT THIS PREDICTS THAT THE MAIN MODEL CANNOT. Production answers "who wins"
plus a single likely method. This answers the richer question the softer
markets are priced on: for each round a fight reaches, what's the chance it
ends THERE, and by WHICH method? Chain those per-round hazards and you get a
full joint distribution over (round, method), with P(decision) falling out as
the probability of surviving every round.

WHY A HAZARD MODEL RATHER THAN A CLASSIFIER OVER OUTCOMES. Fights are
sequential: a round-3 knockout is only possible if the fight survived rounds
1 and 2. Predicting (round, method) directly ignores that structure. A
discrete-time hazard models the CONDITIONAL probability of ending in round r
GIVEN the fight reached r -- the quantity that's actually comparable across
fights of different scheduled lengths -- and then composes them.

CENSORING, HANDLED PROPERLY. A decision isn't a failed finish to be learned
from as "nothing happened"; it's a fight that survived every round it was
given. Each fight contributes one row per round REACHED, and a decision's
rows all say "survived" with no terminal event. That is right-censoring at
the final bell, and it's why decisions inform the model without being
mislabelled.

COMPETING RISKS. KO and submission are different events, not one "finish"
event -- a grappling-heavy matchup lifts submission hazard without touching
KO hazard. So each round-row is a 3-way outcome (survive / KO / SUB) and the
model is multinomial.

POINT-IN-TIME throughout: every fighter feature reflects only fights that had
already happened, and train/test is a date split, never random.

Run: python3 research_survival_model.py
"""

import math
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.elo import EloRatingSystem
from validate_adjustment_layer import load_dated_fights

RESULTS_PATH = next((p for p in ("data/ufc_fight_results.csv",
                                 "/mnt/user-data/uploads/ufc_fight_results.csv")
                     if os.path.exists(p)), None)
HOLDOUT_START = pd.Timestamp("2019-01-01")
MIN_PRIOR_FIGHTS = 3
SURVIVE, KO, SUB = 0, 1, 2


def classify(method):
    """Terminal event type, None for a decision (censored), 'drop' for oddities."""
    m = str(method).upper()
    if "KO" in m or "TKO" in m:
        return KO
    if "SUB" in m:
        return SUB
    if "DEC" in m:
        return None
    return "drop"


class Career:
    """Point-in-time finishing and durability profile."""

    def __init__(self):
        self.t = {}

    def get(self, f):
        t = self.t.get(f)
        if not t or t["fights"] < MIN_PRIOR_FIGHTS:
            return None
        n = t["fights"]
        return {"ko_rate": t["ko_wins"] / n, "sub_rate": t["sub_wins"] / n,
                # Durability -- how often this fighter has BEEN finished. The
                # single strongest input to a finish model: a chinny fighter
                # raises KO hazard regardless of who's opposite him.
                "ko_lost": t["ko_losses"] / n, "sub_lost": t["sub_losses"] / n}

    def update(self, winner, loser, ev):
        for f in (winner, loser):
            self.t.setdefault(f, {"fights": 0, "ko_wins": 0, "sub_wins": 0,
                                  "ko_losses": 0, "sub_losses": 0})
            self.t[f]["fights"] += 1
        if ev == KO:
            self.t[winner]["ko_wins"] += 1
            self.t[loser]["ko_losses"] += 1
        elif ev == SUB:
            self.t[winner]["sub_wins"] += 1
            self.t[loser]["sub_losses"] += 1


def build_rows():
    """One row per (fight, round REACHED) -- the expansion that encodes sequence."""
    fights = load_dated_fights()
    res = pd.read_csv(RESULTS_PATH)
    res.columns = [c.strip() for c in res.columns]
    fmt = {(str(r["EVENT"]).strip(), str(r["BOUT"]).strip()): str(r.get("TIME FORMAT", ""))
           for r in res.to_dict("records")}

    elo, career, rows = EloRatingSystem(), Career(), []
    for f in fights.itertuples(index=False):
        ev = classify(f.method)
        if ev == "drop":
            continue
        # load_dated_fights gives elapsed seconds, not a round number.
        # A round is 300s, so the round a fight ENDED in is however many
        # 5-minute blocks it ran into: 900s exactly = end of round 3,
        # 901s = into round 4. ceil() gives exactly that.
        try:
            end_round = max(1, math.ceil(float(f.duration_sec) / 300.0))
        except (TypeError, ValueError):
            continue
        # Scheduled length from ESPN's own TIME FORMAT ("3 Rnd (5-5-5)"), else
        # inferred -- assuming 3 for a championship fight that ended early
        # would silently truncate its later rounds out of the dataset.
        raw = fmt.get((str(f.event).strip(), str(f.bout).strip()), "")
        scheduled = 5 if raw.strip().startswith("5") else (5 if end_round > 3 else 3)

        c1, c2 = career.get(f.fighter_1), career.get(f.fighter_2)
        if c1 and c2:
            gap = abs(elo.get_rating(f.fighter_1) - elo.get_rating(f.fighter_2))
            for r in range(1, min(end_round, scheduled) + 1):
                rows.append({
                    "date": f.date, "round": r, "scheduled": scheduled,
                    # Offense meeting the opponent's vulnerability -- the
                    # product is the mechanism, not either raw rate alone.
                    "ko_press": c1["ko_rate"] * c2["ko_lost"] + c2["ko_rate"] * c1["ko_lost"],
                    "sub_press": c1["sub_rate"] * c2["sub_lost"] + c2["sub_rate"] * c1["sub_lost"],
                    "ko_rate_sum": c1["ko_rate"] + c2["ko_rate"],
                    "sub_rate_sum": c1["sub_rate"] + c2["sub_rate"],
                    "durability": c1["ko_lost"] + c2["ko_lost"],
                    "elo_gap": gap / 400.0,
                    "y": ev if (r == end_round and ev is not None) else SURVIVE,
                })
        loser = f.fighter_2 if f.winner == f.fighter_1 else f.fighter_1
        elo.update_ratings(f.winner, loser, method=str(f.method))
        career.update(f.winner, loser, ev if ev in (KO, SUB) else None)
    return pd.DataFrame(rows)


FEATURES = ["round", "scheduled", "ko_press", "sub_press",
            "ko_rate_sum", "sub_rate_sum", "durability", "elo_gap"]


def fight_distribution(model, base_row, scheduled):
    """Chain per-round hazards into the full (round, method) joint + P(decision)."""
    surv, out = 1.0, {}
    for r in range(1, scheduled + 1):
        x = dict(base_row); x["round"] = r; x["scheduled"] = scheduled
        p = model.predict_proba(pd.DataFrame([x])[FEATURES])[0]
        out[(r, "KO/TKO")] = surv * p[KO]
        out[(r, "SUB")] = surv * p[SUB]
        surv *= p[SURVIVE]
    out["decision"] = surv
    return out


def main():
    if not RESULTS_PATH:
        print("Need ufc_fight_results.csv in data/ (local-only file).")
        return
    print("Expanding fights into per-round observations...")
    df = build_rows()
    train, test = df[df["date"] < HOLDOUT_START], df[df["date"] >= HOLDOUT_START]
    print(f"  {len(df)} round-rows ({len(train)} train / {len(test)} holdout)")
    print(f"  outcome mix: survive {(df.y==SURVIVE).mean():.1%}, "
          f"KO {(df.y==KO).mean():.1%}, SUB {(df.y==SUB).mean():.1%}")

    model = LogisticRegression(max_iter=3000)
    model.fit(train[FEATURES], train["y"])

    # Baseline: empirical per-round hazard, ignoring who is fighting. A model
    # that can't beat this has learned nothing about matchups.
    base = train["y"].value_counts(normalize=True).reindex([SURVIVE, KO, SUB]).fillna(0).to_numpy()
    proba, y, eps = model.predict_proba(test[FEATURES]), test["y"].to_numpy(), 1e-12
    model_ll = -np.mean(np.log(np.clip(proba[np.arange(len(y)), y], eps, 1)))
    base_ll = -np.mean(np.log(np.clip(base[y], eps, 1)))

    print(f"\n{'='*68}\nHOLDOUT ({HOLDOUT_START.year}+) — per-round conditional log-loss\n{'='*68}")
    print(f"  empirical base rates : {base_ll:.4f}")
    print(f"  hazard model         : {model_ll:.4f}   "
          f"({'BETTER' if model_ll < base_ll else 'WORSE'} by {abs(base_ll - model_ll):.4f})")

    print("\nWhat it learned (log-odds; positive raises that hazard):")
    for cls, name in ((KO, "KO/TKO"), (SUB, "SUB")):
        top = sorted(zip(FEATURES, model.coef_[cls]), key=lambda t: -abs(t[1]))[:4]
        print(f"  {name:7} " + ", ".join(f"{f} {c:+.2f}" for f, c in top))

    dist = fight_distribution(model, test.iloc[0][FEATURES].to_dict(), 3)
    print("\nExample joint distribution, one holdout fight (3 rounds):")
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1])[:5]:
        print(f"   {(k if isinstance(k, str) else f'R{k[0]} {k[1]}'):14} {v:.1%}")
    print(f"   sums to {sum(dist.values()):.3f} — a proper distribution")


if __name__ == "__main__":
    main()
