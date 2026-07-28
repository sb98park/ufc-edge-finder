"""
Do ROUND-LEVEL trends improve the time-to-finish hazard model?

THE GAP THIS TESTS. The hazard model currently sees only CAREER AGGREGATES:
overall finish rate, overall times-finished. Those say nothing about SHAPE.
Two fighters with identical career KO rates are very different bets if one
starts fast and fades while the other builds -- and shape is exactly what a
model of "which round does this end" should care about.

The 41k-row per-round file has that shape and currently feeds only the
Fight Facts display, nothing predictive.

FEATURES ADDED, all point-in-time:
  fade        career round-3+ striking output as a fraction of round-1
              output. Below 1.0 = fades; above = builds.
  absorb_late damage TAKEN in later rounds relative to round 1. A fighter
              who ships more as the fight goes on is a rising KO risk in a
              way career durability alone cannot express.
  pace        career significant strikes landed per round. High-pace fights
              end early more often, independent of who is finishing whom.

Each is differenced across the pair, matching how the existing features are
built, and every one is accumulated chronologically so a fight is scored
only on what was known before it.

Run: python3 research_round_features.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import research_survival_model as R
from src.elo import EloRatingSystem

STATS_PATH = next((p for p in ("data/ufc_fight_stats.csv",
                               "/mnt/user-data/uploads/ufc_fight_stats.csv")
                   if __import__("os").path.exists(p)), None)


def _landed(cell):
    try:
        return int(str(cell).split(" of ")[0])
    except (ValueError, AttributeError, IndexError):
        return 0


class RoundShape:
    """Career round-by-round profile, accumulated point-in-time."""

    def __init__(self):
        self.t = {}

    def get(self, f):
        t = self.t.get(f)
        if not t or t["fights"] < 3 or t["r1_out"] <= 0:
            return None
        late_rounds = max(1, t["late_n"])
        return {
            # <1 fades, >1 builds
            "fade": (t["late_out"] / late_rounds) / (t["r1_out"] / max(1, t["r1_n"])),
            "absorb_late": (t["late_abs"] / late_rounds) / max(1.0, t["r1_abs"] / max(1, t["r1_n"])),
            "pace": t["tot_out"] / max(1, t["rounds"]),
        }

    def update(self, f, per_round):
        t = self.t.setdefault(f, {"fights": 0, "rounds": 0, "tot_out": 0,
                                  "r1_out": 0, "r1_abs": 0, "r1_n": 0,
                                  "late_out": 0, "late_abs": 0, "late_n": 0})
        t["fights"] += 1
        for rnd, landed, absorbed in per_round:
            t["rounds"] += 1
            t["tot_out"] += landed
            if rnd == 1:
                t["r1_out"] += landed; t["r1_abs"] += absorbed; t["r1_n"] += 1
            elif rnd >= 3:
                t["late_out"] += landed; t["late_abs"] += absorbed; t["late_n"] += 1


def build():
    stats = pd.read_csv(STATS_PATH)
    stats.columns = [c.strip() for c in stats.columns]
    stats["rnd"] = stats["ROUND"].astype(str).str.extract(r"(\d+)").astype(float)
    stats["landed"] = stats["SIG.STR."].map(_landed)

    # per (event, bout, fighter) -> [(round, landed, absorbed)]
    by_fight = {}
    for r in stats.to_dict("records"):
        key = (str(r["EVENT"]).strip(), str(r["BOUT"]).strip())
        by_fight.setdefault(key, []).append((str(r["FIGHTER"]).strip(), r["rnd"], r["landed"]))

    fights = R.load_dated_fights()
    elo, career, shape, rows = EloRatingSystem(), R.Career(), RoundShape(), []
    for f in fights.itertuples(index=False):
        ev = R.classify(f.method)
        if ev == "drop":
            continue
        try:
            end_round = max(1, int(np.ceil(float(f.duration_sec) / 300.0)))
        except (TypeError, ValueError):
            continue

        c1, c2 = career.get(f.fighter_1), career.get(f.fighter_2)
        s1, s2 = shape.get(f.fighter_1), shape.get(f.fighter_2)
        if c1 and c2 and s1 and s2:
            gap = abs(elo.get_rating(f.fighter_1) - elo.get_rating(f.fighter_2))
            sched = 5 if end_round > 3 else 3
            for rnd in range(1, min(end_round, sched) + 1):
                rows.append({
                    "date": f.date, "round": rnd, "scheduled": sched,
                    "ko_press": c1["ko_rate"] * c2["ko_lost"] + c2["ko_rate"] * c1["ko_lost"],
                    "sub_press": c1["sub_rate"] * c2["sub_lost"] + c2["sub_rate"] * c1["sub_lost"],
                    "ko_rate_sum": c1["ko_rate"] + c2["ko_rate"],
                    "sub_rate_sum": c1["sub_rate"] + c2["sub_rate"],
                    "durability": c1["ko_lost"] + c2["ko_lost"],
                    "elo_gap": gap / 400.0,
                    # --- new round-shape covariates ---
                    "fade_diff": s1["fade"] - s2["fade"],
                    "absorb_late": s1["absorb_late"] + s2["absorb_late"],
                    "pace_sum": (s1["pace"] + s2["pace"]) / 10.0,
                    "y": ev if (rnd == end_round and ev is not None) else R.SURVIVE,
                })

        loser = f.fighter_2 if f.winner == f.fighter_1 else f.fighter_1
        elo.update_ratings(f.winner, loser, method=str(f.method))
        career.update(f.winner, loser, ev if ev in (R.KO, R.SUB) else None)

        entries = by_fight.get((str(f.event).strip(), str(f.bout).strip()), [])
        for name in (f.fighter_1, f.fighter_2):
            own = [(r, l) for n, r, l in entries if n == name and not pd.isna(r)]
            opp = {r: l for n, r, l in entries if n != name and not pd.isna(r)}
            if own:
                shape.update(name, [(int(r), l, opp.get(r, 0)) for r, l in own])
    return pd.DataFrame(rows)


BASE = R.FEATURES
EXTRA = BASE + ["fade_diff", "absorb_late", "pace_sum"]


def main():
    if not STATS_PATH:
        print("Need ufc_fight_stats.csv (local-only) in data/.")
        return
    df = build()
    tr, te = df[df["date"] < R.HOLDOUT_START], df[df["date"] >= R.HOLDOUT_START]
    print(f"{len(df)} round-rows ({len(tr)} train / {len(te)} holdout)\n")

    eps = 1e-12
    print(f"{'='*66}\nHOLDOUT per-round conditional log-loss\n{'='*66}")
    results = {}
    for label, feats in (("existing features", BASE), ("+ round-shape features", EXTRA)):
        m = LogisticRegression(max_iter=3000).fit(tr[feats], tr["y"])
        p = m.predict_proba(te[feats])
        y = te["y"].to_numpy()
        ll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1)))
        results[label] = (ll, m, feats)
        print(f"  {label:26} {ll:.4f}")
    d = results["existing features"][0] - results["+ round-shape features"][0]
    print(f"\n  improvement: {d:+.4f} "
          f"({'round shape helps' if d > 0 else 'no gain -- career aggregates already capture it'})")

    m, feats = results["+ round-shape features"][1], results["+ round-shape features"][2]
    print("\nWhat the new covariates do (log-odds):")
    for cls, name in ((R.KO, "KO/TKO"), (R.SUB, "SUB")):
        for f in ("fade_diff", "absorb_late", "pace_sum"):
            print(f"  {name:7} {f:12} {m.coef_[cls][feats.index(f)]:+.3f}")


if __name__ == "__main__":
    main()
