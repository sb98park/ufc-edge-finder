"""
Point-in-time validation of the stat-based adjustment layer -- the test
that was impossible until now.

walkforward_backtest.py validates the Elo core but explicitly cannot
validate the style/wrestling/striking adjustments, because those need
career stats AS THEY STOOD AT FIGHT TIME and fighters.csv only has
current-day snapshots. This script closes that gap using the per-round
UFCStats data (ufc_fight_stats.csv + ufc_fight_results.csv from the
Greco1899/scrape_ufc_stats project):

1. Parse per-round stat rows into per-fight, per-fighter totals
   (significant strikes landed/absorbed, takedowns landed/attempted,
   control seconds, fight duration).
2. Date each fight via a UNIQUE fighter-pair join against our own
   data/fight_history.csv. Rematch pairs (same two fighters, multiple
   fights) are DROPPED, not guessed -- a wrongly-ordered rematch would
   poison point-in-time accumulation silently. (Uploading
   ufc_events.csv from the same source would recover these later.)
3. Replay chronologically, maintaining Elo ratings AND per-fighter
   career stat accumulators side by side. Every prediction uses only
   what was known before that fight.
4. Compare Elo-only vs Elo + stat adjustments on a real holdout:
   adjustment weight is chosen on pre-2019 fights only, then evaluated
   frozen on 2019+ -- the same split convention backtest_features.py
   already uses, so the headline number is never tuned on the data it's
   reported against.

The stat adjustments mirror the production signals in matchup_model.py
in spirit (wrestling edge from TD accuracy vs TD defense + control
share; striking edge from SLpM/SApM differentials), expressed as
Elo-point deltas with a total cap, so the question being answered is
the real one: "do point-in-time versions of the signals the live model
uses actually add predictive value over the rating core alone?"

Run: python3 validate_adjustment_layer.py
"""

import math

import pandas as pd

from src.elo import EloRatingSystem

import os

STATS_PATH = next((p for p in ("data/ufc_fight_stats.csv", "/mnt/user-data/uploads/ufc_fight_stats.csv") if os.path.exists(p)), "data/ufc_fight_stats.csv")
RESULTS_PATH = next((p for p in ("data/ufc_fight_results.csv", "/mnt/user-data/uploads/ufc_fight_results.csv") if os.path.exists(p)), "data/ufc_fight_results.csv")
FIGHT_HISTORY_PATH = "data/fight_history.csv"

MIN_PRIOR_STAT_FIGHTS = 3   # both fighters need this many stat-tracked fights
HOLDOUT_START = "2019-01-01"  # tune weight before this date, evaluate frozen after it
ADJ_CAP = 80.0  # Elo points, mirrors the spirit of production's ADJUSTMENT_TOTAL_CAP


def _of_pair(s):
    """'29 of 62' -> (29, 62); returns (0, 0) on '---'/malformed."""
    try:
        landed, attempted = str(s).split(" of ")
        return int(landed), int(attempted)
    except (ValueError, AttributeError):
        return 0, 0


def _ctrl_seconds(s):
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except (ValueError, AttributeError):
        return 0


def _fight_duration_seconds(round_num, time_str):
    try:
        m, s = str(time_str).split(":")
        return (int(round_num) - 1) * 300 + int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def load_per_fight_stats():
    stats = pd.read_csv(STATS_PATH)
    stats.columns = [c.strip() for c in stats.columns]

    parsed = pd.DataFrame({
        "event": stats["EVENT"].str.strip(),
        "bout": stats["BOUT"].str.strip(),
        "fighter": stats["FIGHTER"].str.strip(),
    })
    sig = stats["SIG.STR."].map(_of_pair)
    td = stats["TD"].map(_of_pair)
    parsed["sig_landed"] = [x[0] for x in sig]
    parsed["sig_attempted"] = [x[1] for x in sig]
    parsed["td_landed"] = [x[0] for x in td]
    parsed["td_attempted"] = [x[1] for x in td]
    parsed["ctrl_sec"] = stats["CTRL"].map(_ctrl_seconds)

    per_fight = parsed.groupby(["event", "bout", "fighter"], as_index=False).sum()
    return per_fight


def load_dated_fights():
    results = pd.read_csv(RESULTS_PATH)
    results.columns = [c.strip() for c in results.columns]
    results["EVENT"] = results["EVENT"].str.strip()
    results["BOUT"] = results["BOUT"].str.strip()

    hist = pd.read_csv(FIGHT_HISTORY_PATH)
    hist["date"] = pd.to_datetime(hist["date"])

    def pair_key(a, b):
        return frozenset({str(a).strip().lower(), str(b).strip().lower()})

    hist["pair"] = [pair_key(a, b) for a, b in zip(hist["fighter_a"], hist["fighter_b"])]
    pair_counts = hist["pair"].value_counts()
    unique_pairs = set(pair_counts[pair_counts == 1].index)
    hist_unique = hist[hist["pair"].isin(unique_pairs)]
    pair_to_row = {p: (d, w, m) for p, d, w, m in zip(
        hist_unique["pair"], hist_unique["date"], hist_unique["winner"], hist_unique["method"])}

    bouts = results["BOUT"].str.split(" vs. ", n=1, expand=True)
    results["fighter_1"] = bouts[0].str.strip()
    results["fighter_2"] = bouts[1].str.strip()
    results["pair"] = [pair_key(a, b) for a, b in zip(results["fighter_1"], results["fighter_2"])]
    results["duration_sec"] = [
        _fight_duration_seconds(r, t) for r, t in zip(results["ROUND"], results["TIME"])
    ]

    dated = []
    for _, row in results.iterrows():
        match = pair_to_row.get(row["pair"])
        if match is None:
            continue
        date, winner, method = match
        # Winner name must match one of the two parsed fighters exactly --
        # otherwise (draw/NC or a name-spelling mismatch between sources)
        # skip rather than guess.
        if winner not in (row["fighter_1"], row["fighter_2"]):
            continue
        if row["duration_sec"] is None:
            continue
        dated.append({
            "date": date, "event": row["EVENT"], "bout": row["BOUT"],
            "fighter_1": row["fighter_1"], "fighter_2": row["fighter_2"],
            "winner": winner, "method": method, "duration_sec": row["duration_sec"],
        })
    df = pd.DataFrame(dated).sort_values("date").reset_index(drop=True)
    return df


class StatAccumulator:
    """Career-to-date stats per fighter, updated only after each fight is predicted."""

    def __init__(self):
        self.totals: dict[str, dict] = {}

    def get(self, fighter: str) -> dict | None:
        t = self.totals.get(fighter)
        if not t or t["fights"] < MIN_PRIOR_STAT_FIGHTS or t["seconds"] <= 0:
            return None
        minutes = t["seconds"] / 60.0
        return {
            "slpm": t["sig_landed"] / minutes,
            "sapm": t["sig_absorbed"] / minutes,
            "td_acc": t["td_landed"] / t["td_attempted"] if t["td_attempted"] else None,
            "td_def": 1.0 - (t["td_absorbed"] / t["td_faced"]) if t["td_faced"] else None,
            "ctrl_share": t["ctrl_sec"] / t["seconds"],
        }

    def update(self, fighter: str, own: dict, opp: dict, duration_sec: float):
        t = self.totals.setdefault(fighter, {
            "fights": 0, "seconds": 0.0, "sig_landed": 0, "sig_absorbed": 0,
            "td_landed": 0, "td_attempted": 0, "td_absorbed": 0, "td_faced": 0,
            "ctrl_sec": 0.0,
        })
        t["fights"] += 1
        t["seconds"] += duration_sec
        t["sig_landed"] += own["sig_landed"]
        t["sig_absorbed"] += opp["sig_landed"]
        t["td_landed"] += own["td_landed"]
        t["td_attempted"] += own["td_attempted"]
        t["td_absorbed"] += opp["td_landed"]
        t["td_faced"] += opp["td_attempted"]
        t["ctrl_sec"] += own["ctrl_sec"]


def stat_adjustment(sa: dict, sb: dict) -> float:
    """
    Elo-point adjustment for fighter A from point-in-time stat
    differentials, mirroring the production signals in spirit:
    striking edge (SLpM/SApM) + wrestling edge (TD acc vs opposing TD
    def, control share). Symmetric by construction: swapping A and B
    flips the sign exactly.
    """
    adj = 0.0
    # Striking: net output differential (my landed-minus-absorbed rate vs theirs)
    net_a = sa["slpm"] - sa["sapm"]
    net_b = sb["slpm"] - sb["sapm"]
    adj += (net_a - net_b) * 12.0  # ~1 strike/min net edge ~= 12 Elo points
    # Wrestling: TD accuracy against the opponent's TD defense, both directions
    if sa["td_acc"] is not None and sb["td_def"] is not None:
        adj += max(0.0, sa["td_acc"] - sb["td_def"]) * 60.0
    if sb["td_acc"] is not None and sa["td_def"] is not None:
        adj -= max(0.0, sb["td_acc"] - sa["td_def"]) * 60.0
    # Control share differential
    adj += (sa["ctrl_share"] - sb["ctrl_share"]) * 50.0
    return max(-ADJ_CAP, min(ADJ_CAP, adj))


def main():
    print("Loading per-fight stats..."); per_fight = load_per_fight_stats()
    print("Dating fights via unique-pair join..."); fights = load_dated_fights()
    stat_lookup = {
        (r["event"], r["bout"], r["fighter"]): r
        for r in per_fight.to_dict("records")
    }
    print(f"Fights dated and usable: {len(fights)}")

    elo = EloRatingSystem()
    acc = StatAccumulator()
    holdout_start = pd.Timestamp(HOLDOUT_START)

    # weight=0.0 is the Elo-only control; others scale the stat adjustment
    weights = [0.0, 0.5, 1.0, 1.5, 2.0]
    records = {w: [] for w in weights}

    for _, f in fights.iterrows():
        f1, f2 = f["fighter_1"], f["fighter_2"]
        s1_raw = stat_lookup.get((f["event"], f["bout"], f1))
        s2_raw = stat_lookup.get((f["event"], f["bout"], f2))

        s1, s2 = acc.get(f1), acc.get(f2)
        r1, r2 = elo.get_rating(f1), elo.get_rating(f2)

        if s1 is not None and s2 is not None:
            base_gap = r1 - r2
            adj = stat_adjustment(s1, s2)
            for w in weights:
                gap = base_gap + w * adj
                p1 = 1.0 / (1.0 + 10 ** (-gap / 400.0))
                actual = 1.0 if f["winner"] == f1 else 0.0
                # predicted favorite correctness + probabilistic scores
                fav_p = p1 if p1 >= 0.5 else 1.0 - p1
                fav_won = (actual == 1.0) == (p1 >= 0.5)
                records[w].append({
                    "date": f["date"], "p": p1, "actual": actual,
                    "fav_p": fav_p, "fav_won": fav_won,
                })

        # Update state AFTER predicting (point-in-time discipline)
        loser = f2 if f["winner"] == f1 else f1
        elo.update_ratings(f["winner"], loser, method=str(f["method"]))
        if s1_raw and s2_raw:
            acc.update(f1, s1_raw, s2_raw, f["duration_sec"])
            acc.update(f2, s2_raw, s1_raw, f["duration_sec"])

    def score(rows):
        n = len(rows)
        if n == 0:
            return None
        acc_rate = sum(1 for r in rows if r["fav_won"]) / n
        brier = sum((r["p"] - r["actual"]) ** 2 for r in rows) / n
        eps = 1e-12
        logloss = -sum(
            r["actual"] * math.log(max(r["p"], eps)) + (1 - r["actual"]) * math.log(max(1 - r["p"], eps))
            for r in rows
        ) / n
        return n, acc_rate, brier, logloss

    print(f"\n{'='*70}\nTUNING SPLIT (pre-{HOLDOUT_START[:4]}) -- weight selection happens here ONLY")
    print(f"{'='*70}")
    best_w, best_brier = 0.0, float("inf")
    for w in weights:
        rows = [r for r in records[w] if r["date"] < holdout_start]
        n, a, b, ll = score(rows)
        marker = ""
        if w > 0 and b < best_brier:
            best_brier, best_w = b, w
        if w == 0:
            elo_only_train_brier = b
        print(f"  weight={w:>3}: n={n}, accuracy {a:.1%}, Brier {b:.4f}, log-loss {ll:.4f}")
    if best_brier >= elo_only_train_brier:
        best_w = 0.0
        print(f"\n  -> No positive weight beats Elo-only on the tuning split; selecting weight=0.")
    else:
        print(f"\n  -> Selected weight={best_w} (best tuning-split Brier)")

    print(f"\n{'='*70}\nHOLDOUT ({HOLDOUT_START[:4]}+) -- frozen evaluation, never used for tuning")
    print(f"{'='*70}")
    for w in [0.0, best_w] if best_w != 0.0 else [0.0]:
        rows = [r for r in records[w] if r["date"] >= holdout_start]
        n, a, b, ll = score(rows)
        label = "Elo only (control)" if w == 0.0 else f"Elo + adjustments (weight={w})"
        print(f"  {label}: n={n}, accuracy {a:.1%}, Brier {b:.4f}, log-loss {ll:.4f}")

    print("\nInterpretation: if the adjusted holdout Brier/log-loss beat the")
    print("Elo-only control, the stat layer adds real point-in-time value.")
    print("If not, that is a genuine (and publishable-on-the-site) finding too.")


if __name__ == "__main__":
    main()
