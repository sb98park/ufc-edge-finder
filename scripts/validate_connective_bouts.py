"""
POINT-IN-TIME validation of CONNECTIVE regional bouts at the FULL-MODEL level.

THE QUESTION, AND WHY IT LOOKS ALREADY ANSWERED. src/elo.ufc_only once dropped
every bout with a named promotion from the rating graph, on the argument that
regional opponents have no other results in the graph. Measured against the
spine, that justification holds for a small minority: of 2,497 non-UFC bouts,
2,323 (93%) have at least one corner whose folded name appears in a UFC bout
in data/ufc_fight_results.csv -- they CONNECT an outsider to the graph -- and
only 174 are genuinely unconnected. scripts/validate_spine_cleanup.py measured
the exclusion point-in-time over 9,198 UFC bouts and found it cost +0.00584
Brier at p=0.000, so commit 58d2a51a put every regional bout back at full
weight. That is what ships today.

WHAT WAS STILL OPEN. validate_spine_cleanup scored the RAW ELO CORE
(expected_score on replayed ratings) and says so in its own docstring. The
filter's original justification was a FULL-MODEL failure: build_effective_
ratings counted 18 bouts for Michael Aljarouj while his Elo came from 1, drove
the trust weight to 1.0, and published his opponent at 76% against a truer
57%. The shipped fix keeps the graph and the trust count in agreement, but
nobody had measured whether admitting regional bouts still wins once the
trust ramp, the streak bonus, the debutant branch and the style layer are all
in the loop -- the places the counter-argument actually lives. This harness
closes that gap.

ARMS. Identical scored set (decided UFC bouts), identical adjustment layer,
differing ONLY in which spine rows build ratings, trust counts and streaks --
always together, because their disagreement is what caused the Aljarouj/Sintes
failure, not the inclusion:

    shipped      every bout, full weight (today's code)
    w075/w050/w025  regional bouts at fractional weight: Elo K scaled, trust
                 count and adaptive-K experience incremented fractionally
    connective   regional bouts kept only when a corner appears in a UFC bout
                 (drops the 174 unconnected); binary
    ufconly      the reverted filter: no regional bouts anywhere

The connectivity test uses TODAY'S ufc_fight_results membership, which is the
hypothesis as posed ("the opponent has a UFC record"), not "had one that
night". That is structural future information about graph shape, never about
any scored outcome (admission decisions apply only to non-UFC rows, which are
never scored); the caveat is stated rather than hidden.

MECHANICS, and why the composition is exact rather than approximate.
predict_matchup's probability is sigmoid((eff_a - eff_b + applied_layer)/400)
where applied_layer reads the two roster rows, the weight-class history and
the past spine -- never the ratings. So the layer is computed once per fight
and each arm's probability is composed from its own rating gap plus that
shared layer. The shipped arm's composed value is asserted equal to the
prob_a predict_matchup itself returned, on every scored fight, so if the
layer ever grows a rating dependence this harness fails loudly instead of
comparing arms on a model production never runs (the elo_gap-hardcoded-to-
zero mistake validate_career_method_fallback caught in itself).

PIPELINE FIDELITY. Production's fighters_df is read_csv -> enrich_roster ->
attach_imputed_reach -> reconcile_last_fight_from_history ->
attach_history_coverage. Here: records, last-fight fields and layoff come
from pit_roster (point-in-time by construction, which subsumes reconcile);
rate columns come from build_pit_stats.stats_as_of dated to the fight --
enrich_roster's own per-fighter core with `when` set -- so the style layer's
volume and control branches are reachable, unlike the earlier starved
harnesses; reach imputation uses production's fit on the current roster
(static physicals); history_coverage is recomputed as decided-bouts-held /
record-claimed as of the fight. One deliberate divergence: production falls
back to today's scraped rates for fighters with no stat timeline, and today's
table is exactly what leaked in validate_method_sharpening's first attempt
(AUC 0.804 vs a production 0.720), so here a missing timeline gates the
terms off instead. That starves all arms identically.

RESULTS: NOTHING TO CHANGE. What ships is already the best arm.

FIDELITY ANCHOR FIRST, because two harnesses this week produced confident
nonsense before anyone checked one: 9,185 decided UFC bouts, AUC 0.6589,
accuracy 61.3% against a 50.7% trivial baseline, and 58.5% on the
production-like cut. The sibling full-model harness
(validate_probability_calibration) sits at 62.9%. Close to the sibling and
nowhere near the 0.804 a leaked harness read -- so neither leaking nor
starved, and the tables below are worth reading.

HELD OUT, scored on/after 2024-01-01, n=1,462 over 181 event clusters:

    arm            acc     brier    d.brier       p
    shipped     0.6696   0.20870        --      --
    connective  0.6696   0.20879   +0.00009   0.430
    w075        0.6676   0.20970   +0.00101   0.009
    w050        0.6648   0.21159   +0.00290   0.000
    w025        0.6525   0.21356   +0.00487   0.000
    ufconly     0.6450   0.21637   +0.00768   0.000

Positive d.brier is worse than what ships. Three readings, in order of how
much they matter:

  1. ADMITTING ONLY CONNECTIVE BOUTS BUYS NOTHING. +0.00009 at p = 0.430, and
     the same on the subset where it can bite at all (+0.00001, p = 0.908).
     The weight fit on the pre-2024 half even SELECTED this arm, and it still
     did not separate from shipped out of sample. That is the cleanest kind
     of negative: the arm chosen on the fit half failed on the held-out one.

  2. IT BUYS NOTHING BECAUSE IT IS ALMOST THE SAME ARM. 93% of non-UFC bouts
     are connective, so "keep the connective ones" and "keep all of them"
     differ on 174 rows out of 2,497. The hypothesis was worth testing and is
     nearly a no-op by construction.

  3. DOWN-WEIGHTING IS MONOTONICALLY WORSE, and the reverted filter is worst
     of all -- ufconly costs +0.00768 held out, +0.01005 where a corner has a
     regional bout. Commit 58d2a51a, which put regional bouts back at full
     weight, is confirmed at the FULL-MODEL level and not just at the raw Elo
     core that validate_spine_cleanup measured. That was the open question
     this file was written for, and the answer is that the current
     configuration is right.

WHAT THIS RULES OUT, which is the useful part. Salahdine Parnasse is rated
~64% to win a fight the market prices at -620 (86%), on 24 non-UFC bouts with
five opponents who have UFC records. The graph, the trust ramp, the streak
bonus and the debutant branch were the plausible mechanism, and they are
already configured optimally -- the gap is not there. Look elsewhere: the
contrarian post-mortem on the same day found the model's disagreements with
the market are where nearly all of its error lives, and a 22-point
disagreement is exactly that cohort.

Usage:  python3 scripts/validate_connective_bouts.py
        python3 scripts/validate_connective_bouts.py --since 2015-01-01
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem  # noqa: E402
from src.names import _normalize_name  # noqa: E402
from src.power_rating import (  # noqa: E402
    RATING_CENTER, DEBUT_RATING_SHRINK, compute_stats_rating, _streak_bonus,
    attach_imputed_reach,
)
from src.matchup_model import predict_matchup  # noqa: E402
from scripts.pit_roster import build_fight_index, roster_as_of  # noqa: E402
from scripts.build_pit_stats import (  # noqa: E402
    load_pit_stats, stats_as_of, LIVE_RATE_COLUMNS, _fold as pit_fold,
)
from scripts.harness_stats import (  # noqa: E402
    paired_signflip, corner_flip_key, trivial_baseline, score,
)

HISTORY = "data/fight_history.csv"
FIGHTERS = "data/fighters.csv"
RESULTS = "data/ufc_fight_results.csv"
WC_HISTORY = "data/fighter_weight_class_history.csv"

ARMS = ["shipped", "w075", "w050", "w025", "connective", "ufconly"]
REGIONAL_WEIGHT = {"shipped": 1.0, "w075": 0.75, "w050": 0.50, "w025": 0.25,
                   "ufconly": 0.0}


def _fold(n) -> str:
    return _normalize_name(n)


def _roster_key(n) -> str:
    """THE KEY roster_as_of ACTUALLY USES: plain strip().lower(), no accent
    fold. Keying static_rows with _normalize_name instead silently loses the
    roster row -- physicals, age, the subtracted record -- for every accented
    name, which is a nationality-shaped starvation."""
    return str(n).strip().lower()


class _WeightedElo(EloRatingSystem):
    """EloRatingSystem whose next update can be scaled by a bout weight.

    K is multiplied by the weight (so a half-weight bout moves ratings half
    as far) and the adaptive-K experience count is incremented by the weight
    rather than by one, so a career of half-weight bouts confers half the
    experience. At weight 1.0 this is byte-for-byte the production system.
    """

    def __init__(self):
        super().__init__()
        self.bout_weight = 1.0

    def _k_for(self, fighter: str) -> float:
        return super()._k_for(fighter) * self.bout_weight

    def update_weighted(self, winner, loser, method, weight):
        if weight <= 0.0:
            return
        self.bout_weight = weight
        self.update_ratings(winner, loser, method=method)
        self.bout_weight = 1.0
        if weight < 1.0:
            # update_ratings counted a whole bout of experience; correct it
            # to the fractional bout this arm admitted.
            for f in (winner, loser):
                self.fight_counts[f] -= (1.0 - weight)


def _ufc_name_set() -> set:
    """Folded names of everyone who appears in a UFC bout on record."""
    r = pd.read_csv(RESULTS)
    out = set()
    for bout in r["BOUT"].astype(str):
        for sep in (" vs. ", " vs "):
            if sep in bout:
                a, _, b = bout.partition(sep)
                out.add(_fold(a))
                out.add(_fold(b))
                break
    return out


def _streak_update(streaks: dict, a, b, w) -> None:
    """Mirror of power_rating._current_streaks' per-row behaviour, including
    its handling of a winner string that matches neither corner (that corner
    -- fighter_a -- has its streak reset). Faithful, not endorsed."""
    if not a or not b:
        return
    try:
        if not w:          # None / "" skip; NaN is truthy and falls through
            return
    except (TypeError, ValueError):
        pass
    loser = b if w == a else a
    streaks[w] = streaks.get(w, 0) + 1
    streaks[loser] = 0


def _effective(elo: _WeightedElo, n_prior: float, streak: int, name: str,
               stats_rating: float) -> float:
    """build_effective_ratings' per-fighter arithmetic, on this arm's state."""
    if n_prior <= 0:
        eff = RATING_CENTER + (stats_rating - RATING_CENTER) * DEBUT_RATING_SHRINK
    else:
        weight = min(1.0, n_prior / 4.0)
        eff = weight * elo.get_rating(name) + (1 - weight) * stats_rating
    return eff + _streak_bonus(n_prior, streak)


def _pit_row(name, when, fight_index, static_rows, timelines) -> dict:
    """A production-shaped roster row as of `when`.

    pit_roster supplies record/last-fight/physicals/age; stats_as_of supplies
    the rate columns enrich_roster would (dated, per-fighter fallback to
    nothing rather than to today's scrape -- see module docstring); coverage
    is recomputed point-in-time.
    """
    # today is left at its default (now) so _age_as_of actually walks age
    # back by the elapsed years. The sibling harnesses pass today=when, which
    # zeroes the elapsed time and scores every historical fighter at his
    # CURRENT age -- identical across arms here either way, but the default
    # is what pit_roster documents.
    row = roster_as_of(name, when, fight_index, static_rows)
    static = static_rows.get(_roster_key(name)) or {}

    wc = static.get("weight_class")
    if isinstance(wc, str) and wc.strip():
        row["weight_class"] = wc
    # Production's height->reach imputation, fitted once on the current
    # roster (static physicals are point-in-time safe). Only ever fills a
    # missing reach, same as attach_imputed_reach guarantees.
    if pd.isna(row.get("reach_in", np.nan)):
        imp = static.get("reach_in_imputed")
        if imp is not None and pd.notna(imp):
            row["reach_in_imputed"] = imp

    # stats_as_of compares datetime.date values, as enrich_roster calls it,
    # and load_pit_stats keys timelines with build_pit_stats' own fold.
    tl = timelines.get(pit_fold(name), [])
    s = stats_as_of(tl, when.date(), min_bouts=3)
    if s:
        for col in LIVE_RATE_COLUMNS:
            v = s.get(col)
            row[col] = v if v is not None else float("nan")
    if tl:
        # The sample-size gate's input, set directly from the timeline
        # rather than via roster_as_of's espn_timelines (whose lookup key
        # convention differs from load_pit_stats' fold).
        row["espn_fights"] = sum(1 for t in tl if t["date"].date() < when.date())

    held = sum(1 for d, won, _ in fight_index.get(str(name).strip().lower(), [])
               if d < when and won is not None)
    claimed = int(row.get("wins") or 0) + int(row.get("losses") or 0)
    row["history_coverage"] = (held / claimed) if claimed > 0 else float("nan")
    return row


def _auc(pairs) -> float:
    """Rank-based (Mann-Whitney) AUC with tie handling."""
    ps = [p for p, _ in pairs]
    ys = [y for _, y in pairs]
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    ranks = [0.0] * len(ps)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ps[order[j + 1]] == ps[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    npos = sum(1 for y in ys if y == 1.0)
    nneg = len(ys) - npos
    if not npos or not nneg:
        return float("nan")
    s = sum(ranks[i] for i, y in enumerate(ys) if y == 1.0)
    return (s - npos * (npos + 1) / 2.0) / (npos * nneg)


def _table(recs, label):
    n = len(recs)
    if n < 50:
        print(f"\n=== {label} ===  only {n} fights, skipping")
        return
    clusters = [r["cluster"] for r in recs]
    print(f"\n=== {label} ===  n={n}, {len(set(clusters))} event clusters")
    print(f"  {'arm':<12} {'acc':>7} {'brier':>9} {'logloss':>9} {'d.brier':>10} {'p':>7} {'deff':>6}")
    for arm in ARMS:
        pairs = [(r[arm], r["y"]) for r in recs]
        _, acc, br, ll = score(pairs)
        if arm == "shipped":
            print(f"  {arm:<12} {acc:7.4f} {br:9.5f} {ll:9.5f} {'--':>10} {'--':>7}")
            continue
        deltas = [(r[arm] - r["y"]) ** 2 - (r["shipped"] - r["y"]) ** 2 for r in recs]
        d, p, deff = paired_signflip(deltas, clusters=clusters)
        print(f"  {arm:<12} {acc:7.4f} {br:9.5f} {ll:9.5f} {d:>+10.5f} {p:7.3f} {deff:6.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="only score fights on/after this date")
    ap.add_argument("--cutoff", default="2024-01-01",
                    help="weight fit uses fights before this; held-out table after it")
    args = ap.parse_args()

    fighters = attach_imputed_reach(pd.read_csv(FIGHTERS))
    static_rows = {_roster_key(r["name"]): r.to_dict() for _, r in fighters.iterrows()}

    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    # kind="stable", same reason as elo.build_from_history: quicksort
    # reshuffles within a date and the replay is row order.
    history = (history.dropna(subset=["date"])
               .sort_values("date", kind="stable").reset_index(drop=True))
    fight_index = build_fight_index(history)
    wc_hist = pd.read_csv(WC_HISTORY) if os.path.exists(WC_HISTORY) else None
    timelines = load_pit_stats()

    ufcset = _ufc_name_set()
    promo = history["promotion"].fillna("").astype(str).str.strip()
    is_ufc_col = (promo.eq("") | promo.str.upper().eq("UFC")).to_numpy()

    # THE HYPOTHESIS'S OWN CENSUS, printed so the claim is checkable here.
    non_ufc = history[~is_ufc_col]
    both = one = neither = 0
    for r in non_ufc.itertuples(index=False):
        k = (_fold(r.fighter_a) in ufcset) + (_fold(r.fighter_b) in ufcset)
        both += k == 2
        one += k == 1
        neither += k == 0
    print(f"non-UFC bouts in the spine {len(non_ufc)}: both corners have a UFC "
          f"record {both}, exactly one {one}, neither {neither} "
          f"({(both + one) / max(len(non_ufc), 1):.0%} connective)")

    elo = {a: _WeightedElo() for a in ARMS}
    n_prior = {a: defaultdict(float) for a in ARMS}
    streaks = {a: {} for a in ARMS}
    nonufc_seen = defaultdict(int)

    dates = history["date"].to_numpy()
    recs, n_fail, n_mismatch = [], 0, 0

    for i, r in enumerate(history.itertuples(index=False)):
        a, b = str(r.fighter_a).strip(), str(r.fighter_b).strip()
        w = r.winner
        decided = pd.notna(w) and str(w).strip() != "" and str(w) in (a, b)
        is_ufc = bool(is_ufc_col[i])
        connective = is_ufc or (_fold(a) in ufcset) or (_fold(b) in ufcset)
        when = r.date

        # ---- SCORE FIRST, on the ratings as they stood before this bout ----
        if decided and is_ufc and (not args.since or str(when.date()) >= args.since):
            past = history.iloc[:int(np.searchsorted(dates, when.to_datetime64(),
                                                     side="left"))]
            row_a = _pit_row(a, when, fight_index, static_rows, timelines)
            row_b = _pit_row(b, when, fight_index, static_rows, timelines)
            stats_a = compute_stats_rating(pd.Series(row_a))
            stats_b = compute_stats_rating(pd.Series(row_b))
            eff = {arm: {a: _effective(elo[arm], n_prior[arm][a],
                                       streaks[arm].get(a, 0), a, stats_a),
                         b: _effective(elo[arm], n_prior[arm][b],
                                       streaks[arm].get(b, 0), b, stats_b)}
                   for arm in ARMS}
            try:
                res = predict_matchup(a, b, pd.DataFrame([row_a, row_b]),
                                      eff["shipped"], past, wc_hist, None,
                                      reference_date=when.date())
            except Exception:
                res = None
                n_fail += 1
            p_ship = (res or {}).get("prob_a")
            if p_ship is not None and not math.isnan(p_ship):
                adj = res["adjustment_layer_applied"]
                probs = {}
                for arm in ARMS:
                    gap = eff[arm][a] - eff[arm][b] + adj
                    probs[arm] = 1.0 / (1.0 + 10 ** (-gap / 400.0))
                # THE COMPOSITION MUST REPRODUCE PRODUCTION EXACTLY, or the
                # adjustment layer has grown a rating dependence and every
                # non-shipped arm is being scored on a model that does not
                # exist. Fail loudly rather than compare.
                if abs(probs["shipped"] - p_ship) > 1e-9:
                    n_mismatch += 1
                y = 1.0 if str(w) == a else 0.0
                flip = corner_flip_key(*sorted((_fold(a), _fold(b))), str(when.date()))
                rec = {"date": when, "cluster": str(when.date()),
                       "y": (1.0 - y) if flip else y,
                       "full": (_roster_key(a) in static_rows
                                and _roster_key(b) in static_rows),
                       "affected": nonufc_seen[a] > 0 or nonufc_seen[b] > 0}
                for arm in ARMS:
                    rec[arm] = (1.0 - probs[arm]) if flip else probs[arm]
                recs.append(rec)

        # ---- THEN UPDATE every arm's state with the rows it admits --------
        for arm in ARMS:
            wt = 1.0 if is_ufc else (
                (1.0 if connective else 0.0) if arm == "connective"
                else REGIONAL_WEIGHT[arm])
            if wt <= 0.0:
                continue
            # Trust count mirrors build_effective_ratings' value_counts over
            # the admitted frame: EVERY row counts, decided or not.
            n_prior[arm][a] += wt
            n_prior[arm][b] += wt
            if decided:
                method = getattr(r, "method", "DEC")
                method = method if isinstance(method, str) and method.strip() else "DEC"
                elo[arm].update_weighted(str(w), b if str(w) == a else a, method, wt)
            _streak_update(streaks[arm], a, b, w)
        if not is_ufc:
            nonufc_seen[a] += 1
            nonufc_seen[b] += 1

    if n_mismatch:
        print(f"\nFATAL: composed shipped probability disagreed with "
              f"predict_matchup on {n_mismatch} fights -- the adjustment layer "
              f"reads ratings now; every comparison above the shipped arm is void.")
        return 1
    if not recs:
        print("nothing scored")
        return 1

    # ---- FIDELITY ANCHOR, before any comparison is reported ---------------
    pairs = [(r["shipped"], r["y"]) for r in recs]
    n, acc, br, ll = score(pairs)
    full = [r for r in recs if r["full"]]
    fp = [(r["shipped"], r["y"]) for r in full]
    print(f"\nFIDELITY ANCHOR -- the shipped arm, before any comparison:")
    print(f"  scored {n} decided UFC bouts ({n_fail} predict failures skipped)")
    print(f"  AUC {_auc(pairs):.4f}   accuracy {acc:.1%}   "
          f"trivial baseline {trivial_baseline(pairs):.1%} (should be ~50%)")
    if len(fp) >= 100:
        _, facc, _, _ = score(fp)
        print(f"  production-like cut (both corners on current roster, n={len(fp)}): "
              f"AUC {_auc(fp):.4f}   accuracy {facc:.1%}")
    print("  anchors: sibling full-model harness 62.9% accuracy "
          "(validate_probability_calibration); live graded record 70.6% (72/102); "
          "a leaked harness once read AUC 0.804 against production's 0.720 "
          "(validate_method_sharpening). Materially above ~0.75 AUC here means "
          "leakage; far below the sibling means starvation -- stop and find it "
          "before trusting the tables below.")

    _table(recs, "ALL SCORED")
    _table([r for r in recs if r["affected"]],
           "WHERE IT BITES -- a corner with a prior regional bout")
    _table(full, "PRODUCTION-LIKE CUT -- both corners on the current roster")

    # ---- HELD OUT: weight chosen strictly before the cutoff ---------------
    cut = pd.Timestamp(args.cutoff)
    train = [r for r in recs if r["date"] < cut]
    test = [r for r in recs if r["date"] >= cut]
    if len(train) >= 200 and len(test) >= 200:
        fit_arms = ["shipped", "w075", "w050", "w025", "connective", "ufconly"]
        lls = {arm: score([(r[arm], r["y"]) for r in train])[3] for arm in fit_arms}
        best = min(lls, key=lls.get)
        print(f"\nweight fit on {len(train)} pre-{args.cutoff} fights: "
              + "  ".join(f"{a}={lls[a]:.5f}" for a in fit_arms))
        print(f"  selected: {best}"
              + ("  (the tuned arm IS the shipped control)" if best == "shipped" else ""))
        _table(test, f"HELD OUT -- scored on/after {args.cutoff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
