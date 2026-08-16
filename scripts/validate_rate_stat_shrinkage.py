"""
POINT-IN-TIME validation of the rate-statistic shrinkage in the style layer.

THE CHANGE UNDER TEST. style_matchup_adjustment now scales its striking and
wrestling terms by rate_stat_confidence(), which ramps from 0 to 1 as the
THINNER corner's tracked cage time approaches STYLE_FULL_TRUST_FIGHTS.
Everything else -- height, age, layoff, short notice, durability, submission
threat -- is untouched, because those are biographical facts or record
counts rather than per-minute rates.

WHY IT WAS PROPOSED. On UFC 330 the model made three predictions that
disagreed with the market and lost all three. Two were built on almost no
cage time: Kaue Fernandes (4 fights, 38 min) at 70% against a market 40%,
and Eduardo Chapolin (1 fight, 15 min) at 64% against 46%. Rate stats over
a short sample are not merely uncertain, they are EXTREME -- a fighter who
has not yet had a bad night reads as untouchable -- and the style layer
rewarded exactly that.

WHY THIS HARNESS AND NOT A CARD-BY-CARD READ. Three fights cannot
distinguish a real defect from a bad night, and tuning a model on the card
that annoyed you is how overfitting starts. This borrows
validate_pointintime_stats.py's method wholesale: walk history forward,
predict each fight using only what was known the night before, never let a
fight inform its own prediction.

Both arms see IDENTICAL inputs. They differ only in
matchup_model.STYLE_FULL_TRUST_FIGHTS:

    A. control  -- set to 0, which makes rate_stat_confidence() return 1.0
                   and reproduces the model exactly as it shipped.
    B. shrunk   -- the real value.

Paired on the same fights, so the difference is the change and nothing else.

CAGE TIME IS RECONSTRUCTED POINT-IN-TIME TOO. Reading fight_minutes_total
off today's fighters.csv would leak the future into the very term being
tested: a fighter thin at the time of a 2021 fight is not thin now. Minutes
are summed from the fights strictly before each bout, from the same cached
ESPN responses everything else here uses.

Usage:  python3 scripts/validate_rate_stat_shrinkage.py
        python3 scripts/validate_rate_stat_shrinkage.py --min-prior-fights 5
        python3 scripts/validate_rate_stat_shrinkage.py --sweep
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.elo import EloRatingSystem  # noqa: E402
from src import matchup_model  # noqa: E402
from src.matchup_model import predict_matchup  # noqa: E402
from scripts.validate_pointintime_stats import (  # noqa: E402
    _cached, _fold, _stats_of, build_timelines, stats_as_of,
    CACHE_DIR, ID_MAP, FIGHTERS, HISTORY,
)


def _minutes_of(comp_ref: str) -> float:
    """
    Length of one fight in minutes, from the cached status document.

    The clock counts UP (verified in backfill_espn_fight_stats._fight_minutes:
    two three-round decisions both return period=3, clock=300.0). Getting this
    backwards would turn a round-one finish into 0.8 minutes instead of 4.2.

    Returns 0.0 when the status was never cached, which is honest: an
    uncounted fight makes a fighter look THINNER, so the shrinkage errs
    toward trusting the style layer less. That is the safe direction for a
    change whose entire purpose is to stop over-trusting thin samples.
    """
    competition = comp_ref.split("/competitors/")[0].split("?")[0]
    st = _cached(competition + "/status")
    if not st:
        return 0.0
    period, clock = st.get("period"), st.get("clock")
    if period is None or clock is None:
        return 0.0
    try:
        return (int(period) - 1) * 5.0 + float(clock) / 60.0
    except (TypeError, ValueError):
        return 0.0


def build_minute_timelines(ids: dict) -> dict:
    """{folded_name: [(date, minutes), ...]} sorted, for point-in-time sums."""
    eventlog_tpl = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/eventlog"
    out = defaultdict(list)
    for name, aid in ids.items():
        log = _cached(eventlog_tpl.format(id=aid))
        if not log:
            continue
        ev = log.get("events") or {}
        items = list(ev.get("items") or [])
        try:
            pages = int(ev.get("pageCount") or 1)
        except (TypeError, ValueError):
            pages = 1
        for pg in range(2, pages + 1):
            more = _cached(eventlog_tpl.format(id=aid) + f"?page={pg}")
            items += ((more or {}).get("events") or {}).get("items") or []
        for entry in items:
            if not entry.get("played"):
                continue
            comp_ref = (entry.get("competitor") or {}).get("$ref")
            ev_ref = (entry.get("event") or {}).get("$ref")
            if not comp_ref or not ev_ref:
                continue
            evd = _cached(ev_ref)
            date_s = (evd or {}).get("date")
            if not date_s:
                continue
            try:
                import datetime as dt
                when = dt.datetime.fromisoformat(date_s.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                continue
            out[name].append((when, _minutes_of(comp_ref)))
    for n in out:
        out[n].sort(key=lambda t: t[0])
    return out


def minutes_as_of(timeline, when) -> float:
    return sum(m for d, m in timeline if d < when)


def _paired_brier_test(rows, t, n_boot=4000, seed=12345):
    """
    Paired bootstrap on the PER-FIGHT change in squared error.

    A Brier delta of -0.0009 on 377 fights is not self-evidently signal, and
    the two arms are scored on identical fights -- so the honest test is
    paired, resampling fights rather than treating the arms as independent
    samples. Returns (mean_delta, p_two_sided).

    Deterministic seed: a validation number that moves between runs is not a
    validation number.
    """
    import random
    deltas = [ (pr[t] - y) ** 2 - (pr[0.0] - y) ** 2 for _, y, pr in rows ]
    n = len(deltas)
    if n == 0:
        return 0.0, 1.0
    obs = sum(deltas) / n
    rnd = random.Random(seed)
    # Null: the change has no effect, so a fight's delta is equally likely to
    # have carried the opposite sign. Flip signs rather than resample values.
    hits = 0
    for _ in range(n_boot):
        s = 0.0
        for d in deltas:
            s += d if rnd.random() < 0.5 else -d
        if abs(s / n) >= abs(obs):
            hits += 1
    return obs, hits / n_boot


def _score(pairs):
    n = len(pairs)
    acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in pairs) / n
    return n, acc, brier, ll


def run(threshold_values, min_prior_fights):
    ids = {_fold(r["name"]): str(r["espn_id"]) for _, r in pd.read_csv(ID_MAP).iterrows()}
    print(f"building stat timelines from cache for {len(ids)} fighters...")
    timelines = build_timelines(ids)
    print(f"  {len(timelines)} usable stat timelines")
    print()

    fighters = pd.read_csv(FIGHTERS)
    rows_by_name = {_fold(r["name"]): r for _, r in fighters.iterrows()}
    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")

    elo = EloRatingSystem()
    counts = defaultdict(int)
    # One record per scored fight: (thinner corner's minutes, outcome,
    # {threshold: predicted_prob}). Keeping every arm's prediction on the
    # same row is what lets the subsets below be sliced identically -- a
    # comparison where the two arms see different fights is not a comparison.
    records = []

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()

        if (counts[fa] >= min_prior_fights and counts[fb] >= min_prior_fights
                and fa in timelines and fb in timelines):
            base_a = rows_by_name[fa].to_dict() if fa in rows_by_name else {"name": a}
            base_b = rows_by_name[fb].to_dict() if fb in rows_by_name else {"name": b}
            eff = {a: elo.get_rating(a), b: elo.get_rating(b)}
            sa, sb = stats_as_of(timelines[fa], when), stats_as_of(timelines[fb], when)
            # PRIOR FIGHT COUNT, reconstructed exactly: entries in the
            # stat timeline strictly before this bout. Unlike duration this
            # needs no extra endpoint, so it is complete rather than 27%
            # populated -- see the note on STYLE_FULL_TRUST_FIGHTS.
            ma = sum(1 for d, _ in timelines[fa] if d < when)
            mb = sum(1 for d, _ in timelines[fb] if d < when)

            ra, rb = dict(base_a), dict(base_b)
            for col in ("strike_accuracy_pct", "td_accuracy_pct", "td_defense_pct"):
                ra[col] = None
                rb[col] = None
            ra.update(sa)
            rb.update(sb)
            # POINT-IN-TIME cage time overwrites the roster's current value.
            ra["espn_fights"] = ma
            rb["espn_fights"] = mb
            y = 1.0 if winner == a else 0.0
            frame = pd.DataFrame([ra, rb])

            saved = matchup_model.STYLE_FULL_TRUST_FIGHTS
            probs = {}
            try:
                for t in threshold_values:
                    matchup_model.STYLE_FULL_TRUST_FIGHTS = t
                    try:
                        res = predict_matchup(a, b, frame, eff)
                    except Exception:
                        res = None
                    p = (res or {}).get("prob_a")
                    if p is not None:
                        probs[t] = p
            finally:
                matchup_model.STYLE_FULL_TRUST_FIGHTS = saved
            # Only fights every arm could predict, or the arms are scored on
            # different populations.
            if len(probs) == len(threshold_values):
                records.append((min(ma, mb), y, probs))

        loser = b if winner == a else a
        elo.update_ratings(winner, loser, method=method)
        counts[fa] += 1
        counts[fb] += 1

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prior-fights", type=int, default=3)
    ap.add_argument("--sweep", action="store_true",
                    help="try several thresholds rather than just control vs 90")
    args = ap.parse_args()

    if not os.path.isdir(CACHE_DIR):
        print(f"No {CACHE_DIR}. Run scripts/backfill_espn_fight_stats.py first.")
        sys.exit(1)

    thresholds = [0.0, 3.0, 6.0, 10.0, 15.0] if args.sweep else [0.0, 6.0]
    records = run(thresholds, args.min_prior_fights)
    if not records:
        print("No scorable fights -- is the ESPN cache populated?")
        sys.exit(1)

    def table(rows, title):
        print(f"\n{title}  (n={len(rows)})")
        print(f"  {'STYLE_FULL_TRUST_FIGHTS':<28}{'accuracy':>10}{'Brier':>10}{'log loss':>11}")
        print("  " + "-" * 59)
        base = None
        for t in thresholds:
            pairs = [(pr[t], y) for _, y, pr in rows]
            n, acc, brier, ll = _score(pairs)
            label = f"{t:g}" + ("  (control, current)" if t == 0 else "")
            print(f"  {label:<28}{acc:>9.1%}{brier:>10.4f}{ll:>11.4f}")
            if t == 0:
                base = (acc, brier, ll)
        if base:
            for t in thresholds:
                if t == 0:
                    continue
                pairs = [(pr[t], y) for _, y, pr in rows]
                _, acc, brier, ll = _score(pairs)
                verdict = "BETTER" if brier < base[1] else ("no change" if brier == base[1] else "WORSE")
                delta, p = _paired_brier_test(rows, t)
                sig = "significant" if p < 0.05 else "NOT significant"
                print(f"    {t:g} vs control: acc {acc-base[0]:+.2%}  "
                      f"Brier {brier-base[1]:+.4f}  log loss {ll-base[2]:+.4f}   -> {verdict}"
                      f"   [paired p={p:.3f}, {sig}]")

    table(records, "ALL SCORED FIGHTS")

    # THE SUBSET THE CHANGE IS ACTUALLY ABOUT. On fights where both corners
    # are experienced the weight is 1.0 and every arm is identical, so
    # including them dilutes a real effect toward zero and would hide it.
    # Sliced on the SAME rows for every arm.
    for cut in (3.0, 6.0, 10.0):
        thin = [r for r in records if r[0] < cut]
        if thin:
            table(thin, f"ONLY FIGHTS WITH A CORNER UNDER {cut:g} PRIOR FIGHTS")


if __name__ == "__main__":
    main()
