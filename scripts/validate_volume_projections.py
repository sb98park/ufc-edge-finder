"""
Does the duration-aware strike projection beat just quoting a career average?

THE CLAIM UNDER TEST. Projecting volume as

    per-MINUTE rate  x  expected fight MINUTES

should beat the naive thing a bettor can already do for free -- read a
fighter's career strikes-per-FIGHT off a stats page. The whole argument for
the per-minute detour is that a per-fight average confounds pace with fight
length: a finisher's totals are suppressed by the trait that makes him
dangerous. If that confound doesn't actually cost the naive number anything,
the detour is not worth the complexity and the section should not ship.

WHY THIS TEST CAN ACTUALLY ANSWER, unlike the win/loss ones. Every model idea
so far has been judged on fight outcomes, where one bit of information per
fight means five of six ideas died ambiguously and the survivor needed n=1747
plus a control sweep. Strike counts are CONTINUOUS: each fight carries real
information about how wrong a projection was, so a few hundred fights settle
it rather than hinting.

POINT-IN-TIME THROUGHOUT. Every rate is computed from fights strictly BEFORE
the one being predicted, using the same cache the stats backfill built. The
fight being predicted contributes nothing to its own inputs -- which is the
failure that makes backtest_model.py unusable and is worth restating because
it is invisible when it goes wrong.

Reports mean absolute error, median absolute error, and correlation for both
methods. Lower error and higher correlation win.

Usage (offline; needs the backfill cache):
    python3 scripts/validate_volume_projections.py
    python3 scripts/validate_volume_projections.py --min-prior-minutes 40
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import unicodedata
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prop_projections import expected_fight_minutes  # noqa: E402
from src.method_model import _ROUND_FINISH_SHARE  # noqa: E402

CACHE_DIR = "data/.espn_cache"
ID_MAP = "data/espn_athlete_ids.csv"
EVENTLOG = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/eventlog"


def _fold(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _cached(url: str):
    p = os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _stats_of(comp_ref: str) -> dict:
    d = _cached(comp_ref.split("?")[0].rstrip("/") + "/statistics")
    if not d:
        return {}
    cats = (d.get("splits") or {}).get("categories") or d.get("categories") or []
    return {s.get("name"): s.get("value") for c in cats for s in (c.get("stats") or [])
            if s.get("name") is not None and s.get("value") is not None}


def _minutes_of(comp_ref: str):
    st = _cached(comp_ref.split("/competitors/")[0].split("?")[0] + "/status")
    if not st:
        return None
    period, clock = st.get("period"), st.get("clock")
    if period is None or clock is None:
        return None
    try:
        return (int(period) - 1) * 5.0 + float(clock) / 60.0
    except (TypeError, ValueError):
        return None


def build_timelines(ids: dict) -> dict:
    """{folded name: [(date, sig_strikes_landed, minutes, opponent_id)]} oldest first."""
    tl = defaultdict(list)
    for name, aid in ids.items():
        log = _cached(EVENTLOG.format(id=aid))
        if not log:
            continue
        # ALL PAGES. The eventlog paginates at 25, so any fighter with a
        # longer career was silently truncated to his 25 most recent fights --
        # and since the oldest drop first, every timeline built here started
        # mid-career. Any conclusion drawn from a truncated history is drawn
        # from a different fighter.
        _ev = log.get("events") or {}
        _items = list(_ev.get("items") or [])
        try:
            _pages = int(_ev.get("pageCount") or 1)
        except (TypeError, ValueError):
            _pages = 1
        for _pg in range(2, _pages + 1):
            _more = _cached(EVENTLOG.format(id=aid) + f"?page={_pg}")
            _items += ((_more or {}).get("events") or {}).get("items") or []
        for e in _items:
            if not e.get("played"):
                continue
            cr = (e.get("competitor") or {}).get("$ref")
            er = (e.get("event") or {}).get("$ref")
            if not cr or not er:
                continue
            ev = _cached(er)
            ds = (ev or {}).get("date")
            if not ds:
                continue
            try:
                when = dt.datetime.fromisoformat(ds.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                continue
            st = _stats_of(cr)
            mins = _minutes_of(cr)
            ssl = st.get("sigStrikesLanded")
            if ssl is None or mins is None or mins <= 0:
                continue
            # The OPPONENT's id, so the expected duration can depend on who
            # is across the cage. Without it the projection collapses into
            # the naive average -- see the note in main().
            opp_id = None
            clist = _cached(cr.split("/competitors/")[0].split("?")[0] + "/competitors")
            for item in (clist or {}).get("items") or []:
                ref = item.get("$ref", "")
                if f"/competitors/{aid}" not in ref and "/competitors/" in ref:
                    opp_id = ref.split("/competitors/")[1].split("?")[0].rstrip("/")
                    break
            tl[name].append((when, float(ssl), float(mins), opp_id))
    for n in tl:
        tl[n].sort(key=lambda t: t[0])
    return tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prior-minutes", type=float, default=25.0)
    args = ap.parse_args()

    if not os.path.isdir(CACHE_DIR):
        print(f"No {CACHE_DIR}. Run scripts/backfill_espn_fight_stats.py first.")
        sys.exit(1)

    ids = {_fold(r["name"]): str(r["espn_id"]) for _, r in pd.read_csv(ID_MAP).iterrows()}
    print(f"building timelines from cache for {len(ids)} fighters...")
    tl = build_timelines(ids)
    print(f"  {len(tl)} fighters with usable timelines "
          f"({sum(len(v) for v in tl.values())} fights)\n")

    shares3 = _ROUND_FINISH_SHARE[3]
    # id -> folded name, so an opponent's competitor id can reach their timeline.
    by_id = {str(v): k for k, v in ids.items()}

    def prior_finish_rate(fighter_name, before):
        """How often this fighter's PRIOR bouts ended early. None if unknown."""
        hist = [f for f in tl.get(fighter_name, []) if f[0] < before]
        if not hist:
            return None
        return sum(1 for f in hist if f[2] < 14.5) / len(hist)

    rows = []
    for name, fights in tl.items():
        for i, (when, actual_ssl, actual_min, opp_id) in enumerate(fights):
            prior = fights[:i]
            prior_min = sum(f[2] for f in prior)
            if prior_min < args.min_prior_minutes:
                continue
            prior_ssl = sum(f[1] for f in prior)

            # NAIVE: the number a stats page gives you -- career strikes per
            # FIGHT, no duration correction at all.
            naive = prior_ssl / len(prior)

            # PROJECTION: per-minute rate x expected minutes, where the
            # expected duration depends on BOTH fighters.
            #
            # This is the whole feature. Using only this fighter's own prior
            # finish rate makes the projection algebraically IDENTICAL to the
            # naive career average -- rate x mean(own durations) is exactly
            # strikes-per-fight -- so it could not possibly win. The duration
            # correction only earns anything where the expected length of THIS
            # fight differs from the fighter's own norm, which is precisely
            # what the opponent contributes: a durable decision fighter
            # lengthens it, a finisher shortens it.
            rate = prior_ssl / prior_min
            own_fr = sum(1 for f in prior if f[2] < 14.5) / len(prior)
            opp_fr = prior_finish_rate(by_id.get(str(opp_id), ""), when) if opp_id else None
            # A fight ends early if EITHER man ends it, so combine as the
            # complement of neither finishing rather than by averaging.
            finish_prob = own_fr if opp_fr is None else 1 - (1 - own_fr) * (1 - opp_fr)
            exp_min = expected_fight_minutes(finish_prob, shares3, 3)
            projected = rate * exp_min

            # The RANGE is what the display actually claims, so it is what
            # must be checked. Point-estimate accuracy is dominated by
            # duration variance no method resolves -- a round-one finish
            # against a decision is a 4x swing -- so MAE mostly measures the
            # sport, not the model. A 20-80 interval that contains the truth
            # ~60% of the time is an honest interval; one that contains it 30%
            # of the time is lying to a bettor sizing a position.
            exp_min_lo = expected_fight_minutes(min(finish_prob + 0.25, 1.0), shares3, 3)
            exp_min_hi = expected_fight_minutes(max(finish_prob - 0.25, 0.0), shares3, 3)
            lo, hi = rate * exp_min_lo, rate * exp_min_hi
            rows.append((actual_ssl, naive, projected, min(lo, hi), max(lo, hi)))

    if not rows:
        print("No scorable fights. Is the cache populated?")
        sys.exit(1)

    n = len(rows)
    def stats(idx):
        errs = [abs(a - p[idx]) for a, *p in [(r[0], r[1], r[2]) for r in rows]]
        return errs
    naive_err = [abs(a - nv) for a, nv, _, _, _ in rows]
    proj_err = [abs(a - pr) for a, _, pr, _, _ in rows]

    def corr(xs, ys):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        return num / (dx * dy) if dx and dy else 0.0

    actual = [r[0] for r in rows]
    print(f"{'method':<34}{'MAE':>9}{'median AE':>12}{'corr':>9}")
    print("-" * 64)
    for label, errs, preds in (
        ("naive: career strikes per fight", naive_err, [r[1] for r in rows]),
        ("projection: rate x E[minutes]", proj_err, [r[2] for r in rows]),
    ):
        mae = sum(errs) / n
        med = sorted(errs)[n // 2]
        print(f"{label:<34}{mae:>9.2f}{med:>12.2f}{corr(actual, preds):>9.3f}")

    better = sum(1 for pe, ne in zip(proj_err, naive_err) if pe < ne)
    print(f"\nn = {n} fights")
    print(f"projection closer on {better}/{n} ({better/n:.1%}) of fights")

    inside = sum(1 for a, _, _, lo, hi in rows if lo <= a <= hi)
    print(f"\nRANGE CALIBRATION (what the UI actually claims)")
    print(f"  actual inside the stated range: {inside}/{n} ({inside/n:.1%})")
    print(f"  an honest 20-80 style interval should land near 60%.")
    if inside / n < 0.45:
        print("  READ: the range is TOO NARROW -- it will read as confident and be wrong.")
    elif inside / n > 0.80:
        print("  READ: the range is TOO WIDE -- technically honest, but so loose it")
        print("        cannot help anyone decide against a line.")
    else:
        print("  READ: the interval is roughly honest, which is the property that")
        print("        matters for a number shown next to a book's line.")

    d = [ne - pe for ne, pe in zip(naive_err, proj_err)]
    md = sum(d) / n
    sd = (sum((x - md) ** 2 for x in d) / max(n - 1, 1)) ** 0.5
    se = sd / (n ** 0.5)
    t = md / se if se else 0.0
    print(f"paired mean error reduction {md:+.3f} strikes  (SE {se:.3f}, t {t:+.2f})")
    if t >= 2:
        print("READ: the duration correction earns its place.")
    elif t <= -2:
        print("READ: the duration correction makes projections WORSE. Do not ship it.")
    else:
        print("READ: no distinguishable difference. The per-minute detour is not "
              "paying for itself on this sample -- which is a reason not to ship "
              "the section, not a reason to look for a better statistic.")


if __name__ == "__main__":
    main()
