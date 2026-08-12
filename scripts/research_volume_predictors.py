"""
Which strike-volume predictor actually works? Point-in-time, several candidates.

WHY THIS EXISTS. The first attempt tested ONE predictor -- per-minute rate x
expected duration -- found it no better than a career average (corr 0.257 vs
0.293, MAE 31.2 vs 30.8), and I generalised that into "strike volume is
unpredictable". That was too broad. It showed the DURATION correction doesn't
pay; it said nothing about the levers we deliberately deferred.

The obvious untested one is the opponent. A high-output striker facing someone
who smothers and grapples lands far fewer; the first test adjusted DURATION
for the opponent but left the RATE opponent-blind, which is where the effect
should live. Recency weighting -- the one validated model improvement this
project has, worth +1.37pp on picks -- also went unapplied.

CANDIDATES, all scored identically on the same fights:
  1  career strikes per FIGHT              the number a stats page gives you
  2  rate x E[minutes]                     the already-failed version, for continuity
  3  recency-weighted rate x E[minutes]    does recency help here as it did on picks?
  4  opponent-adjusted rate x E[minutes]   scale by how much the opponent absorbs
  5  recency + opponent                    both levers together

The opponent adjustment is a ratio to the league norm, the standard form:

    expected_rate = own_rate x (opponent_absorbed_rate / league_mean_absorbed)

so facing someone who gets hit 20% more than average scales output up 20%.

THE BAR. Beat candidate 1 on CORRELATION, which is what matters for a market:
ranking fights correctly is most of the job, and MAE is dominated by duration
variance no method resolves. A correlation that doesn't clearly exceed 0.293
means the feature does not ship, and that is a real answer rather than a
prompt to keep hunting for a statistic that flatters it.
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

# Same 18-month half-life as the validated recency work. Reusing the swept
# value rather than introducing a second decay rate to defend.
HALF_LIFE_DAYS = 548.0


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


def _stats_of(ref: str) -> dict:
    d = _cached(ref.split("?")[0].rstrip("/") + "/statistics")
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
    """{name: [(date, own_landed, absorbed, minutes, opp_id)]} oldest first."""
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
            mine = _stats_of(cr)
            mins = _minutes_of(cr)
            ssl = mine.get("sigStrikesLanded")
            if ssl is None or mins is None or mins <= 0:
                continue
            opp_ref, opp_id = None, None
            clist = _cached(cr.split("/competitors/")[0].split("?")[0] + "/competitors")
            for item in (clist or {}).get("items") or []:
                ref = item.get("$ref", "")
                if f"/competitors/{aid}" not in ref and "/competitors/" in ref:
                    opp_ref = ref
                    opp_id = ref.split("/competitors/")[1].split("?")[0].rstrip("/")
                    break
            absorbed = (_stats_of(opp_ref).get("sigStrikesLanded") if opp_ref else None) or 0.0
            tl[name].append((when, float(ssl), float(absorbed), float(mins), opp_id))
    for n in tl:
        tl[n].sort(key=lambda t: t[0])
    return tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prior-minutes", type=float, default=25.0)
    args = ap.parse_args()

    if not os.path.isdir(CACHE_DIR):
        print(f"No {CACHE_DIR}. Run the backfill first.")
        sys.exit(1)

    ids = {_fold(r["name"]): str(r["espn_id"]) for _, r in pd.read_csv(ID_MAP).iterrows()}
    print(f"building timelines for {len(ids)} fighters...")
    tl = build_timelines(ids)
    by_id = {str(v): k for k, v in ids.items()}
    print(f"  {len(tl)} usable ({sum(len(v) for v in tl.values())} fights)\n")

    # League mean absorbed-per-minute, the denominator of the opponent ratio.
    tot_abs = sum(f[2] for v in tl.values() for f in v)
    tot_min = sum(f[3] for v in tl.values() for f in v)
    league_abs_rate = tot_abs / tot_min if tot_min else 1.0
    print(f"league mean absorbed/min: {league_abs_rate:.3f}\n")

    shares3 = _ROUND_FINISH_SHARE[3]

    def prior_rates(name, before):
        """(landed/min, absorbed/min, recency landed/min, finish rate, n) from prior fights."""
        hist = [f for f in tl.get(name, []) if f[0] < before]
        if not hist:
            return None
        mins = sum(f[3] for f in hist)
        if mins < args.min_prior_minutes:
            return None
        landed = sum(f[1] for f in hist)
        absorbed = sum(f[2] for f in hist)
        wl = wm = 0.0
        for when, l, a, m, _ in hist:
            w = 0.5 ** (max((before - when).days, 0) / HALF_LIFE_DAYS)
            wl += l * w
            wm += m * w
        return (landed / mins, absorbed / mins,
                (wl / wm) if wm > 0 else landed / mins,
                sum(1 for f in hist if f[3] < 14.5) / len(hist), len(hist))

    preds = defaultdict(list)
    actuals = []
    for name, fights in tl.items():
        for i, (when, actual, _absorbed, _mins, opp_id) in enumerate(fights):
            me = prior_rates(name, when)
            if me is None:
                continue
            own_rate, _own_abs, own_recent, own_fr, n_prior = me

            opp_name = by_id.get(str(opp_id), "")
            opp = prior_rates(opp_name, when) if opp_name else None
            opp_fr = opp[3] if opp else None
            opp_abs_rate = opp[1] if opp else None

            fp = own_fr if opp_fr is None else 1 - (1 - own_fr) * (1 - opp_fr)
            exp_min = expected_fight_minutes(fp, shares3, 3)

            # Opponent scaling: how hittable is this specific opponent,
            # relative to the league. Clamped so one freakish opponent line
            # cannot triple a projection.
            if opp_abs_rate and league_abs_rate:
                ratio = max(0.6, min(1.6, opp_abs_rate / league_abs_rate))
            else:
                ratio = 1.0

            hist_prior = [f for f in tl[name] if f[0] < when]
            preds["1 career strikes per fight"].append(
                sum(f[1] for f in hist_prior) / len(hist_prior))
            preds["2 rate x E[min]"].append(own_rate * exp_min)
            preds["3 recency rate x E[min]"].append(own_recent * exp_min)
            preds["4 opponent-adj rate x E[min]"].append(own_rate * ratio * exp_min)
            preds["5 recency + opponent"].append(own_recent * ratio * exp_min)
            actuals.append(actual)

    n = len(actuals)
    if not n:
        print("No scorable fights.")
        sys.exit(1)

    def corr(xs, ys):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        return num / (dx * dy) if dx and dy else 0.0

    print(f"{'predictor':<32}{'MAE':>9}{'medAE':>9}{'corr':>9}")
    print("-" * 60)
    base_corr = None
    for label in sorted(preds):
        p = preds[label]
        errs = [abs(a - x) for a, x in zip(actuals, p)]
        c = corr(actuals, p)
        if label.startswith("1 "):
            base_corr = c
        print(f"{label:<32}{sum(errs)/n:>9.2f}{sorted(errs)[n//2]:>9.2f}{c:>9.3f}")

    print(f"\nn = {n} fights")
    best = max(preds, key=lambda k: corr(actuals, preds[k]))
    bc = corr(actuals, preds[best])
    print(f"best: {best}  (corr {bc:.3f} vs baseline {base_corr:.3f})")
    if bc > base_corr + 0.05:
        print("READ: a real improvement over the stats-page number. Worth revisiting "
              "the feature, with an EMPIRICAL range built from these residuals.")
    else:
        print("READ: nothing beats the career average by enough to matter. The "
              "feature does not ship -- and the levers we deferred were not the "
              "missing piece, which is the answer we came for.")


if __name__ == "__main__":
    main()
