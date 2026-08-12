"""
POINT-IN-TIME validation of the striking / takedown terms.

THE THING THIS PROJECT COULD NEVER DO UNTIL NOW. Both existing harnesses say
so themselves: backtest_model.py opens by admitting it predicts historical
fights using TODAY's fighters.csv, so every prediction already knows that
fight's outcome; walkforward_backtest.py is honest but validates only the Elo
core, and its docstring names the blocker -- the style layer needs "career
stats as they stood at fight time", called "a future data project".

ESPN's per-fight statistics are that project's missing piece. Because each
bout's numbers arrive separately WITH a date, a fighter's striking accuracy
as of any past night is just the fights before it. Nothing has to be
reconstructed or assumed. And scripts/backfill_espn_fight_stats.py has
already cached every one of those responses, so this runs offline.

WHAT IT COMPARES, and why that is the right question. Not "is the model any
good" -- that is already established (adjustments 57.7% vs Elo-only 55.9%).
The open question is narrower: do the ESPN striking/takedown columns EARN
their place, now that they are the only thing standing between the model and
the hardcoded 45 / 20 / 65 defaults? So two runs over identical fights:

    A. stats WITHHELD  -- both corners blank, so the both-corners gating in
       matchup_model zeroes those terms. This is what the model has actually
       been doing for months.
    B. stats POINT-IN-TIME -- each fighter's numbers as of the night before.

Same Elo, same fights, same everything else. The gap between A and B is what
the columns are worth, measured rather than assumed.

READ THE RESULT HONESTLY. A Brier improvement of a few thousandths is noise,
not a win -- this project has killed five ideas on exactly that distinction,
and the one that survived (recency weighting) moved Brier 0.2345 -> 0.2329
with a monotonic control. If B does not clearly beat A, the columns are
decoration and the honest move is to say so.

Usage (offline; run the backfill first so the cache is warm):
    python3 scripts/validate_pointintime_stats.py
    python3 scripts/validate_pointintime_stats.py --min-prior-fights 3
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
import unicodedata
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.elo import EloRatingSystem  # noqa: E402
from src.matchup_model import predict_matchup  # noqa: E402

CACHE_DIR = "data/.espn_cache"
ID_MAP = "data/espn_athlete_ids.csv"
HISTORY = "data/fight_history.csv"
FIGHTERS = "data/fighters.csv"

# Same floors as the backfill. A percentage off a tiny denominator is noise
# wearing a number's clothes, and letting it in here would flatter the very
# thing being tested.
MIN_SIG_STRIKES_ATT = 100
MIN_TD_ATT = 5
MIN_TD_ATT_FACED = 5


def _fold(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _cached(url: str):
    import hashlib
    p = os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _stats_of(comp_ref: str) -> dict:
    data = _cached(comp_ref.split("?")[0].rstrip("/") + "/statistics")
    if not data:
        return {}
    cats = (data.get("splits") or {}).get("categories") or data.get("categories") or []
    return {s.get("name"): float(s.get("value"))
            for c in cats for s in (c.get("stats") or [])
            if s.get("name") is not None and s.get("value") is not None}


def build_timelines(ids: dict) -> dict:
    """
    {folded name: [(date, per-fight counters), ...]} sorted oldest first.

    Cache-only: a fighter whose eventlog was never fetched simply has no
    timeline and falls back to blank stats, which the gating then treats as
    "say nothing" -- exactly the production behaviour.
    """
    timelines = defaultdict(list)
    eventlog_tpl = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/eventlog"
    for name, aid in ids.items():
        log = _cached(eventlog_tpl.format(id=aid))
        if not log:
            continue
        # ALL PAGES. The eventlog paginates at 25 and drops the OLDEST fights
        # first, so every timeline replayed here began mid-career -- which for
        # a point-in-time harness is the one thing that must not happen: the
        # "prior fights" a prediction is scored against were incomplete in a
        # way that grows with a fighter's experience.
        _ev = log.get("events") or {}
        _items = list(_ev.get("items") or [])
        try:
            _pages = int(_ev.get("pageCount") or 1)
        except (TypeError, ValueError):
            _pages = 1
        for _pg in range(2, _pages + 1):
            _more = _cached(eventlog_tpl.format(id=aid) + f"?page={_pg}")
            _items += ((_more or {}).get("events") or {}).get("items") or []
        for entry in _items:
            if not entry.get("played"):
                continue
            comp_ref = (entry.get("competitor") or {}).get("$ref")
            ev_ref = (entry.get("event") or {}).get("$ref")
            if not comp_ref or not ev_ref:
                continue
            ev = _cached(ev_ref)
            date_s = (ev or {}).get("date")
            if not date_s:
                continue
            try:
                when = dt.datetime.fromisoformat(date_s.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                continue
            mine = _stats_of(comp_ref)
            if not mine:
                continue
            # Opponent's takedowns = this fighter's takedown defence.
            opp = {}
            comp_url = comp_ref.split("/competitors/")[0].split("?")[0]
            clist = _cached(comp_url + "/competitors")
            for item in (clist or {}).get("items") or []:
                ref = item.get("$ref", "")
                if f"/competitors/{aid}" not in ref:
                    opp = _stats_of(ref)
                    break
            timelines[name].append((when, {
                "ssl": mine.get("sigStrikesLanded", 0.0),
                "ssa": mine.get("sigStrikesAttempted", 0.0),
                "tdl": mine.get("takedownsLanded", 0.0),
                "tda": mine.get("takedownsAttempted", 0.0),
                "opp_tdl": opp.get("takedownsLanded", 0.0),
                "opp_tda": opp.get("takedownsAttempted", 0.0),
            }))
    for n in timelines:
        timelines[n].sort(key=lambda t: t[0])
    return timelines


def stats_as_of(timeline, when) -> dict:
    """Aggregate ONLY fights strictly before `when`. This is the whole point."""
    tot = defaultdict(float)
    for date, c in timeline:
        if date >= when:
            break
        for k, v in c.items():
            tot[k] += v
    out = {}
    if tot["ssa"] >= MIN_SIG_STRIKES_ATT:
        out["strike_accuracy_pct"] = tot["ssl"] / tot["ssa"] * 100
    if tot["tda"] >= MIN_TD_ATT:
        out["td_accuracy_pct"] = tot["tdl"] / tot["tda"] * 100
    if tot["opp_tda"] >= MIN_TD_ATT_FACED:
        out["td_defense_pct"] = 100 - tot["opp_tdl"] / tot["opp_tda"] * 100
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prior-fights", type=int, default=3)
    args = ap.parse_args()

    if not os.path.isdir(CACHE_DIR):
        print(f"No {CACHE_DIR}. Run scripts/backfill_espn_fight_stats.py first.")
        sys.exit(1)

    ids = {_fold(r["name"]): str(r["espn_id"]) for _, r in pd.read_csv(ID_MAP).iterrows()}
    print(f"building stat timelines from cache for {len(ids)} fighters...")
    timelines = build_timelines(ids)
    print(f"  {len(timelines)} fighters have a usable timeline "
          f"({sum(len(v) for v in timelines.values())} fights)\n")

    fighters = pd.read_csv(FIGHTERS)
    rows_by_name = {_fold(r["name"]): r for _, r in fighters.iterrows()}

    history = pd.read_csv(HISTORY)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date")

    elo = EloRatingSystem()
    counts = defaultdict(int)
    scored = {"withheld": [], "pointintime": []}
    paired = []
    preds = {}

    for _, f in history.iterrows():
        a, b, winner, method = f["fighter_a"], f["fighter_b"], f["winner"], f["method"]
        fa, fb = _fold(a), _fold(b)
        when = f["date"].to_pydatetime()

        if (counts[fa] >= args.min_prior_fights and counts[fb] >= args.min_prior_fights
                and fa in timelines and fb in timelines):
            # A ROSTER ROW IS NOT REQUIRED. The first version demanded both
            # fighters appear in fighters.csv, which is the CURRENT 220-name
            # roster -- so retired fighters were dropped even with a fully
            # cached timeline, and 820 scorable fights collapsed to 160.
            # Nothing needed it: this is a PAIRED comparison where the two
            # arms differ ONLY in the three stat columns, so an identical
            # minimal base row on both sides changes neither arm's result
            # relative to the other. If anything it isolates the stat columns
            # more cleanly, because the other terms (height, durability, sub
            # threat) gate themselves off when their inputs are absent.
            # DICTS, not Series. A Series built from {"name": x} infers a
            # string dtype, so assigning a float stat into it raises
            # TypeError under pandas 2.x/3.x -- the same typed-empty-column
            # trap that bites the flag columns elsewhere in this project.
            # Building plain dicts and letting DataFrame() infer types at the
            # end sidesteps it, and costs nothing.
            base_a = rows_by_name[fa].to_dict() if fa in rows_by_name else {"name": a}
            base_b = rows_by_name[fb].to_dict() if fb in rows_by_name else {"name": b}
            # Elo as it stood the moment before this fight, mirroring
            # walkforward_backtest -- predict BEFORE updating.
            eff = {a: elo.get_rating(a), b: elo.get_rating(b)}

            # Computed once, before the arms, so the stratum label and the
            # pointintime arm are guaranteed to describe the same numbers.
            sa = stats_as_of(timelines[fa], when)
            sb = stats_as_of(timelines[fb], when)

            for label, use_stats in (("withheld", False), ("pointintime", True)):
                ra, rb = dict(base_a), dict(base_b)
                for col in ("strike_accuracy_pct", "td_accuracy_pct", "td_defense_pct"):
                    ra[col] = None
                    rb[col] = None
                if use_stats:
                    ra.update(sa)
                    rb.update(sb)
                try:
                    res = predict_matchup(a, b, pd.DataFrame([ra, rb]), eff)
                except Exception:
                    res = None
                if res:
                    p_a = res.get("prob_a")
                    if p_a is not None:
                        scored[label].append((p_a, 1.0 if winner == a else 0.0))
                        preds[label] = p_a
            # Keep the pair together for the paired test below. Only fights
            # where BOTH arms produced a prediction are comparable.
            if "withheld" in preds and "pointintime" in preds:
                y = 1.0 if winner == a else 0.0
                # STRATUM = how many of the three columns BOTH corners had.
                # The gating means a term only fires when both sides have it,
                # so a fight where one corner is missing everything cannot
                # show an effect no matter how good the data is -- yet it
                # still dilutes the average. Counting the columns present on
                # BOTH sides is the honest way to find where the signal, if
                # any, actually lives.
                shared = sum(1 for c in ("strike_accuracy_pct", "td_accuracy_pct", "td_defense_pct")
                             if c in sa and c in sb)
                paired.append((preds["withheld"], preds["pointintime"], y, shared))
            preds.clear()

        loser = b if winner == a else a
        elo.update_ratings(winner, loser, method=method)
        counts[fa] += 1
        counts[fb] += 1

    print(f"{'variant':<16}{'n':>7}{'accuracy':>11}{'Brier':>10}{'log loss':>11}")
    print("-" * 55)
    out = {}
    for label in ("withheld", "pointintime"):
        pairs = scored[label]
        if not pairs:
            print(f"{label:<16}{0:>7}   no scored fights")
            continue
        n = len(pairs)
        acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
        brier = sum((p - y) ** 2 for p, y in pairs) / n
        ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)) for p, y in pairs) / n
        out[label] = (n, acc, brier, ll)
        print(f"{label:<16}{n:>7}{acc:>10.1%}{brier:>10.4f}{ll:>11.4f}")

    if len(out) == 2:
        n0, a0, b0, l0 = out["withheld"]
        n1, a1, b1, l1 = out["pointintime"]
        print(f"\ndelta            {'':>7}{a1 - a0:>+10.1%}{b1 - b0:>+10.4f}{l1 - l0:>+11.4f}")
        print("\nLower Brier and log loss are better; higher accuracy is better.")

        # PAIRED TEST -- the correct one for this design, and far more
        # powerful than comparing two marginal Briers. The arms run on
        # IDENTICAL fights with identical Elo, differing only in three
        # columns, so most predictions come out byte-identical and contribute
        # exactly zero difference. Treating the arms as independent samples
        # (the marginal SE printed above) buries a real effect under the
        # variance of the fights themselves, which cancels here.
        diffs = [ (pw - y) ** 2 - (pp - y) ** 2 for pw, pp, y, _ in paired ]
        changed = [d for d in diffs if abs(d) > 1e-12]
        if diffs:
            n_d = len(diffs)
            mean_d = sum(diffs) / n_d
            var = sum((d - mean_d) ** 2 for d in diffs) / max(n_d - 1, 1)
            sd = var ** 0.5
            se = sd / (n_d ** 0.5) if n_d else float("inf")
            t = mean_d / se if se else 0.0
            print(f"\nPAIRED TEST (Brier improvement per fight, + means the stats helped)")
            print(f"  fights compared        {n_d}")
            print(f"  predictions that MOVED {len(changed)} ({len(changed)/n_d:.0%})")
            print(f"  mean improvement       {mean_d:+.5f}")
            print(f"  paired SE              {se:.5f}")
            print(f"  t                      {t:+.2f}")
            if abs(t) >= 2.0:
                print(f"  READ: |t| >= 2 -- the effect is distinguishable from noise, "
                      f"and it {'HELPS' if mean_d > 0 else 'HURTS'}.")
            else:
                print(f"  READ: |t| < 2 -- still not distinguishable from noise, even "
                      f"paired. The columns are not demonstrably earning their place.")
            if changed:
                mc = sum(changed) / len(changed)
                print(f"  among fights that moved, mean improvement {mc:+.5f}")

            # STRATIFIED: the effect should concentrate where both corners
            # actually have the data. If it does not -- if the 3-column
            # stratum looks no better than the 0-column one -- that is
            # evidence the apparent gain is noise, because the 0-column
            # stratum is a group where these columns CANNOT do anything.
            # That stratum is the built-in control this test otherwise lacks.
            print(f"\n  BY SHARED COLUMNS (how many of the three BOTH corners had)")
            print(f"    {'cols':>4}{'n':>7}{'moved':>8}{'mean improvement':>19}{'t':>8}")
            for k in (0, 1, 2, 3):
                grp = [((pw - y) ** 2 - (pp - y) ** 2) for pw, pp, y, sh in paired if sh == k]
                if not grp:
                    continue
                ng = len(grp)
                mg = sum(grp) / ng
                if ng > 1:
                    sdg = (sum((d - mg) ** 2 for d in grp) / (ng - 1)) ** 0.5
                    seg = sdg / (ng ** 0.5)
                    tg = mg / seg if seg else 0.0
                else:
                    tg = 0.0
                movedg = sum(1 for d in grp if abs(d) > 1e-12)
                print(f"    {k:>4}{ng:>7}{movedg:>8}{mg:>+19.5f}{tg:>+8.2f}")
            print("    (0 columns is the CONTROL: the terms cannot fire there, so any "
                  "apparent effect in that row is pure noise and sets the scale.)")
        # SAMPLE SIZE GATES THE VERDICT, not just effect size. The first
        # version declared "the columns earn their place" off n=160 and a
        # 0.0015 Brier gain -- while ACCURACY moved 2.5% the other way. At
        # n=160 the standard error on Brier is ~0.02, so that gain is under a
        # tenth of one SE: noise in both directions at once. It also compared
        # itself to recency weighting, which was n=1747 WITH a monotonic
        # control sweep. This project's own history is the warning: 50 live
        # picks told the opposite story from 2,931.
        MIN_N_FOR_VERDICT = 1500
        if n1 < MIN_N_FOR_VERDICT:
            se = (b0 * (1 - b0) / n1) ** 0.5 if 0 < b0 < 1 else 0.02
            print(f"READ: INCONCLUSIVE at n={n1}. Rough SE on Brier is ~{se:.4f}, so a "
                  f"difference of {abs(b1 - b0):.4f} is ~{abs(b1 - b0) / se:.2f} SE -- "
                  f"indistinguishable from noise. Cache more fighters "
                  f"(backfill --from-history) and re-run; aim for n >= {MIN_N_FOR_VERDICT}.")
            if (b1 < b0) != (a1 > a0):
                print("      Note Brier and accuracy disagree in direction, which is the "
                      "signature of noise rather than a small real effect.")
        elif b1 < b0 - 0.001:
            print("READ: the columns earn their place -- a Brier gain of this size, at "
                  "this sample, is the same order as the validated recency-weighting change.")
        elif b1 > b0 + 0.001:
            print("READ: the columns make the model WORSE point-in-time. That is a real "
                  "result and worth acting on, not explaining away.")
        else:
            print("READ: the difference is within noise (|dBrier| < 0.001). The columns "
                  "are not demonstrably earning anything yet -- which is not the same as "
                  "being harmful, since real measurements still beat hardcoded defaults "
                  "on any fight where the two corners differ.")


if __name__ == "__main__":
    main()
