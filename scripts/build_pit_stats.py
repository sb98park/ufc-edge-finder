"""
Per-fight striking and grappling statistics, dated -- the table that makes
the wrestling and striking terms measurable.

WHY THIS EXISTS. audit_term_coverage.py reports wrestling_adjustment and
striking_adjustment firing on 22% and 82% of live predictions and 0% of
backtested ones. They read strike_accuracy_pct, td_accuracy_pct and
td_defense_pct, which live only in data/fighters.csv -- a snapshot of the
CURRENT roster, carrying today's career averages. Feeding those to a 2019
fight is look-ahead, so pit_roster deliberately omits them, so the terms gate
off, so every backtest verdict this project has published was measured on a
model missing its two most-used style terms.

That was recorded here as a permanent data limitation. It is not one.
data/ufc_fight_stats.csv holds 41,340 per-ROUND rows over 8,647 bouts and
2,702 fighters, with control time populated on 99.9% of them. Everything the
terms need is there; it needed aggregating and dating.

WHAT THIS WRITES. data/pit_stats.csv, one row per (fighter, bout):

    name, date, event, bout, opponent,
    sig_str_landed, sig_str_att, sig_str_absorbed, sig_str_faced,
    td_landed, td_att, td_faced, td_stuffed,
    kd_for, kd_against, sub_att, ctrl_seconds, fight_seconds

Rates are deliberately NOT precomputed. A point-in-time average must sum the
components across a fighter's prior bouts and divide once -- averaging
per-fight percentages weights a 15-second fight the same as a 25-minute one.
stats_as_of() below does it correctly; storing a percentage column would
invite the wrong thing.

DATING, which is the part with no obvious solution. ufc_fight_stats has no
date, and neither does ufc_fight_results -- only an EVENT name. But
fight_history has dates and fighter names, so an event can be dated by
matching any of its bouts to a history row and taking the modal date. All 781
events resolve this way.

One trap cost the first attempt: results EVENT strings carry a TRAILING
SPACE and the stats file's do not, so the two join to exactly zero rows
before stripping and all 781 after. Both sides are stripped here.

REMATCHES are why the join key is (event, bout) rather than a name pair. 201
name-pairs are rematches covering 422 bouts; keyed on names alone they would
collapse onto one date and hand a fighter his own future performance.

RESULT. 17,524 fighter-bout rows over 8,626 bouts and 2,702 fighters, dated
1994-03-11 to 2026-07-18. Population sanity, which is how a silently-wrong
ETL gets caught: sig-strike accuracy 44.8% against the UFC's published ~45%,
takedown accuracy 37.5% against ~35-40%, mean fight length 10.6 minutes
against ~11. Per-fighter, Makhachev reads 91% takedown defence and 51%
control time and Holloway reads 6.9 significant strikes landed per minute
with 6% control -- their actual, well-known signatures rather than plausible
numbers.

With this wired in, audit_term_coverage.py reports wrestling and striking
firing in a backtest for the first time, at 56% each. There are now no dark
terms.

ONE CONSEQUENCE WORTH STATING, because it inverts the usual complaint.
Wrestling fires on 56% of backtested fights against 22% of live ones, because
fighters.csv has control_time_pct populated on 0.0% of the roster and
td_accuracy_pct on 71.9%, while this table carries takedown attempts on 65%
of bouts and control time on 82%. The BACKTEST model is now better fed than
production on exactly the terms this was built to measure. That is the
opposite of the usual direction and it means a verdict from these harnesses
is no longer a lower bound on production -- for the wrestling term it may be
an upper one. The clean fix is for production to read its rates from here
too, which is a separate change and needs its own validation.

Usage:  python3 scripts/build_pit_stats.py
        python3 scripts/build_pit_stats.py --out data/pit_stats.csv
"""

import argparse
import collections
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATS = "data/ufc_fight_stats.csv"
RESULTS = "data/ufc_fight_results.csv"
HISTORY = "data/fight_history.csv"
OUT = "data/pit_stats.csv"


def _fold(n) -> str:
    return str(n).strip().lower()


def _landed_att(cell) -> tuple[float, float]:
    """'29 of 62' -> (29, 62). Anything unparseable -> (0, 0)."""
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)\s*$", str(cell))
    if not m:
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2))


def _mmss(cell) -> float:
    """'2:02' -> 122.0 seconds. '--' and NaN -> 0.0."""
    m = re.match(r"\s*(\d+):(\d{1,2})\s*$", str(cell))
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + int(m.group(2))


def _num(cell) -> float:
    try:
        v = float(cell)
        return 0.0 if v != v else v
    except (TypeError, ValueError):
        return 0.0


def _fight_seconds(row) -> float:
    """
    Elapsed time from the finishing round and clock.

    ROUND is the round it ended in and TIME the clock at that moment, so a
    round-3 decision at 5:00 is 15 minutes. Rounds are five minutes except in
    the handful of legacy formats below, which are read off TIME FORMAT
    rather than assumed -- '1 Rnd + 2OT (15-3-3)' has a fifteen-minute first
    round and would otherwise be scored as five.
    """
    rnd, clock, fmt = row.get("ROUND"), row.get("TIME"), str(row.get("TIME FORMAT") or "")
    try:
        rnd = int(rnd)
    except (TypeError, ValueError):
        return 0.0
    lengths = [int(x) for x in re.findall(r"(\d+)", fmt.split("(")[-1])] if "(" in fmt else []
    if not lengths:
        lengths = [5]
    # pad with the last known round length
    while len(lengths) < rnd:
        lengths.append(lengths[-1])
    completed = sum(lengths[: max(rnd - 1, 0)]) * 60.0
    return completed + _mmss(clock)


def event_dates() -> dict:
    """
    {stripped EVENT name: date} via any bout that matches fight_history.

    Modal rather than first, so one mis-entered history row cannot re-date a
    whole card.
    """
    h = pd.read_csv(HISTORY)
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"])
    by_pair = collections.defaultdict(list)
    for r in h.itertuples(index=False):
        by_pair[frozenset((_fold(r.fighter_a), _fold(r.fighter_b)))].append(r.date)

    res = pd.read_csv(RESULTS)
    votes = collections.defaultdict(collections.Counter)
    for r in res.itertuples(index=False):
        parts = [x.strip() for x in str(r.BOUT).split(" vs. ")]
        if len(parts) != 2:
            continue
        for d in by_pair.get(frozenset((_fold(parts[0]), _fold(parts[1]))), []):
            votes[str(r.EVENT).strip()][d.date()] += 1
    return {e: c.most_common(1)[0][0] for e, c in votes.items() if c}


def bout_durations() -> dict:
    """{(event, bout): elapsed seconds}."""
    res = pd.read_csv(RESULTS)
    out = {}
    for r in res.to_dict("records"):
        key = (str(r["EVENT"]).strip(), str(r["BOUT"]).strip())
        out[key] = _fight_seconds(r)
    return out


def build() -> pd.DataFrame:
    st = pd.read_csv(STATS)
    dates = event_dates()
    durations = bout_durations()

    # Sum the per-round rows into one record per (event, bout, fighter).
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in st.to_dict("records"):
        event, bout = str(r["EVENT"]).strip(), str(r["BOUT"]).strip()
        fighter = str(r["FIGHTER"]).strip()
        if not fighter or fighter == "nan":
            continue
        a = agg[(event, bout, fighter)]
        sl, sa = _landed_att(r.get("SIG.STR."))
        tl, ta = _landed_att(r.get("TD"))
        a["sig_str_landed"] += sl
        a["sig_str_att"] += sa
        a["td_landed"] += tl
        a["td_att"] += ta
        a["kd_for"] += _num(r.get("KD"))
        a["sub_att"] += _num(r.get("SUB.ATT"))
        a["ctrl_seconds"] += _mmss(r.get("CTRL"))
        a["rounds"] += 1

    # Pair each fighter with the other corner so absorbed/faced can be filled
    # from the OPPONENT's own line -- the quantity td_defense_pct and the
    # strikes-absorbed rate are actually about.
    by_bout = collections.defaultdict(list)
    for (event, bout, fighter), vals in agg.items():
        by_bout[(event, bout)].append((fighter, vals))

    rows = []
    for (event, bout), corners in by_bout.items():
        if len(corners) != 2:
            continue      # a bout with one line cannot supply an opponent
        date = dates.get(event)
        if date is None:
            continue
        seconds = durations.get((event, bout), 0.0)
        for i, (fighter, v) in enumerate(corners):
            opp_name, o = corners[1 - i]
            rows.append({
                "name": fighter,
                "date": date,
                "event": event,
                "bout": bout,
                "opponent": opp_name,
                "sig_str_landed": v["sig_str_landed"],
                "sig_str_att": v["sig_str_att"],
                "sig_str_absorbed": o["sig_str_landed"],
                "sig_str_faced": o["sig_str_att"],
                "td_landed": v["td_landed"],
                "td_att": v["td_att"],
                "td_faced": o["td_att"],
                "td_stuffed": o["td_att"] - o["td_landed"],
                "kd_for": v["kd_for"],
                "kd_against": o["kd_for"],
                "sub_att": v["sub_att"],
                "ctrl_seconds": v["ctrl_seconds"],
                "fight_seconds": seconds,
            })
    return pd.DataFrame(rows).sort_values(["date", "event", "bout", "name"])


# ---------------------------------------------------------------- consumption

def load_pit_stats(path: str = OUT) -> dict:
    """{folded name: [row dicts]} sorted oldest first, for point-in-time reads."""
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")
    out = collections.defaultdict(list)
    for r in d.to_dict("records"):
        out[_fold(r["name"])].append(r)
    return out


def stats_as_of(timeline: list, when, min_bouts: int = 3) -> dict:
    """
    A fighters.csv-shaped stat dict from bouts strictly BEFORE `when`.

    Components are summed and divided once, rather than averaging per-fight
    percentages -- a 15-second fight and a 25-minute one carry very different
    amounts of evidence and a mean of ratios pretends otherwise.

    Returns {} below min_bouts, which makes style_matchup_adjustment's
    both-corners gate refuse the term rather than trust three minutes of cage
    time. That gate is why this returns nothing instead of a default.
    """
    prior = [r for r in timeline if r["date"].date() < when] if timeline else []
    if len(prior) < min_bouts:
        return {}
    s = {k: sum(float(r.get(k) or 0) for r in prior) for k in (
        "sig_str_landed", "sig_str_att", "sig_str_absorbed", "sig_str_faced",
        "td_landed", "td_att", "td_faced", "td_stuffed",
        "kd_for", "kd_against", "ctrl_seconds", "fight_seconds")}
    minutes = s["fight_seconds"] / 60.0
    out = {"espn_fights": len(prior)}
    if s["sig_str_att"] > 0:
        out["strike_accuracy_pct"] = 100.0 * s["sig_str_landed"] / s["sig_str_att"]
    if s["td_att"] > 0:
        out["td_accuracy_pct"] = 100.0 * s["td_landed"] / s["td_att"]
    if s["td_faced"] > 0:
        out["td_defense_pct"] = 100.0 * s["td_stuffed"] / s["td_faced"]
    if minutes > 0:
        out["slpm"] = s["sig_str_landed"] / minutes
        out["sapm"] = s["sig_str_absorbed"] / minutes
        out["td_per_15"] = 15.0 * s["td_landed"] / minutes
        out["control_time_pct"] = 100.0 * s["ctrl_seconds"] / s["fight_seconds"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    for f in (STATS, RESULTS, HISTORY):
        if not os.path.exists(f):
            print(f"missing {f}")
            sys.exit(1)

    df = build()
    if df.empty:
        print("built nothing -- check the EVENT join")
        sys.exit(1)
    df.to_csv(args.out, index=False)

    print(f"wrote {args.out}: {len(df)} fighter-bout rows, "
          f"{df['bout'].nunique()} bouts, {df['name'].nunique()} fighters")
    print(f"  dated {df['date'].min()} to {df['date'].max()}")
    for col in ("sig_str_att", "td_att", "ctrl_seconds", "fight_seconds"):
        print(f"  {col:<16} non-zero on {(df[col] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
