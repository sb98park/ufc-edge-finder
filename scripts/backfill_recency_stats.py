"""
Write RECENCY-WEIGHTED career stats into data/fighters.csv.

WHY. Production's slpm / sapm / td_per_15 are ALL-TIME career means, so a
fighter's 2016 form counts exactly as much as their 2025 form. Measured
across 1,061 fighters with 6+ tracked fights: the median within-career drift
in significant strikes per minute is 0.62 of the BETWEEN-fighter standard
deviation, and 32% of fighters drift by more than a full sd. An all-time mean
therefore often describes a blend of two materially different fighters.

VALIDATED BEFORE BUILDING (research_recency_weighting.py). Half-life swept on
the tuning split with all-time included as a control; the control lost
MONOTONICALLY (12mo 0.2377 ... all-time 0.2396). Frozen holdout n=1747:
58.6% -> 59.9% accuracy, Brier 0.2345 -> 0.2329. An 18-month half-life won.

ARCHITECTURE, deliberately the same as scripts/backfill_td_rate.py: the big
per-fight CSVs stay local-only, but fighters.csv IS committed -- so this
computes locally and writes columns that scheduled Actions builds can use
without ever needing the source data.

Columns written (all suffixed _r to sit ALONGSIDE the all-time values rather
than overwrite them, so production can fall back and the two can be compared
on a live card before anything is switched over):
    slpm_r, sapm_r, td_per_15_r

Usage:
    python3 scripts/backfill_recency_stats.py            # dry run
    python3 scripts/backfill_recency_stats.py --apply
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.card_matcher import _normalize_name  # noqa: E402

FIGHTERS = "data/fighters.csv"
HALF_LIFE_DAYS = 18 * 30.44          # the value the sweep selected
MIN_FIGHTS = 3
MIN_MINUTES = 15.0

STATS = next((p for p in ("data/ufc_fight_stats.csv",
                          "/mnt/user-data/uploads/ufc_fight_stats.csv") if os.path.exists(p)), None)
RESULTS = next((p for p in ("data/ufc_fight_results.csv",
                            "/mnt/user-data/uploads/ufc_fight_results.csv") if os.path.exists(p)), None)


def _of_pair(cell):
    try:
        return int(str(cell).split(" of ")[0])
    except (ValueError, AttributeError, IndexError):
        return 0


def _duration(round_num, time_str):
    try:
        m, s = str(time_str).split(":")
        return (int(round_num) - 1) * 300 + int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def main():
    apply = "--apply" in sys.argv
    if not STATS or not RESULTS:
        print("Need ufc_fight_stats.csv AND ufc_fight_results.csv in data/ (local-only files).")
        sys.exit(1)

    res = pd.read_csv(RESULTS); res.columns = [c.strip() for c in res.columns]
    stats = pd.read_csv(STATS); stats.columns = [c.strip() for c in stats.columns]

    # Fight duration and DATE, both keyed by (event, bout). The date is what
    # makes weighting possible at all -- the all-time version never needed it.
    meta = {}
    for r in res.to_dict("records"):
        d = _duration(r.get("ROUND"), r.get("TIME"))
        if d and d > 0:
            meta[(str(r["EVENT"]).strip(), str(r["BOUT"]).strip())] = d
    # Reuse the harness's dating join rather than reimplementing it: it
    # resolves fights by unique fighter-PAIR against fight_history.csv and is
    # the same code path every validated experiment used. A hand-rolled join
    # here returned zero matches, which is exactly the kind of silent
    # mismatch that makes a backfill look like it ran fine.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from validate_adjustment_layer import load_dated_fights
    dated = load_dated_fights()
    dates = {}
    for f in dated.itertuples(index=False):
        dates[(str(f.event).strip(), str(f.bout).strip())] = f.date

    # Accumulate per fighter: (date, sig_landed, sig_absorbed, td_landed, seconds)
    per = {}
    for key, group in pd.DataFrame(stats).groupby(["EVENT", "BOUT"]):
        k = (str(key[0]).strip(), str(key[1]).strip())
        dur = meta.get(k)
        if not dur:
            continue
        names = [str(n).strip() for n in group["FIGHTER"].unique()]
        if len(names) != 2:
            continue
        when = dates.get(k)
        if when is None or pd.isna(when):
            continue
        tot = {n: {"sig": 0, "td": 0} for n in names}
        for r in group.to_dict("records"):
            n = str(r["FIGHTER"]).strip()
            tot[n]["sig"] += _of_pair(r.get("SIG.STR."))
            tot[n]["td"] += _of_pair(r.get("TD"))
        for n in names:
            opp = names[1] if n == names[0] else names[0]
            per.setdefault(n, []).append((when, tot[n]["sig"], tot[opp]["sig"], tot[n]["td"], dur))

    today = pd.Timestamp.today().normalize()
    out = {}
    for name, fights in per.items():
        if len(fights) < MIN_FIGHTS:
            continue
        wsig = wabs = wtd = wsec = wcount = 0.0
        for when, sig, absorbed, td, sec in fights:
            w = 0.5 ** (max(0.0, (today - when).days) / HALF_LIFE_DAYS)
            wcount += w
            wsig += w * sig; wabs += w * absorbed; wtd += w * td; wsec += w * sec
        mins = wsec / 60.0
        # EFFECTIVE-SAMPLE GUARD. The old floor was a quarter of one fight in
        # weighted minutes, which is far too loose: with an 18-month half-life
        # a fighter whose bouts were 3-5 years ago carries weights of 0.10-0.25
        # each, so they can clear that bar while contributing well under a
        # single effective fight. A rate from that slice is noise, and it can
        # land on exactly 0.0 -- which then reads as "this fighter never scores
        # takedowns" and silently inflates the opponent's wrestling edge.
        # Real case: a fighter came out at td_per_15 = 0.0 against 0.592
        # all-time, and it moved a Lock of the Week pick.
        # Falling back to the all-time value is strictly better than a
        # confident number computed from almost nothing.
        if mins < MIN_MINUTES or wcount < 1.5:
            continue
        out[_normalize_name(name)] = {
            "slpm_r": round(wsig / mins, 3),
            "sapm_r": round(wabs / mins, 3),
            "td_per_15_r": round(wtd / mins * 15.0, 3),
        }

    fighters = pd.read_csv(FIGHTERS)
    cols = {c: [] for c in ("slpm_r", "sapm_r", "td_per_15_r")}
    matched = 0
    for name in fighters["name"]:
        v = out.get(_normalize_name(str(name)))
        if v:
            matched += 1
        for c in cols:
            cols[c].append(v[c] if v else "")

    print(f"half-life: {HALF_LIFE_DAYS/30.44:.0f} months (selected by the tuning sweep)")
    print(f"fighters with a weighted profile: {len(out)}")
    print(f"fighters.csv rows: {len(fighters)} | matched: {matched} ({matched/len(fighters):.0%})")
    got = [v for v in cols["slpm_r"] if v != ""]
    if got:
        got.sort()
        print(f"slpm_r: min {got[0]:.2f} | median {got[len(got)//2]:.2f} | max {got[-1]:.2f}")
        # A sanity contrast: recency-weighted values SHOULD differ from all-time.
        if "slpm" in fighters.columns:
            cmp = [(a, b) for a, b in zip(fighters["slpm"], cols["slpm_r"])
                   if b != "" and pd.notna(a)]
            if cmp:
                diff = sum(abs(float(a) - float(b)) for a, b in cmp) / len(cmp)
                print(f"mean |all-time - recency| for slpm: {diff:.2f}  "
                      f"(near 0 would mean the weighting isn't doing anything)")

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return
    for c, vals in cols.items():
        fighters[c] = vals
    fighters.to_csv(FIGHTERS, index=False)
    print(f"\nWritten. Commit data/fighters.csv so Actions builds get the columns.")


if __name__ == "__main__":
    main()
