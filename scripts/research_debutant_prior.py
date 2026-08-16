"""
What is a pre-UFC record actually worth?

THE QUESTION. A debutant has no UFC history, so build_effective_ratings
falls back entirely to compute_stats_rating, which reads their career
win/loss record. That treats 7-0 on the regional scene and 7-0 in PFL as
the same evidence. Anthony Wint (7-0, DWCS) came out at 1781 against a
debut opponent at 1500 -- a 281-point gap and an 85% pick, on a fight
between two men the UFC has never seen.

WHAT THIS MEASURES, and it needs no new data. data/ufc_fight_results.csv is
UFC-only, 8,784 bouts, newest-first -- so a fighter's LAST appearance in it
is their UFC debut, and everything in their career record beyond their UFC
bouts happened before it. That gives, for every roster fighter:

    pre-UFC record  = career record - UFC record
    debut outcome   = did they win their first UFC fight

Then: feed the pre-UFC record to compute_stats_rating exactly as production
would for a debutant, convert to a win probability against a neutral 1500
opponent, and compare against what actually happened.

THE RESULT (n=263):

    model implies   actually wins   gap        n
    56.4%           55.6%           -0.9%      9
    65.8%           70.2%           +4.4%     47
    74.9%           63.2%          -11.7%    163
    82.1%           45.5%          -36.6%     44

    overall         implied 73.8%   actual 61.2%   -12.6%
    top end 75%+                    actual 59.0%   -20.0%

The error is not a constant offset -- it grows with the record and INVERTS
at the top. Debutants the model rates at 82% win less than half their
debuts. A glossy undefeated regional record is, at the extreme, negative
evidence.

Two supporting cuts, same direction:

  pre-UFC win% 70-85%  ->  debut win 57.5%
  pre-UFC win% 85%+    ->  debut win 62.4%     (barely moves)

  pre-UFC 1-7 bouts    ->  debut win 64.4%
  pre-UFC 8-14 bouts   ->  debut win 62.6%
  pre-UFC 15+ bouts    ->  debut win 54.2%     (more experience, worse)

A padded record and a hard-earned one look identical in W-L, which is
exactly what an organisation label would separate -- and why the raw record
carries so little signal on its own.

CAVEATS, stated because the headline number is dramatic. The top bucket is
n=44. The finish rate in each prior is estimated from the fighter's CAREER
ratio rather than their pre-UFC one, since the splits are not stored
separately (an earlier version zeroed it, which charged every debutant the
same -60 penalty and shifted every implied probability down by a constant).
And the comparison is against a neutral 1500 opponent, not the real one, so
it measures how the RATING CURVE behaves, not any specific matchup.

Usage:  python3 scripts/research_debutant_prior.py
"""

import collections
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.power_rating import compute_stats_rating, RATING_CENTER  # noqa: E402

UFC_RESULTS = "data/ufc_fight_results.csv"
FIGHTERS = "data/fighters.csv"


def ufc_bouts_by_fighter(path=UFC_RESULTS):
    """
    {folded_name: [(row_index, won), ...]} sorted OLDEST FIRST.

    The file is newest-first (row 0 is a 2026 card, the last row is UFC 2 in
    1994), verified against fight_history dates rather than assumed, so a
    descending row index is ascending time.
    """
    d = pd.read_csv(path)
    out = collections.defaultdict(list)
    for i, (bout, outcome) in enumerate(zip(d["BOUT"], d["OUTCOME"])):
        parts = [x.strip() for x in str(bout).split(" vs. ")]
        if len(parts) != 2 or outcome not in ("W/L", "L/W"):
            continue      # draws and no-contests carry no win to attribute
        out[parts[0].lower()].append((i, outcome == "W/L"))
        out[parts[1].lower()].append((i, outcome == "L/W"))
    for n in out:
        out[n].sort(key=lambda t: -t[0])
    return out


def build_debutants(fighters_path=FIGHTERS):
    fighters = pd.read_csv(fighters_path)
    bouts = ufc_bouts_by_fighter()
    rows = []
    for _, row in fighters.iterrows():
        b = bouts.get(str(row["name"]).strip().lower())
        if not b:
            continue
        ufc_w = sum(1 for _, w in b if w)
        ufc_l = len(b) - ufc_w
        cw, cl = row.get("wins"), row.get("losses")
        if pd.isna(cw) or pd.isna(cl):
            continue
        pre_w, pre_l = int(cw) - ufc_w, int(cl) - ufc_l
        # A negative remainder means the two sources disagree about this
        # fighter. Skipped rather than clamped -- clamping would invent a
        # 0-0 pre-UFC career and quietly bias the low end.
        if pre_w < 0 or pre_l < 0 or pre_w + pre_l == 0:
            continue
        kw, sw = row.get("ko_wins"), row.get("sub_wins")
        finish_ratio = (((float(kw) + float(sw)) / float(cw))
                        if pd.notna(kw) and pd.notna(sw) and float(cw) > 0 else 0.4)
        prior = pd.Series({
            "wins": pre_w, "losses": pre_l,
            "ko_wins": finish_ratio * pre_w, "sub_wins": 0,
            "reach_in": row["reach_in"] if pd.notna(row.get("reach_in")) else 70,
        })
        rating = compute_stats_rating(prior)
        first3 = [w for _, w in b[:3]]
        rows.append({
            "name": row["name"], "pre_w": pre_w, "pre_l": pre_l,
            "pre_n": pre_w + pre_l, "pre_pct": pre_w / (pre_w + pre_l),
            "rating": rating,
            "implied": 1 / (1 + 10 ** (-(rating - RATING_CENTER) / 400)),
            "debut_win": b[0][1],
            "first3": sum(first3) / len(first3),
        })
    return pd.DataFrame(rows)


def main():
    if not os.path.exists(UFC_RESULTS):
        print(f"No {UFC_RESULTS}.")
        sys.exit(1)
    r = build_debutants()
    print(f"UFC debutants with a reconstructable pre-UFC record: n={len(r)}")
    print(f"overall debut win rate: {r['debut_win'].mean():.1%}\n")

    print("CALIBRATION OF THE DEBUTANT PRIOR")
    print(f"  {'implied':<12}{'actual':<12}{'gap':<12}{'n'}")
    print("  " + "-" * 42)
    for lo, hi in [(0, .60), (.60, .70), (.70, .80), (.80, 1.01)]:
        g = r[(r["implied"] >= lo) & (r["implied"] < hi)]
        if len(g) >= 8:
            print(f"  {g['implied'].mean():<12.1%}{g['debut_win'].mean():<12.1%}"
                  f"{g['debut_win'].mean() - g['implied'].mean():<+12.1%}{len(g)}")
    print(f"\n  overall: implied {r['implied'].mean():.1%}  actual {r['debut_win'].mean():.1%}"
          f"  gap {r['debut_win'].mean() - r['implied'].mean():+.1%}")

    print("\nBY PRE-UFC WIN RATE")
    for lo, hi, lab in [(0, .70, "under 70%"), (.70, .85, "70-85%"), (.85, 1.01, "85%+")]:
        g = r[(r["pre_pct"] >= lo) & (r["pre_pct"] < hi)]
        if len(g) >= 5:
            print(f"  {lab:<12} n={len(g):>3}  debut {g['debut_win'].mean():.1%}  first-3 {g['first3'].mean():.1%}")

    print("\nBY PRE-UFC EXPERIENCE")
    for lo, hi, lab in [(1, 8, "1-7 bouts"), (8, 15, "8-14 bouts"), (15, 100, "15+ bouts")]:
        g = r[(r["pre_n"] >= lo) & (r["pre_n"] < hi)]
        if len(g) >= 5:
            print(f"  {lab:<12} n={len(g):>3}  debut {g['debut_win'].mean():.1%}  first-3 {g['first3'].mean():.1%}")


if __name__ == "__main__":
    main()
