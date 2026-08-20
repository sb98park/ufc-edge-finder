"""
Do feeder promotions predict UFC debut performance? Measured: no.

THE QUESTION. A debutant's starting rating comes from compute_stats_rating,
which reads their career W-L. The obvious complaint is that 7-0 in PFL and
7-0 on the regional circuit are not the same evidence, so the natural fix is
a per-promotion Elo offset -- tier the feeders, add points for the good ones.
This measures whether the data supports that. It does not.

WHERE PROVENANCE COMES FROM. Nothing in data/*.csv names a promotion:
fight_history.csv is date/fighter_a/fighter_b/winner/method, and
ufc_fight_results.csv is UFC-only. But data/.espn_cache holds 2,926 cached
ESPN event objects whose `name` is "Bellator MMA: Bellator 123", and whose
competitors carry ESPN athlete ids that join to data/espn_athlete_ids.csv.
That recovers 24,367 event-competitor rows across 2,769 roster fighters, and
for 1,254 of them a pre-debut non-UFC event -- their feeder promotion.

WHY IT STILL CANNOT BE TIERED.

1. FRAGMENTATION. Those 1,254 debuts spread across 378 organisations after
   normalising event numbers away ("Invicta FC 32" -> "Invicta FC"). Only 14
   reach n>=15, six reach n>=30, one reaches n>=50. The confidence intervals
   swallow every plausible effect:

       Invicta FC   n=65   52.3%   [40.2%, 64.5%]
       RFA          n=39   38.5%   [23.2%, 53.7%]
       ROC          n=36   58.3%   [42.2%, 74.4%]
       CWFC         n=24   62.5%   [43.1%, 81.9%]
       MFC          n=19   10.5%   [-3.3%, 24.3%]

2. NO SIGNAL. Across the 14 organisations with n>=15 (N=380, pooled debut
   win rate 46.3%), chi2 = 20.46 on 13 df, p = 0.084. The between-promotion
   spread is not distinguishable from noise.

3. WRONG ERA. Median debut date in that sample is 2014-08-23, and only 8.7%
   debuted after 2020. It is dominated by promotions that no longer exist --
   Strikeforce, IFL, MFC, RFA, TPF. The modern feeder landscape is nearly
   absent: Bellator n=13, Rizin n=12, and Dana White's Contender Series is
   excluded by construction because it is UFC-run. Tiers fitted here would
   describe a feeder ecosystem that stopped existing a decade ago.

AND THE ALTERNATIVE THAT DOES NOT NEED PROMOTION LABELS ALSO FAILS. Rather
than trusting a promotion's brand, count how many of a debutant's pre-UFC
opponents themselves went on to fight in the UFC -- strength of schedule,
measured directly, per fighter, immune to the fragmentation above:

    prior UFC-calibre opponents    n     debut win rate
    0                              88    63.6%
    1                             160    49.4%
    2                              43    67.4%
    3+                             28    64.3%

A chi2 across those four buckets gives p = 0.048, which looks like a result
until you notice the pattern is not ordered: the zero bucket is the second
HIGHEST. The hypothesis that would justify an Elo offset is a monotonic one,
and the Cochran-Armitage trend test on it gives z = +0.353, p = 0.724. The
chi2 was an unordered wiggle. Reporting it as a finding would have been a
multiple-comparisons artefact -- four tests were run against this question.

WHAT THIS LEAVES. scripts/research_debutant_prior.py already showed the
pre-UFC record prior does not merely add noise, it INVERTS at the top: a
debutant the model rates at 82.1% wins 45.5% of the time (n=44). And
scripts/validate_debutant_shrink.py swept the correction point-in-time over
3,966 debut fights and found it cancels (p=0.162). Three independent attempts
-- shrink the record, tier the promotion, weight the opposition -- have now
failed to improve the debutant RATING.

That is consistent, and it points somewhere specific: the defect is not the
number, it is the confidence attached to it. The same validation run found
that a 60-80% pick in a debut fight wins about 55% (n=1,757, overstated by
13-16pp). Fixing that lives in the reporting layer beside
MIN_RECORD_FOR_HIGH_CONFIDENCE, not in the rating.

Run: python3 scripts/research_feeder_promotions.py
"""

from __future__ import annotations

import collections
import csv
import datetime
import glob
import json
import math
import re
import sys

sys.path.insert(0, ".")
from src.fighter_history import fold_name

CACHE = "data/.espn_cache/*.json"
MIN_ORG_N = 15          # below this a promotion's rate is unreadable noise
UFC_PREFIXES = ("ufc", "noche ufc", "dana white's contender series",
                "the ultimate fighter")


def normalise_org(name: str) -> str:
    """'Invicta FC 32' -> 'Invicta FC'. Event numbers are not organisations."""
    s = re.sub(r"\s+\d+$", "", name.strip())
    s = re.sub(r"\s+(19|20)\d\d$", "", s)
    s = re.sub(r"\s*[-:]\s*.*$", "", s)
    s = re.sub(r"\s+[IVXLC]+$", "", s)
    return re.sub(r"\s+\d+$", "", s).strip()


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of chi-square. Avoids a scipy dependency for one call."""
    if df % 2 == 0:
        term = math.exp(-x / 2); total = term
        for i in range(1, df // 2):
            term *= x / (2 * i); total += term
        return min(1.0, total)
    z = math.sqrt(x)
    total = math.erfc(z / math.sqrt(2))
    term = math.sqrt(2 / math.pi) * z * math.exp(-x / 2)
    for i in range(1, (df - 1) // 2 + 1):
        total += term; term *= x / (2 * i + 1)
    return min(1.0, total)


def load_debuts() -> dict:
    """{folded name: (debut date, won)} from the ufcstats spine."""
    dates = {}
    for row in csv.DictReader(open("data/ufc_event_details.csv", encoding="utf-8")):
        try:
            dates[row["EVENT"].strip()] = datetime.datetime.strptime(
                row["DATE"].strip(), "%B %d, %Y").strftime("%Y-%m-%d")
        except (ValueError, KeyError):
            continue
    bouts = collections.defaultdict(list)
    for row in csv.DictReader(open("data/ufc_fight_results.csv", encoding="utf-8")):
        parts = [p.strip() for p in str(row.get("BOUT", "")).split(" vs. ")]
        if len(parts) != 2:
            continue
        date = dates.get(row["EVENT"].strip())
        outcome = row.get("OUTCOME", "").strip()
        if not date or outcome not in ("W/L", "L/W"):
            continue
        for i, who in enumerate(parts):
            won = (i == 0) if outcome == "W/L" else (i == 1)
            bouts[fold_name(who)].append((date, int(won)))
    return {k: sorted(v, key=lambda x: x[0])[0] for k, v in bouts.items() if v}


def load_feeders() -> dict:
    """{roster name: [(date, promotion), ...]} from the cached ESPN events."""
    ids = {str(r["espn_id"]).strip(): r["name"].strip()
           for r in csv.DictReader(open("data/espn_athlete_ids.csv", encoding="utf-8"))}
    out = collections.defaultdict(list)
    for path in glob.glob(CACHE):
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not (isinstance(doc, dict) and "competitions" in doc and "name" in doc):
            continue
        name = str(doc.get("name") or "")
        date = str(doc.get("date") or "")[:10]
        if not date:
            continue
        promo = name.split(":")[0].strip() if ":" in name else name.strip()
        for comp in (doc.get("competitions") or []):
            for cp in (comp.get("competitors") or []):
                aid = str(cp.get("id") or "")
                if aid in ids:
                    out[ids[aid]].append((date, promo))
    return out


def main() -> int:
    debuts = load_feeders(), load_debuts()
    appearances, debut = debuts
    rows = []
    for name, events in appearances.items():
        key = fold_name(name)
        if key not in debut:
            continue
        ddate, won = debut[key]
        prior = [(d, p) for d, p in events
                 if d < ddate and not any(p.lower().startswith(u) for u in UFC_PREFIXES)]
        if not prior:
            continue
        prior.sort()
        rows.append({"name": name, "debut": ddate, "won": won,
                     "feeder": normalise_org(prior[-1][1])})

    by = collections.defaultdict(list)
    for r in rows:
        by[r["feeder"]].append(r)
    big = {p: v for p, v in by.items() if len(v) >= MIN_ORG_N}

    print(f"debutants with a recoverable feeder promotion: {len(rows)}")
    print(f"distinct organisations: {len(by)}   with n>={MIN_ORG_N}: {len(big)}\n")
    print(f"   {'organisation':26} {'n':>4} {'debut win%':>11} {'95% CI':>18}")
    for p, v in sorted(big.items(), key=lambda kv: -len(kv[1])):
        n = len(v); rate = sum(x["won"] for x in v) / n
        se = math.sqrt(rate * (1 - rate) / n)
        print(f"   {p[:26]:26} {n:4d} {rate:10.1%}   [{rate - 1.96 * se:5.1%},{rate + 1.96 * se:6.1%}]")

    total = sum(len(v) for v in big.values())
    wins = sum(x["won"] for v in big.values() for x in v)
    p0 = wins / total
    chi = sum((sum(x["won"] for x in v) - len(v) * p0) ** 2 / (len(v) * p0 * (1 - p0))
              for v in big.values())
    df = len(big) - 1
    pval = chi2_sf(chi, df)
    print(f"\n   pooled debut win rate {p0:.1%} over N={total}")
    print(f"   chi2 = {chi:.2f}, df = {df}, p = {pval:.3f}")
    print(f"   -> {'SIGNAL' if pval < 0.05 else 'not distinguishable from noise'}")

    recent = sum(1 for v in big.values() for r in v if r["debut"] >= "2020-01-01")
    median = sorted(r["debut"] for v in big.values() for r in v)[total // 2]
    print(f"\n   median debut date in the tierable sample: {median}")
    print(f"   debuts since 2020: {recent} of {total} ({recent / total:.1%})")
    print("\nSee the module docstring: the alternative that needs no promotion "
          "label fails too (trend test z=+0.35, p=0.72).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
