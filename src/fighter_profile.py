"""
Per-fighter attribute percentiles, ranked against the whole UFC roster.

WHAT THIS IS FOR, AND WHAT IT DELIBERATELY IS NOT. The scout row used to list
the model's matchup factors -- "+ Wrestling", "+ Durability" -- which is the
same decomposition the probability waterfall prints directly beneath it.
Measured on a full build, 191 of 210 labels (91%) appeared in both places with
the same numbers. The badges were matchup-relative by construction
(build_factor_badges compares fighter A against fighter B), so they could
never say anything the waterfall did not already say better.

This answers the other question: not "why does the model favour him tonight"
but "what kind of fighter is this". Percentiles are absolute -- a fighter's
rank does not move when his opponent changes -- so nothing here can duplicate
the waterfall even in principle.

WHY UFC BOUTS ONLY. Same reason as src/ufc_method_rates.py, which documents it
at length: a prospect builds a regional record finishing weak opposition, so
pre-UFC history pads wins and adds almost no losses. Every fighter arrives
looking more dangerous and more durable than he is. ufcstats does not publish
regional per-round data anyway, so this is a constraint and a preference at
the same time.

THE THREE-BOUT FLOOR, MEASURED RATHER THAN ASSUMED. Ranking a fighter on his
first k bouts and comparing with where he eventually settles, across the 631
fighters with 10 or more bouts:

    bouts    median error    share more than 25 percentile points wrong
      1         27pp                        54%
      2         21pp                        43%
      3         18pp                        37%
      6         11pp                        21%

A one-bout profile is a coin flip. Dropping the floor to 1 would lift
all-fights coverage from 65% to 87%, and every point of that gain would be a
number wrong more often than not -- a confident wrong answer is worse than an
honest blank, so the floor stays at 3.

WHY SHRINKAGE ON TOP OF THE FLOOR. Three bouts is the floor, not a guarantee:
36 of the 131 profiled fighters sit between 3 and 5 bouts, where the table
above still shows 37% to 25% badly wrong. Every rate is therefore pulled
toward the roster average in proportion to how little cage time backs it,
which buys roughly one extra bout of accuracy (at one bout, 54% -> 41% badly
wrong) and stops a fighter screaming "11th percentile" off three fights.

WHAT CANNOT BE FIXED. Of 167 booked fighters, 36 have no profile, and all 36
are simply short of bouts -- their pit_stats count matches their
ufc_fight_results count exactly, so not one is a scrape or name-match failure.
Twelve have never fought in the UFC at all. The gap concentrates where it
matters least: 89% of main events and 88% of co-mains are fully profiled
against 33% of early prelims, which is close to the definition of an early
prelim.
"""

from __future__ import annotations

import csv
from bisect import bisect_left

from src.fighter_history import fold_name

PIT = "data/pit_stats.csv"

# Below this a profile is not shown at all. See the table above.
MIN_UFC_BOUTS = 3

# Shrinkage strength, in minutes of an average fighter. 15 is one full
# three-round fight, so a debutant's rate is weighted 50/50 against the roster
# and a ten-fight veteran is barely touched.
PRIOR_MINUTES = 15.0

_COUNTS = ("sig_str_landed", "sig_str_att", "sig_str_absorbed", "td_landed",
           "td_att", "td_faced", "td_stuffed", "kd_for", "kd_against",
           "sub_att", "ctrl_seconds", "fight_seconds")


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# label -> (function of summed counts -> rate or None, higher_is_better)
# Six attributes, chosen so that three of them -- takedown volume, takedown
# defence and control -- appear on no other panel of the fight card. The radar
# chart already carries striking pace, knockdown rate and damage absorbed.
def _rate_strike_acc(c):
    return 100.0 * c["sig_str_landed"] / c["sig_str_att"] if c["sig_str_att"] > 0 else None


def _per15(c, key):
    mins = c["fight_seconds"] / 60.0
    return 15.0 * c[key] / mins if mins > 0 else None


ATTRIBUTES = [
    ("Strike acc",  lambda c: _rate_strike_acc(c),                                      True),
    ("KO power",    lambda c: _per15(c, "kd_for"),                                      True),
    ("TD volume",   lambda c: _per15(c, "td_att"),                                      True),
    ("TD defence",  lambda c: (100.0 * c["td_stuffed"] / c["td_faced"]) if c["td_faced"] > 0 else None, True),
    ("Control",     lambda c: (100.0 * c["ctrl_seconds"] / c["fight_seconds"]) if c["fight_seconds"] > 0 else None, True),
    # Fewer knockdowns absorbed is better, so this one inverts on ranking.
    ("Chin",        lambda c: _per15(c, "kd_against"),                                  False),
]


def _aggregate(path: str = PIT) -> dict:
    """{folded name: {count columns summed, 'bouts': n}} over every UFC bout."""
    out: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = fold_name(row.get("name", ""))
                if not key:
                    continue
                rec = out.setdefault(key, {c: 0.0 for c in _COUNTS} | {"bouts": 0})
                rec["bouts"] += 1
                for c in _COUNTS:
                    rec[c] += _num(row.get(c))
    except FileNotFoundError:
        return {}
    return out


def _shrunk(counts: dict, pop_rate: float, fn) -> float | None:
    """
    A fighter's rate pulled toward the roster average by how little cage time
    backs it. Weight is minutes / (minutes + PRIOR_MINUTES), so the prior
    fades out on its own as a career accumulates rather than at a cliff.
    """
    raw = fn(counts)
    mins = counts["fight_seconds"] / 60.0
    if mins <= 0:
        return None
    if raw is None:
        # The attribute never came up -- no takedowns faced, say. The roster
        # average is a better guess than omitting the row, and with no
        # observations it IS the estimate.
        return pop_rate
    w = mins / (mins + PRIOR_MINUTES)
    return w * raw + (1.0 - w) * pop_rate


def build_profiles(names, path: str = PIT) -> dict:
    """
    {folded name: {"bouts": n, "pct": {label: 0-100} or None}} for `names`.

    "pct" is None below MIN_UFC_BOUTS; "bouts" is always the true count, so the
    UI can say "2 fights" rather than showing an unexplained blank.
    """
    agg = _aggregate(path)
    if not agg:
        return {}

    # Population totals give the prior each fighter is shrunk toward.
    totals = {c: sum(r[c] for r in agg.values()) for c in _COUNTS}
    pop_rates = {label: (fn(totals) or 0.0) for label, fn, _ in ATTRIBUTES}

    # Rank against every fighter who clears the floor, not just the booked
    # ones -- a percentile against 14 people on this card would mean nothing,
    # and would move every week as the card changed.
    eligible = [k for k, r in agg.items() if r["bouts"] >= MIN_UFC_BOUTS]
    index: dict[str, list] = {}
    for label, fn, _ in ATTRIBUTES:
        vals = [v for v in (_shrunk(agg[k], pop_rates[label], fn) for k in eligible) if v is not None]
        index[label] = sorted(vals)

    wanted = {fold_name(n): n for n in names if str(n).strip()}
    out = {}
    for folded in wanted:
        rec = agg.get(folded)
        bouts = rec["bouts"] if rec else 0
        if not rec or bouts < MIN_UFC_BOUTS:
            out[folded] = {"bouts": bouts, "pct": None}
            continue
        pct = {}
        for label, fn, higher_better in ATTRIBUTES:
            v = _shrunk(rec, pop_rates[label], fn)
            col = index[label]
            if v is None or not col:
                continue
            p = 100.0 * bisect_left(col, v) / len(col)
            pct[label] = int(round(p if higher_better else 100.0 - p))
        out[folded] = {"bouts": bouts, "pct": pct or None}
    return out


def summarise(profiles: dict) -> dict:
    got = sum(1 for v in profiles.values() if v["pct"])
    return {"profiled": got, "total": len(profiles),
            "no_bouts": sum(1 for v in profiles.values() if v["bouts"] == 0)}
