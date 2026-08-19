"""
Per-fighter method rates computed from UFC bouts only.

WHY THIS EXISTS. The method model was trained on UFC-only, point-in-time
career profiles (research_survival_model.py, the Career class: ko_wins/fights
where both count UFC bouts and update() advances as the walk steps forward).
It was SERVED from data/fighters.csv, which holds whole-career records
including the regional circuit.

Both halves of the fraction came from different universes, and they moved in
opposite directions. Measured over the 127 booked fighters with both:

    mean career fights 21.9     mean UFC fights 10.4        2.10x

    own_ko_rate    served 0.3700   trained 0.2480    1.49x too HIGH
    own_sub_rate   served 0.1653   trained 0.1025    1.61x too HIGH
    opp_ko_lost    served 0.0603   trained 0.0982    0.61x too LOW

The split is not noise, it is the shape of a fighting career: a prospect
builds a regional record winning by finish against weak opposition, and gets
signed on the back of it. Pre-UFC history therefore pads the numerator on WINS
and the denominator on everything, while adding almost no losses -- nobody is
signed off a losing streak. So every fighter arrived looking more finishing
AND more durable than he is, at the same time.

Effect on what the site published, across 51 booked fights:

    P(KO)          +2.75pp     t =  3.50
    P(submission)  +2.20pp     t =  5.85
    P(decision)    -4.95pp     t = -7.07     too low on 43 of 51

THE PART THAT MATTERS MOST. The market's one measured inefficiency is that it
underprices decisions by 4.03pp (implied 45.77% against an actual 49.80%,
z=5.98, n=5,464). Served on career totals the model underpriced decisions by
4.95pp -- it was reproducing the market's own error, at the market's own
magnitude, and so could not see the single edge this project has found.

THIS IS THE SECOND HALF OF A FIX ALREADY MADE. compute_divisional_method_priors
in matchup_model.py carries the same finding for DIVISIONAL rates and says so
in its docstring, with a table showing Lightweight at a true 0.30/0.22/0.48
against a roster-derived 0.44/0.32/0.25. That fix landed for divisions and
never reached the per-fighter rates.

WHY NOT REFIT THE MODEL ON CAREER TOTALS INSTEAD. The training walk is
point-in-time by construction, so a 2019 prediction sees only pre-2019 fights.
fighters.csv is a snapshot of today. Refitting to match it would train on
future-contaminated features -- trading a known bias for a leak, which is a
worse trade. It would also invalidate every constant tuned against UFC-only
rates (DURABILITY_SCALE, the method grid, the divisional priors above).
"""

from __future__ import annotations

import csv
import os
import unicodedata

RESULTS = "data/ufc_fight_results.csv"

# Below this, a fighter's own rates are noise -- one UFC fight won by KO reads
# as a 100% KO rate. Mirrors MIN_PRIOR_FIGHTS in the training walk, which
# returns None rather than a rate for exactly the same reason.
MIN_UFC_FIGHTS = 3

_STROKE_FOLD = str.maketrans({
    "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D",
    "ħ": "h", "Ħ": "H", "ŧ": "t", "Ŧ": "T", "ß": "ss",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ı": "i",
})


def _fold(name) -> str:
    """
    Same fold as the rest of the project. ufcstats publishes ASCII while the
    roster carries the real spelling, so without stripping diacritics every
    accented fighter resolves to nothing -- the defect that left 13 roster
    fighters with no control-time data at all.
    """
    s = str(name).strip().translate(_STROKE_FOLD)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.encode("ascii", "ignore").decode().lower()


def _method_kind(method) -> str | None:
    m = str(method).upper()
    if "KO" in m or "TKO" in m:
        return "ko"
    if "SUB" in m:
        return "sub"
    return None


_CACHE: dict | None = None
_CACHE_KEY = None


def load_ufc_records(path: str = RESULTS) -> dict:
    """
    {folded name: {fights, ko_wins, sub_wins, ko_losses, sub_losses}} over
    every resolved UFC bout. Cached on the file's mtime and size, so a spine
    refresh is picked up without a restart and a normal run reads the file
    once rather than once per fight.
    """
    global _CACHE, _CACHE_KEY
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        key = (path, None, None)
    if _CACHE is not None and _CACHE_KEY == key:
        return _CACHE

    table: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                parts = [p.strip() for p in str(row.get("BOUT", "")).split(" vs. ")]
                if len(parts) != 2:
                    continue
                outcome = str(row.get("OUTCOME", "")).strip()
                # Only W/L and L/W are decided. Draws and no-contests carry no
                # method information about either fighter and must not inflate
                # the denominator -- they would depress every rate silently.
                if outcome == "W/L":
                    winner, loser = parts
                elif outcome == "L/W":
                    loser, winner = parts
                else:
                    continue
                for who in (winner, loser):
                    table.setdefault(_fold(who), {
                        "fights": 0, "ko_wins": 0, "sub_wins": 0,
                        "ko_losses": 0, "sub_losses": 0,
                    })["fights"] += 1
                kind = _method_kind(row.get("METHOD"))
                if kind:
                    table[_fold(winner)][f"{kind}_wins"] += 1
                    table[_fold(loser)][f"{kind}_losses"] += 1
    except FileNotFoundError:
        table = {}

    _CACHE, _CACHE_KEY = table, key
    return table


def ufc_method_rates(name, table: dict | None = None, min_fights: int = MIN_UFC_FIGHTS):
    """
    (ko_rate, sub_rate, ko_lost, sub_lost) over UFC bouts, or None when the
    fighter has too little UFC history to say anything.

    None rather than a number on purpose. The career fallback that used to
    stand here always produced *something*, and that something was wrong in a
    known direction -- a confident answer built from regional-circuit finishes.
    A caller that gets None can reach for the divisional prior, which is
    computed from real UFC bouts and is right on average.
    """
    tbl = table if table is not None else load_ufc_records()
    rec = tbl.get(_fold(name))
    if not rec or rec["fights"] < min_fights:
        return None
    n = float(rec["fights"])
    return (rec["ko_wins"] / n, rec["sub_wins"] / n,
            rec["ko_losses"] / n, rec["sub_losses"] / n)


def divisional_fallback_rates(priors: dict, weight_class):
    """
    Stand-in rates for a fighter with too little UFC history.

    HALF the divisional rate on each side, which is the arithmetic and not a
    fudge: a divisional KO rate is the share of BOUTS ending that way, and a
    bout has a winner and a loser. In a division where 30% of fights end by
    knockout, the average fighter wins 15% of his fights by KO and loses 15%
    of them the same way. Splitting it evenly is what makes an unknown fighter
    average rather than either a finisher or a victim.
    """
    from src.matchup_model import divisional_prior_for
    # Keys are "KO/TKO" and "SUB" -- the labels the priors are built with,
    # not lowercase shorthands. A wrong key here would silently return the
    # fallback float for every division and look like it worked.
    ko = divisional_prior_for(priors, weight_class, "KO/TKO", 0.30)
    sub = divisional_prior_for(priors, weight_class, "SUB", 0.20)
    return (ko / 2.0, sub / 2.0, ko / 2.0, sub / 2.0)


def rates_or_prior(name, priors: dict, weight_class, table: dict | None = None):
    """UFC rates where they exist, the divisional prior where they don't."""
    got = ufc_method_rates(name, table)
    if got is not None:
        return got
    return divisional_fallback_rates(priors, weight_class)
