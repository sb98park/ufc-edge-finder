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
import datetime as _dt
import os
import unicodedata

from src.names import _normalize_name

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
    # PUNCTUATION TOO, which this did not do while its own docstring called it
    # "the same fold as the rest of the project". ufcstats writes "Benoit
    # Saint Denis" and "Abdul Rakhman Yakhyaev"; fighters.csv carries the
    # hyphens. Neither name is accented, so the diacritic machinery above was
    # irrelevant -- a single hyphen was the whole gap, and it resolved both
    # men to zero bouts. Abdul-Rakhman Yakhyaev is booked for UFC 333 and
    # rendered a scouting drawer from 0 bouts when 3 exist, while
    # has_measured_method_rates returned False and gated him out of method
    # legs on a false premise.
    #
    # Delegates the punctuation/whitespace half to card_matcher, the canonical
    # fold, rather than restating it -- but keeps _STROKE_FOLD first, because
    # NFKD does not decompose the Scandinavian stroke and card_matcher would
    # drop it.
    s = str(name).strip().translate(_STROKE_FOLD)
    return _normalize_name(s)


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


_TOKEN_INDEX: dict | None = None
_TOKEN_INDEX_FOR: int | None = None


def _BY_TOKENS(table: dict) -> dict:
    """Token-sorted view of the records table, rebuilt when the table changes."""
    global _TOKEN_INDEX, _TOKEN_INDEX_FOR
    if _TOKEN_INDEX is None or _TOKEN_INDEX_FOR != id(table):
        _TOKEN_INDEX = {" ".join(sorted(k.split())): v for k, v in table.items()}
        _TOKEN_INDEX_FOR = id(table)
    return _TOKEN_INDEX


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
    if rec is None:
        # NAME ORDER, as a fallback only. _fold strips accents but keeps token
        # order, and this table is built from UFCStats, which writes many
        # Chinese fighters family-name-first: it holds "Wang Cong" while our
        # roster, our results and ESPN all say "Cong Wang". Exact matching
        # returned nothing for her, so she fell to the divisional prior with
        # six real UFC bouts on file.
        #
        # Safe because it is checked ONLY after an exact miss, and because no
        # two distinct fighters in this table collide under a token-sorted key
        # -- verified across all 2,733 of them. Do not promote it above the
        # exact lookup: an exact hit is always the stronger claim.
        rec = _BY_TOKENS(tbl).get(" ".join(sorted(_fold(name).split())))
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


def has_measured_method_rates(name, table: dict | None = None,
                              min_fights: int = MIN_UFC_FIGHTS) -> bool:
    """
    Whether this fighter's method rates are MEASURED or merely assumed.

    rates_or_prior answers the same question by returning something either
    way, which is right for producing a number and wrong for deciding whether
    to bet on it. Two debutants in the same division receive identical rates
    by construction -- divisional_fallback_rates halves the divisional prior
    for both sides -- so the method model's inputs for that bout carry no
    fighter-specific information at all, and its output is the divisional
    base rate wearing a fight's name.

    Measured on the 2026-08-29 China card: 11 of 26 fighters had fewer than
    three UFC bouts, and that card's P(decision) spread across 13 fights was
    0.121, the narrowest of the last six cards (median 0.234). The spread
    alone does not prove causation -- across those six cards the correlation
    between share-of-known-fighters and spread was -0.15, i.e. nothing -- but
    the construction argument does not need the correlation: where both sides
    fall back, there is provably no fighter-level signal to have.
    """
    return ufc_method_rates(name, table, min_fights) is not None


def rates_or_prior(name, priors: dict, weight_class, table: dict | None = None):
    """UFC rates where they exist, the divisional prior where they don't."""
    got = ufc_method_rates(name, table)
    if got is not None:
        return got
    return divisional_fallback_rates(priors, weight_class)


# ---------------------------------------------------------------- point in time

EVENTS = "data/ufc_event_details.csv"
_DATED_CACHE = None
_DATED_KEY = None


def load_dated_ufc_bouts(path: str = RESULTS, events_path: str = EVENTS) -> dict:
    """
    {folded name: [(date, is_win, kind), ...]} ascending, over decided UFC bouts.

    THE SAME PARSE AS load_ufc_records, deliberately sharing _method_kind, the
    W/L outcome gate and _fold. A validation harness that re-derives the
    serving path measures its own re-derivation: the first attempt at this
    scored AUC 0.804 against the production model's 0.720, which is a backtest
    beating the live model and therefore a receipt for leakage, not a result.

    The only thing added is a DATE, which ufc_fight_results.csv does not carry
    -- it lives in ufc_event_details.csv, joined on EVENT.

    A BOUT WHOSE EVENT IS NOT IN THAT FILE IS DROPPED, because a bout that
    cannot be placed in time cannot be windowed by one. Measured at 27 of
    8,859 rows (0.30%), on three events -- two recent cards the event file has
    not caught up with, and one Road to UFC. So a fighter who fought on one of
    those carries a denominator one or two short of production's, which
    tests/test_pit_method_rates.py measures rather than assumes: everyone else
    must match to the last decimal.

    Cached on both files' mtime and size, like the undated table.
    """
    global _DATED_CACHE, _DATED_KEY
    try:
        key = tuple((p, os.stat(p).st_mtime_ns, os.stat(p).st_size)
                    for p in (path, events_path))
    except OSError:
        key = (path, events_path, None)
    if _DATED_CACHE is not None and _DATED_KEY == key:
        return _DATED_CACHE

    # NORMALISED TO ISO ON THE WAY IN. ufc_event_details.csv writes
    # "August 15, 2026", and the windowing below compares date STRINGS -- so
    # left raw, "August..." sorts above every 4-digit year and the very first
    # bout looks later than any cutoff. Every fighter then read as having no
    # UFC history at all, which is a silent None rather than an error, and the
    # harness would have scored a model that knew nothing.
    dates: dict[str, str] = {}
    try:
        with open(events_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ev, dt = str(row.get("EVENT", "")).strip(), str(row.get("DATE", "")).strip()
                if not ev or not dt:
                    continue
                try:
                    dates[ev] = _dt.datetime.strptime(dt, "%B %d, %Y").strftime("%Y-%m-%d")
                except ValueError:
                    # Already ISO, or a shape this does not know. An unparsed
                    # date is dropped below rather than guessed at.
                    dates[ev] = dt if len(dt) >= 10 and dt[4] == "-" else ""
    except FileNotFoundError:
        dates = {}

    out: dict[str, list] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                parts = [p.strip() for p in str(row.get("BOUT", "")).split(" vs. ")]
                if len(parts) != 2:
                    continue
                outcome = str(row.get("OUTCOME", "")).strip()
                if outcome == "W/L":
                    winner, loser = parts
                elif outcome == "L/W":
                    loser, winner = parts
                else:
                    continue
                when = dates.get(str(row.get("EVENT", "")).strip())
                if not when:
                    continue          # undateable bout cannot be placed in time
                kind = _method_kind(row.get("METHOD"))
                out.setdefault(_fold(winner), []).append((when, True, kind))
                out.setdefault(_fold(loser), []).append((when, False, kind))
    except FileNotFoundError:
        out = {}
    for v in out.values():
        v.sort(key=lambda t: t[0])

    _DATED_CACHE, _DATED_KEY = out, key
    return out


def ufc_method_rates_as_of(name, when, table: dict | None = None,
                           min_fights: int = MIN_UFC_FIGHTS):
    """
    ufc_method_rates restricted to bouts STRICTLY BEFORE `when`.

    Same tuple, same denominator (fights, not wins), same min_fights gate and
    the same None-means-unknown contract, so a caller can swap one for the
    other and change nothing but the window.

    `when` is compared as an ISO date string, which is what both files store
    and what sorts correctly without parsing 17,000 dates per call.
    """
    tbl = table if table is not None else load_dated_ufc_bouts()
    key = _fold(name)
    bouts = tbl.get(key)
    if bouts is None:
        # Same token-order fallback as the undated lookup, and only after an
        # exact miss, for the same UFCStats family-name-first reason.
        want = " ".join(sorted(key.split()))
        for k, v in tbl.items():
            if " ".join(sorted(k.split())) == want:
                bouts = v
                break
    if not bouts:
        return None
    cut = str(when)[:10]
    fights = ko_w = sub_w = ko_l = sub_l = 0
    for d, won, kind in bouts:
        if d[:10] >= cut:
            break                      # sorted, so nothing later can qualify
        fights += 1
        if kind == "ko":
            ko_w += won
            ko_l += not won
        elif kind == "sub":
            sub_w += won
            sub_l += not won
    if fights < min_fights:
        return None
    n = float(fights)
    return (ko_w / n, sub_w / n, ko_l / n, sub_l / n)


def rates_or_prior_as_of(name, when, priors: dict, weight_class,
                         table: dict | None = None):
    """rates_or_prior with the clock wound back. Same fallback, same halving."""
    got = ufc_method_rates_as_of(name, when, table)
    if got is not None:
        return got
    return divisional_fallback_rates(priors, weight_class)
