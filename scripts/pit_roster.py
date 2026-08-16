"""
Point-in-time fighter rows: a roster as it stood the night before any fight.

WHY THIS EXISTS. Both backtest harnesses in this project score a model that
is not the production model, and say so. validate_pointintime_stats.py feeds
`{"name": x}` for anyone missing from the current roster; validate_market_
blend.py does the same and reports the consequence -- only 401 of 10,692
history rows (4%) have BOTH corners on the 293-name roster, so ~96% of
scored fights run with every record-, age- and physical-dependent style term
gated off. That model reads ~54.5% where the production one logs 74.4%.

walkforward_backtest.py's docstring calls fixing this "a future data
project". This is that project, for the parts that are genuinely
reconstructable:

  EXACT, from fight_history alone -- no network, no cache:
    wins / losses / draws
    ko_wins / sub_wins / dec_wins / ko_losses / sub_losses / dec_losses
    last_fight_date          (drives the layoff and quick-return terms)

  EXACT, from the ESPN stat timeline:
    espn_fights              (drives anything gated on sample size)

  STATIC -- current values are correct for any past date:
    height_in / reach_in / stance

  DERIVED:
    age                      (current age minus years elapsed; ±1yr, which
                              matters only at the age-cliff boundary)

  NOT RECONSTRUCTABLE, deliberately left absent so their terms gate off
  exactly as they do in production for an unknown fighter:
    missed_weight_count, short_notice, control_time_pct, weight-class
    history, and the recency-weighted *_r columns.

THE HONEST LIMIT. 21% of history rows carry no method (regional bouts
imported without one), so method SPLITS are undercounted for fighters whose
early career sits in that gap. Win/loss totals are unaffected. A fighter's
reconstructed splits can therefore sum to less than their reconstructed
record, which is correct rather than broken: it says "we know he won 12, we
know how 9 of them ended".
"""

import datetime as dt
from collections import defaultdict

import pandas as pd

# fight_history's method column is mostly terse codes with a few long-form
# strays (19 "Decision - Unanimous", 15 "Submission"). Matched by prefix
# rather than equality so a new spelling degrades to "no split" instead of
# being silently miscounted as a decision.
_KO = ("ko", "tko")
_SUB = ("sub",)
_DEC = ("dec",)


def _method_bucket(method) -> str | None:
    if method is None or (isinstance(method, float) and pd.isna(method)):
        return None
    m = str(method).strip().lower()
    if not m or m == "nan":
        return None
    if m.startswith(_DEC):
        return "dec"
    if m.startswith(_SUB):
        return "sub"
    if m.startswith(_KO) or "ko/tko" in m:
        return "ko"
    return None          # DQ, NC, anything unrecognised: counted in W/L, no split


def build_fight_index(history_df: pd.DataFrame) -> dict:
    """
    {folded_name: [(date, won: bool, bucket: str|None), ...]} sorted ascending.

    Per-fighter lists rather than one global timeline: a query then scans
    only that fighter's own handful of bouts, which keeps ~5,000 lookups
    cheap instead of rescanning 10,700 rows each time.
    """
    idx = defaultdict(list)
    df = history_df.dropna(subset=["date"])
    for r in df.itertuples(index=False):
        a, b, w = str(r.fighter_a).strip(), str(r.fighter_b).strip(), str(r.winner).strip()
        bucket = _method_bucket(getattr(r, "method", None))
        when = r.date.to_pydatetime() if hasattr(r.date, "to_pydatetime") else r.date
        for name in (a, b):
            idx[name.lower()].append((when, name.lower() == w.lower(), bucket))
    for n in idx:
        idx[n].sort(key=lambda t: t[0])
    return idx


_REC_FIELDS = ("wins", "losses", "ko_wins", "sub_wins", "dec_wins",
               "ko_losses", "sub_losses", "dec_losses")


def record_as_of(fight_index: dict, name: str, when, current: dict | None = None) -> dict:
    """
    Record as it stood before `when`.

    TWO METHODS, AND THE CHOICE MATTERS. fighters.csv stores CAREER records
    -- every promotion -- while fight_history is overwhelmingly UFC. Volkov
    is 40-11 in the roster and 14-5 in history. So counting history forward
    produces a UFC-only record, which is not the quantity production feeds
    the model, and a backtest built on it would be differently unfair rather
    than fair.

      SUBTRACT (preferred, used when the fighter has a current roster row):
        start from the true career total and remove the bouts known to have
        happened on or after `when`. Exact for the subtraction, and lands in
        the same units production uses.

      ACCUMULATE (fallback, for everyone off the current 293-name roster):
        count history forward. Undercounts a career that began elsewhere,
        but it is what exists, and it beats the status quo of handing the
        model no record at all.

    `record_source` is returned so analysis can stratify on it rather than
    quietly averaging two different measurements together.
    """
    key = str(name).strip().lower()
    fights = fight_index.get(key, [])

    before, after, last, last_won, last_bucket = _split_counts(fights, when)

    def _finish(out: dict) -> dict:
        out["last_fight_date"] = last.strftime("%Y-%m-%d") if last else None
        # LAST-FIGHT OUTCOME AND METHOD. Both are already in the index and
        # neither was being emitted, which left quick_return_penalty unable to
        # fire on a single backtested fight -- it gates on
        # last_fight_result == "L" and last_fight_method in (KO/TKO, SUB), and
        # a row lacking those columns fails the gate silently. That looked
        # like a data limitation and was a missing assignment.
        if last is not None:
            out["last_fight_result"] = "L" if last_won is False else "W"
            out["last_fight_method"] = {
                "ko": "KO/TKO", "sub": "SUB", "dec": "DEC"}.get(last_bucket)
        return out

    if current:
        out = {}
        usable = True
        for f in _REC_FIELDS:
            cur = current.get(f)
            if cur is None or (isinstance(cur, float) and pd.isna(cur)):
                usable = False
                break
            # RAW, UNCLAMPED. The clamp used to happen here, which is what
            # made the guard below vacuous.
            out[f] = int(cur) - after[f]
        if usable:
            # A career total that cannot cover its own known subsequent
            # fights means the two sources disagree about this fighter;
            # fall through to accumulate rather than emit a clamped
            # number that looks authoritative.
            #
            # THIS GUARD WAS DEAD. It read:
            #   if out["wins"] >= before["wins"] * 0 and out["losses"] >= 0
            # and both operands had already been through max(0, ...), so
            # `>= 0` and `>= anything * 0` were unconditionally true and the
            # documented fall-through was unreachable. Every negative
            # remainder was silently clamped to 0 and returned as
            # "subtracted", which is exactly the authoritative-looking
            # clamped number the comment says not to emit.
            #
            # One visible consequence: with the method splits each clamped
            # independently, ko_losses + sub_losses could exceed losses,
            # driving finish_loss_rate above 1.0 and durability_adjustment to
            # +/-240 -- twice its documented maximum -- on 0.7% of corners.
            if all(v >= 0 for v in out.values()) and \
                    out["ko_losses"] + out["sub_losses"] <= out["losses"] and \
                    out["ko_wins"] + out["sub_wins"] <= out["wins"]:
                out["record_source"] = "subtracted"
                return _finish(out)

    out = dict(before)
    out["record_source"] = "accumulated"
    return _finish(out)


def _split_counts(fights, when):
    """(counts before, counts on/after, last date before, its result, its method)."""
    before = {f: 0 for f in _REC_FIELDS}
    after = {f: 0 for f in _REC_FIELDS}
    last = last_won = last_bucket = None
    for d, won, bucket in fights:
        tgt = before if d < when else after
        if d < when:
            last, last_won, last_bucket = d, won, bucket
        if won:
            tgt["wins"] += 1
            if bucket:
                tgt[f"{bucket}_wins"] += 1
        else:
            tgt["losses"] += 1
            if bucket:
                tgt[f"{bucket}_losses"] += 1
    return before, after, last, last_won, last_bucket


def _age_as_of(current_age, when, today=None) -> float | None:
    """
    Current age walked back by the elapsed years.

    Approximate by up to a year, because fighters.csv stores age rather than
    a birthdate. That resolution is irrelevant to every term except the age
    cliff, where a fighter within a year of the threshold could land on
    either side -- accepted, and better than the alternative of omitting age
    entirely, which would gate the term off for the whole backtest and hide
    whatever signal it carries.
    """
    if current_age is None or pd.isna(current_age):
        return None
    today = today or dt.datetime.now()
    years = (today - when).days / 365.25
    aged = float(current_age) - years
    return round(aged, 1) if aged > 15 else None      # nonsense before their teens


def roster_as_of(name, when, fight_index, static_rows, espn_timelines=None, today=None) -> dict:
    """
    A fighters.csv-shaped dict for `name` as of `when`.

    static_rows: {folded_name: row_dict} from the CURRENT fighters.csv, read
    only for height/reach/stance/age -- never for record or stats, which are
    exactly the fields contaminated by the future.
    """
    folded = str(name).strip().lower()
    static = static_rows.get(folded) or {}
    row = {"name": name}
    row.update(record_as_of(fight_index, name, when, current=static or None))

    for col in ("height_in", "reach_in", "stance"):
        v = static.get(col)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            row[col] = v
    row["age"] = _age_as_of(static.get("age"), when, today)

    if espn_timelines is not None:
        tl = espn_timelines.get(folded)
        if tl is not None:
            row["espn_fights"] = sum(1 for d, _ in tl if d < when)

    return row
