"""
Free / member payload partition.

REDACTION AT THE DATA LAYER, NOT CONDITIONALS IN THE TEMPLATE.

The obvious way to build a paywall into a Jinja site is to thread a `tier`
variable through and wrap each paid element in `{% if tier == 'member' %}`.
That was the original plan here and it is the wrong one. The template renders
model output at roughly sixty sites across 1,334 Jinja expressions, and the
model's rationale and waterfall sit INSIDE the same fight card as the tale of
the tape, so the wrapping is fine-grained rather than structural. Sixty
conditionals is sixty chances to forget one, every forgotten one ships paid
content inside the free payload, and nothing about the rendered page makes the
omission obvious.

Stripping the fields from the data instead means a forgotten template branch
cannot leak anything, because there is nothing left to render. It also gives
the leak test something exact to assert against: the free payload must not
contain any of the values this module removed.

Jinja's default Undefined is lenient -- `fight.preview.waterfall` where
preview is None evaluates to Undefined, which is falsy -- so removing a field
degrades to "renders nothing" rather than raising. That is what makes this
approach safe without touching the template at all.

THE BOUNDARY (agreed with the product owner):

                        analytics          model layer
    graded fights       free               free  <- the track record IS the pitch
    ungraded fights     free               MEMBER

A record nobody can audit is not a record, so every graded call stays public.
What the free tier cannot see is the read on a fight that has not happened.

GRADED, NOT "IN THE future_events LIST". The first version of this keyed off
that list and was wrong in a way worth recording: `future_events` is the
"Coming Up" section of LATER cards, while the card actually being sold renders
from `events[0]`. Redacting future_events left every pick on this weekend's
card sitting in the free payload, which is the exact failure this module
exists to prevent -- caught by scripts/check_free_build.py on its first real
run, which is the entire argument for that check being a hard gate.

Keying on whether a fight has a recorded winner is both correct and
self-maintaining: a card stops being paid content the moment it is graded, with
no date arithmetic, no timezone edge cases, and nothing to remember to flip.

The one deliberate exception is the banner's aggregate confidence strip
("1 HIGH / 7 MED / 5 LOW"). It reveals the shape of the card without revealing
any individual pick, and it is the free tier's only signal that there is
something behind the paywall worth having.
"""

from __future__ import annotations

import copy
import re

# Keys on a single prop/edge row that ARE model output. Anything derived from
# the model's probability belongs here too -- ev_pct and edge_pct are the
# model measured against the book, so publishing them publishes the model.
MODEL_ROW_FIELDS = frozenset({
    "model_prob",
    "model_fair_odds",
    "blended_prob",
    "combined_prob",
    "combined_prob_raw",
    "ev_pct",
    "edge_pct",
    "predicted_method",
    "predicted_favorite",
    "rationale",
    "is_model",
    "low_sample",
})

# Whole context keys that exist only to carry model output about the upcoming
# card. Emptied rather than removed, so the template's `{% for %}` loops and
# `| length` calls keep working and simply render nothing.
MEMBER_ONLY_CONTEXT = (
    "lock_picks",
    "bankroll_parlays",
    "lotto_parlays",
    "model_legs",
    "standout_props",
    "disagreement_props",
)

# Deliberately NOT redacted, and each for a reason worth stating:
#   events                    past cards, picks included -- the proof
#   track_record, units_*     the public record
#   calibration_svg           how well-calibrated the model has been, historically
#   notable_movements*        market data, not model output
#   countdown_confidence_*    aggregate only, no individual pick (see above)
#   moneyline_chart           book prices; the best free teaser on the page


def _collect(value, sink: set) -> None:
    """Record a removed scalar so the leak test can assert on it later."""
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value)
        if text and text.lower() not in ("none", "nan", "0", "0.0"):
            sink.add(text)
    elif isinstance(value, dict):
        for v in value.values():
            _collect(v, sink)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect(v, sink)


# Values distinctive enough that finding one in the payload PROVES a leak.
#
# This is an allowlist, not a denylist, and that direction matters. The first
# version removed things that looked unsafe and kept the rest, which meant the
# manifest filled up with Polymarket token IDs, build dates and book prices --
# all of them free by design, all of them reported as leaks. Hundreds of false
# failures is not a strict check, it is a check nobody will read.
#
# What can only have come from the model is a raw high-precision number:
# 0.6291, -42.673. A book quotes "+102", the page prints "2026-08-22", and
# "Anthony Hernandez vs Gregory Rodrigues" is the name of a fight anyone can
# look up -- none of those are secrets, and all three were being reported as
# leaks because they sit inside the structures the redaction walks.
#
# TEXT IS DELIBERATELY NOT ASSERTED ON. An earlier version also collected any
# string of 25+ characters, reasoning that rationale sentences are
# distinctive. They are, but so are fight labels and market names, which live
# in the same dictionaries and are public. Rationale does not need a value
# assertion anyway: it only ever renders inside the WHY block, and the marker
# check already fails on `data-nav="why"` appearing in an ungraded card. Two
# checks, each precise about a different thing, beat one check that is vague
# about both.
_HIGH_PRECISION = re.compile(r"^-?\d+\.\d{3,}$")


def _assertable(values: set) -> set:
    return {v for v in values if _HIGH_PRECISION.match(v)}


def _redact_row(row: dict, sink: set) -> dict:
    """Drop every model-derived key from one prop row."""
    for key in MODEL_ROW_FIELDS:
        if key in row:
            _collect(row[key], sink)
    return {k: v for k, v in row.items() if k not in MODEL_ROW_FIELDS}


def is_graded(fight: dict) -> bool:
    """
    Has this fight happened and been scored?

    A graded fight is free in full, model read included -- that record is the
    product's whole claim to being worth paying for.
    """
    return bool(fight.get("winner") or fight.get("result_label"))


def redact_fight(fight: dict, sink: set | None = None) -> dict:
    """
    Strip the model layer from ONE upcoming fight, leaving its analytics.

    Returns a copy. The caller may be holding the same fight dict that a
    member build will render, and mutating it in place would redact that
    build too -- a bug that would look like the paywall working perfectly
    right up until nobody could see what they paid for.
    """
    sink = set() if sink is None else sink
    out = copy.deepcopy(fight)

    # The entire model preview: pick, confidence, method, waterfall, rationale.
    _collect(out.get("preview"), sink)
    out["preview"] = None

    # Per-market rows keep their market, selection, book and price -- all of
    # which are market facts -- and lose the model's opinion of them.
    if isinstance(out.get("edges"), list):
        out["edges"] = [_redact_row(r, sink) for r in out["edges"]]

    groups = out.get("market_groups")
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, dict) and isinstance(group.get("rows"), list):
                group["rows"] = [_redact_row(r, sink) for r in group["rows"]]

    return out


def redact_context(context: dict) -> tuple[dict, list[str]]:
    """
    Free-tier view of the full render context.

    Every event list is walked, and each fight is judged individually on
    whether it has been graded. A part-way-through card therefore reveals each
    fight as its result lands, which is the behaviour a reader expects on a
    live fight night and falls out of the rule for free.

    Returns the redacted context and every scalar value that was removed, so
    scripts/check_free_build.py can assert none of them survived into the
    rendered payload. Deriving that list here rather than restating it in the
    checker is what stops the two drifting apart.
    """
    out = dict(context)
    sink: set[str] = set()

    for key in MEMBER_ONLY_CONTEXT:
        if key in out:
            _collect(out[key], sink)
            out[key] = []

    # BOTH lists. `events` holds the card being sold right now; future_events
    # holds the ones after it. Missing the first was the original bug.
    for list_key in ("events", "future_events"):
        source = out.get(list_key)
        if not isinstance(source, list):
            continue
        rebuilt = []
        for event in source:
            ev = dict(event)
            if isinstance(ev.get("fights"), list):
                ev["fights"] = [
                    f if is_graded(f) else redact_fight(f, sink)
                    for f in ev["fights"]
                ]
            rebuilt.append(ev)
        out[list_key] = rebuilt

    out["tier"] = "free"
    return out, sorted(sink), sorted(_assertable(sink))
