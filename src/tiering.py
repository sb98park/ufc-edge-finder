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
    # THE PLAYS THEMSELVES ARE THE PRODUCT. plays_card and plays_rows are
    # this weekend's staked bets -- the single most valuable model output on
    # the site, and the reason to pay for it. plays_record is deliberately NOT
    # here: it is the settled record of bets already made and belongs with
    # track_record among the things a free reader gets to audit before paying.
    "plays_card",
    "plays_rows",
    # THIS WAS NOT REDACTED, AND IT WAS PUBLISHING THE MODEL LAYER FOR FREE.
    # Measured on the shipped free payload: "Denise Gomes vs Yan Xiaonan ·
    # Moneyline +153 · 62.4% model confidence", the full rationale, and the
    # edge spelled out -- "a cushion of 22.9 points". That is the pick, the
    # probability and the disagreement, which is precisely what the wall
    # exists to hold back, given away on the section next to the one that
    # asks for money.
    #
    # AND THE LEAK CHECK COULD NOT SEE IT. check_free_build asserts that every
    # value the redaction REMOVED is absent from the free payload -- it reads
    # the manifest the redaction writes, so it tracks that redaction
    # automatically. Which is exactly its blind spot: it catches a redaction
    # that has broken, and is structurally incapable of catching one that was
    # never written. A key missing from this tuple contributes nothing to the
    # manifest, so there is nothing to assert and the gate passes green.
    "favorite_picks",
    "bankroll_parlays",
    "model_legs",
    "standout_props",
    "disagreement_props",
)

# KEYS THAT ARE MOSTLY FREE AND CARRY ONE MEMBER-ONLY LIMB.
#
# whats_new_snapshot is the "what changed since you last looked" banner. Its
# `movements` list is market data and belongs to everyone. Its `standout` list
# is built FROM standout_props, before redaction, and was shipping the model's
# five clearest reads to the free payload with the edge attached:
#
#     {"key": "Rei Tsuruya|Moneyline", "label": "Rei Tsuruya Moneyline",
#      "edge_pct": -6.05}
#
# Emptying the whole key would take the movement banner down with it, so the
# limb is removed and the rest survives.
MEMBER_ONLY_SUBKEYS = {
    "whats_new_snapshot": ("standout",),
}

# EVERY OTHER CONTEXT KEY, NAMED ON PURPOSE.
#
# This tuple exists because of how the favorite_picks leak was found -- which
# is to say, by accident, months late, by a person reading a page. The leak
# gate could not have found it: check_free_build asserts that every value the
# redaction REMOVED is absent from the free payload, so it catches a redaction
# that has broken and is structurally blind to one that was never written. A
# key missing from MEMBER_ONLY_CONTEXT contributes nothing to the manifest,
# so there is nothing to assert and the gate passes green.
#
# The fix is not a smarter heuristic -- a heuristic is what we already had,
# and it was "someone will remember". It is a CLOSED SET: every key in the
# render context must appear in exactly one of these lists, and a key in
# neither fails the build. Adding a context key now forces the question "is
# this the model layer?" at the moment it is added, by someone holding the
# reason in their head, instead of leaving it to be discovered in production.
#
# Each group below says why the whole group is free.
FREE_CONTEXT = (
    # THE RECORD, and the whole argument for paying. A reader has to be able
    # to audit what the model has already done before being asked for money.
    "track_record", "calibration_svg", "units_sparkline_svg",
    "units_timeseries_svg", "plays_record", "countdown_confidence_counts",
    # Settled bets only, expressed as a multiple of where it started. It
    # carries no pick and no currency -- see src/bankroll for why it is a
    # ratio rather than a sum.
    "bankroll",
    # SETTLED PLAYS ONLY, and summarise_by_event is what enforces it -- it
    # drops every ungraded row before this key is built. A graded card's bets
    # are the public record; an ungraded card's rows are label, price and
    # stake on fights that have not happened, which is the model layer. This
    # comment previously claimed "the tier check below", and there was no such
    # check; the filter is now in the function and there is a test on it.
    "plays_events",

    # MARKET DATA. Prices and how they moved. Not model output -- these come
    # from the books and the exchange, and are the best free teaser we have.
    "notable_movements", "notable_movements_upcoming",

    # THE CARD ITSELF, redacted FIGHT BY FIGHT rather than wholesale, so a
    # graded fight stays visible and an upcoming one loses its model layer.
    # See redact_fight.
    "events", "future_events",

    # WHO IS FIGHTING, WHERE, AND WHEN. Public facts about a scheduled event.
    "analytics_source_event", "countdown_city", "countdown_label",
    "countdown_matchup", "countdown_series", "countdown_target_iso",
    "countdown_venue", "days_since_event", "event_full_name", "event_matchup",
    "event_short_name", "fight_schedule_json", "just_concluded_json",
    "espn_live_fight_key_json",

    # SCOUTING AND HISTORY. Measured facts about fighters -- rates, records,
    # trivia. Free by design: the landing page sells the tape and the scout
    # rails as permanently free, and this is what backs that claim.
    "fighter_history_json", "fighter_rates_json", "fun_facts",
    "fun_facts_by_fighter",

    # BUILD PLUMBING. Timestamps, the tier flag itself, error states, and
    # coverage reporting. No model output anywhere in here.
    "generated_at", "generated_at_date", "generated_at_short",
    "generated_at_time_only", "live_error", "results_coverage", "source",
    "tier", "unmatched", "whats_new_snapshot",
)

# Deliberately NOT redacted, and each for a reason worth stating:
#   events                    past cards, picks included -- the proof
#   track_record, units_*     the public record
#   calibration_svg           how well-calibrated the model has been, historically
#   notable_movements*        market data, not model output
#   countdown_confidence_*    aggregate only, no individual pick (see above)
#   moneyline_chart           book prices; the best free teaser on the page
#   plays_record              settled bets only -- the record, not the picks


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

    # THE CLOSED SET, CHECKED BEFORE ANYTHING IS REMOVED. See FREE_CONTEXT.
    # This raises rather than warns: a warning in a build log is how the last
    # one shipped for months.
    unclassified = sorted(set(out) - set(MEMBER_ONLY_CONTEXT) - set(FREE_CONTEXT))
    if unclassified:
        raise RuntimeError(
            "src/tiering.py does not know whether these render-context keys are "
            f"the model layer: {unclassified}.\n"
            "Add each one to MEMBER_ONLY_CONTEXT (it is a pick, a probability, "
            "an edge or a stake on a fight that has not happened) or to "
            "FREE_CONTEXT (it is a public fact, market data, or the graded "
            "record). Refusing to guess: guessing is how favorite_picks shipped "
            "the model's picks on the free payload."
        )

    for key in MEMBER_ONLY_CONTEXT:
        if key in out:
            _collect(out[key], sink)
            out[key] = []

    # One limb, not the whole key -- see MEMBER_ONLY_SUBKEYS.
    for key, subkeys in MEMBER_ONLY_SUBKEYS.items():
        holder = out.get(key)
        if not isinstance(holder, dict):
            continue
        trimmed = dict(holder)
        for sub_key in subkeys:
            if sub_key in trimmed:
                _collect(trimmed[sub_key], sink)
                trimmed[sub_key] = []
        out[key] = trimmed

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
