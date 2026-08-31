"""
Fighter and bout identity: the one place a name is folded.

CLAUDE.md s4 counts ~12 name-folding helpers in this project and says not to
add a thirteenth. This module exists so that the count can go DOWN: it is a
leaf with no project imports, so anything can depend on it. card_matcher
re-exports both names, so every existing `from src.card_matcher import ...`
keeps working.

It lives here rather than in card_matcher because card_matcher pulls in
model_preview, which pulls in ufc_method_rates -- so the two modules that most
needed the canonical fold could not import it without a cycle, and each had
grown its own punctuation-blind copy instead.
"""

import re
import unicodedata

import pandas as pd


def _normalize_name(name: str) -> str:
    """
    Strips accents and standardizes punctuation so minor spelling differences
    between sources (e.g. Polymarket listing 'Benoît Saint Denis' while our
    data has 'Benoit Saint-Denis') don't cause a real fight to silently miss
    its match and get dumped into 'unmatched' instead.
    """
    # Coerce first. Edge rows come from several finders with different key
    # sets -- fight-level markets never set "opponent" -- so a DataFrame built
    # from a mix of them fills the gap with NaN, which is a FLOAT and blows up
    # unicodedata.normalize. Returning "" for a missing name lets the caller's
    # set comparison simply fail to match, which is the correct outcome, and
    # is far better than a crash three frames away from the cause.
    if name is None or not isinstance(name, str):
        try:
            if pd.isna(name):
                return ""
        except (TypeError, ValueError):
            pass
        name = str(name) if name is not None else ""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    # COLLAPSE THE RUNS THIS FUNCTION ITSELF CREATES. Substituting punctuation
    # with a space turns "Ode' Osbourne" into "ode  osbourne" -- two spaces --
    # which .strip() does not touch, so it never equalled "ode osbourne" and
    # the two spellings compared as different fighters. " ".join(split())
    # collapses internal runs as well as the ends.
    #
    # Measured before changing: across all 5,272 distinct names in
    # fight_history.csv plus fighters.csv this merges exactly ONE group,
    # {"Ode Osbourne", "Ode' Osbourne"}, who is one man. The change can only
    # ever make two names compare EQUAL, which is the direction this function
    # exists to move in.
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", normalized.lower()).split())


def fight_key(fighter_a, fighter_b, date) -> tuple:
    """The identity of one bout: an unordered folded pair plus its date.

    THE ONE KEY FOR THE SPINE. Four write paths and the deduper each built
    this themselves and three of them folded punctuation differently from
    _normalize_name above, so "Benoit Saint-Denis" and "Benoit Saint Denis"
    were one fighter to the card path and two to the dedupe path. The file
    accumulated 231 double-written bouts that
    `scripts/dedupe_fight_history.py` reported as clean, and src/elo.py
    replays raw names, so each one was scored twice against a phantom node.
    It moved the published probability on a main event by 3.3 points.

    UNORDERED, because cards get re-scraped with the corners swapped. Dated,
    because the pair alone carries no event and a rematch would collide with
    the first meeting (CLAUDE.md s4). Callers that need a tolerance window
    should compare against fight_key(a, b, date +/- 1 day) rather than
    inventing their own key -- see dedupe_fight_history.
    """
    return (frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)}),
            str(date)[:10])


