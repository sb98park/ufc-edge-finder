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



# CANONICAL SPELLINGS for fighters the sources genuinely disagree about, where
# no fold can bridge the gap. Folding handles accents, punctuation and token
# order; it cannot handle a MIDDLE NAME, because "jose miguel delgado" and
# "jose delgado" differ by a whole token and matching on first+last alone
# would happily merge two different people.
#
# Jose Delgado, found 2026-09-01 by the owner noticing an empty scouting
# drawer on the Noche card. He was split across every file at once:
# fight_history held 4 bouts under one spelling and 14 under the other -- two
# separate nodes in the Elo graph for one man -- while all 22 rows of his
# per-bout striking and grappling stats sat under the spelling nothing on the
# card pointed at. His rating was built from 14 of 18 bouts and his drawer
# from none.
#
# Deliberately a short explicit list rather than a cleverer fold. Across all
# 369 roster rows this is the ONLY such pair (369 distinct folded names), so
# the cost of being explicit is one line and the cost of being clever is
# merging two real fighters. scripts/check_card_data_coverage.py reports new
# candidates rather than leaving the next one to be found by chance.
NAME_ALIASES = {
    "jose miguel delgado": "Jose Delgado",
}


def canonical_name(name) -> str:
    """The spelling this project stores, for a name a source spells otherwise.

    Returns the input unchanged when there is no alias, so it is safe to call
    on the way in from any source.
    """
    if name is None:
        return name
    text = str(name).strip()
    if not text:
        return name
    key = " ".join(re.sub(r"[^a-z0-9 ]", " ",
                          unicodedata.normalize("NFKD", text)
                          .encode("ascii", "ignore").decode().lower()).split())
    return NAME_ALIASES.get(key, text)


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
    folded = " ".join(re.sub(r"[^a-z0-9 ]", " ", normalized.lower()).split())
    # An alias resolves to the canonical spelling, then folds -- so every
    # fold-based lookup in the project (the scouting drawer, method rates,
    # fight_key, coverage) sees one fighter where the sources see two.
    alias = NAME_ALIASES.get(folded)
    if alias:
        return " ".join(re.sub(r"[^a-z0-9 ]", " ",
                               unicodedata.normalize("NFKD", alias)
                               .encode("ascii", "ignore").decode().lower()).split())
    return folded


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


