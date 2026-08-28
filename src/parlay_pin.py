"""
One slip per card per tier, chosen once and then held.

WHY THIS EXISTS. The builder runs on every render -- every five minutes --
and re-picked whatever combination ranked first on the prices in front of it.
Over a single card (Hernandez vs. Rodrigues) that produced 51 distinct
bankroll slips and 93 distinct lotto slips; 20 of the bankroll ones existed
for exactly one build, roughly five minutes on screen. A recommendation that
changes ninety-three times is not a recommendation, and there is no object
there to grade either: "how did the lotto do on that card" has ninety-three
different answers, none of them the one anybody acted on.

So the LEGS are chosen once and never swapped. Prices still move -- each
build re-quotes the pinned legs against the current pool, so what a reader
sees is live -- but the selection is frozen at first publication, which is
also the price the record settles at (see combined_decimal_first in
parlay_ledger).

DELIBERATELY NO ESCAPE VALVE, unlike the landing page's held chart, which
hands the slot to a materially better candidate. A better slip appearing on
Friday is exactly the churn this exists to stop; letting it in would restore
the behaviour under a politer name.

A LEG WHOSE FIGHT IS CANCELLED IS ALSO NOT REPLACED. A book voids that leg
and settles the rest, the live tracker in site.html already grades it that
way, and swapping in a replacement would be shopping for a result after the
fight was already off.

Because the legs hold, _slip_id() -- which hashes fight ids and leg labels --
returns the same value on every build. The ledger keys on slip_id, so a
pinned card contributes exactly one row per tier instead of dozens, and its
`renders` count becomes a measure of how long the slip stood rather than of
how fast the builder churned.
"""

import json
import os
from datetime import datetime, timezone

from src.parlay_builder import (BANKROLL_MIN_LEG_PROB, BETTABLE_VENUES,
                                leg_still_eligible)

PIN_PATH = "data/pinned_parlays.json"

# How many cards' pins to keep. Purely a file-size bound -- a pin is only ever
# read while its own card is current, so anything older is dead weight. Twelve
# is roughly three months of cards, which is long enough that a card returning
# from a postponement still finds its pin.
MAX_EVENTS = 12


def _identity(leg: dict) -> str:
    """
    A leg's identity, independent of prose.

    fight_key plus the grading conditions, which together name the exact
    market. NOT the label: parlay_ledger already makes the point that "prose
    is not a protocol", and a wording change must not silently look like a
    different leg and unpin the card.
    """
    return json.dumps(
        [leg.get("fight_key") or "", leg.get("conditions") or []],
        sort_keys=True, ensure_ascii=False,
    )


def load(path: str = PIN_PATH) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save(pins: dict, path: str = PIN_PATH) -> None:
    """Atomic, and never fatal -- a pin that cannot be written must not take
    the build down with it. The cost of failing is one card that re-picks."""
    try:
        trimmed = dict(sorted(
            pins.items(),
            key=lambda kv: max((t.get("pinned_at", "") for t in kv[1].values()), default=""),
            reverse=True,
        )[:MAX_EVENTS])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(trimmed, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[parlay_pin] not written ({exc}) -- continuing")


def _requote(identities: list[str], pieces: list[dict],
             prefer: str | None = None) -> dict | None:
    """
    Rebuild the pinned slip at today's prices, or None if any leg has left
    the pool.

    Partial re-quoting is deliberately not offered. A slip priced from four
    live legs and one week-old one is a number no book would honour, and it
    would be indistinguishable on the page from a fully live one.
    """
    from src.parlay_builder import _combine

    # ONE BOOK, OR NO SLIP. _identity is fight_key plus conditions and says
    # nothing about where a leg was priced -- deliberately, because prose is
    # not a protocol. That makes it the wrong key to rebuild a slip from on
    # its own: the pool carries the same leg from several books, the first
    # one wins per identity, and the slip comes back spread across two of
    # them. It did. The bankroll pin for Nurmagomedov vs. Song was re-quoted
    # into Liu Ce at FanDuel alongside Lawrence Lui at DraftKings, under a
    # single slip_id that had been wholly DraftKings when it was pinned, and
    # the slip-level venue check upstream passed it because FanDuel is a real
    # book. Nobody can place that ticket.
    #
    # So the search is per venue and a venue has to cover EVERY leg. The
    # pinned book is tried first, which keeps a slip on the book it was
    # pinned at whenever that book still quotes the whole thing, rather than
    # letting it drift to whichever one happens to sort first.
    # AND THE LEG STILL HAS TO BE ONE THE MODEL WOULD PICK TODAY.
    #
    # _find_parlays screens candidates on model_prob >= BANKROLL_MIN_LEG_PROB
    # when a slip is BUILT. Nothing applied that screen when a pinned slip was
    # RE-QUOTED, and this function is the re-quote: it matches on identity,
    # which is fight_key plus conditions and says nothing about what the model
    # currently thinks. So a leg the model had turned against kept its place
    # and simply got a new price.
    #
    # The live slip for Nurmagomedov vs. Song was holding Under 2.5 rounds on
    # Perez vs Sumudaerji at model 0.4282, with Over 2.5 -- the other side of
    # the same market -- at 0.5718. Pinning is a commitment to a SLIP, not a
    # licence to keep a leg the model has stopped believing in.
    by_venue: dict[str, dict[str, dict]] = {}
    for p in pieces:
        venue = p.get("source") or p.get("venue")
        if venue not in BETTABLE_VENUES:
            continue
        if not leg_still_eligible(p):
            continue
        by_venue.setdefault(venue, {}).setdefault(_identity(p), p)

    order = ([prefer] if prefer in by_venue else []) + sorted(
        v for v in by_venue if v != prefer)
    for venue in order:
        table = by_venue[venue]
        matched = [table.get(i) for i in identities]
        if any(m is None for m in matched):
            continue
        combined = _combine(tuple(matched))
        # A PRICELESS SLIP IS NOT A RE-QUOTE. _combine returns None for the
        # price when a leg it was handed has none, and the caller writes
        # whatever comes back straight into the snapshot -- which is how the
        # pin ended up published with american=None on every leg. Falling
        # through to the stored snapshot is the honest answer: that one has
        # the price it was pinned at.
        if combined and combined.get("combined_american") is not None:
            return combined
    return None


def hold(event_name: str | None, tier: str, fresh: list[dict],
         pieces: list[dict], path: str = PIN_PATH) -> list[dict]:
    """
    The slip this card is committed to for `tier`, as a 0- or 1-item list so
    it drops straight into the template where the builder's output used to.

    First call for a card pins whatever the builder just chose. Every call
    after returns that same selection, re-quoted if it still can be and
    served from the stored snapshot if it cannot.
    """
    if not event_name:
        return fresh[:1]

    pins = load(path)
    entry = (pins.get(event_name) or {}).get(tier)

    if not entry:
        if not fresh:
            return []
        slip = fresh[0]
        pins.setdefault(event_name, {})[tier] = {
            "pinned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "identities": [_identity(l) for l in (slip.get("legs") or [])],
            "snapshot": slip,
        }
        save(pins, path)
        print(f"[parlay_pin] pinned {tier} for {event_name}: "
              f"{slip.get('combined_american'):+d}, {len(slip.get('legs') or [])} legs")
        return [slip]

    # A PIN IS NOT A LICENCE TO IGNORE A LATER RULE.
    #
    # The snapshot fallback below exists so a leg that is momentarily
    # unpriced does not destroy a card's slip. But "the leg is gone" is also
    # exactly what happens when a validity check REJECTS it -- the rounds
    # coherence gate in polymarket_source drops a whole ladder, and a pinned
    # slip built from that ladder would then be served from its snapshot
    # forever, outliving the finding that condemned it.
    #
    # So a pin is re-examined against the rules that exist NOW, not only the
    # ones that existed when it was taken. A snapshot whose venue is not
    # bettable, or which carries no venue at all because it predates the
    # field, is dropped and re-pinned rather than held.
    snap = entry.get("snapshot") or {}
    snap_venue = snap.get("venue") or next(
        (l.get("source") for l in (snap.get("legs") or []) if l.get("source")), None)
    # THE STORED SNAPSHOT GETS THE SAME TEST THE POOL DOES. Checking only the
    # slip-level venue was not enough: that field is stamped from the first
    # leg, so a slip whose legs came from two books still reports one of them
    # and passes. This pin did -- venue FanDuel over Liu Ce at FanDuel and
    # Lawrence Lui at DraftKings, after being pinned wholly at DraftKings.
    leg_venues = {l.get("source") for l in (snap.get("legs") or []) if l.get("source")}
    if snap_venue not in BETTABLE_VENUES or len(leg_venues) > 1:
        why = (f"venue {snap_venue!r} is not one a slip can be placed at"
               if snap_venue not in BETTABLE_VENUES else
               f"its legs are split across {sorted(leg_venues)}, which is not one ticket")
        print(f"[parlay_pin] dropping the {tier} pin for {event_name}: {why}")
        pins.get(event_name, {}).pop(tier, None)
        save(pins, path)
        return hold(event_name, tier, fresh, pieces, path)

    identities = entry.get("identities") or []

    # A LEG THE MODEL HAS TURNED AGAINST DROPS THE WHOLE PIN, rather than
    # falling through to "cannot re-quote" and serving the stored snapshot.
    #
    # That distinction is the entire point. The stored snapshot is what we
    # want when a leg is DELISTED -- the slip existed, at that price, and the
    # market moving on does not unmake it. It is exactly what we do not want
    # when the leg is still quoted and the model has simply changed its mind:
    # serving it then means publishing a bet the model now disagrees with,
    # under a slip_id that implies it still stands behind it.
    #
    # Checked against the CURRENT pool, because the snapshot's own legs carry
    # no model_prob -- _combine does not keep one.
    _now_by_identity = {_identity(p): p for p in pieces}
    _turned = []
    for ident in identities:
        piece = _now_by_identity.get(ident)
        if piece is None:
            continue                    # delisted -- a different case, handled below
        if not leg_still_eligible(piece):
            _turned.append((piece.get("label") or ident,
                            piece.get("model_prob"), piece.get("model_prob_raw")))
    if _turned:
        for label, ranked, raw in _turned:
            print(f"[parlay_pin] dropping the {tier} pin for {event_name}: "
                  f"{label} no longer qualifies (ranked {ranked}, raw model {raw})")
        pins.get(event_name, {}).pop(tier, None)
        save(pins, path)
        return hold(event_name, tier, fresh, pieces, path)

    requoted = _requote(identities, pieces, prefer=snap_venue) if identities else None

    if requoted is None:
        snapshot = entry.get("snapshot")
        # Says which of the two failure modes it is, because they mean
        # opposite things: a leg genuinely delisted is worth knowing about,
        # a pin written before this field existed is not.
        print(f"[parlay_pin] {tier} for {event_name}: cannot re-quote "
              f"({'a leg left the pool' if identities else 'no identities stored'})"
              f" -- serving the pinned snapshot")
        return [snapshot] if snapshot else []

    entry["snapshot"] = requoted
    pins[event_name][tier] = entry
    save(pins, path)
    return [requoted]
