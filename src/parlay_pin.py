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


def _requote(identities: list[str], pieces: list[dict]) -> dict | None:
    """
    Rebuild the pinned slip at today's prices, or None if any leg has left
    the pool.

    Partial re-quoting is deliberately not offered. A slip priced from four
    live legs and one week-old one is a number no book would honour, and it
    would be indistinguishable on the page from a fully live one.
    """
    from src.parlay_builder import _combine

    by_identity = {}
    for p in pieces:
        by_identity.setdefault(_identity(p), p)

    matched = []
    for ident in identities:
        piece = by_identity.get(ident)
        if piece is None:
            return None
        matched.append(piece)
    return _combine(tuple(matched))


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

    identities = entry.get("identities") or []
    requoted = _requote(identities, pieces) if identities else None

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
