"""
The PAPER record: what the discretionary rule would have played, graded.

Nothing here is ever staked. No number this module produces reaches the
bankroll, the published units total, the landing page curve, or the track
record. It exists to answer one question with evidence instead of argument.

WHY IT EXISTS

card_plays.DISCRETIONARY_PLAYS is documented as "a pause, not a verdict --
turn it back on when the discretionary moneylines have enough of a record to
argue with." Nothing was accumulating that record. `shelved` was recomputed
every render and thrown away, and the `shortlist` stats in track_record are
the LADDER's own record, not the discretionary one. So the stated condition
for lifting the pause could never be met, and the decision stayed a matter
of opinion for as long as the flag stayed off.

WHAT THE COUNTERFACTUAL SAID, and why it is not enough on its own

Replaying the hurdle over the graded history (2026-09-15, 128 graded picks)
put Medium Confidence at -8.80U on 17 qualifying picks, a 23.5% hit rate
against 56.9% for the tier as a whole, and 15 of those 17 were plus-money
underdogs. That is the Elo compression described in HURDLE_MONEYLINE
reaching the staking rule: an EV hurdle hunts for model-over-market gaps,
and in the weak tiers that gap is mostly model error.

It is still a RECONSTRUCTION. It re-derives prices and blends from a log
written for another purpose, it cannot see the venue gate or the card
budget as they stood on the night, and it rests on n=17. This file records
the real thing, decided at build time from the same candidates and the same
budget as the live card, and settled from the same results map.

HOW IT STAYS HONEST

Every row goes through plays_ledger's own writers, so the paper record
inherits set-once pricing exactly as the real one has it: a row's stake,
price and probabilities are fixed the moment it is first written, and only
last_seen, closing_odds, result, units_result, graded_at and void_reason
may ever change. A paper record you can revise after seeing the outcome is
not evidence of anything, and this is the whole reason the module is a
delegation rather than a fresh CSV writer.

It is deliberately NOT gated hard. scripts/check_plays_ledger reports
violations here as a warning and keeps its non-zero exit for the real
ledger alone: a bookkeeping file about bets nobody placed must never be
able to freeze a site that real money is staked against.
"""

from src.plays_ledger import (  # noqa: F401 -- re-exported deliberately
    committed_for, grade_rows, load as _load, play_id, record_plays as _record,
    summarise as _summarise, summarise_by_event as _summarise_by_event,
    void_stale, write_graded as _write,
)

SHADOW_LEDGER_PATH = "data/shadow_ledger.csv"


def load(path: str = SHADOW_LEDGER_PATH) -> list[dict]:
    return _load(path)


def record(card: dict, now: str, live_prices: dict | None = None,
           path: str = SHADOW_LEDGER_PATH) -> list[dict]:
    """
    Merge this render's PAPER plays into the shadow ledger.

    Takes the card dict build_card_plays returns and swaps `shadow_plays` in
    for `plays`, so the ledger sees the same shape it always has.
    """
    paper = dict(card or {}, plays=(card or {}).get("shadow_plays") or [])
    return _record(paper, now, live_prices=live_prices, path=path)


def write_graded(rows: list[dict], path: str = SHADOW_LEDGER_PATH) -> None:
    _write(rows, path)


def summarise(rows: list[dict]) -> dict:
    return _summarise(rows)


def summarise_by_event(rows: list[dict]) -> dict:
    return _summarise_by_event(rows)
