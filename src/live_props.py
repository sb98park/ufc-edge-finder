"""
Shared live-odds fetching logic. Polymarket is the source; The Odds API is
a last resort only if it returns nothing at all.

DRAFTKINGS WAS REMOVED. It used to be tried alongside Polymarket and merged
in to fill (fighter-pair, market) gaps. In practice it filled none: the
scrape hit DraftKings' undocumented site endpoints, which now refuse it, so
every build called it, caught the failure, logged a warning and carried on
with Polymarket alone. The generated site recorded "Polymarket" and nothing
else as its source.

Keeping it cost a failed request on every one of ~288 builds a day and left
253 lines of sportsbook-scraping code in the tree that looked load-bearing
and wasn't. If DraftKings odds are wanted again the route is a licensed
feed, not that scraper -- The Odds API already carries DraftKings
moneylines under a bookmaker key on the plan this project already uses.
"""

import os

import pandas as pd

from src.polymarket_source import fetch_polymarket_ufc_props
from src.live_odds import fetch_mma_odds, to_upcoming_rows


def _pair_key(row: dict) -> frozenset | None:
    """Normalized fighter-pair key used to match the same fight across different sources."""
    from src.card_matcher import _normalize_name  # local import avoids any load-order issues

    fighter_a, fighter_b = row.get("fighter_a"), row.get("fighter_b")
    if not fighter_a or not fighter_b:
        return None
    return frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)})


def _bet_key(row: dict) -> tuple:
    """
    Full identity of a specific bet -- fighter pair + market + exact selection,
    not just the fight.

    SOURCE IS PART OF THE IDENTITY. The same bet quoted by Polymarket, by
    DraftKings and by FanDuel is three rows that must all survive, because
    they answer different questions: the vig-free midpoint is the FAIR line
    the model is measured against, and the two book prices are what can
    actually be bet. Deduping across sources would silently keep whichever
    happened to arrive first and throw away either the reference or the
    bettable price.

    Within a single source a duplicate is still a duplicate -- Polymarket
    genuinely lists the same fight under two market objects at different
    prices -- so the de-duplication below still has work to do.
    """
    pair = _pair_key(row) or frozenset({row.get("fighter_a"), row.get("fighter_b")})
    return (pair, row.get("market"), row.get("selection"),
            row.get("selection_method"), row.get("source"))


def get_live_props(known_fighters=None) -> tuple[pd.DataFrame, str]:
    """
    known_fighters: names on the cards we're tracking. Passed down to
    Polymarket's discovery filter so a fight event is recognised by WHO is
    in it rather than by whether the title happens to contain "UFC" -- a
    live run matched 0 of 200 events because of exactly that assumption.
    """
    sources_used = []
    if known_fighters:
        try:
            from src.polymarket_source import set_known_fighters
            set_known_fighters(known_fighters)
        except Exception:
            pass    # discovery still works on its original "ufc" + "vs" rule
    pm_rows = []

    try:
        pm_rows = fetch_polymarket_ufc_props()
        if pm_rows:
            sources_used.append("Polymarket")
    except Exception as exc:
        print(f"[warn] Polymarket fetch failed ({exc})")

    if not pm_rows:
        try:
            events = fetch_mma_odds()
            rows = to_upcoming_rows(events)
            if rows:
                # STAMPED HERE TOO, and this is the case the flag exists for.
                # This path returns real sportsbook moneylines, which carry
                # the book's margin; the Polymarket path below returns
                # peer-to-peer midpoints, which carry none. Treating the two
                # as one "Book" price silently compares a vigged number
                # against a vig-free one, and the whole edge calculation is
                # the difference between a model probability and exactly this
                # number.
                for r in rows:
                    r.setdefault("source", "The Odds API")
                    r.setdefault("source_is_vig_free", False)
                return pd.DataFrame(rows), "The Odds API (moneyline only)"
        except Exception as exc:
            # Both sources failed. This used to raise RuntimeError, which
            # killed the ENTIRE site generation -- a transient network blip
            # at the wrong moment meant no site update at all, including
            # everything that doesn't depend on live odds (track record,
            # fighter data, schedules). The empty-DataFrame path below
            # already exists for the quieter "sources answered but had no
            # rows" case, and everything downstream (including the
            # template's "Couldn't fetch live odds right now" notice)
            # already handles it correctly -- so a total outage should take
            # that same graceful path, just with a louder log line.
            print(f"[warn] BOTH odds sources failed (Polymarket, The Odds API) "
                  f"-- generating site without live odds. Last error: {exc}")
        return pd.DataFrame(), "no source returned data"

    # Polymarket rows are kept as-is (no-vig, more trustworthy pricing). This
    # used to merge in DraftKings rows for any (fighter-pair, market) combo
    # Polymarket hadn't covered; with that source gone there is nothing to
    # merge, but the de-duplication below still matters -- Polymarket alone
    # can list the same fight under two separate markets.
    # THE BOOK PRICES, alongside rather than instead of Polymarket. These
    # carry vig and are bettable, so they are the BOOK; Polymarket keeps its
    # role as the FAIR line precisely because it has no margin in it. Both are
    # needed and neither substitutes for the other -- see src/rundown_source.
    #
    # Metered, so it fails silently and runs on its own slower clock. A second
    # source must never be able to reduce what the first one already provides.
    rd_rows = []
    if os.environ.get("RUNDOWN_API_KEY"):
        try:
            from src.rundown_source import fetch_rundown_ufc_odds
            dates = sorted({str(r.get("start_date") or "")[:10]
                            for r in pm_rows if r.get("start_date")})
            dates = [d for d in dates if len(d) == 10]
            if dates:
                rd_rows = fetch_rundown_ufc_odds(dates)
                if rd_rows:
                    sources_used.append("DraftKings/FanDuel")
        except Exception as exc:
            print(f"[warn] TheRundown fetch failed ({exc}) -- continuing on Polymarket alone")

    combined_rows = pm_rows + rd_rows

    # STAMP THE SOURCE ON EVERY ROW.
    #
    # Until now the provenance of a price existed only as one global label
    # ("Odds via Polymarket"), so a row could not say where it came from. That
    # was survivable with a single source. It stops being survivable the
    # moment a second one carries different markets, because "Book" then means
    # different things on different rows and the reader cannot tell which.
    #
    # It matters more here than in most products because the sources are not
    # merely different books, they are different KINDS of price. Polymarket
    # and Kalshi are peer-to-peer and vig-free; DraftKings and FanDuel carry
    # roughly 4.5% on a moneyline and 20% on a method grid. An edge computed
    # against one is not comparable to an edge computed against the other, and
    # averaging or ranking them together silently mixes the two.
    for row in combined_rows:
        row.setdefault("source", "Polymarket")
        # Vig-free sources need a different treatment everywhere a price is
        # converted to a probability, so the flag travels with the row rather
        # than being re-derived from the name at each call site.
        row.setdefault("source_is_vig_free", True)

    # Final safety net: the same specific bet can show up twice at two
    # different prices (confirmed live) -- most likely from Polymarket
    # having two separate market listings covering the same fight. Keep
    # only the first occurrence of each exact bet.
    seen: dict[tuple, dict] = {}
    dupes_removed = 0
    upgrades = 0
    for row in combined_rows:
        key = _bet_key(row)
        if key not in seen:
            seen[key] = row
            continue
        dupes_removed += 1
        existing = seen[key]
        # Prefer whichever duplicate actually has a usable clob_token_id --
        # blindly keeping "whichever came first" was silently discarding
        # rows with real chart data in favor of rows without it, for no
        # reason other than list order (confirmed live: this is exactly
        # why McGregor vs Holloway's chart fell back to sparse tracking
        # data while every other fight got full CLOB history).
        if not existing.get("clob_token_id") and row.get("clob_token_id"):
            seen[key] = row
            upgrades += 1
    deduped = list(seen.values())
    if dupes_removed:
        msg = f"[live_props] removed {dupes_removed} duplicate bet(s) (same fighter/market/selection, different price)"
        if upgrades:
            msg += f", upgraded {upgrades} to keep the copy with a working clob_token_id"
        print(msg)

    source_label = " + ".join(sources_used) if len(sources_used) > 1 else sources_used[0]
    return pd.DataFrame(deduped), source_label
