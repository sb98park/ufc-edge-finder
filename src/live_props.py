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

import json
import os
from datetime import datetime, timezone

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


def _record_source_health(pm_rows, rd_rows, dates) -> None:
    """
    Write a one-line, committed record of which sources actually returned
    prices this build.

    WHY THIS IS WORTH A FILE. TheRundown was wired in, configured with a live
    secret, and never once called -- the date list it needed was built from a
    field Polymarket does not have, so it sat behind an empty `if` for days
    while the site kept quoting Polymarket and looking healthy. Nothing in the
    repo recorded which feeds contributed, the build log is the only place it
    would have shown, and logs are not readable after the fact without the gh
    CLI. The bug was eventually caught sideways, from source labels in the
    parlay ledger.

    Row counts per source are three lines of code and turn "is the second feed
    alive?" from an investigation into a `cat`. Committed by the workflow's
    `git add -A -- data/`, so the answer is in git history rather than in a
    log that has aged out.

    Never raises: a build must not fail over its own bookkeeping.
    """
    try:
        from collections import Counter
        counts = Counter(r.get("source") or "Polymarket" for r in pm_rows)
        counts.update(r.get("source") or "unknown" for r in rd_rows)
        payload = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows_by_source": dict(sorted(counts.items())),
            "rundown_key_set": bool(os.environ.get("RUNDOWN_API_KEY")),
            "rundown_dates_requested": list(dates or []),
        }
        os.makedirs("data", exist_ok=True)
        with open("data/source_health.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"[source_health] {payload['rows_by_source']}")
    except Exception as exc:
        print(f"[source_health] not written ({exc}) -- continuing")


def _today() -> str:
    """UTC date as YYYY-MM-DD, for discarding cards that have already run."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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
    rd_dates: list[str] = []
    if os.environ.get("RUNDOWN_API_KEY"):
        try:
            from src.rundown_source import fetch_rundown_ufc_odds
            # DATES COME FROM THE CARDS, NOT FROM THE PRICES.
            #
            # This read `r.get("start_date") for r in pm_rows` and Polymarket
            # rows have no such key -- it is not in their schema at all. So
            # `dates` was unconditionally empty, `if dates` was never true,
            # and fetch_rundown_ufc_odds was NEVER CALLED in production. No
            # exception, no log line, nothing: the integration sat behind an
            # empty list looking installed. It was caught only because every
            # moneyline and totals leg in the published slate still said
            # "Polymarket", and _devig_and_shop falls back to the reference
            # price exactly when no book quoted.
            #
            # known_fighters already carries (fighter_a, fighter_b, date) --
            # the date was in this function's own argument the whole time.
            dates = sorted({str(t[2])[:10] for t in (known_fighters or [])
                            if isinstance(t, (tuple, list)) and len(t) > 2 and t[2]}
                           - {"None", "nan", ""})
            dates = [d for d in dates if len(d) == 10 and d >= _today()]

            # ONE DATE, AND THE QUOTA IS WHY. The free tier allows 20,000 data
            # points a day, where a point is one participant x one line x one
            # book. With main_line=true a 13-fight card is roughly 104 points
            # (52 moneyline + 52 totals), so at this source's own 15-minute
            # clock -- 96 pulls a day -- one date costs ~10,000 points. Two
            # dates would sit exactly at the cap and nine (this repo tracks
            # eight future cards alongside the live one) would be ~90,000,
            # four and a half times over.
            #
            # The nearest card is also the only one the edges, standouts and
            # slips are built for; the rest are "Coming Up" listings that
            # quote no prices. So this is a cheap constraint, not a sacrifice.
            dates = dates[:1]
            rd_dates = dates

            if dates:
                rd_rows = fetch_rundown_ufc_odds(dates)
                if rd_rows:
                    sources_used.append("DraftKings/FanDuel")
                else:
                    print(f"[rundown] key is set and {dates[0]} was requested, "
                          f"but no book prices came back")
            else:
                # NEVER SILENT AGAIN. A configured, metered source returning
                # nothing is a fault; a configured source that is never even
                # called is a worse one, and the only reason the last failure
                # went unnoticed for days is that this branch printed nothing.
                print("[rundown] RUNDOWN_API_KEY is set but no usable card date "
                      f"was derived from {len(known_fighters or [])} tracked bout(s) "
                      "-- source skipped")
        except Exception as exc:
            print(f"[warn] TheRundown fetch failed ({exc}) -- continuing on Polymarket alone")

    _record_source_health(pm_rows, rd_rows, rd_dates)

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

def record_edge_health(edges_df) -> None:
    """
    Merge edge-level provenance into data/source_health.json.

    row counts alone proved TheRundown was ALIVE (28 DraftKings rows, 13
    FanDuel) while every published leg still said Polymarket -- so the prices
    were arriving and losing. Arriving and being selected are separate
    questions and the row counts only answer the first.

    This answers the second: which book actually won each bet, and how many
    books were shopped. `best_book: Polymarket` on a moneyline means no
    vig-bearing source quoted BOTH sides of that fight, since _devig_and_shop
    excludes vig-free prices from shopping and falls back to the reference
    only when the quote list is empty.
    """
    try:
        if edges_df is None or getattr(edges_df, "empty", True):
            return
        payload = {}
        try:
            with open("data/source_health.json", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            pass
        two_way = edges_df[edges_df["market"].astype(str).str.startswith(
            ("Moneyline", "Total Rounds"))] if "market" in edges_df else edges_df
        def _counts(col):
            if col not in two_way:
                return {}
            return {str(k): int(v) for k, v in two_way[col].value_counts().items()}
        payload["edges_two_way"] = {
            "n": int(len(two_way)),
            "best_book": _counts("best_book"),
            "books_quoting": _counts("books_quoting"),
        }
        with open("data/source_health.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"[source_health] best_book {payload['edges_two_way']['best_book']} "
              f"books_quoting {payload['edges_two_way']['books_quoting']}")
    except Exception as exc:
        print(f"[source_health] edge health not written ({exc}) -- continuing")

