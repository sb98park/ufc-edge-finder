"""
DraftKings and FanDuel prices, via TheRundown.

WHY THIS EXISTS. Every edge on this site was computed against a Polymarket
midpoint -- a peer-to-peer, vig-free number that no sportsbook will honour.
The premise of the product is "what the book offers versus what the model
thinks", and the book side of that comparison was a price nobody could bet.
This supplies the real one.

WHAT IT ACTUALLY DELIVERS, verified against a live card rather than assumed:

    market_id 1  moneyline   present on 7 of 8 fights
    market_id 3  totals      present on 7 of 8 fights

and nothing else. TheRundown's catalogue lists eighteen markets for sport 7
including `method_of_victory_double_chance` (1371) and `fight_to_start_round`
(1369) -- exactly the two markets worth having -- but requesting them for a
real card returns no data. They are DEFINED, not SERVED, at least on the free
tier. Do not re-add them to MARKET_IDS on the strength of the catalogue; that
was checked and it comes back empty.

A useful accident during that check: the one fight with no markets at all was
Kody Steele vs Gauge Young, which this repo had independently flagged as
cancelled. The feed agreed without being asked.

THREE OPERATIONAL FACTS THAT ARE NOT OPTIONAL.

1. CLOUDFLARE BLOCKS THE DEFAULT PYTHON USER AGENT. A request without a
   browser-like UA returns 403, while the identical request from curl
   succeeds. This is not documented anywhere; it cost an hour.

2. THE QUOTA IS THE BINDING CONSTRAINT, not the rate limit. The free tier
   allows 20,000 "data points" a day, where a point is one participant x one
   line x one book. A 13-fight card is roughly:

       moneyline   2 fighters x 2 books                  =  52
       totals      2 sides x ~3 lines x 2 books          = 156

   about 208 points. At the site's 5-minute rebuild cadence that is 288 pulls
   a day and ~59,900 points -- three times over the cap. So this source runs
   on its OWN slower clock (see MIN_SECONDS_BETWEEN_PULLS) and requests
   main_line=true, which collapses totals to the headline line and cuts the
   bill by more than half.

3. COVERAGE IS PATCHY PER MARKET. A totals block came back carrying only
   affiliate 19 (DraftKings) while the moneyline on the same fight carried
   both. Never assume both books quoted a market; a row that silently shows
   one book where the reader expects two is worse than one that says so.

WHAT THIS SOURCE IS FOR, AND WHAT POLYMARKET BECOMES. These prices carry vig
and are bettable, so they are the BOOK. Polymarket keeps its place as the
FAIR line -- the cleanest available estimate of true probability, precisely
because it has no margin in it. Two different jobs, and the provenance column
on every row is what keeps them from being confused for one another.
"""

import os
import time

import requests

from src.odds_utils import american_to_implied_prob

BASE = "https://therundown.io/api/v2"
SPORT_MMA = 7

# Cloudflare 403s the default python UA. See the module docstring.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Verified against the live /affiliates endpoint, which needs no auth.
AFFILIATES = {19: "DraftKings", 23: "FanDuel", 22: "BetMGM"}
BOOK_AFFILIATES = (19, 23)          # the two the reader actually bets

# Only what is actually served. See the docstring before adding to this.
MARKET_IDS = (1, 3)
_MARKET_NAME = {1: "Moneyline", 3: "TotalRounds"}

# Own clock, slower than the site's 5-minute rebuild, because the daily point
# budget rather than the per-second limit is what binds.
MIN_SECONDS_BETWEEN_PULLS = 15 * 60
_last_pull = {"at": 0.0, "rows": None}


def _get(path: str, timeout: int = 20):
    key = os.environ.get("RUNDOWN_API_KEY")
    if not key:
        raise RuntimeError("RUNDOWN_API_KEY is not set")
    r = requests.get(f"{BASE}{path}",
                     headers={"X-TheRundown-Key": key, "User-Agent": _UA,
                              "Accept": "application/json"},
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


def _rows_from_event(ev: dict) -> list[dict]:
    """Flatten one event's markets into the row shape live_props expects."""
    teams = ev.get("teams") or []
    if len(teams) < 2:
        return []
    # is_away first, to match how the rest of the repo orders a pairing.
    away = next((t for t in teams if t.get("is_away")), teams[0])
    home = next((t for t in teams if t.get("is_home")), teams[-1])
    fa, fb = away.get("name"), home.get("name")
    if not fa or not fb:
        return []
    fight_id = f"{fa}|{fb}"

    out = []
    for m in ev.get("markets") or []:
        market = _MARKET_NAME.get(m.get("market_id"))
        if not market:
            continue
        for p in m.get("participants") or []:
            pname = p.get("name")
            for line in p.get("lines") or []:
                value = line.get("value")
                for aff_id, price in (line.get("prices") or {}).items():
                    book = AFFILIATES.get(int(aff_id))
                    if not book:
                        continue
                    am = price.get("price")
                    if am is None:
                        continue
                    if market == "Moneyline":
                        selection, sel_method = pname, ""
                    else:
                        # "Over"/"Under" plus the line value, matching the
                        # selection strings the rest of the pipeline parses.
                        selection = f"{pname} {value}" if value else pname
                        sel_method = str(value or "")
                    out.append({
                        "fight_id": fight_id, "fighter_a": fa, "fighter_b": fb,
                        "market": market, "selection": selection,
                        "selection_method": sel_method,
                        "odds_american": float(am),
                        # PROVENANCE. These are vigged, bettable prices -- the
                        # opposite of the Polymarket rows they sit beside, and
                        # the whole reason the flag travels per row.
                        "source": book,
                        "source_is_vig_free": False,
                        "is_main_line": bool(price.get("is_main_line")),
                        "price_updated_at": price.get("updated_at"),
                    })
    return out


def fetch_rundown_ufc_odds(dates: list[str], force: bool = False) -> list[dict]:
    """
    Moneyline and totals for the given YYYY-MM-DD dates, from DraftKings and
    FanDuel. Returns [] on any failure -- a metered second source must never
    be able to take a site build down with it.

    Cached on its own clock: repeated calls inside MIN_SECONDS_BETWEEN_PULLS
    return the previous result rather than spending quota.
    """
    now = time.time()
    if not force and _last_pull["rows"] is not None \
            and now - _last_pull["at"] < MIN_SECONDS_BETWEEN_PULLS:
        return _last_pull["rows"]

    mkts = ",".join(str(m) for m in MARKET_IDS)
    affs = ",".join(str(a) for a in BOOK_AFFILIATES)
    rows: list[dict] = []
    for i, d in enumerate(dates):
        if i:
            time.sleep(1.1)          # documented limit is 1 req/sec
        try:
            # main_line=true is a QUOTA decision, not a display one: alternate
            # totals lines multiply the point cost by roughly three for a
            # market the site shows one line of anyway.
            data = _get(f"/sports/{SPORT_MMA}/events/{d}"
                        f"?market_ids={mkts}&affiliate_ids={affs}&main_line=true")
        except Exception as exc:
            print(f"[rundown] {d} failed ({exc}) -- continuing without it")
            continue
        for ev in data.get("events") or []:
            rows.extend(_rows_from_event(ev))

    books = {r["source"] for r in rows}
    print(f"[rundown] {len(rows)} price(s) across {len(dates)} date(s) "
          f"from {', '.join(sorted(books)) if books else 'no book'}")
    _last_pull.update(at=now, rows=rows)
    return rows


def best_book_price(rows: list[dict], fight_id: str, market: str,
                    selection: str) -> dict | None:
    """
    The best available price for one bet across the books, and which book has
    it. "Best" is the highest implied payout, i.e. the LOWEST implied
    probability -- the same bet for less money.

    This is what line shopping is, and it is the one edge on this page that
    needs no model to be real: taking +116 over +114 on the identical
    selection is free, and the only way to see it is to hold two books
    side by side.
    """
    cands = [r for r in rows
             if r["fight_id"] == fight_id and r["market"] == market
             and r["selection"] == selection]
    if not cands:
        return None
    best = min(cands, key=lambda r: american_to_implied_prob(r["odds_american"]))
    return {
        "book": best["source"],
        "odds_american": best["odds_american"],
        "implied": american_to_implied_prob(best["odds_american"]),
        "all_books": {r["source"]: r["odds_american"] for r in cands},
        # A single-book quote is not a shopped price, and the reader should be
        # told rather than left to assume both were checked.
        "books_quoting": len(cands),
    }
