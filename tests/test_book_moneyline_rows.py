"""
The Odds API supplies BETTABLE book moneylines, not a median.

Two separate bugs meet in this file, and both were silent.

  fallback-only   src/live_props ran this source only `if not pm_rows`.
                  Polymarket always returns rows, so it never ran: its cache
                  sat fifteen days unrefreshed against a two-hour TTL while
                  the 09-19 card carried zero bettable moneylines and the
                  Plays section read "no plays this week". DraftKings was
                  quoting the High Confidence co-main at -375 the whole time.

  the median      to_upcoming_rows averages across bookmakers and stamps
                  "The Odds API". That is a fine reference number and an
                  unplaceable bet -- card_plays only stakes a venue in
                  BETTABLE_VENUES, so such a row is fetched, shown, and then
                  refused at the gate.

Synthetic fixtures only; nothing here reads data/ or the network.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live_odds import (BETTABLE_BOOK_KEYS, ODDS_CACHE_MAX_SERVE_SEC,  # noqa: E402
                           ODDS_CACHE_TTL_SEC, _event_dates,
                           to_book_moneyline_rows)
from src.parlay_builder import BETTABLE_VENUES  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def event(a, b, when, **books):
    return {
        "home_team": a, "away_team": b, "commence_time": when,
        "bookmakers": [
            {"key": k, "last_update": when,
             "markets": [{"key": "h2h", "outcomes": [
                 {"name": a, "price": v[0]}, {"name": b, "price": v[1]}]}]}
            for k, v in books.items()
        ],
    }


# --- every venue produced must be one the staking gate accepts ------------
# If these two lists ever drift, the symptom is a price that is fetched,
# rendered, and then refused -- which looks exactly like no price at all.
check("every mapped venue is bettable", set(BETTABLE_BOOK_KEYS.values()) <= BETTABLE_VENUES)

# --- per book, not a median ----------------------------------------------
evs = [event("Arman Tsarukyan", "Mauricio Ruffy", "2026-09-20T03:30:00Z",
             draftkings=(-375, 295), fanduel=(-390, 280), betrivers=(-385, 280))]
rows = to_book_moneyline_rows(evs)
by = {(r["source"], r["selection"]): r["odds_american"] for r in rows}
check("DraftKings keeps its OWN price", by.get(("DraftKings", "Arman Tsarukyan")) == -375)
check("FanDuel keeps its OWN price", by.get(("FanDuel", "Arman Tsarukyan")) == -390)
check("the true median appears nowhere", -385.0 not in by.values())
check("an unbettable book is dropped", not any(r["source"] == "BetRivers" for r in rows))
check("both sides of the fight are priced", len(rows) == 4)
check("book prices are never flagged vig-free",
      all(r["source_is_vig_free"] is False for r in rows))
check("every row is a moneyline", all(r["market"] == "Moneyline" for r in rows))

# --- the duplicate-event trap --------------------------------------------
# One real response carried Tsarukyan three times: 09-10, 09-20 and 12-31.
# Pricing the card off the wrong listing is worse than having no price.
dupes = [
    event("Arman Tsarukyan", "Mauricio Ruffy", "2026-09-10T00:00:00Z", draftkings=(-380, 303)),
    event("Arman Tsarukyan", "Mauricio Ruffy", "2026-09-20T03:30:00Z", draftkings=(-375, 295)),
    event("Justin Gaethje", "Arman Tsarukyan", "2026-12-31T05:00:00Z", draftkings=(385, -500)),
]
picked = to_book_moneyline_rows(dupes, dates=["2026-09-19"])
check("exactly one listing survives the filter", len(picked) == 2)
check("  ...and it is the right one", {r["odds_american"] for r in picked} == {-375, 295})

# A card is dated by its US Eastern day, so a 03:30Z main event belongs to the
# DAY BEFORE in UTC terms. Matching UTC alone misfiles most of the schedule.
check("a 03:30Z start belongs to the previous ET day",
      "2026-09-19" in _event_dates("2026-09-20T03:30:00Z"))
check("  ...and its UTC day is kept too, for cards abroad",
      "2026-09-20" in _event_dates("2026-09-20T03:30:00Z"))
check("an afternoon-UTC start still matches its own day",
      "2026-09-19" in _event_dates("2026-09-19T18:00:00Z"))
check("no dates filter means no filtering", len(to_book_moneyline_rows(dupes)) == 6)
check("a date that matches nothing yields nothing",
      to_book_moneyline_rows(dupes, dates=["2026-01-01"]) == [])
check("an unparseable start time is not a match", _event_dates("whenever") == set())

# --- degenerate payloads must not raise ----------------------------------
for label, payload in (
        ("no events", []),
        ("None", None),
        ("event with no teams", [{"commence_time": "2026-09-20T03:30:00Z", "bookmakers": []}]),
        ("no bookmakers key", [{"home_team": "A", "away_team": "B"}]),
        ("outcome with no price", [{"home_team": "A", "away_team": "B",
                                    "bookmakers": [{"key": "draftkings", "markets": [
                                        {"key": "h2h", "outcomes": [{"name": "A"}]}]}]}]),
        ("a name not in the fight", [{"home_team": "A", "away_team": "B",
                                      "bookmakers": [{"key": "draftkings", "markets": [
                                          {"key": "h2h", "outcomes": [
                                              {"name": "Someone Else", "price": -200}]}]}]}]),
):
    try:
        check(f"{label} -> no rows, no crash", to_book_moneyline_rows(payload) == [])
    except Exception as exc:                      # noqa: BLE001
        check(f"{label} -> no rows, no crash ({exc})", False)

# A non-h2h market must never be read as a moneyline.
totals = [{"home_team": "A", "away_team": "B", "commence_time": "2026-09-20T03:30:00Z",
           "bookmakers": [{"key": "draftkings", "markets": [
               {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 1.5}]}]}]}]
check("a totals market yields no moneyline row", to_book_moneyline_rows(totals) == [])

# --- the quota and freshness bounds --------------------------------------
# 500 requests a month on the free tier. Anything under ~1.4h overruns it.
_calls_per_month = (24 * 30 * 3600) / ODDS_CACHE_TTL_SEC
check("the TTL fits inside the free tier", _calls_per_month <= 500)
check("  ...without being so slow the price is meaningless", ODDS_CACHE_TTL_SEC <= 6 * 3600)
# Fifteen days is what this bound exists to have refused.
check("the stale-serve bound is shorter than the outage it missed",
      ODDS_CACHE_MAX_SERVE_SEC < 15 * 86400)
check("  ...and longer than one refresh cycle", ODDS_CACHE_MAX_SERVE_SEC > ODDS_CACHE_TTL_SEC)

print(f"{'PASS' if not fail else 'FAIL'}: test_book_moneyline_rows -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
