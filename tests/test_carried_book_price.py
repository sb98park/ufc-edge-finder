"""
A ladder pick must not be dropped because one fetch came back thin.

The venue gate refuses any play not priced at a bettable book, which is
right: a record measured at prices nobody offers is worthless. But the feed
does not quote continuously, and the published ladder promises EVERY Lock
and High Confidence pick. A subscriber cannot reproduce a record that
silently skips one because DraftKings missed a fighter for an hour.

So a ladder pick may carry a recent book quote. The window is the whole
safety argument: the ladder stakes a FLAT tier size regardless of price, so
a carried quote changes nothing about whether we bet or how much -- it
changes the price the ledger CLAIMS, and therefore the return it reports.

Measured on the 2026-09-19 card: the co-main and main event carried book
quotes 358 HOURS old with one history point each, while sixteen other
fighters had quotes 2.1 hours old with thirty points. A window loose enough
to stake the first group would publish a fortnight-old price as available.

Synthetic fixtures in a temp dir; nothing here reads data/.
"""

import datetime as dt
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_plays import (BOOK_PRICE_MAX_AGE_HOURS, _book_key,  # noqa: E402
                            _carried_book_price, _load_last_book_prices,
                            _remember_book_prices)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


NOW = dt.datetime(2026, 9, 14, 12, 0, tzinfo=dt.timezone.utc)
ago = lambda h: (NOW - dt.timedelta(hours=h)).isoformat(timespec="seconds")


def store(**entries):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "last_book_price.json")
    with open(p, "w") as fh:
        json.dump(entries, fh)
    return p


# --- remembering ----------------------------------------------------------
d = tempfile.mkdtemp()
path = os.path.join(d, "s.json")
_remember_book_prices([
    {"market": "Moneyline", "fighter": "Arman Tsarukyan", "odds_american": -500,
     "best_book": "DraftKings"},
    {"market": "Moneyline", "fighter": "Someone Else", "odds_american": 120,
     "best_book": "Polymarket"},                      # reference: not remembered
    {"market": "Method: KO/TKO", "fighter": "Arman Tsarukyan", "odds_american": -150,
     "best_book": "DraftKings"},                      # not a moneyline
], NOW.isoformat(timespec="seconds"), path)
s = _load_last_book_prices(path)
check("a bettable moneyline is remembered", _book_key("Arman Tsarukyan", "Moneyline") in s)
check("a reference price is NOT remembered", len(s) == 1)
check("...and the venue is kept", s[_book_key("Arman Tsarukyan", "Moneyline")]["venue"] == "DraftKings")
check("...and the price", s[_book_key("Arman Tsarukyan", "Moneyline")]["odds"] == -500)

# --- carrying, and the window --------------------------------------------
key = _book_key("Arman Tsarukyan", "Moneyline")
fresh = _load_last_book_prices(store(**{key: {"odds": -500, "venue": "DraftKings", "at": ago(2)}}))
got = _carried_book_price("Arman Tsarukyan", "Moneyline", NOW, fresh)
check("a 2-hour-old quote is carried", got == (-500.0, "DraftKings"))

edge = _load_last_book_prices(store(**{key: {"odds": -500, "venue": "DraftKings",
                                             "at": ago(BOOK_PRICE_MAX_AGE_HOURS - 0.5)}}))
check("just inside the window is carried",
      _carried_book_price("Arman Tsarukyan", "Moneyline", NOW, edge) is not None)

old = _load_last_book_prices(store(**{key: {"odds": -500, "venue": "DraftKings",
                                            "at": ago(BOOK_PRICE_MAX_AGE_HOURS + 0.5)}}))
check("just outside the window is NOT carried",
      _carried_book_price("Arman Tsarukyan", "Moneyline", NOW, old) is None)

# THE REAL CASE that prompted this: a fortnight-old quote must never stake.
ancient = _load_last_book_prices(store(**{key: {"odds": -500, "venue": "DraftKings", "at": ago(358)}}))
check("the 358-hour quote on the 09-19 co-main is NOT carried",
      _carried_book_price("Arman Tsarukyan", "Moneyline", NOW, ancient) is None)

# --- name folding, so a variant spelling still finds its quote ------------
folded = _load_last_book_prices(store(**{_book_key("Jose Miguel Delgado", "Moneyline"):
                                         {"odds": -200, "venue": "FanDuel", "at": ago(1)}}))
check("a quote is found across a name variant",
      _carried_book_price("Jose Delgado", "Moneyline", NOW, folded) == (-200.0, "FanDuel"))

# --- degenerate records must not raise -----------------------------------
for label, rec in (("no timestamp", {"odds": -500, "venue": "DraftKings"}),
                   ("unparseable timestamp", {"odds": -500, "venue": "DK", "at": "nope"}),
                   ("no odds", {"venue": "DraftKings", "at": ago(1)}),
                   ("not a dict", "garbage")):
    st = _load_last_book_prices(store(**{key: rec}))
    check(f"{label} -> None, not a crash",
          _carried_book_price("Arman Tsarukyan", "Moneyline", NOW, st) is None)

check("an unknown selection is None",
      _carried_book_price("Nobody At All", "Moneyline", NOW, fresh) is None)
check("a missing store file is empty, not an error",
      _load_last_book_prices("/nonexistent/nope.json") == {})

# --- the store stays bounded ---------------------------------------------
# It is COMMITTED state rewritten every ~5 minutes, so an entry that can never
# be carried again is permanent diff noise. Every write prunes.
_stale = BOOK_PRICE_MAX_AGE_HOURS + 1
_p = store(**{
    _book_key("Long Gone", "Moneyline"): {"odds": -200, "venue": "DraftKings", "at": ago(_stale)},
    _book_key("Still Fresh", "Moneyline"): {"odds": -200, "venue": "DraftKings", "at": ago(1)},
    _book_key("Unreadable", "Moneyline"): {"odds": -200, "venue": "DraftKings", "at": "nope"},
})
_remember_book_prices(
    [{"market": "Moneyline", "fighter": "New Guy", "odds_american": -150,
      "best_book": "FanDuel"}],
    NOW.isoformat(timespec="seconds"), _p)
_after = _load_last_book_prices(_p)
check("the aged-out entry is dropped", _book_key("Long Gone", "Moneyline") not in _after)
check("  ...and so is the unreadable one", _book_key("Unreadable", "Moneyline") not in _after)
check("  ...while one inside the window survives", _book_key("Still Fresh", "Moneyline") in _after)
check("  ...alongside the quote just seen", _book_key("New Guy", "Moneyline") in _after)
check("  ...and nothing else is invented", len(_after) == 2)

# A build that sees no bettable quote at all still tidies -- otherwise a dead
# card's entries sit in the file until something happens to overwrite them.
_p2 = store(**{_book_key("Long Gone", "Moneyline"):
               {"odds": -200, "venue": "DraftKings", "at": ago(_stale)}})
_remember_book_prices([], NOW.isoformat(timespec="seconds"), _p2)
check("an empty build still prunes", _load_last_book_prices(_p2) == {})

# The window has to stay meaningfully shorter than the fortnight it exists
# to reject, and long enough to cover a feed that pulls every ~35 minutes.
check("the window is bounded", 1 <= BOOK_PRICE_MAX_AGE_HOURS <= 24)

print(f"{'PASS' if not fail else 'FAIL'}: test_carried_book_price -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
