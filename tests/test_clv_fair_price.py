"""
CLV must not be graded on a fair probability that cannot belong to its price.

book_fair_prob is a cross-source de-vigged consensus; odds_american is the
single best bettable quote. They differ by the vig plus a little line
shopping -- a couple of points. Twenty rows in predictions_log disagreed by
more than fifty, all from 2026-08-18 onward, and the error ran AGAINST the
site: the published record read 39/73 and -2.9pts where the price basis gives
43/73 and -0.2.
"""
import sys

sys.path.insert(0, ".")
from src.track_record import (FAIR_PRICE_TOLERANCE,        # noqa: E402
                             _clv_result, _fair_agrees_with_price)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


# The real corrupt pairs.
check("Tsuruya's 0.449 against -800 is rejected", not _fair_agrees_with_price(0.449, -800))
check("Hasan's 0.406 against -850 is rejected", not _fair_agrees_with_price(0.406, -850))
check("Aoriqileng's 0.477 against +370 is rejected", not _fair_agrees_with_price(0.477, 370))

# Normal vig-sized gaps must survive, or every honest row loses its fair basis.
check("a de-vigged favourite is accepted", _fair_agrees_with_price(0.872, -800))
check("a de-vigged underdog is accepted", _fair_agrees_with_price(0.213, 370))
check("exactly at the tolerance is accepted",
      _fair_agrees_with_price(0.889 - FAIR_PRICE_TOLERANCE, -800))

# Missing or unparseable inputs are not evidence of a fault.
for bad in (None, "", "n/a", float("nan")):
    check(f"{bad!r} fair is not treated as corrupt", _fair_agrees_with_price(bad, -800))
    check(f"{bad!r} odds is not treated as corrupt", _fair_agrees_with_price(0.5, bad))

# End to end: a corrupt pair must fall back to the price basis, not grade on fair.
corrupt = _clv_result(pick_odds=-650, closing_odds=-800,
                      pick_fair_prob=0.451, closing_fair_prob=0.449)
check("a corrupt pair does not grade on the fair basis",
      corrupt is None or corrupt.get("basis") == "price")
check("and the price basis says the line moved TOWARD the pick",
      corrupt is not None and corrupt.get("clv_pct", 0) > 0)

clean = _clv_result(pick_odds=-650, closing_odds=-800,
                    pick_fair_prob=0.845, closing_fair_prob=0.872)
check("a coherent pair still grades on the fair basis",
      clean is not None and clean.get("basis") == "fair")

# The shipped ledger must carry no corrupt pair through to a fair grade.
import csv                                                   # noqa: E402
import pathlib                                               # noqa: E402
graded_on_bad_fair = 0
with pathlib.Path("data/predictions_log.csv").open(newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        res = _clv_result(r.get("pick_odds"), r.get("closing_odds"),
                          r.get("pick_fair_prob"), r.get("closing_fair_prob"))
        if res and res.get("basis") == "fair" and not _fair_agrees_with_price(
                r.get("closing_fair_prob"), r.get("closing_odds")):
            graded_on_bad_fair += 1
check("no shipped row is graded on a fair that disagrees with its price",
      graded_on_bad_fair == 0)

print(f"test_clv_fair_price: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
