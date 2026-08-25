"""
The bankroll, as a multiple of where it started.

WHY THIS EXISTS. A unit has always been "1% of bankroll", and the bankroll was
a number nobody was tracking -- so a unit meant 1% of an imaginary fixed sum
forever. Every figure on the site was therefore a FLAT-STAKE result, which is
not how a bankroll is built. Replayed over the graded record, the same picks
at the same unit sizes return +55.9% flat and +70.1% compounded; ladder-only
returns +60.3% flat and +78.8% compounded. Roughly fourteen points of growth
were being left on the table by the accounting alone.

EXPRESSED AS A MULTIPLE, NOT A CURRENCY. This file never learns what the
bankroll is worth, and it should not: the number is the reader's, it changes
when they deposit or withdraw, and a site that stored it would be storing a
financial detail it has no business holding. A multiple answers the only
question the record needs to answer -- has this grown, and by how much -- and
it is identical for someone starting at 500 and someone starting at 50,000.

    stake_fraction = units * UNIT_AS_BANKROLL_PCT      (1U = 1% of CURRENT)
    win   -> multiple *= 1 + stake_fraction * (decimal_odds - 1)
    loss  -> multiple *= 1 - stake_fraction
    void  -> unchanged; money that was never at risk did not move

FORWARD-ONLY, AND SEPARATE FROM THE PUBLISHED RECORD. data/predictions_log.csv
and the +63.44U it produces are a flat-stake history and stay exactly as they
are -- restating them on a compounding basis would rewrite a published number,
which is the one thing this project does not do. This starts at 1.0000 on the
day the plays ledger starts and describes the plays only.

ORDER DOES NOT CHANGE THE MULTIPLE, AND IT DOES CHANGE THE PATH. The multiple
is a product of per-bet factors, and multiplication commutes -- so a win
before a loss and a loss before a win land on exactly the same number. What
the sequence changes is everything measured ALONG the way: the peak, the
drawdown from it, and the cash each bet actually risked. Those are the figures
a bettor feels, so settlement is applied in GRADED order and the file records
how far it has got, rather than being recomputed from an unordered set.

(This docstring said the opposite until a test disagreed with it. The test was
right: the arithmetic is commutative and the intuition that "a loss early
costs less" is about cash, not about the multiple.)
"""

from __future__ import annotations

import json
import os

from src.odds_utils import american_to_decimal
from src.plays import UNIT_AS_BANKROLL_PCT

STATE_PATH = "data/bankroll.json"

# What a fresh bankroll is worth, in its own terms. Every figure derived from
# here is a ratio, so the choice of 1.0 is cosmetic -- it just makes "1.42x"
# read the way a bettor would say it.
STARTING_MULTIPLE = 1.0


def load(path: str = STATE_PATH) -> dict:
    if not os.path.exists(path):
        return {"multiple": STARTING_MULTIPLE, "settled": [], "peak": STARTING_MULTIPLE,
                "max_drawdown_pct": 0.0}
    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    state.setdefault("multiple", STARTING_MULTIPLE)
    state.setdefault("settled", [])
    state.setdefault("peak", state["multiple"])
    state.setdefault("max_drawdown_pct", 0.0)
    return state


def save(state: dict, path: str = STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
        fh.write("\n")


def apply_settled(state: dict, rows: list[dict]) -> dict:
    """
    Fold every newly graded play into the bankroll, once each.

    `rows` are plays_ledger rows. Only those with a result are considered, and
    a play_id already in `settled` is skipped -- this runs on every build, and
    a bankroll that compounded the same win twice would be fiction within a
    week.
    """
    seen = set(state.get("settled") or [])
    multiple = float(state.get("multiple", STARTING_MULTIPLE))
    peak = float(state.get("peak", multiple))
    max_dd = float(state.get("max_drawdown_pct", 0.0))

    # Graded order, because compounding is path-dependent. graded_at is written
    # once per play and never revised.
    pending = sorted(
        (r for r in rows
         if (r.get("result") or "").lower() in ("won", "lost")
         and r.get("play_id") not in seen),
        key=lambda r: (str(r.get("graded_at") or ""), str(r.get("play_id"))),
    )

    for row in pending:
        try:
            units = float(row.get("units") or 0)
            price = float(row.get("odds_american"))
        except (TypeError, ValueError):
            continue
        if units <= 0:
            continue
        fraction = units * UNIT_AS_BANKROLL_PCT
        if (row.get("result") or "").lower() == "won":
            multiple *= 1.0 + fraction * (american_to_decimal(price) - 1.0)
        else:
            multiple *= 1.0 - fraction
        seen.add(row.get("play_id"))
        peak = max(peak, multiple)
        if peak > 0:
            max_dd = max(max_dd, (peak - multiple) / peak * 100.0)

    return {"multiple": round(multiple, 6), "settled": sorted(seen),
            "peak": round(peak, 6), "max_drawdown_pct": round(max_dd, 2)}


def summarise(state: dict) -> dict:
    """What the page prints. Growth is against the start, not against peak."""
    multiple = float(state.get("multiple", STARTING_MULTIPLE))
    return {
        "multiple": round(multiple, 4),
        "growth_pct": round((multiple / STARTING_MULTIPLE - 1.0) * 100.0, 1),
        "settled": len(state.get("settled") or []),
        "max_drawdown_pct": round(float(state.get("max_drawdown_pct", 0.0)), 1),
        # A unit is 1% of the CURRENT bankroll, so this is what one is worth
        # now relative to the first bet placed. It is the whole point of the
        # file: at 1.42x, a 5U bet risks 42% more cash than it did on day one.
        "unit_vs_start": round(multiple, 4),
    }
