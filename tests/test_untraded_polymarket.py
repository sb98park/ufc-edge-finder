"""
An untraded Polymarket book must never become an edge baseline.

Polymarket returns a placeholder near 0.5 for a market nobody has touched.
The guard for this existed and required the price to be EXACTLY 0.5, so it
caught almost nothing: on UFC 331 Pantoja vs Van every one of the 26 props
had zero volume and $4-40 of liquidity, quoting 0.5/0.5, 0.51/0.49 AND
0.505/0.495. Only the exact ones were dropped. The rest reached the page as
-102fair / -104fair / -106fair, and the model differenced against them --
36 published edges from -37.6% to +10.4% on markets nobody had traded, 8 of
them positive, which is the shape that reads as value.

The discriminator is VOLUME, not the price. A maker quoting 0.67/0.33 with
real liquidity and no trades yet is an opinion worth carrying; a book pinned
at ~0.5 with $8 behind it is a placeholder. These tests hold both halves,
because widening the band without keeping the volume condition would throw
away genuine quotes.

Synthetic fixtures only; nothing here hits the network or reads data/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.polymarket_source import _classify_and_parse_market  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


TITLE = "UFC 331: Alexandre Pantoja vs. Joshua Van (Flyweight, Main Card)"


def mkt(pa, pb, volume, question="Will the fight be won by KO or TKO?"):
    return {"question": question,
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps([str(pa), str(pb)]),
            "volumeNum": volume}


def kept(pa, pb, volume, question="Will the fight be won by KO or TKO?"):
    return bool(_classify_and_parse_market(mkt(pa, pb, volume, question), TITLE))


# --- the regression: near-0.5 placeholders with no volume ----------------
check("exact 0.5/0.5 with no volume is dropped", not kept(0.5, 0.5, 0))
check("0.505/0.495 with no volume is dropped", not kept(0.505, 0.495, 0))
check("0.51/0.49 with no volume is dropped", not kept(0.51, 0.49, 0))
check("0.485/0.515 with no volume is dropped", not kept(0.485, 0.515, 0))
check("0.519/0.481 with no volume is dropped", not kept(0.519, 0.481, 0))

# --- what must SURVIVE ---------------------------------------------------
# A maker quote away from 0.5 is a real opinion even with no trades yet.
check("0.67/0.33 with no volume is kept", kept(0.67, 0.33, 0))
check("0.665/0.335 with no volume is kept", kept(0.665, 0.335, 0))
check("0.69/0.31 with no volume is kept", kept(0.69, 0.31, 0))

# Volume is the discriminator: a TRADED market near 0.5 is a real pick'em and
# must not be filtered. The Pantoja ($831) and Vera ($1,700) moneylines both
# sit here, and dropping them would be a different bug.
check("0.48/0.52 WITH volume is kept", kept(0.48, 0.52, 831.42))
check("0.495/0.505 WITH volume is kept", kept(0.495, 0.505, 1699.66))
check("0.5/0.5 WITH volume is kept", kept(0.5, 0.5, 500.0))
check("even $1 of volume is enough to keep it", kept(0.505, 0.495, 1.0))

# The band is bounded -- it must not swallow a genuine near-coin-flip quote.
check("0.55/0.45 with no volume is kept", kept(0.55, 0.45, 0))
check("0.45/0.55 with no volume is kept", kept(0.45, 0.55, 0))

# 'volume' as a fallback key, and a missing key meaning zero.
check("legacy 'volume' key is honoured",
      not _classify_and_parse_market(
          {"question": "Will the fight be won by submission?",
           "outcomes": json.dumps(["Yes", "No"]),
           "outcomePrices": json.dumps(["0.505", "0.495"]),
           "volume": 0}, TITLE))
check("absent volume key counts as untraded",
      not _classify_and_parse_market(
          {"question": "Will the fight be won by submission?",
           "outcomes": json.dumps(["Yes", "No"]),
           "outcomePrices": json.dumps(["0.505", "0.495"])}, TITLE))

print(f"{'PASS' if not fail else 'FAIL'}: test_untraded_polymarket -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
