"""
The drift monitor must tell a moving SCALE apart from a flat run of CARDS.

That distinction is the entire reason the check compares against the market
instead of watching the model alone, so it is what these tests are for. A
monitor that fires whenever the UFC books three quiet Fight Nights gets muted,
and a muted alarm is worth less than no alarm.

Synthetic fixtures throughout, deliberately. tests/test_source_health.py once
asserted on the live source_health.json and froze CI for over two hours when
that file legitimately changed; nothing here reads data/.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from check_scale_drift import analyse, _per_card, BASELINE_CARDS, TRAILING_CARDS  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def cards(spec):
    """spec: list of (model_p90, market_p90, clearing, fights)."""
    return [{"event": f"E{i}", "date": f"2026-01-{i+1:02d}", "fights": f,
             "priced": f, "model_p90": m, "market_p90": k,
             "gap_p90": round(m - k, 4), "clearing_floor": c,
             "selectivity": c / f}
            for i, (m, k, c, f) in enumerate(spec)]


def log_for(cs, prob=0.80):
    return pd.DataFrame([{"event_name": c["event"], "favorite_prob": prob}
                         for c in cs for _ in range(c["fights"])])


N = BASELINE_CARDS

# 1. THE CARDS GOT FLATTER, NOT THE SCALE. Model and market fall together, so
#    the gap barely moves -- the floor selecting far less is honest.
flat = cards([(0.88, 0.86, 3, 14)] * N + [(0.72, 0.70, 0, 14)] * TRAILING_CARDS)
r = analyse(flat, log_for(flat))
check("a flat run of cards is not called drift", r["verdict"] != "drifted")
check("...and it is still surfaced as worth watching", r["verdict"] == "watch")
check("...because selectivity did collapse", r["off_band"] is True)
check("...while the gap held", abs(r["gap_move"]) <= 0.04)

# 2. THE SCALE MOVED. Market holds, model falls. This is the real thing.
drift = cards([(0.88, 0.86, 3, 14)] * N + [(0.72, 0.86, 0, 14)] * TRAILING_CARDS)
r = analyse(drift, log_for(drift))
check("model falling against a flat market is called drift", r["verdict"] == "drifted")
check("...with the move reported negative", r["gap_move"] < -0.04)

# 3. NOTHING HAPPENING stays quiet.
calm = cards([(0.86, 0.85, 3, 14)] * (N + TRAILING_CARDS))
r = analyse(calm, log_for(calm))
check("a steady scale is ok", r["verdict"] == "ok")
check("...and reports no move", abs(r["gap_move"]) < 1e-9)

# 4. A MODEL GETTING MORE CONFIDENT than the market is drift too -- the floor
#    would start crowning locks it was never calibrated to crown.
up = cards([(0.80, 0.85, 1, 14)] * N + [(0.95, 0.85, 8, 14)] * TRAILING_CARDS)
r = analyse(up, log_for(up))
check("drift in the confident direction is caught", r["verdict"] == "drifted")
check("...reported positive", r["gap_move"] > 0.04)

# 5. Selectivity is weighted by PICKS, not averaged over cards.
uneven = cards([(0.86, 0.85, 2, 4)] * N + [(0.86, 0.85, 2, 40)] * TRAILING_CARDS)
r = analyse(uneven, log_for(uneven))
check("selectivity weights a 40-fight card above a 4-fight one",
      abs(r["base_sel"] - 0.5) < 1e-9 and abs(r["tail_sel"] - 0.05) < 1e-9)

# 6. THE STALE ORPHAN. An event the order map cannot place is dropped rather
#    than sorted to one end of the series. Real case: a card renamed mid-week
#    leaves its old name in predictions_log with frozen probabilities.
log = pd.DataFrame(
    [{"event_name": "Real", "favorite_prob": 0.8, "pick_odds": -200, "opponent_odds": 170}] * 8
    + [{"event_name": "Orphan", "favorite_prob": 0.9, "pick_odds": -200, "opponent_odds": 170}] * 8)
got = _per_card(log, {"Real": "2026-03-01"})
check("an unplaceable event is excluded", [c["event"] for c in got] == ["Real"])

# 7. A card with too few priced fights cannot contribute a p90.
thin = pd.DataFrame([{"event_name": "Tiny", "favorite_prob": 0.8,
                      "pick_odds": -200, "opponent_odds": 170}] * 3)
check("a 3-fight card is not given a p90", _per_card(thin, {"Tiny": "2026-03-01"}) == [])

# 8. Missing prices must not crash the fold, and must not fake a market p90.
nan_odds = pd.DataFrame([{"event_name": "NoLines", "favorite_prob": 0.8,
                          "pick_odds": None, "opponent_odds": None}] * 9)
check("a card with no prices at all is skipped, not defaulted",
      _per_card(nan_odds, {"NoLines": "2026-03-01"}) == [])

print(f"test_scale_drift: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
