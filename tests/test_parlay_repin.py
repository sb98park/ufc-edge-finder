"""
A pin that moves within a card must retire the slip it replaced.

pinned_at is never cleared, deliberately -- that is what keeps a slip
gradeable after the pin rotates to the NEXT card, and it was introduced
because three cards' worth of slips had become permanently ungradeable.

The side effect is that a pin moving WITHIN a card leaves two slips stamped
for the same (event, tier) and grades both. On 2026-09-05 that turned one
card into a 1-1 parlay record: Wood+Benouaich held the slot from 09-04,
Page+Benouaich replaced it the next morning, and the ledger settled both.
Only one was ever the standing commitment.

These tests pin the two properties against each other, because a fix that
loses the second one re-breaks what the stamp was for:
  1. a replaced slip inside one card is retired and never graded
  2. a slip whose card is simply OVER stays gradeable forever

Synthetic fixtures and a temp ledger; nothing here reads data/.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parlay_grader import grade_pinned, summarise, void_stale_slips  # noqa: E402
from src.parlay_ledger import load  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


EVENT = "Test Card: A vs. B"
NOW = "2026-09-05 12:00 PM ET"


# Who each pick is booked against, so a leg can carry a real fight_key.
OPP = {"Wood": "Andrusca", "Benouaich": "Montenegro",
       "Page": "Ruziboev", "Ziam": "Sola"}


def slip(sid, legs, **kw):
    r = {"slip_id": sid, "event": EVENT, "tier": "bankroll",
         "legs": [{"label": f"{n} ML", "market": "Moneyline", "selection": n,
                   "fight_key": f"{n}|{OPP[n]}",
                   "conditions": [{"kind": "winner", "fighter": n}]} for n in legs],
         "combined_decimal": 2.5, "first_seen": "2026-09-04T00:00:00+00:00",
         "last_seen": "2026-09-05T00:00:00+00:00"}
    r.update(kw)
    return r


def pins_for(sid, pinned_at, event=EVENT, tier="bankroll"):
    return {event: {tier: {"pinned_at": pinned_at, "snapshot": {"slip_id": sid}}}}


def run(rows, pins, now=NOW, results=None):
    d = tempfile.mkdtemp()
    lp, rp = os.path.join(d, "l.jsonl"), os.path.join(d, "r.csv")
    with open(lp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(rp, "w") as fh:
        fh.write("event_name,fighter_a,fighter_b,winner,method,end_round,end_time\n")
        for line in (results or []):
            fh.write(line + "\n")
    grade_pinned(pins, now, path=lp, results_path=rp)
    return {r["slip_id"]: r for r in load(lp)}


# --- 1. the regression: pin moves within one card -------------------------
old = slip("OLD", ["Wood", "Benouaich"], pinned_at="2026-09-04T20:51:30+00:00")
new = slip("NEW", ["Page", "Benouaich"])
out = run([old, new], pins_for("NEW", "2026-09-05T14:03:56+00:00"))

check("the replaced slip is retired", bool(out["OLD"].get("superseded_at")))
check("and it names its replacement", out["OLD"].get("superseded_by") == "NEW")
check("the standing slip is not retired", not out["NEW"].get("superseded_at"))
check("the standing slip keeps its pin stamp", bool(out["NEW"].get("pinned_at")))

# Retired is NOT void: a void is a bet that existed and was refunded, this is
# a bet that stopped existing before the card.
check("a retired slip takes no result", not (out["OLD"].get("result") or "").strip())
check("a retired slip takes no units", out["OLD"].get("units_result") in (None, ""))

# --- 2. it must not grade even once both legs have results ----------------
res = [f"{EVENT},Wood,Andrusca,Andrusca,Decision,3,5:00",
       f"{EVENT},Benouaich,Montenegro,Benouaich,Submission,2,0:38",
       f"{EVENT},Page,Ruziboev,Page,Decision,3,5:00"]
out2 = run([dict(old), dict(new)], pins_for("NEW", "2026-09-05T14:03:56+00:00"), results=res)
check("retired slip stays ungraded even with every leg resolved",
      not (out2["OLD"].get("result") or "").strip())
check("the standing slip still grades normally",
      (out2["NEW"].get("result") or "") in ("cashed", "dead", "void"))

# The published record must therefore show ONE slip for this card, not two.
s = summarise(list(out2.values()))
check("one card yields one graded slip", s["n"] == 1)
check("and one event", s["events"] == 1)

# --- 3. THE PROPERTY THE FIX MUST NOT LOSE --------------------------------
# The pin rotating to the NEXT card must leave last week's slip gradeable.
# This is what pinned_at exists for; a fix that retires on "not currently
# pinned" instead of "replaced within its own (event, tier)" breaks it.
lastweek = slip("LASTWEEK", ["Wood", "Benouaich"], pinned_at="2026-08-29T00:00:00+00:00")
out3 = run([lastweek], pins_for("OTHER", "2026-09-05T00:00:00+00:00",
                                event="Some Other Card"), results=res)
check("a slip whose card is over is NOT retired", not out3["LASTWEEK"].get("superseded_at"))
check("and it still grades after the pin rotates away",
      (out3["LASTWEEK"].get("result") or "") in ("cashed", "dead", "void"))

# A different TIER on the same card is its own slot, not a replacement.
a = slip("TIER_A", ["Page"], tier="bankroll", pinned_at="2026-09-04T00:00:00+00:00")
b = slip("TIER_B", ["Wood"], tier="longshot")
pins = pins_for("TIER_A", "2026-09-04T00:00:00+00:00")
pins[EVENT]["longshot"] = {"pinned_at": "2026-09-05T00:00:00+00:00",
                           "snapshot": {"slip_id": "TIER_B"}}
out4 = run([a, b], pins)
check("a second tier does not retire the first", not out4["TIER_A"].get("superseded_at"))
check("nor the other way round", not out4["TIER_B"].get("superseded_at"))

# --- 4. idempotence: the cron runs this every 5 minutes -------------------
first = run([dict(old), dict(new)], pins_for("NEW", "2026-09-05T14:03:56+00:00"))
stamp = first["OLD"]["superseded_at"]
again = run(list(first.values()), pins_for("NEW", "2026-09-05T14:03:56+00:00"),
            now="2026-09-05 01:00 PM ET")
check("the retirement timestamp is not rewritten on a later pass",
      again["OLD"]["superseded_at"] == stamp)

# --- 5. a retired slip must not collect a stale void either ---------------
stale = slip("STALE", ["Wood"], pinned_at="2026-09-01T00:00:00+00:00",
             superseded_at=NOW, superseded_by="NEW",
             last_seen="2026-09-01T00:00:00+00:00")
changed = void_stale_slips([stale], "2026-09-10 12:00 PM ET")
check("void_stale skips a retired slip", changed == [])
check("so it never acquires a void result", not (stale.get("result") or "").strip())

print(f"{'PASS' if not fail else 'FAIL'}: test_parlay_repin -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
