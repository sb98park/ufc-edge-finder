"""
The paper record has to be as hard to revise as the real one.

A shadow ledger exists to settle an argument with evidence: whether the
discretionary tiers (Medium, Low, props) are worth staking. That evidence is
worth nothing if a row can be rewritten after the fight, so the paper rows go
through plays_ledger's own set-once writers rather than a fresh CSV path.

What must hold:
  - a paper row is never a ladder play, and never reaches the real ledger
  - the paper card respects the CARD BUDGET, or it flatters itself
  - stake, price and probabilities are set once; only the settle fields move
  - grading settles paper rows from the same results map as the real ones
  - nothing here can take the build down

Synthetic fixtures and tmpdirs only; nothing reads or writes data/.
"""

import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_plays import build_card_plays as _build  # noqa: E402
from src.plays import MAX_UNITS_PER_CARD  # noqa: E402
from src.plays_ledger import committed_for  # noqa: E402
import src.shadow_ledger as shadow  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "card_nurmagomedov_song.json")
_STORE = os.path.join(tempfile.mkdtemp(prefix="shadow_"), "last_book_price.json")


def card_at_book():
    """The fixture priced at a real book, so anything on it can be staked."""
    raw = json.load(open(FIXTURE))
    for f in raw.get("fights") or []:
        for e in f.get("edges") or []:
            e["source"] = e["best_book"] = "DraftKings"
    return raw


def build(**kw):
    return _build(card_at_book(), book_price_path=_STORE, **kw)


built = build()

# --- the paper card is separate from the real one ------------------------
real_labels = {p["label"] for p in built["plays"]}
paper = built["shadow_plays"]
check("the paper card has plays to record", len(paper) > 0)
check("no paper play is a ladder play", not any(p.get("on_ladder") for p in paper))
check("no paper play is also a real play",
      not ({p["label"] for p in paper} & real_labels))
check("every paper play carries a stake", all(p["units"] > 0 for p in paper))

# --- it respects the budget ----------------------------------------------
# select_card's ceiling is the whole point: a paper card that ignored it
# would stack plays the real rule would never have allowed, and then claim
# the returns. Ladder plays are caps_exempt but still SPEND.
check("the paper card is narrower than the raw shelved list",
      len(paper) < len(built["shelved"]))
check("real plus paper stays inside the card ceiling",
      sum(p["units"] for p in paper) + sum(p["units"] for p in built["plays"])
      <= MAX_UNITS_PER_CARD)

# --- recording ------------------------------------------------------------
d = tempfile.mkdtemp(prefix="shadow_ledger_")
LED = os.path.join(d, "shadow_ledger.csv")
REAL = os.path.join(d, "plays_ledger.csv")

rows = shadow.record(built, "2026-09-15 10:00 AM ET", path=LED)
check("rows are written", len(rows) == len(paper))
check("the file exists", os.path.exists(LED))
check("THE REAL LEDGER IS NOT TOUCHED", not os.path.exists(REAL))

# KEYED BY play_id, NOT LABEL. Two fights on one card can both offer
# "Fight ends inside the distance"; keying on the label silently collapses
# them and the count check passes for the wrong reason.
first = {r["play_id"]: dict(r) for r in shadow.load(LED)}
check("every paper play is on file", len(first) == len(paper))

# --- set-once -------------------------------------------------------------
# Re-record with every price moved and a different stake. The row must not
# move: this is what makes the record evidence rather than a story.
moved = copy.deepcopy(built)
for p in moved["shadow_plays"]:
    p["odds_american"] = float(p["odds_american"]) - 500.0
    p["units"] = 99.0
    p["blended_prob"] = 0.999
shadow.record(moved, "2026-09-15 06:00 PM ET", path=LED)
after = {r["play_id"]: dict(r) for r in shadow.load(LED)}

check("no row is added on a re-record", len(after) == len(first))
frozen_ok = True
for pid, was in first.items():
    now = after[pid]
    for field in ("units", "odds_american", "blended_prob", "model_prob",
                  "fair_prob", "to_win", "tier", "published_at"):
        if str(was[field]) != str(now[field]):
            frozen_ok = False
            print(f"    moved: {pid} {field} {was[field]!r} -> {now[field]!r}")
check("stake, price and probabilities are all set once", frozen_ok)
check("  ...while last_seen still advances",
      any(after[k]["last_seen"] != first[k]["last_seen"] for k in first))

# --- grading --------------------------------------------------------------
# Settled from a results map, exactly like the real ledger.
rows = shadow.load(LED)
target = rows[0]
results = {frozenset({target["fighter_a"].strip().lower(),
                      target["fighter_b"].strip().lower()}):
           {"winner": target["fighter_a"], "method": "Decision - Unanimous"}}
n = shadow.grade_rows(rows, results, "2026-09-16 09:00 AM ET")
check("grading settles at least one paper row", n >= 1)
shadow.write_graded(rows, LED)
graded = [r for r in shadow.load(LED) if r.get("result")]
check("the settled row persists", len(graded) >= 1)
s = shadow.summarise(shadow.load(LED))
check("the summary counts what settled", s["settled"] == len(graded))
check("  ...and reports notional units", isinstance(s["units"], float))

# --- it can never take the build down -------------------------------------
for label, bad in (("a card with no shadow_plays", {"event_name": "x", "plays": []}),
                   ("an empty dict", {}),
                   ("None", None)):
    try:
        shadow.record(bad, "2026-09-15 10:00 AM ET",
                      path=os.path.join(d, f"junk_{abs(hash(label))}.csv"))
        check(f"{label} -> no crash", True)
    except Exception as exc:                      # noqa: BLE001
        check(f"{label} -> no crash ({exc})", False)

check("load on a missing file is empty, not an error",
      shadow.load("/nonexistent/nope.csv") == [])

# --- the budget carries across renders ------------------------------------
# committed_for feeds the paper card its own prior rows, so a second render
# does not re-spend a budget the first one already used.
_com = committed_for(built["event_name"], shadow.load(LED),
                     event_date=built.get("event_date"))
check("the paper ledger reports its own committed rows", len(_com) >= 1)
again = build(shadow_committed=_com)
_again_ids = {shadow.play_id(again["event_name"], *p["fight_key"].split("|", 1),
                             p["market"], p["selection"])
              for p in again["shadow_plays"]}
check("a second render adds nothing already on paper", not (_again_ids & set(first)))

print(f"{'PASS' if not fail else 'FAIL'}: test_shadow_ledger -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
