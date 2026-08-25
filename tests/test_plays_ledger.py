"""
The ledger, and the one property it exists to guarantee: a published play is
never restated.

Everything here runs against a temp file. The real ledger is append-in-spirit
and must never be a test fixture.
"""

import sys, os, csv, tempfile, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plays_ledger import (  # noqa: E402
    record_plays, load, committed_for, grade_rows, summarise, play_id, fight_key,
)
from src.card_plays import build_card_plays  # noqa: E402

FAILURES = []


def check(label, got, want, tol=1e-6):
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:58s} got {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "card_nurmagomedov_song.json")
card = build_card_plays(json.load(open(FIXTURE)))
tmp = os.path.join(tempfile.mkdtemp(), "plays_ledger.csv")

print("\nfirst publication")
rows = record_plays(card, "2026-08-25T12:00:00+00:00", path=tmp)
check("every play is written", len(rows), len(card["plays"]))
check("  ...and re-reads from disk", len(load(tmp)), len(card["plays"]))
check("nothing is graded on the way in",
      all(not r["result"] for r in rows), True)
first = {r["play_id"]: dict(r) for r in load(tmp)}

print("\nTHE PROPERTY: a published play is never restated")
# Re-run the same card with every price moved and every stake halved -- what a
# later render looks like once the line has come in. The ledger must be
# indifferent to all of it.
moved = json.loads(json.dumps(card))
for p in moved["plays"]:
    p["odds_american"] = int(p["odds_american"]) - 40
    p["units"] = round(p["units"] / 2, 1)
    p["blended_prob"] = 0.999
rows = record_plays(moved, "2026-08-27T09:00:00+00:00", path=tmp)
check("no duplicate rows", len(rows), len(card["plays"]))
sample = rows[0]
check("the price it was taken at survives",
      sample["odds_american"], first[sample["play_id"]]["odds_american"])
check("the stake survives", sample["units"], first[sample["play_id"]]["units"])
check("the probability survives",
      sample["blended_prob"], first[sample["play_id"]]["blended_prob"])
check("published_at survives", sample["published_at"], "2026-08-25T12:00:00+00:00")
check("  ...but last_seen advances", sample["last_seen"], "2026-08-27T09:00:00+00:00")

print("\nTHE ROUND TRIP, which is the property the gate actually enforces")
# This is the test that was missing, and its absence froze the live site.
# load() coerces to Python types; the writer wrote those straight back, so
# is_prop went "1" -> "True" and an empty closing_odds went "" -> "None" on
# the SECOND render. Every row rewritten and no longer equal to itself, which
# check_plays_ledger.py correctly calls a restated play -- so the gate failed
# every build from then on. A single write-and-read proves nothing here; only
# repeated ones do.
import shutil
rt = tmp + ".roundtrip"
shutil.copy(tmp, rt)
_immutable = [k for k in load(rt)[0]
              if k not in ("last_seen", "closing_odds", "result", "units_result", "graded_at")]
_before = {r["play_id"]: {k: r[k] for k in _immutable} for r in load(rt)}
for _i in range(3):
    _rows = load(rt)
    record_plays({"event_name": _rows[0]["event_name"], "event_date": _rows[0]["event_date"],
                  "plays": []}, f"2026-08-2{_i + 5}T00:00:00+00:00", path=rt)
_after = {r["play_id"]: {k: r[k] for k in _immutable} for r in load(rt)}
_drift = [(pid, k) for pid, was in _before.items()
          for k, v in was.items() if _after.get(pid, {}).get(k) != v]
check("three renders change nothing that was set once", len(_drift), 0)
with open(rt, encoding="utf-8") as _fh:
    _text = _fh.read()
check("  ...and no Python repr leaks into the file",
      ("True" in _text) or ("None" in _text) or ("False" in _text), False)

print("\nclosing price is recorded, and the settled tick is refused")
pid = rows[0]["play_id"]
record_plays(card, "2026-08-30T22:00:00+00:00",
             live_prices={pid: -180}, path=tmp)
check("a real closing price lands", load(tmp)[0]["closing_odds"], -180)
record_plays(card, "2026-08-30T23:59:00+00:00",
             live_prices={pid: -19900}, path=tmp)
# A market does not delist when the horn sounds; it prints ~0.999 while it
# resolves. That is the RESULT wearing a price, and CLV is the one number
# claiming to show edge independently of results.
check("  ...and the post-horn tick does not", load(tmp)[0]["closing_odds"], -180)

print("\ncommitted plays spend the card's budget")
committed = committed_for(card["event_name"], load(tmp))
check("every live play is committed", len(committed), len(card["plays"]))
check("  ...carrying its units",
      sum(c["units"] for c in committed), card["total_units"])
# Re-running the selector against a card that is already fully committed must
# not find room to bet the same fights again.
again = build_card_plays(json.load(open(FIXTURE)), committed=committed)
check("nothing is bet twice", len(again["plays"]), 0)
check("  ...and the total still counts what is already down",
      again["total_units"], card["total_units"])

print("\ngrading")
ledger = load(tmp)
ml = [r for r in ledger if r["market"] == "Moneyline"][0]
results = {fight_key(ml["fighter_a"], ml["fighter_b"]):
           {"winner": ml["selection"], "method": "KO/TKO", "end_round": 2, "end_time": "1:30"}}
n = grade_rows(ledger, results, "2026-08-31T04:00:00+00:00")
check("only the fights we have results for settle", n > 0, True)
graded = [r for r in ledger if r["result"]]
check("  ...and the winner is paid at its own price",
      float([r for r in graded if r["market"] == "Moneyline"][0]["units_result"]) > 0, True)
check("ungraded rows stay ungraded",
      any(not r["result"] for r in ledger), True)

print("\na cancelled fight voids at zero, it does not lose")
ledger2 = load(tmp)
target = ledger2[0]
grade_rows(ledger2, {fight_key(target["fighter_a"], target["fighter_b"]): {"cancelled": True}},
           "2026-08-31T04:00:00+00:00")
voided = [r for r in ledger2 if r["result"] == "void"]
check("it is marked void", len(voided) > 0, True)
check("  ...at zero units", float(voided[0]["units_result"]), 0.0)
check("  ...and stops spending the card's budget",
      len(committed_for(card["event_name"], ledger2)) < len(ledger2), True)

print("\nthe fight key does not care which corner is which")
ledger3 = load(tmp)
t3 = [r for r in ledger3 if r["market"] == "Moneyline"][0]
grade_rows(ledger3, {fight_key(t3["fighter_b"], t3["fighter_a"]):
                     {"winner": t3["selection"], "method": "Decision - Unanimous",
                      "end_round": 3, "end_time": "5:00"}},
           "2026-08-31T04:00:00+00:00")
check("a re-scraped card with swapped corners still settles",
      bool([r for r in ledger3 if r["play_id"] == t3["play_id"]][0]["result"]), True)

print("\nAN UNGRADED CARD NEVER REACHES THE PER-EVENT SUMMARY")
# summarise_by_event feeds `plays_events`, which src/tiering classifies FREE.
# That is only safe because ungraded rows are dropped here: a play on a fight
# that has not happened is label, price and stake -- the model layer exactly.
from src.plays_ledger import summarise_by_event  # noqa: E402
_open_only = [{"event_name": "Card A", "result": "", "units": 5.0,
               "odds_american": -190, "units_result": None, "label": "X Moneyline"}]
check("a card with only open plays is absent entirely",
      summarise_by_event(_open_only), {})
_mixed = _open_only + [{"event_name": "Card A", "result": "won", "units": 10.0,
                        "odds_american": 100, "units_result": 10.0, "label": "Y Moneyline"}]
_sum = summarise_by_event(_mixed)
check("  ...and a part-graded card shows only what has settled",
      len(_sum["Card A"]["plays"]), 1)
check("  ...so no open label can ride along",
      any(p.get("label") == "X Moneyline" for p in _sum["Card A"]["plays"]), False)
check("  ...with the staked total counting the settled play only",
      _sum["Card A"]["staked"], 10.0)

print("\nthe record counts settled rows only")
s = summarise(ledger)
check("wins and losses add up to settled", s["won"] + s["lost"], s["settled"])
check("a void is in neither", summarise(voided)["settled"], 0)
check("ROI is against what was actually staked",
      s["roi_pct"] is not None and s["staked"] > 0, True)

print("\nplay_id is stable and specific")
check("same bet, same id",
      play_id("E", "a", "b", "Moneyline", "a"), play_id("E", "a", "b", "Moneyline", "a"))
check("a rematch is a different bet",
      play_id("E2", "a", "b", "Moneyline", "a") != play_id("E", "a", "b", "Moneyline", "a"), True)
check("two markets on one fight do not collide",
      play_id("E", "a", "b", "Moneyline", "a") != play_id("E", "a", "b", "Method: KO/TKO", "a"), True)

print("\n" + ("-" * 70))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print("the ledger holds")
