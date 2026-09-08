"""
The things that failed silently must reach the page.

Three failures in three days surfaced only as an absence: a fight wrongly
cancelled at 04:08, a landing page that stopped rebuilding while the build
exited 0, and a hard test gate that held the refresh down for eleven hours.
All three WERE recorded -- source_health's `steps` block has carried step
outcomes since it was built. It is published to the GitHub Actions run
summary, which is the one place nobody opens.

build_health_alerts measures nothing new. It reads that same record and
hands it to the page. These tests are about the two properties that decide
whether anyone ever benefits: it must be SILENT on a healthy build, or it
becomes furniture; and it must not miss a stranded pick.

Synthetic fixtures in a temp dir; nothing here reads data/.
"""

import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import generate_site as gs  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def setup(steps, cards, picks):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "source_health.json"), "w") as fh:
        json.dump({"steps": steps}, fh)
    preds = os.path.join(d, "predictions_log.csv")
    with open(preds, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["event_name", "fighter_a", "fighter_b", "voided"])
        w.writeheader()
        for a, b, v in picks:
            w.writerow({"event_name": "E", "fighter_a": a, "fighter_b": b, "voided": v})
    return d, pd.DataFrame(cards), preds


def run(steps=None, cards=None, picks=None):
    d, df, preds = setup(steps or {}, cards or [], picks or [])
    old = gs.DATA_DIR
    gs.DATA_DIR = d
    try:
        return gs.build_health_alerts(df, preds)
    finally:
        gs.DATA_DIR = old


OK_STEP = {"ok": True, "consecutive_failures": 0, "last_ok": "2026-09-08T06:00:00+00:00"}


# --- SILENCE ON A HEALTHY BUILD -------------------------------------------
# The property that decides whether the strip is ever read.
check("a healthy build produces no alerts",
      run(steps={"backfill_history": OK_STEP, "merge_results": OK_STEP}) == [])
check("no steps recorded yet is not an alert", run(steps={}) == [])
check("a missing health file is not an alert", run() == [])

# --- A FAILING STEP -------------------------------------------------------
a = run(steps={"backfill_history": {"ok": False, "consecutive_failures": 4,
                                    "last_ok": "2026-09-05T22:10:00+00:00",
                                    "error": "NameError: espn_id"}})
check("a failing step is reported", len(a) == 1)
check("...with its streak", "4 runs in a row" in a[0]["text"])
check("...named", "backfill_history" in a[0]["text"])
check("...and when it last worked", "2026-09-05 22:10" in a[0]["detail"])
check("...carrying the error", "NameError" in a[0]["detail"])
check("a 4-run streak is severe", a[0]["severe"] is True)

one = run(steps={"grade_prop_prices": {"ok": False, "consecutive_failures": 1,
                                       "last_ok": "2026-09-08T06:00:00+00:00"}})
check("a single failure is reported but not severe",
      len(one) == 1 and one[0]["severe"] is False)
check("singular grammar for one run", "1 run in a row" in one[0]["text"])
check("a step that never succeeded says so",
      "never" in run(steps={"x": {"ok": False, "consecutive_failures": 2}})[0]["detail"])

# --- A CANCELLED FIGHT THAT STILL CARRIES A PICK --------------------------
# The 2026-09-07 case: the orphan branch cancelled a live bout and the only
# trace was one line in a CI log.
cards = [{"event_name": "Noche", "fighter_a": "Ramiro Jimenez",
          "fighter_b": "Rodrigo Vera", "cancelled": "True"}]
a = run(cards=cards, picks=[("Ramiro Jimenez", "Rodrigo Vera", "")])
check("a cancelled fight with a published pick is reported", len(a) == 1)
check("...and is severe", a[0]["severe"] is True)
check("...naming both fighters",
      "Ramiro Jimenez" in a[0]["text"] and "Rodrigo Vera" in a[0]["text"])

# A cancellation with no pick behind it is routine and must stay quiet.
check("a cancelled fight with no pick is not an alert",
      run(cards=cards, picks=[("Someone Else", "Another Guy", "")]) == [])
# A live fight with a pick is the normal case.
check("a live fight with a pick is not an alert",
      run(cards=[dict(cards[0], cancelled="")], picks=[("Ramiro Jimenez", "Rodrigo Vera", "")]) == [])
# An already-voided pick is not stranded -- it has been dealt with.
check("a voided pick on a cancelled fight is not an alert",
      run(cards=cards, picks=[("Ramiro Jimenez", "Rodrigo Vera", "True")]) == [])
# Corners get swapped between scrapes (s4), so matching must not be positional.
check("the pick matches with the corners swapped",
      len(run(cards=cards, picks=[("Rodrigo Vera", "Ramiro Jimenez", "")])) == 1)
# And name variants must fold, or a stranded pick goes unreported.
check("the pick matches across a name variant",
      len(run(cards=[{"event_name": "N", "fighter_a": "Jose Miguel Delgado",
                      "fighter_b": "Jean Silva", "cancelled": "True"}],
              picks=[("Jose Delgado", "Jean Silva", "")])) == 1)

# --- both classes at once -------------------------------------------------
both = run(steps={"backfill_history": {"ok": False, "consecutive_failures": 4}},
           cards=cards, picks=[("Ramiro Jimenez", "Rodrigo Vera", "")])
check("both classes surface together", len(both) == 2)
check("...and are distinguishable", {x["kind"] for x in both} == {"step", "cancelled"})

print(f"{'PASS' if not fail else 'FAIL'}: test_health_alerts -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
