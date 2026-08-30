"""
The model must not read absence as evidence.

WHAT WENT WRONG. Michael Aljarouj was priced for 2026-09-05 off ONE bout.
fighters.csv has him 13-3; fight_history.csv held one of those sixteen, from
2021. layoff_penalty read a 5.47-year layoff and charged -89.3 rating points.
His real last fight was 2025-04-12 -- worth -8.0. Eighty-one points of the
card's second-largest model-vs-market gap were an artefact of our own
coverage, and nothing in the pipeline said so: a human found it by opening
Tapology on a phone, five days out.

The asymmetry is the argument. last_fight_date is the newest bout WE HOLD, so
on a partial history it is a LOWER BOUND on the fighter's activity -- the real
last fight can only be more recent. Both consumers only ever subtract points.
So a gap in our data can manufacture ring rust and can never remove any.

These tests pin the guard, and pin the two things that must NOT change: a
caller who supplies no coverage column, and a fighter who is genuinely
inactive with a complete history.
"""

import sys, os, datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import matchup_model as mm  # noqa: E402

FAILURES = []
REF = dt.date(2026, 9, 5)


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:64s} got {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def row(**kw):
    base = {"name": "X", "last_fight_date": "2021-03-18", "wins": 13, "losses": 3}
    base.update(kw)
    return pd.Series(base)


print("\ncoverage is bouts held over bouts claimed, from two independent sources")
fighters = pd.DataFrame([
    {"name": "Michael Aljarouj", "wins": 13, "losses": 3},
    {"name": "Fully Covered", "wins": 2, "losses": 0},
    {"name": "No Record", "wins": None, "losses": None},
    {"name": "Debutant", "wins": 0, "losses": 0},
])
history = pd.DataFrame([
    {"fighter_a": "Michael Aljarouj", "fighter_b": "Azat Maksum"},
    {"fighter_a": "Fully Covered", "fighter_b": "Someone"},
    {"fighter_a": "Other", "fighter_b": "Fully Covered"},
])
cov = mm.attach_history_coverage(fighters, history).set_index("name")["history_coverage"]
check("1 of 16 bouts", round(float(cov["Michael Aljarouj"]), 3), 0.062)
check("2 of 2 bouts", float(cov["Fully Covered"]), 1.0)
check("no record is unmeasurable, not complete", pd.isna(cov["No Record"]), True)
check("nor is 0-0", pd.isna(cov["Debutant"]), True)

print("\nand a partial history stops the layoff penalty being invented")
check("shipped behaviour with no coverage column at all",
      round(mm.layoff_penalty(row(), REF), 1), -89.3)
check("unchanged when coverage says we hold it all",
      round(mm.layoff_penalty(row(history_coverage=1.0), REF), 1), -89.3)
check("unchanged when coverage is unmeasurable (NaN)",
      round(mm.layoff_penalty(row(history_coverage=float("nan")), REF), 1), -89.3)
check("suppressed at 1 of 16", mm.layoff_penalty(row(history_coverage=0.0625), REF), 0.0)
check("suppressed just under the floor",
      mm.layoff_penalty(row(history_coverage=0.599), REF), 0.0)
check("NOT suppressed at the floor",
      round(mm.layoff_penalty(row(history_coverage=0.60), REF), 1), -89.3)

print("\na genuinely inactive fighter we hold in full is still penalised")
check("5 years out, complete history",
      round(mm.layoff_penalty(row(history_coverage=1.0,
                                  last_fight_date="2021-09-05"), REF), 1), -80.0)
check("and the cap still binds",
      mm.layoff_penalty(row(history_coverage=1.0,
                            last_fight_date="1990-01-01"), REF), -mm.LAYOFF_PENALTY_CAP)

print("\nthe quick-return penalty rides on the same date and follows it")
ko = {"last_fight_result": "L", "last_fight_method": "KO/TKO",
      "last_fight_date": "2026-06-05", "wins": 13, "losses": 3}
check("a real quick return after a knockout still costs",
      mm.quick_return_penalty(pd.Series({**ko, "history_coverage": 1.0}), REF) < 0, True)
check("the same shape on a history we cannot see is not charged",
      mm.quick_return_penalty(pd.Series({**ko, "history_coverage": 0.06}), REF), 0.0)

print("\nand the layoff we report is the layoff we used")
check("no number is displayed for a partial history",
      mm.layoff_years(row(history_coverage=0.06), REF), None)

print("\nthe alarm and the model agree on who is thin")
from scripts import check_card_data_coverage as cc  # noqa: E402
check("one floor, not two", cc.COVERAGE_FLOOR, mm.HISTORY_COVERAGE_FLOOR)

print("\n" + ("-" * 78))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print("absence is not evidence, and evidence is still evidence")
