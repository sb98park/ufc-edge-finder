"""
The spine's replay order, its promotion filter, and the reconciliation
between fight_history.csv and fighters.csv.

Each check here corresponds to a defect that was live on 2026-08-31.
"""
import datetime as dt
import subprocess
import sys
import unicodedata

import pandas as pd

sys.path.insert(0, ".")
import pathlib                                                      # noqa: E402

from src.elo import EloRatingSystem, ufc_only                       # noqa: E402
from src.matchup_model import (attach_history_coverage,              # noqa: E402
                               layoff_years,
                               reconcile_last_fight_from_history)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


# ---------------------------------------------------------------- promotion
# THE FILTER IS NO LONGER APPLIED TO THE RATING GRAPH. It was, for one day.
# Measured point-in-time over 9,198 UFC bouts, excluding regional bouts cost
# +0.00584 Brier at p=0.000 -- a fighter with ten regional wins really is
# better than the 1500 default, so the alternative to a biased estimate was no
# estimate (scripts/validate_spine_cleanup.py). elo and power_rating now both
# replay every row; fun_facts still filters, because comparability between
# published superlatives is a different question from prediction.
H = pd.DataFrame([
    {"date": "2020-01-01", "fighter_a": "A", "fighter_b": "B", "winner": "A", "method": "DEC", "promotion": ""},
    {"date": "2021-01-01", "fighter_a": "A", "fighter_b": "R", "winner": "A", "method": "KO/TKO", "promotion": "Regional"},
])
r = EloRatingSystem().build_from_history(H)
check("a regional opponent IS in the rating pool -- it carries signal", "R" in r)
check("a regional bout moves the rating", r["A"] > 1500.0)

check("ufc_only still exists for fun_facts", callable(ufc_only))
check("ufc_only still drops a promoted row", len(ufc_only(H)) == 1)
check("ufc_only is a no-op without the column",
      len(ufc_only(H.drop(columns=["promotion"]))) == len(H))
check("fun_facts still applies it",
      "ufc_only(" in pathlib.Path("src/fun_facts.py").read_text(encoding="utf-8"))

# THE TWO MUST AGREE. Their disagreement -- elo scoring UFC-only while
# power_rating counted every row -- is what published Sintes at 76% against a
# truer 57%. Whichever way the filter goes, it goes for both.
_elo_src = pathlib.Path("src/elo.py").read_text(encoding="utf-8")
_pr_src = pathlib.Path("src/power_rating.py").read_text(encoding="utf-8")
_elo_filters = "ufc_only(fight_history_df)" in _elo_src
_pr_filters = "ufc_only(history_df)" in _pr_src
check("elo and power_rating agree about which bouts count",
      _elo_filters == _pr_filters)

# ------------------------------------------------------------- stable sort
# Non-stable sorting reshuffles rows within a date; Elo replays row by row,
# so an unstable sort silently rewrites ratings with no data change.
real = pd.read_csv("data/fight_history.csv")
base = EloRatingSystem().build_from_history(real)
resorted = EloRatingSystem().build_from_history(real.sort_values("date", kind="stable"))
check("stable re-sort of the real spine changes nothing",
      all(abs(resorted[k] - base[k]) < 1e-9 for k in base))

# ------------------------------------------------------------- no duplicates
def fold(n):
    s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


keys = [(frozenset({fold(a), fold(b)}), str(d)[:10])
        for a, b, d in zip(real["fighter_a"], real["fighter_b"], real["date"])]
check("no fight is recorded twice in the shipped spine",
      len(keys) == len(set(keys)))
check("spine is in date order", real["date"].astype(str).is_monotonic_increasing)

# --------------------------------------------------------------- reconcile
F = pd.DataFrame([{"name": "A", "last_fight_date": "2019-01-01",
                   "last_fight_opponent": "old", "last_fight_result": "W",
                   "wins": 2, "losses": 0}])
out = reconcile_last_fight_from_history(F, H)
check("last_fight_date moves forward to the newest spine bout",
      str(out.loc[0, "last_fight_date"]) == "2021-01-01")
check("opponent carried across", out.loc[0, "last_fight_opponent"] == "R")

# It must NEVER move backwards: the term only ever subtracts points, so an
# over-old date invents ring rust while an over-recent one merely under-charges.
F_new = F.copy()
F_new.loc[0, "last_fight_date"] = "2025-01-01"
check("a more recent fighters.csv value is left alone",
      str(reconcile_last_fight_from_history(F_new, H).loc[0, "last_fight_date"]) == "2025-01-01")

# A no contest round-trips through the CSV as NaN, and NaN is TRUTHY --
# `winner or ""` yields the string "nan" and grades the NC as a loss.
H_nc = pd.DataFrame([{"date": "2022-01-01", "fighter_a": "A", "fighter_b": "Z",
                      "winner": float("nan"), "method": "NC", "promotion": "Regional"}])
check("a no contest is recorded as NC, not L",
      reconcile_last_fight_from_history(F, H_nc).loc[0, "last_fight_result"] == "NC")
check("a no contest still counts as activity",
      str(reconcile_last_fight_from_history(F, H_nc).loc[0, "last_fight_date"]) == "2022-01-01")

# ---------------------------------------------------------------- coverage
# Denominator is wins+losses, so the numerator must also exclude NC/draws --
# otherwise coverage exceeds 1.0 and reads as holding more than exists.
F_cov = pd.DataFrame([{"name": "A", "wins": 1, "losses": 1}])
H_cov = pd.DataFrame([
    {"date": "2020-01-01", "fighter_a": "A", "fighter_b": "B", "winner": "A", "method": "DEC"},
    {"date": "2020-06-01", "fighter_a": "A", "fighter_b": "C", "winner": "C", "method": "DEC"},
    {"date": "2021-01-01", "fighter_a": "A", "fighter_b": "D", "winner": "", "method": "NC"},
])
check("no contests are excluded from coverage",
      abs(float(attach_history_coverage(F_cov, H_cov).loc[0, "history_coverage"]) - 1.0) < 1e-9)

# ------------------------------------------------ the case that found it all
f = attach_history_coverage(
    reconcile_last_fight_from_history(pd.read_csv("data/fighters.csv"), real), real)
a = f[f["name"] == "Michael Aljarouj"]
if a.empty:
    print("  (skipped Aljarouj checks -- not on the roster)")
else:
    a = a.iloc[0]
    check("Aljarouj holds his full decided record",
          abs(float(a["history_coverage"]) - 1.0) < 1e-9)
    check("Aljarouj's last fight is the 2025 no contest",
          str(a["last_fight_date"])[:10] == "2025-04-12" and a["last_fight_result"] == "NC")
    yrs = layoff_years(a, dt.date(2026, 9, 5))
    check("layoff is read, and is ~1.4y rather than 5.5y", yrs is not None and 1.3 < yrs < 1.5)
    check("his regional run stays out of the rating pool",
          "Ronny Gomez" not in EloRatingSystem().build_from_history(real))

# ------------------------------------- connected-history propagation
# power_rating's blend weight and streak count must be built from THE SAME
# rows elo replayed. That is the invariant; which rows those are is decided in
# one place and tested above.
from src.power_rating import build_effective_ratings                # noqa: E402

mixed = pd.DataFrame([
    {"date": "2020-01-01", "fighter_a": "P", "fighter_b": "Q", "winner": "P", "method": "DEC", "promotion": ""},
    {"date": "2021-01-01", "fighter_a": "P", "fighter_b": "R1", "winner": "P", "method": "KO/TKO", "promotion": "Regional"},
    {"date": "2022-01-01", "fighter_a": "P", "fighter_b": "R2", "winner": "P", "method": "KO/TKO", "promotion": "Regional"},
    {"date": "2023-01-01", "fighter_a": "P", "fighter_b": "R3", "winner": "P", "method": "KO/TKO", "promotion": "Regional"},
])
roster = pd.DataFrame([{"name": "P", "wins": 4, "losses": 0, "reach_in": 70.0,
                        "height_in": 70.0, "age": 30}])
elo_r = EloRatingSystem().build_from_history(mixed)
check("regional bouts now count toward the Elo blend weight",
      abs(build_effective_ratings(roster, elo_r, mixed)["P"]
          - build_effective_ratings(roster, elo_r, ufc_only(mixed))["P"]) > 1e-9)

print(f"test_spine_integrity: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
