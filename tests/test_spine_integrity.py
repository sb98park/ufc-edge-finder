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
from src.elo import EloRatingSystem                                  # noqa: E402
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
H = pd.DataFrame([
    {"date": "2020-01-01", "fighter_a": "A", "fighter_b": "B", "winner": "A", "method": "DEC", "promotion": ""},
    {"date": "2021-01-01", "fighter_a": "A", "fighter_b": "R", "winner": "A", "method": "KO/TKO", "promotion": "Regional"},
])
r = EloRatingSystem().build_from_history(H)
check("regional opponent never enters the rating pool", "R" not in r)
r_ufc_only = EloRatingSystem().build_from_history(H[H["promotion"] == ""])
check("regional bout does not move the UFC rating", abs(r["A"] - r_ufc_only["A"]) < 1e-9)

# "UFC" spelled explicitly must behave exactly like a blank.
H2 = H.copy()
H2.loc[0, "promotion"] = "UFC"
check("explicit UFC == blank", abs(EloRatingSystem().build_from_history(H2)["A"] - r["A"]) < 1e-9)

# A frame with no promotion column at all is the pre-change world.
check("absent promotion column is a no-op",
      abs(EloRatingSystem().build_from_history(H.drop(columns=["promotion"]))["A"] - r["A"]) > 0)

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
# Elo excludes regional bouts, so everything that is BLENDED WITH or COMPARED
# AGAINST Elo must exclude them too. Counting them in build_effective_ratings
# moved Aljarouj's blend weight from 0.25 to 1.0 and made the model fully
# trust an Elo built from one fight.
from src.elo import ufc_only                                        # noqa: E402
from src.power_rating import build_effective_ratings                # noqa: E402

mixed = pd.DataFrame([
    {"date": "2020-01-01", "fighter_a": "P", "fighter_b": "Q", "winner": "P", "method": "DEC", "promotion": ""},
    {"date": "2021-01-01", "fighter_a": "P", "fighter_b": "R1", "winner": "P", "method": "KO/TKO", "promotion": "Regional"},
    {"date": "2022-01-01", "fighter_a": "P", "fighter_b": "R2", "winner": "P", "method": "KO/TKO", "promotion": "Regional"},
    {"date": "2023-01-01", "fighter_a": "P", "fighter_b": "R3", "winner": "P", "method": "KO/TKO", "promotion": "Regional"},
])
check("ufc_only keeps blanks and drops promoted rows", len(ufc_only(mixed)) == 1)
check("ufc_only is a no-op without the column",
      len(ufc_only(mixed.drop(columns=["promotion"]))) == len(mixed))

roster = pd.DataFrame([{"name": "P", "wins": 4, "losses": 0, "reach_in": 70.0,
                        "height_in": 70.0, "age": 30}])
elo_r = EloRatingSystem().build_from_history(mixed)
eff_mixed = build_effective_ratings(roster, elo_r, mixed)
eff_ufc = build_effective_ratings(roster, elo_r, ufc_only(mixed))
check("regional bouts do not raise the Elo blend weight",
      abs(eff_mixed["P"] - eff_ufc["P"]) < 1e-9)

# The streak bonus has the same exposure: a regional win run is not a UFC one.
solo = pd.DataFrame([{"name": "P", "wins": 1, "losses": 0, "reach_in": 70.0,
                      "height_in": 70.0, "age": 30}])
one_ufc = build_effective_ratings(solo, elo_r, mixed.head(1))
check("streak bonus counts connected wins only",
      abs(build_effective_ratings(solo, elo_r, mixed)["P"] - one_ufc["P"]) < 1e-9)

print(f"test_spine_integrity: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
