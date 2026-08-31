"""
The reach fallback: what compute_stats_rating uses when reach is unknown.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.power_rating import (attach_imputed_reach,       # noqa: E402
                              compute_stats_rating)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


roster = pd.read_csv("data/fighters.csv")
out = attach_imputed_reach(roster)

check("column added", "reach_in_imputed" in out.columns)
check("never overrides a real measurement",
      out[out["reach_in"].notna()]["reach_in_imputed"].isna().all())
check("every missing reach gets an estimate",
      out[out["reach_in"].isna()]["reach_in_imputed"].notna().all())
check("estimates are physically plausible",
      out["reach_in_imputed"].dropna().between(58, 88).all())
check("real reaches are untouched",
      out["reach_in"].equals(roster["reach_in"]))

# The fallback must beat the flat 70 on the fighters we can check.
known = roster.dropna(subset=["reach_in", "height_in"])
slope, intercept = np.polyfit(known["height_in"], known["reach_in"], 1)
fit_err = (intercept + slope * known["height_in"] - known["reach_in"]).abs().mean()
flat_err = (known["reach_in"] - 70).abs().mean()
check("height fit beats the flat 70 on held-out truth", fit_err < flat_err)
check("and by a wide margin", fit_err < flat_err / 2)

# A frame with no imputed column must behave exactly as before: 70.
base = pd.Series({"name": "X", "wins": 5, "losses": 5, "height_in": 70.0,
                  "reach_in": np.nan, "weight_class": "Lightweight", "age": 30})
at_70 = pd.Series(dict(base, reach_in=70.0))
check("absent imputed column falls back to 70 exactly",
      abs(compute_stats_rating(base) - compute_stats_rating(at_70)) < 1e-9)

# With the column present it must be used, and worth 4 points an inch.
imputed = pd.Series(dict(base, reach_in_imputed=74.0))
check("imputed reach is used",
      abs(compute_stats_rating(imputed) - compute_stats_rating(at_70) - 16.0) < 1e-9)

# A real reach always wins over an estimate, even a contradictory one.
both = pd.Series(dict(base, reach_in=72.0, reach_in_imputed=80.0))
check("a real reach beats an estimate",
      abs(compute_stats_rating(both)
          - compute_stats_rating(pd.Series(dict(base, reach_in=72.0)))) < 1e-9)

# NaN in the imputed column must not poison the rating (CLAUDE.md s4).
nan_imp = pd.Series(dict(base, reach_in_imputed=np.nan))
check("NaN imputed value falls back to 70, not NaN",
      not np.isnan(compute_stats_rating(nan_imp))
      and abs(compute_stats_rating(nan_imp) - compute_stats_rating(at_70)) < 1e-9)

# A fighter with neither reach nor height still gets the division backstop.
no_h = attach_imputed_reach(pd.DataFrame([
    dict(name="Y", wins=1, losses=1, reach_in=np.nan, height_in=np.nan,
         weight_class="Heavyweight"),
    *[dict(name=f"H{i}", wins=1, losses=1, reach_in=78.0 + (i % 3),
           height_in=74.0 + (i % 5), weight_class="Heavyweight") for i in range(40)],
]))
check("no height falls back to the division mean",
      pd.notna(no_h.loc[0, "reach_in_imputed"])
      and 77 < float(no_h.loc[0, "reach_in_imputed"]) < 81)

# Too few fighters to fit against: leave everything alone rather than guess.
tiny = attach_imputed_reach(pd.DataFrame([
    dict(name="Z", wins=1, losses=0, reach_in=np.nan, height_in=70.0,
         weight_class="Lightweight")]))
check("a frame too small to fit is left unimputed",
      tiny["reach_in_imputed"].isna().all())

# A roster with no spread in height must not fit a line through noise.
flat_h = attach_imputed_reach(pd.DataFrame([
    dict(name="W", wins=1, losses=1, reach_in=np.nan, height_in=70.0,
         weight_class="Lightweight"),
    *[dict(name=f"L{i}", wins=1, losses=1, reach_in=71.0 + (i % 3),
           height_in=70.0, weight_class="Lightweight") for i in range(40)],
]))
est = flat_h.loc[0, "reach_in_imputed"]
check("degenerate height spread falls back to the division mean",
      pd.notna(est) and 70 < float(est) < 74)

print(f"test_reach_fallback: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
