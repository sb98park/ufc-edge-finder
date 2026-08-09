"""
Do the new ESPN-derived stats sit on the SAME SCALE as the old ufcstats ones?

WHY THIS AND NOT A FULL BACKTEST. The striking and wrestling terms were
validated with these columns populated (adjustment layer 57.7% vs 55.9%
Elo-only). The columns then went dark and the terms fell back to hardcoded
constants -- 45 / 20 / 65 -- which describe no fighter who has ever
competed. Restoring real measurements returns the model to the configuration
that was validated; it is not a new, unproven feature.

What genuinely could break is SCALE. STRIKING_ADVANTAGE_SCALE and
WRESTLING_ADVANTAGE_SCALE were tuned against ufcstats numbers. If ESPN counts
significant strikes differently, or if takedown defence derived from the
opponent's attempts sits systematically higher or lower than ufcstats'
published figure, then every individual value is correct while the
DIFFERENTIALS -- which is all the model consumes -- change magnitude.

So compare the two sources on the fighters who have both: the committed
fighters.csv (ufcstats era) against the working tree (ESPN). Close agreement
means the constants still hold and this can ship. A systematic offset means
recalibrate before shipping.

Run from the repo root, after the backfill has written but BEFORE committing:
    python3 scripts/compare_stat_sources.py
"""

import io
import subprocess
import sys

import pandas as pd

COLS = ["strike_accuracy_pct", "td_accuracy_pct", "td_defense_pct"]


def git_show(path: str) -> pd.DataFrame | None:
    try:
        out = subprocess.run(["git", "show", f"HEAD:{path}"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return pd.read_csv(io.StringIO(out))


def main():
    new = pd.read_csv("data/fighters.csv")
    old = git_show("data/fighters.csv")
    if old is None:
        print("Could not read the committed fighters.csv via git show.")
        sys.exit(1)

    merged = old.merge(new, on="name", suffixes=("_old", "_new"))
    print(f"{len(merged)} fighters present in both versions\n")

    any_overlap = False
    for c in COLS:
        co, cn = f"{c}_old", f"{c}_new"
        if co not in merged or cn not in merged:
            print(f"{c}: column missing on one side")
            continue
        both = merged[[co, cn, "name"]].dropna()
        if both.empty:
            print(f"{c}: no fighter has BOTH an old and a new value "
                  f"-- nothing to compare (expected if the backfill only "
                  f"touched fighters who had no old value).")
            continue
        any_overlap = True
        diff = both[cn] - both[co]
        print(f"{c}  (n={len(both)})")
        print(f"    old mean {both[co].mean():6.2f}   new mean {both[cn].mean():6.2f}")
        print(f"    mean difference   {diff.mean():+6.2f}  <- systematic offset if far from 0")
        print(f"    median |difference| {diff.abs().median():5.2f}")
        print(f"    correlation        {both[co].corr(both[cn]):5.3f}  <- near 1.0 = same ranking")
        worst = both.reindex(diff.abs().sort_values(ascending=False).index).head(3)
        for _, r in worst.iterrows():
            print(f"      biggest gap: {r['name']:<22} {r[co]:.1f} -> {r[cn]:.1f}")
        print()

    if not any_overlap:
        print("No overlap on any column. That means the old ufcstats values were "
              "already gone for everyone the backfill touched, so there is nothing "
              "to calibrate against here -- judge the new values on their own "
              "plausibility instead (a 20-fight veteran landing ~40% of significant "
              "strikes is the sanity anchor).")
        return

    print("READING THIS: a mean difference near zero with correlation near 1.0 means "
          "the two sources agree and the tuned scale constants still apply -- ship it. "
          "A consistent offset in one direction means every differential is inflated "
          "or deflated by roughly that much, so the matching *_ADVANTAGE_SCALE should "
          "be adjusted by the inverse ratio before this goes live.")


if __name__ == "__main__":
    main()
