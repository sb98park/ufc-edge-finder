"""
Loader for data/external_odds.csv, with its difference columns rebuilt.

WHY THIS EXISTS. Three of the fifteen `*_dif` columns in that file INVERT
their sign partway through the dataset:

    age_dif, loss_dif, lose_streak_dif    R - B   before 2020-05-11  (100.0%)
                                          B - R   after              (99.8%)
    the other twelve                      B - R   throughout

AND THE SIGN FLIP IS NOT THE ONLY DAMAGE. Rebuilding every column exposed
four more that disagree with their own raw pairs without any sign pattern --
simply wrong values on a minority of rows: ko_dif 332 of 7,177 (4.6%),
avg_td_dif 132 (2.1%), avg_sub_att_dif 108 (1.7%), sig_str_dif 61 (1.0%),
plus single-row noise in reach/height/win_streak. That is the argument for
recomputing all fifteen rather than patching the three that flip.

Verified directly against the raw R_/B_ pairs, and independently reproduced
twice during the underdog-alpha research that turned it up. The splice sits
between 2020-05-09 and 2020-05-13.

WHAT THAT DOES TO ANYTHING TRAINED ON IT. A model spanning that boundary is
fed a variable that means the opposite thing in each half, so the fitted
coefficient is a blend of two contradictory relationships and lands near
zero however strong the real effect is. Worse for this repo specifically:
the boundary falls INSIDE the train side of a 2023 train/test split, so the
corruption is concentrated where the model learns and absent where it is
scored -- the shape most likely to produce a confident wrong answer.

THE FIX IS NOT TO PATCH THE CSV. That file is a scrape; the next refresh
would silently reintroduce the flip and a hand-edited copy would hide it.
Every difference column is instead RECOMPUTED here from the raw pair it
claims to summarise, so the stored values are never trusted at all and the
repair cannot be forgotten by a future caller.

CONVENTION: B - R, matching the twelve columns that were already consistent
and therefore whatever prior work used them.

A NOTE FOR ANYONE MODELLING THIS FILE. The Red corner beats its own
de-vigged closing price by +2.25pp over n=6,916, and Red is essentially just
the ufcstats bout-listing order carried into the odds data. That is a
scraping convention with no causal content, and it is large enough to
masquerade as a finding: an age effect measured without controlling for it
comes out +4.92pp when the younger fighter is Red and +0.26pp when Blue.
Control for corner, or randomise it, before believing any result from here.
"""

import numpy as np
import pandas as pd

PATH = "data/external_odds.csv"

# Each difference column and the raw pair it is defined from. Anything not
# listed here is left exactly as found -- silently "fixing" a column whose
# definition is unknown would be the same mistake in the other direction.
DIF_SOURCES = {
    "lose_streak_dif": ("R_current_lose_streak", "B_current_lose_streak"),
    "win_streak_dif": ("R_current_win_streak", "B_current_win_streak"),
    "longest_win_streak_dif": ("R_longest_win_streak", "B_longest_win_streak"),
    "win_dif": ("R_wins", "B_wins"),
    "loss_dif": ("R_losses", "B_losses"),
    "total_round_dif": ("R_total_rounds_fought", "B_total_rounds_fought"),
    "total_title_bout_dif": ("R_total_title_bouts", "B_total_title_bouts"),
    "ko_dif": ("R_win_by_KO/TKO", "B_win_by_KO/TKO"),
    "sub_dif": ("R_win_by_Submission", "B_win_by_Submission"),
    "height_dif": ("R_Height_cms", "B_Height_cms"),
    "reach_dif": ("R_Reach_cms", "B_Reach_cms"),
    "age_dif": ("R_age", "B_age"),
    "sig_str_dif": ("R_avg_SIG_STR_landed", "B_avg_SIG_STR_landed"),
    "avg_sub_att_dif": ("R_avg_SUB_ATT", "B_avg_SUB_ATT"),
    "avg_td_dif": ("R_avg_TD_landed", "B_avg_TD_landed"),
}


def american_to_implied(odds) -> float:
    o = float(odds)
    return (-o / (-o + 100.0)) if o < 0 else (100.0 / (o + 100.0))


def load_external_odds(path: str = PATH, verbose: bool = True) -> pd.DataFrame:
    """
    The file with every difference column recomputed as B - R, a parsed date,
    and the de-vigged closing probability attached.

    Adds:
      fair_red   de-vigged closing probability for the Red corner
      overround  the closing pair's implied sum, before de-vigging
      red_won    1/0, NaN when the bout was not a Red/Blue decision
    """
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    repaired = []
    for col, (r_col, b_col) in DIF_SOURCES.items():
        if col not in df.columns or r_col not in df.columns or b_col not in df.columns:
            continue
        rebuilt = df[b_col] - df[r_col]
        both = df[[col, r_col, b_col]].notna().all(axis=1)
        differs = both & ~np.isclose(df[col].astype(float),
                                     rebuilt.astype(float), equal_nan=True)
        if differs.any():
            repaired.append((col, int(differs.sum()), int(both.sum())))
        df[col] = rebuilt

    if verbose and repaired:
        print("[external_odds] difference columns rebuilt from raw pairs "
              "(stored values disagreed):")
        for col, bad, tot in sorted(repaired, key=lambda t: -t[1]):
            print(f"[external_odds]   {col:<24} {bad:>5} of {tot:>5} rows "
                  f"({bad / tot * 100:.1f}%)")

    ok = df["R_odds"].notna() & df["B_odds"].notna()
    p_r = df["R_odds"].where(ok).map(american_to_implied, na_action="ignore")
    p_b = df["B_odds"].where(ok).map(american_to_implied, na_action="ignore")
    # Assigned in one concat rather than three inserts: the frame is already
    # 118 columns and adding them one at a time fragments it enough for pandas
    # to warn on every single load.
    extra = pd.DataFrame({
        "overround": p_r + p_b,
        "fair_red": p_r / (p_r + p_b),
        "red_won": np.where(df["Winner"].eq("Red"), 1.0,
                            np.where(df["Winner"].eq("Blue"), 0.0, np.nan)),
    }, index=df.index)
    return pd.concat([df, extra], axis=1).copy()
