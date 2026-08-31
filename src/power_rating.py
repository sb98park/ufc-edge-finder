"""
Fixes a real gap in pure Elo: a fighter with zero recorded fights against
anyone already in fight_history.csv sits at the default rating forever,
so every matchup between two "isolated" fighters comes out exactly 50/50
no matter how different they actually are.

This blends Elo (when there's enough connected fight history to trust it)
with a stats-based power rating built from career record, finish rate,
and physical attributes -- so a fighter with a 26-7 record and a 54% KO
rate isn't rated identically to a 3-0 fighter just because neither has
fought anyone in our Elo graph yet.
"""

import pandas as pd

from src.elo import ufc_only

RATING_CENTER = 1500.0

# How much of a NO-HISTORY fighter's record-derived rating to keep. 1.0 is
# the shipped behaviour: trust the pre-UFC record at full strength. Lower
# values pull a debutant toward the neutral centre.
#
# scripts/research_debutant_prior.py measured what that record is worth
# across 263 debuts. The prior is not merely noisy, it inverts at the top:
#
#     model implies   actually wins    n
#     74.9%           63.2%           163
#     82.1%           45.5%            44
#
# A debutant rated at 82% wins less than half the time, because a padded
# regional record and a hard-earned one are identical in W-L.
#
# THE BACKTEST DID NOT JUSTIFY MOVING IT, so this stays at 1.0 and the
# branch below is a no-op. scripts/validate_debutant_shrink.py swept it
# point-in-time over 3,966 debut fights and the effect cancels: against a
# RATED opponent 0.5 gives Brier -0.0007 (p=0.040), but with BOTH corners
# debuting it gives +0.0007, and 0.0 is significantly WORSE (+0.0066,
# p=0.009, accuracy 71.8% -> 70.0%). Overall, p=0.162.
#
# The reason is a scale mismatch, not a bad prior. Against Elo the stats
# curve is on the wrong scale and shrinking narrows it; against another
# stats-curve rating the scale already matches and shrinking only compresses
# a correct gap. A shrink conditional on the OPPONENT captures the whole
# gain and costs nothing -- but build_effective_ratings assigns one rating
# per fighter and never sees a matchup, so there is nowhere to put it.
#
# The real defect that run surfaced is calibration, not rating: a 60-80%
# pick in a debut fight wins about 55% (n=1,757, overstated by 13-16pp).
# That belongs with MIN_RECORD_FOR_HIGH_CONFIDENCE in the reporting layer.
#
# Kept as a named constant rather than deleted so the finding stays
# attached to the line it is about.
DEBUT_RATING_SHRINK = 1.0


def compute_stats_rating(row: pd.Series) -> float:
    """
    A rough power rating on the same numeric scale as Elo (centered at 1500),
    built purely from career stats. Not a substitute for real fight-by-fight
    history -- just a reasonable prior when that history doesn't exist yet.

    Every field read here is explicitly NaN-checked, not just defaulted via
    .get() -- .get(col, default) only falls back when the COLUMN is entirely
    absent from the row, not when it's present but holds NaN, which is the
    actual shape missing data takes in fighters.csv (the column always
    exists in the schema; individual fighters are just missing a value).
    Confirmed live: this was silently producing a NaN power rating for any
    fighter missing reach_in or a win-method breakdown, which then became
    their fallback Elo rating (since a new/obscure fighter typically has no
    connected fight_history.csv entries yet), corrupting every downstream
    prediction and ultimately crashing the parlay builder on a NaN price.
    """
    wins = row["wins"] if pd.notna(row.get("wins")) else 0
    losses = row["losses"] if pd.notna(row.get("losses")) else 0
    reach_in = row["reach_in"] if pd.notna(row.get("reach_in")) else 70

    # NO RECORD IS NOT A BAD RECORD. With wins = losses = 0 the two lines
    # below read as win_pct 0.0 and finish_rate 0.0 -- a fighter who lost
    # every bout and never finished anyone -- because max(x, 1) was there to
    # stop a division by zero, not to say anything about the fighter. The
    # result is a DOUBLE penalty for data we simply do not have.
    #
    # Terrance Chatman, UFC debut 2026-08-22, is 5-1 as a professional and
    # sits in fighters.csv as 0-0 because ESPN has no page for him. That
    # scored him 1439 -- only 66 points above a genuinely 0-5 fighter, and
    # 358 below his 7-0 opponent. The gap alone contributed +38.7pp, made it
    # an 89.7% pick, and crowned it Lock of the Week. His real 5-1 record
    # scores 1613, an 88-point gap and a completely different fight.
    #
    # The model already knew the number was nonsense: method_model logged
    # "elo_gap=0.895 is far outside the training range [0.003, 0.469] --
    # clipping" on exactly this matchup.
    #
    # An unknown fighter belongs at the neutral prior, which is what every
    # other term here degrades to when its input is missing. Reach still
    # applies if we have it -- that is real measured data, not an inference
    # from an empty record.
    if wins + losses == 0:
        return RATING_CENTER + 4.0 * (reach_in - 70)

    total_fights = wins + losses
    win_pct = wins / total_fights
    # THE FINISH-RATE TERM IS GONE, and it is not coming back without new
    # evidence. It was:
    #
    #     finish_rate = (ko_wins + sub_wins) / max(wins, 1)
    #     rating += 150.0 * (finish_rate - 0.4)
    #
    # IT COULD NEVER BE EVIDENCED, because of where this function is used.
    # build_effective_ratings blends Elo against this rating with
    # weight = min(1, n_prior/4), so at four or more connected fights the stats
    # rating is multiplied by ZERO. This is a low-experience prior and nothing
    # else. Across all of fight_history, every one of the 11,469 fighter-
    # appearances where it still counts has three or fewer wins, and 6,932 of
    # them -- 60% -- have NONE, where finish_rate is 0 by construction and the
    # term fired a flat -60. So it was only ever asked the question in the one
    # regime where it cannot be answered.
    #
    # THAT ALSO MADE AN ALMOST-UNKNOWN FIGHTER RATE BELOW A COMPLETELY UNKNOWN
    # ONE. The wins+losses == 0 guard above returns the neutral prior, but 0-1
    # missed it and landed at 1423 against a debutant's 1500. Found via Michael
    # Aljarouj, carried as 0-1 when he is 13-3: it inflated his opponent's
    # published probability by 23.6 points and only the thin-record label cap
    # kept it off Lock of the Week.
    #
    # MEASURED, point-in-time, paired arms on identical fights
    # (scripts/validate_finish_rate_weight.py), 9,193 fights where a corner had
    # under four wins -- the population where this term operates at all:
    #
    #     arm                       acc      brier    d.brier     p
    #     shipped (unweighted)   0.6681    0.21559        --     --
    #     weighted k=8           0.6820    0.21245   -0.00313  0.000
    #     weighted k=15          0.6817    0.21229   -0.00330  0.000
    #     REMOVED                0.6808    0.21210   -0.00348  0.000
    #
    # Removal wins both proper scoring rules, and beats every weighted arm
    # pairwise (k=8 p=0.001, k=15 p=0.015, k=30 p=0.037) -- the penalty
    # shrinking as k grows, i.e. the nearer a weighting came to deleting the
    # term, the better it did. On the well-evidenced cut (both corners 4+ wins,
    # 2,667 fights) every arm is byte-identical, delta 0.00000, because the
    # whole rating is already multiplied by zero there. So there is no cost
    # elsewhere to weigh against the gain.
    #
    # The ko_wins / sub_wins reads that fed it are gone with it. The COLUMNS
    # stay in fighters.csv and are still used elsewhere -- the method model and
    # ufc_method_rates both read them -- but nothing in this function does.

    # experience damps how much we trust a small sample (a 3-0 record
    # shouldn't swing as hard as a 26-7 record even at similar win%)
    experience_weight = min(1.0, total_fights / 15.0)

    rating = RATING_CENTER
    rating += 500.0 * (win_pct - 0.5) * experience_weight
    rating += 4.0 * (reach_in - 70)

    return rating


def build_effective_ratings(
    fighters_df: pd.DataFrame,
    elo_ratings: dict[str, float],
    history_df: pd.DataFrame,
    min_fights_to_trust_elo: int = 4,
) -> dict[str, float]:
    """
    For each fighter: if they have enough *connected* fight history for Elo
    to mean something, blend toward Elo as that count grows. Otherwise, rely
    on the stats-based rating instead of the meaningless flat default.
    """
    # CONNECTED history only -- the docstring's word, and it is load-bearing.
    # Both numbers below are compared against, or blended with, Elo, which is
    # built from the UFC-only subgraph. Counting a regional bout here while
    # excluding it there makes the two disagree about the same fighter: it
    # raises the blend weight toward an Elo that no extra fight informed.
    connected = ufc_only(history_df)
    fight_counts = pd.concat([
        connected["fighter_a"] if "fighter_a" in connected else pd.Series(dtype=str),
        connected["fighter_b"] if "fighter_b" in connected else pd.Series(dtype=str),
    ]).value_counts()

    streaks = _current_streaks(connected)

    effective = {}
    for _, row in fighters_df.iterrows():
        name = row["name"]
        stats_rating = compute_stats_rating(row)
        n_fights_tracked = int(fight_counts.get(name, 0))

        if n_fights_tracked == 0:
            # THE DEBUTANT BRANCH. No connected history at all, so the record
            # is the only evidence -- and the least reliable kind. Shrunk
            # toward the centre by DEBUT_RATING_SHRINK; at 1.0 this is
            # exactly the previous behaviour.
            effective[name] = RATING_CENTER + (stats_rating - RATING_CENTER) * DEBUT_RATING_SHRINK
        else:
            weight = min(1.0, n_fights_tracked / min_fights_to_trust_elo)
            elo_r = elo_ratings.get(name, RATING_CENTER)
            effective[name] = weight * elo_r + (1 - weight) * stats_rating

        effective[name] += _streak_bonus(n_fights_tracked, streaks.get(name, 0))

    return effective


def _current_streaks(history_df: pd.DataFrame) -> dict[str, int]:
    """Consecutive wins as at the most recent fight, per fighter."""
    if history_df is None or history_df.empty:
        return {}
    if "winner" not in history_df.columns:
        return {}
    df = history_df
    if "date" in df.columns:
        df = df.sort_values("date")
    streak: dict[str, int] = {}
    for r in df.itertuples(index=False):
        a = getattr(r, "fighter_a", None)
        b = getattr(r, "fighter_b", None)
        w = getattr(r, "winner", None)
        if not a or not b or not w:
            continue
        loser = b if w == a else a
        streak[w] = streak.get(w, 0) + 1
        streak[loser] = 0
    return streak


# Validated in research_experience_adjustment.py on a frozen 2019+ holdout
# (n=3430): Brier 0.2432 -> 0.2402, accuracy 55.9% -> 56.9%, and the
# sign-flipped control was clearly WORSE (0.2506), so the shape is structure
# rather than noise. Every experience cohort improved.
#
# WHAT THIS CORRECTS. Elo accumulates over fights, so a rating reflects how
# many you have won more than who you beat. Measured against 5,978 closing
# lines, the model sat 6.6% BELOW the market on fighters with 0-3 UFC fights
# and 9.8% ABOVE it on those with 16+ -- a 16-point monotonic swing.
#
# ONLY THE STREAK TERM SURVIVED VALIDATION. A fight-count term was fit
# alongside it and the sweep chose ZERO for it: the gradient against the
# market is real, but on its own it does not convert into better
# predictions. Being newly PROVEN -- few fights and currently winning -- is
# what the rating lags on. That distinction only surfaced because the fix
# was judged on outcomes rather than on closing the gap it was fit to.
#
# The 16+ cohort improved MOST (-0.0099) despite receiving no bonus itself,
# because the correction lands on the risers they face. That cohort was
# picking at 50.2% -- a coin flip -- before this.
#
# ---------------------------------------------------------------------------
# RE-VALIDATED AND HALVED, 20 -> 10 (scripts/validate_streak_bonus.py).
#
# Everything above was honest when written and is now the OLDEST validation in
# the model. Four things have since moved underneath it -- point-in-time
# wrestling and striking rates from data/pit_stats.csv, rebuilt divisional
# method priors, the durability Beta shrink, and reference_date reaching the
# style layer's recency term. All four measure recent form, which is what a
# streak bonus is a PROXY for, so the question is whether the proxy still adds
# anything once the real thing is present.
#
# Point-in-time, two disjoint windows, corners randomised, bootstrap clustered
# by card. Deltas are Brier against the arm named in the column:
#
#     arm    vs k=20 recent    vs k=20 prior    vs k=0 recent   vs k=0 prior
#     k=0    +0.00029 (.77)    -0.00182 (.076)        --             --
#     k=10   -0.00034 (.49)    -0.00126 (.016)   -0.00064 (.23)  +0.00055 (.28)
#     k=30   +0.00111 (.018)   +0.00183 (.0003)       --             --
#
# THREE THINGS ARE ESTABLISHED AND ONE IS NOT.
#
# 1. The term must not grow. k = 30 is significantly worse in BOTH windows,
#    the only result here significant in both.
# 2. k = 10 beats the shipped k = 20 in both windows. It is the only arm never
#    beaten by anything, which is what it is being shipped on.
# 3. The bonus only ever touches its intended cohort: fights where the thinner
#    corner has 9+ prior bouts score BYTE-IDENTICALLY across all four arms, in
#    both windows. That is the check that the gate works, not a result.
#
# NOT ESTABLISHED: that the term earns its place at all. k = 10 against k = 0
# FLIPS SIGN between the windows (-0.00064, then +0.00055), neither
# significant. That is the signature of no effect, and it means an earlier
# recommendation to ablate this to zero is no better supported than keeping
# it. Halving is what the evidence actually carries: it is the only change
# that improved on the shipped value twice, and it also halves the term's
# worst-case contribution from 120 rating points to 60 -- which matters more
# here than anywhere else in the model, because this is added to the EFFECTIVE
# RATING and is therefore the one term not bounded by ADJUSTMENT_TOTAL_CAP.
#
# If a future baseline makes this flip negative in both windows, retire it.
# Right now the honest reading is that it is nearly inert and too large.
STREAK_K = 10
STREAK_MAX = 6
STREAK_APPLIES_UNDER = 8


def _streak_bonus(n_fights: int, streak: int) -> float:
    """
    Rating points for a low-experience fighter on an active win streak.

    Capped at six wins and restricted to fighters under nine tracked
    fights: the cohort evidence only covers that range, and an uncapped
    term would extrapolate into territory nothing was measured in.
    """
    if n_fights > STREAK_APPLIES_UNDER or streak <= 0:
        return 0.0
    return STREAK_K * min(int(streak), STREAK_MAX)
