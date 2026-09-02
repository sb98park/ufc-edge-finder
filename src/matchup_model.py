"""
'Styles make fights.' A raw rating gap between two fighters misses a real
dynamic: a strong wrestler with good takedown accuracy against a striker
with weak takedown defense has a stylistic advantage the base rating alone
won't capture, and a fighter who's been finished by strikes repeatedly
brings real durability risk into their next fight, independent of their
overall record.

This layer takes the Elo/stats blended rating from power_rating.py and
nudges it based on:
  1. Takedown accuracy vs. opponent's takedown defense (wrestling advantage)
  2. Striking accuracy differential (volume/precision advantage)
  3. Durability: how often each fighter has been finished before, and by
     which method -- a proxy for whether a given attack is likely to work
     against them specifically, not just in general

None of this replaces real film study or a trained analyst's eye -- it's a
systematic way to weight publicly available stats a bit closer to how
people actually reason about matchups, instead of just comparing records.
"""

import datetime as dt
import math
import os

import pandas as pd

from src.elo import canonical_method
from src.names import canonical_name

# How many Elo-equivalent rating points a fully-realized stylistic
# advantage is worth. Tuned to be meaningful but not dominate the base
# rating gap entirely -- these are secondary signals, not the headline.
WRESTLING_ADVANTAGE_SCALE = 300.0  # legacy fallback path only (see wrestling_adj)
# Rating points per 1.0 takedown-per-15-min differential. Set so the term's
# typical magnitude matches what head_to_head_adjustment.py validated (its
# TD-rate coefficient of 10.0 carried a tuned blend weight of 1.5).
TD_RATE_ADVANTAGE_SCALE = 15.0
STRIKING_ADVANTAGE_SCALE = 150.0
DURABILITY_SCALE = 120.0
# REJECTED BY ITS OWN BACKTEST, kept at the no-op value with the verdict
# recorded at the term in style_matchup_adjustment. 0.0 is byte-identical to
# not having the term. Do not raise this without reading that note first --
# the underlying signal is real and the conclusion is still that it does not
# belong here.
CHIN_SCALE = 0.0

# PSEUDO-COUNTS for the durability rate's Beta shrink toward the division's
# own finish rate. 0.0 is the shipped behaviour: the raw career ratio at full
# strength.
#
# WHY IT WAS PROPOSED. finish_loss_rate is (ko_losses + sub_losses) / losses,
# an UNSHRUNK ratio whose denominator is frequently 1. One stoppage on a
# 12-1 record reads as a 100% finish-loss rate -- a fighter who has never
# survived -- and drives the term to its full +/-120, which is the largest
# single contribution in the style layer and 80% of ADJUSTMENT_TOTAL_CAP on
# its own. The 0-loss case is already guarded; the 1-loss and 2-loss cases
# are not, and they are common.
#
# k pseudo-observations at the divisional base rate:
#     (finish_losses + k * base) / (losses + k)
# so a 1-1 record moves most of the way to the base and a 20-8 record barely
# moves at all -- which is the correct amount of scepticism in each case.
#
# VALIDATED AND ON. scripts/validate_durability_shrink.py swept k point-in-
# time over two DISJOINT windows, with full context so the scored model has
# its recency term:
#
#     window                    k=2 Brier    p        accuracy
#     recent 2,500 (n=1834)      -0.0030   0.011       +0.22%
#     prior  2,500 (n=1663)      -0.0052   0.000       +1.44%
#
# Significant in both, and every arm (k = 2, 5, 10, 20) improved Brier and
# log loss in every subset -- 16 of 16 in the same direction across the first
# window alone. Under the null those signs would scatter.
#
# The response curve is FLAT in k, which is itself the diagnosis: the damage
# is the 0-or-1 ratio at a denominator of 1, and any pseudo-count pulls it
# off the rails. k = 2 is the least intervention that captures the effect and
# was significant in both windows, so it is the one that ships. A 1-loss
# fighter finished once goes from a 100% finish-loss rate to 68%.
#
# Accuracy IMPROVES in both windows, so this is not the calibration-for-
# accuracy trade the first measurement suggested. Those first numbers
# (-0.0021 / -0.0027, accuracy -0.22% / +0.30%) were taken while
# _shrunk_finish_loss_rate still pinned 0-loss fighters at 0.0, which handed
# every undefeated fighter a better chin than anyone who had ever lost.
# Removing that discontinuity roughly doubled the Brier gain and flipped the
# accuracy sign -- the shrink was being measured while fighting an artifact
# it had itself created.
DURABILITY_SHRINK_K = 2.0
VOLUME_DIFFERENTIAL_SCALE = 40.0  # rating points per 1.0 SLpM-SApM differential gap

# Cage time at which a fighter's RATE statistics earn full weight in the
# style layer. Below it, the striking and wrestling terms are scaled down
# toward zero, handing the fight back to the rating gap.
#
# WHY. Rate stats are per-minute and per-fight averages, so a short sample
# does not merely make them uncertain -- it makes them EXTREME. A fighter
# with four bouts and 38 minutes of tracked time can read as 1.42 strikes
# absorbed per minute with zero knockdowns absorbed, numbers no established
# fighter posts, because he has not yet had the bad night that would move
# them. The style layer then reads that as an enormous edge.
#
# The both-corners gating below already stops a real number being compared
# against a DEFAULT. It does nothing about a real number computed from
# almost nothing, which is a different failure and the one that bit on
# UFC 330: the model's three disagreements with the market all went against
# it, and two were built on 38 and 15 minutes of cage time respectively
# (Kaue Fernandes at 70% against a market 40%, Eduardo Chapolin at 64%
# against 46%). Both lost.
#
# 90 minutes is six full three-round fights. Deliberately a smooth ramp
# rather than a cliff: fighter_profile draws nothing below MIN_UFC_BOUTS = 3,
# which is right for a chart that must either show a number or not, but a
# hard threshold in a continuous model just relocates the discontinuity.
#
# Set to 0 to disable the scaling entirely -- used by the backtest harness
# to run the unweighted model as a control arm.
#
# COUNTED IN FIGHTS, NOT MINUTES, and that choice is load-bearing. Minutes
# are the better measure in principle -- four five-round fights carry more
# information than four first-round finishes -- but they cannot be
# reconstructed point-in-time from the cached ESPN data: fight duration comes
# from a per-competition status endpoint, and 73% of those were never cached.
# A first version of this keyed on fight_minutes_total and its backtest was
# meaningless as a result: the reconstruction returned 0 for most fights, the
# weight collapsed to ~0 nearly everywhere, and what actually got measured
# was "switch the style layer off", which this project already established is
# worse (57.7% with adjustments vs 55.9% Elo-only).
#
# Fight COUNT is reconstructable exactly -- it is just the timeline entries
# before a date -- so the validation can test what production runs. It is
# also what fighter_profile already gates on (MIN_UFC_BOUTS = 3), which keeps
# one notion of "enough data" across the codebase rather than two.
# OFF (0.0) -- IMPLEMENTED, MEASURED, NOT JUSTIFIED YET.
#
# scripts/validate_rate_stat_shrinkage.py walks 3,181 point-in-time fights
# and compares this against an unweighted control. Every threshold, every
# subset, the paired bootstrap comes back NOT significant:
#
#     subset                     best threshold   Brier      p
#     all 3,181 fights                  3        -0.0001   0.248
#     corner under 3 prior fights       3        -0.0009   0.263
#     corner under 6 prior fights       3        -0.0003   0.252
#     corner under 10 prior fights      3        -0.0002   0.242
#
# The direction is consistently right -- threshold 3 never hurts, and helps
# most on exactly the thin-sample fights the theory points at -- but 377
# fights cannot separate that from luck, and 0.248 is not close.
#
# Notably 6.0 (the value first proposed, from reasoning rather than
# measurement) is NEUTRAL overall and beaten by 3.0 everywhere. That is the
# same failure mode as the picks this was meant to fix: a confident number
# derived from a plausible story rather than from data.
#
# Left in place rather than deleted because the mechanism is written and
# tested, and the sample only grows. Re-run the harness after another
# season; if the thin-fight subset reaches significance at threshold 3, set
# this to 3.0 and nothing else needs to change. Setting it to any positive
# value enables the scaling.
STYLE_FULL_TRUST_FIGHTS = 0.0


def rate_stat_confidence(row_a: pd.Series, row_b: pd.Series) -> float:
    """
    0.0-1.0 multiplier for style terms built from rate statistics.

    Governed by the THINNER corner. A differential is only as trustworthy as
    its weaker side: pairing 24 tracked fights against 1 does not average out
    to a reliable comparison, it produces a confident-looking number about a
    fighter nobody has measured.
    """
    if STYLE_FULL_TRUST_FIGHTS <= 0:
        return 1.0

    def _fights(row) -> float:
        v = row.get("espn_fights")
        try:
            return float(v) if pd.notna(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    thinner = min(_fights(row_a), _fights(row_b))
    return max(0.0, min(1.0, thinner / STYLE_FULL_TRUST_FIGHTS))
SUBMISSION_THREAT_SCALE = 60.0  # rating points per 1.0 sub-win-rate differential

# Southpaw-vs-orthodox is a real, documented edge in striking sports --
# most fighters train far more often against orthodox opponents, so a
# southpaw sees a less familiar look more often than the reverse. It's a
# real but modest effect in the research, not a dominant one, so this is
# calibrated well below the primary factors above (roughly comparable to
# a single year past a fighter's age-decline threshold, not a heavy
# thumb on the scale). Switch-stance fighters can choose their angle, so
# they get the same small edge against either pure stance.
STANCE_MISMATCH_BONUS = 18.0

# Base width (in probability points) for the uncertainty band before
# dividing by sqrt(thinner_record + 1) -- 0.30 means a 0-fight matchup
# gets roughly a +/-30pp band, narrowing to roughly +/-6pp by the time
# the thinner record reaches 20+ fights.
UNCERTAINTY_BASE = 0.30

# A recent win/loss (from fight_history.csv specifically, not the
# aggregate career record) gets a small rating nudge that decays to zero
# over RECENT_FORM_DECAY_YEARS -- deliberately modest since this is a
# single data point, not an aggregate trend.
RECENT_FORM_SCALE = 10.0  # per decayed win/loss unit, summed over last 3 fights (backtest-validated, July 2026)
RECENT_FORM_DECAY_YEARS = 2.0
RECENT_FORM_LOOKBACK = 3

# Height advantage in the adjustment layer. Deliberately small: reach
# (which correlates strongly with height) is already worth 4 pts/inch in
# the stats-based power rating, so a large height weight here would
# double-count the same physical edge. NOT backtest-validatable with
# current data (fight_history.csv has no physical stats), so this is a
# documented heuristic, kept conservative for exactly that reason.
HEIGHT_ADVANTAGE_SCALE = 2.0

# Short-notice fights: fighters accepting a bout on short notice
# (typically < ~30 days, no full camp) win measurably less often --
# roughly a 10 percentage-point drop is the commonly reported figure in
# published analyses of UFC short-notice outcomes. ~10pp at even odds
# corresponds to ~70 Elo points (dP/dgap near 50% is ~0.144pp per
# point). NOT backtest-validatable (fight_history.csv has no notice
# data); literature-derived rather than fitted, flagged per-fighter via
# the short_notice column in fighters.csv at card-research time and
# cleared automatically once that fighter's bout result is logged.
# INERT, AND PERMANENTLY UNVALIDATABLE AT THIS GRANULARITY. Fires on 0% live
# and 0% in backtest. No public source publishes bout-agreement dates, so the
# "days of notice" this scales cannot be reconstructed for any historical
# fight -- not by better scraping, not by any API identified in a dedicated
# research pass. At 70 points it would be the second-largest term in the
# layer if it ever fired.
#
# The reachable weaker question is the binary "was this fighter a
# replacement", from Wikipedia event prose, and even that needs a
# hand-labelled recall estimate before a harness should read it. Recorded
# here so the number is not mistaken for a measured one.
SHORT_NOTICE_PENALTY = 70.0

# Safety rail on the ADJUSTMENT LAYER (style factors + recent form), NOT
# on the Elo/base rating gap. Evidence basis: the walk-forward backtest
# (3,713 point-in-time fights) showed the Elo core is well-calibrated
# but rarely confident -- it said 70%+ only 16 times and never said
# 80%+. Extreme final numbers therefore come almost entirely from the
# adjustment stack, which is exactly the part that CANNOT be backtested
# yet (needs historical stat snapshots). Until those magnitudes are
# fitted to evidence rather than hand-tuned, the stack's total influence
# is clipped: +/-150 rating points can move a coin flip to at most
# ~70/30, so no pile of unvalidated modifiers can manufacture extreme
# confidence on its own. Revisit after the extended walk-forward fits
# the layoff/age magnitudes (the two factors this bites most).
ADJUSTMENT_TOTAL_CAP = 150.0

# Ring rust: no penalty for a normal 6-12 month camp cycle. Beyond a year
# away, each additional year away costs more. WEIGHT REVISED by the July
# 2026 walk-forward backtest: the previous 60 pts/yr scored WORSE than no
# layoff penalty at all on both the pre-2019 train split and the 2019+
# held-out split -- the penalty as sized was actively hurting predictions.
# The data tolerates 0-20 pts/yr (differences within noise across that
# range); 20 keeps the domain-motivated factor alive at the highest
# level the evidence supports rather than the loudest level intuition
# suggested.
LAYOFF_GRACE_YEARS = 1.0
LAYOFF_PENALTY_PER_YEAR = 20.0
LAYOFF_PENALTY_CAP = 300.0


def _get(row: pd.Series, col: str, default: float) -> float:
    return float(row[col]) if col in row and pd.notna(row[col]) else default


# Below this share of a fighter's own claimed bouts, last_fight_date stops
# being a fact about them and becomes a fact about us. Measured, not picked:
# see scripts/validate_layoff_coverage_guard.py. 0.60 is the LARGEST threshold
# that is still a bit-for-bit no-op on well-covered fights.
HISTORY_COVERAGE_FLOOR = 0.60


def history_is_partial(row) -> bool:
    """
    Do we hold materially fewer bouts than this fighter says they have had?

    A row with no `history_coverage` column reads as covered, so every caller
    that does not supply it -- every test, every harness, rationale.py -- gets
    exactly the behaviour it had before this existed.
    """
    cov = row.get("history_coverage") if hasattr(row, "get") else None
    if cov is None or (isinstance(cov, float) and pd.isna(cov)):
        return False
    try:
        return float(cov) < HISTORY_COVERAGE_FLOOR
    except (TypeError, ValueError):
        return False


def reconcile_last_fight_from_history(fighters_df, fight_history_df):
    """Move last_fight_date forward when the spine holds a more recent bout.

    These two files disagree and nothing reconciled them. last_fight_date is
    written by fighter_backfill (from ESPN, and only when the cell is empty)
    and by results_fetcher (when a card we watched grades). Neither reads
    fight_history.csv -- so a bout can be sitting in the spine while
    fighters.csv still reports a fight from years earlier, and every
    history-derived term goes on believing the older date.

    Michael Aljarouj is the case that found this. ESPN's athlete eventlog has
    two entries for him: a 2021 Brave CF loss and his upcoming UFC debut. His
    real last fight was 2025-04-12. Correcting the spine alone would not have
    helped, because nothing carried the correction across.

    ONE-DIRECTIONAL, ON PURPOSE. This only ever moves the date FORWARD. Both
    consumers of it (layoff_penalty, quick_return_penalty) only ever subtract
    rating points, so a date that is too old manufactures ring rust that the
    fighter has not earned, while a date that is too recent can at worst fail
    to charge for a layoff that is real. Given a disagreement we cannot
    adjudicate, the direction that under-punishes is the honest one.
    """
    if "last_fight_date" not in fighters_df.columns:
        return fighters_df
    hist = fight_history_df.copy()
    hist["_d"] = pd.to_datetime(hist["date"], errors="coerce")
    newest = {}
    _methods = (hist["method"] if "method" in hist.columns
                else pd.Series([""] * len(hist), index=hist.index))
    for col, other in (("fighter_a", "fighter_b"), ("fighter_b", "fighter_a")):
        for name, opp, when, winner, method in zip(
                hist[col], hist[other], hist["_d"],
                hist.get("winner", pd.Series([""] * len(hist))), _methods):
            if pd.isna(when):
                continue
            key = str(name).strip().lower()
            if key not in newest or when > newest[key][0]:
                # An NC still happened; the cage time is what layoff reads.
                # NOT `winner or ""` -- a blank winner round-trips through the
                # CSV as NaN, which is TRUTHY, so that idiom yields the string
                # "nan" and grades every no contest as a loss (CLAUDE.md s4).
                won = "" if pd.isna(winner) else str(winner).strip()
                res = "NC" if not won else ("W" if won == str(name).strip() else "L")
                newest[key] = (when, str(opp), res, method)

    out = fighters_df.copy()
    moved = 0
    for i, row in out.iterrows():
        hit = newest.get(str(row.get("name", "")).strip().lower())
        if hit is None:
            continue
        when, opp, res, method = hit
        held = pd.to_datetime(row.get("last_fight_date"), errors="coerce")
        if pd.notna(held) and held >= when:
            continue
        out.at[i, "last_fight_date"] = when.date().isoformat()
        out.at[i, "last_fight_opponent"] = opp
        out.at[i, "last_fight_result"] = res
        # THE METHOD MUST TRAVEL WITH THE REST OF THE ROW. Moving the date,
        # the opponent and the result while leaving the method behind
        # published Axel Sola -- co-main of 2026-09-05 -- as "W by Decision
        # (Unanimous) against Ismael Bonfim" when that bout was a submission;
        # the method was lifted from a different fight four months earlier.
        #
        # Normalised on the way in, because quick_return_penalty matches
        # ("KO/TKO", "SUB") EXACTLY and the spine holds 19 rows spelled
        # "Submission". Writing that verbatim would have silently switched
        # off a penalty that currently fires correctly.
        if pd.notna(method) and str(method).strip():
            out.at[i, "last_fight_method"] = canonical_method(str(method))
        moved += 1
    if moved:
        print(f"[reconcile] {moved} fighter(s) had a more recent bout in the spine "
              f"than in fighters.csv; last_fight_date moved forward")
    return out


def attach_history_coverage(fighters_df, fight_history_df):
    """
    Add a `history_coverage` column: bouts we hold / bouts they claim.

    The two numbers come from independent places, which is the only reason the
    comparison means anything. `wins + losses` in fighters.csv is sourced from
    ESPN's career record (or a hand correction); the count is of rows in
    fight_history.csv. When they agree we have the whole career. When they do
    not, every history-derived term -- layoff, recent form, streaks, Elo -- is
    reading a sample we can see is partial, and we can say so instead of
    quietly treating absence as evidence.

    NaN, not 1.0, for a fighter with no usable record: unmeasurable is not the
    same claim as complete, and history_is_partial reads NaN as "do not
    intervene".

    ONLY DECIDED BOUTS COUNT, on both sides of the ratio. The denominator is
    `wins + losses`, which excludes no contests and draws, so counting them in
    the numerator compares different things and can exceed 1.0 -- Michael
    Aljarouj's two no contests put him at 18/16 = 1.12, which reads as holding
    more of his career than exists.

    Returns the frame; safe to call on a frame that already has the column.
    """
    if fighters_df is None or getattr(fighters_df, "empty", True):
        return fighters_df
    counts = {}
    if fight_history_df is not None and not getattr(fight_history_df, "empty", True):
        hist = fight_history_df
        if "winner" in hist.columns:
            hist = hist[hist["winner"].notna()
                        & hist["winner"].astype(str).str.strip().ne("")]
        for col in ("fighter_a", "fighter_b"):
            if col not in hist.columns:
                continue
            for n in hist[col].dropna().astype(str):
                key = n.strip().lower()
                counts[key] = counts.get(key, 0) + 1

    def _cov(row):
        w, l = row.get("wins"), row.get("losses")
        if pd.isna(w) or pd.isna(l):
            return float("nan")
        claimed = int(w) + int(l)
        if claimed <= 0:
            return float("nan")     # a 0-0 row says nothing about coverage
        return counts.get(str(row.get("name")).strip().lower(), 0) / claimed

    fighters_df = fighters_df.copy()
    fighters_df["history_coverage"] = fighters_df.apply(_cov, axis=1)
    return fighters_df


def layoff_years(row: pd.Series, reference_date: dt.date | None = None) -> float | None:
    if "last_fight_date" not in row or pd.isna(row["last_fight_date"]):
        return None
    # AN INCOMPLETE HISTORY CANNOT PROVE A LAYOFF. last_fight_date is the
    # newest bout WE HOLD, so on a partial history it is a lower bound on the
    # fighter's real activity -- the true last fight can only be more recent,
    # never older. Both consumers of this number only ever subtract points, so
    # a gap in our data can manufacture ring rust but can never remove any.
    # That asymmetry is the whole argument; the threshold is the only part
    # that needed measuring.
    #
    # Michael Aljarouj, priced 2026-09-05: fighters.csv has him 13-3, the
    # spine has one of those sixteen bouts, from 2021. The model read a
    # 5.47-year layoff and charged -89.3 points. His real last fight was
    # 2025-04-12 -- 1.40 years, worth -8.0.
    #
    # Measured over 675 point-in-time fights with a corner below the floor:
    # Brier 0.23790 -> 0.23557 (-0.00233, p=0.004), accuracy 0.5985 -> 0.6044.
    # On the 2,013 fully-covered fights it is identical to five decimal places
    # (delta 0.00000, p=1.000), which is the check that matters: it does
    # nothing where it has no business doing anything.
    if history_is_partial(row):
        return None
    reference_date = reference_date or dt.date.today()
    last_fight = pd.to_datetime(row["last_fight_date"]).date()
    return (reference_date - last_fight).days / 365.25


def layoff_penalty(row: pd.Series, reference_date: dt.date | None = None) -> float:
    years_away = layoff_years(row, reference_date)
    if years_away is None or years_away <= LAYOFF_GRACE_YEARS:
        return 0.0
    penalty = LAYOFF_PENALTY_PER_YEAR * (years_away - LAYOFF_GRACE_YEARS)
    return -min(penalty, LAYOFF_PENALTY_CAP)


# Coming back too SOON after being finished carries real, documented risk --
# the opposite problem from ring rust. Six months is a rough dividing line.
QUICK_RETURN_THRESHOLD_YEARS = 0.5
QUICK_RETURN_PENALTY_CAP = 150.0


def quick_return_penalty(row: pd.Series, reference_date: dt.date | None = None) -> float:
    if row.get("last_fight_result") != "L" or row.get("last_fight_method") not in ("KO/TKO", "SUB"):
        return 0.0  # only a finish loss carries this specific risk, not a decision loss
    years_away = layoff_years(row, reference_date)
    if years_away is None or years_away >= QUICK_RETURN_THRESHOLD_YEARS:
        return 0.0
    severity = (QUICK_RETURN_THRESHOLD_YEARS - years_away) / QUICK_RETURN_THRESHOLD_YEARS
    return -severity * QUICK_RETURN_PENALTY_CAP


# The age cliff hits divisions very differently. Speed/output-dependent
# lighter divisions (where reflexes and recovery matter most) see a steep
# decline past 35; heavyweight and light heavyweight fighters, where power
# and experience matter more than raw speed, often peak or sustain well
# into their late 30s.
# THE UFC HAS NO MEN'S STRAWWEIGHT. The 115lb division is women-only, so the
# "Strawweight" key below was covering a division that does not exist while
# the one that does -- "Women's Strawweight" -- fell through to the 37
# default. Both labels appear in fighters.csv (4 and 8 fighters), which is
# the same real division recorded two ways.
#
# The women's divisions were missing entirely, so Women's Flyweight and
# Women's Bantamweight also defaulted to 37 while the men's divisions at the
# SAME weight got 35 -- inverting this table's own rationale that lighter
# fighters decline earlier. On the booked card that is 11 of 76 fights.
#
# Thresholds mirror the men's division at the same weight, since the
# rationale is about weight rather than sex. Catchweight is deliberately
# absent: it is not a division, it is a one-off contracted weight, and the
# default is the honest answer for it.
AGE_CLIFF_START = {
    "Flyweight": 35, "Bantamweight": 35, "Featherweight": 35,
    "Lightweight": 37, "Welterweight": 37, "Middleweight": 37,
    "Light Heavyweight": 39, "Heavyweight": 40,
    "Women's Strawweight": 35, "Women's Flyweight": 35, "Women's Bantamweight": 35,
    "Women's Featherweight": 35,
}
AGE_CLIFF_DEFAULT_START = 37  # for any weight class not explicitly listed

# Labels that mean the same real division. Applied before every divisional
# lookup so one fighter cannot sit under two age cliffs and two method priors
# depending on which spelling their row happens to carry.
DIVISION_ALIASES = {
    "Strawweight": "Women's Strawweight",
    "Women's Straw Weight": "Women's Strawweight",
}


def normalize_division(weight_class) -> str | None:
    """Canonical division label, or None when it is genuinely unknown."""
    if weight_class is None or (isinstance(weight_class, float) and pd.isna(weight_class)):
        return None
    s = str(weight_class).strip()
    if not s or s.lower() in ("unknown", "nan", "n/a"):
        return None
    return DIVISION_ALIASES.get(s, s)
AGE_CLIFF_PENALTY_PER_YEAR = 25.0
AGE_CLIFF_PENALTY_CAP = 200.0


def age_cliff_penalty(row: pd.Series) -> float:
    age = row.get("age")
    if pd.isna(age):
        return 0.0  # no penalty when age isn't known -- better than guessing wrong
    # THE DIVISION ONLY SETS THE THRESHOLD; the age is real data either way.
    # `not weight_class` used to gate the whole term off for an empty string
    # while letting NaN through to the default -- two spellings of "unknown"
    # taking opposite paths. Unknown now consistently means "use the default
    # cliff", which keeps a real age signal on the 114 of 310 roster fighters
    # who carry no division rather than discarding it.
    weight_class = normalize_division(row.get("weight_class"))
    cliff_age = AGE_CLIFF_START.get(weight_class, AGE_CLIFF_DEFAULT_START)
    years_past_cliff = float(age) - cliff_age
    if years_past_cliff <= 0:
        return 0.0
    return -min(AGE_CLIFF_PENALTY_PER_YEAR * years_past_cliff, AGE_CLIFF_PENALTY_CAP)


# Missing weight is a documented red flag -- often reflecting a rushed or
# broken training camp, not just a one-off scale mistake -- and it also
# means the opponent gets an automatic strength/size advantage on fight
# night after rehydration. Data note: this field defaults to 0 (no known
# instances) for the current roster; populating real history requires
# per-fighter weigh-in research this build doesn't have time to do exhaustively.
# INERT, AND NOT VALIDATABLE. audit_term_coverage.py reports this term firing
# on 0% of live fights AND 0% of backtested ones -- the column it reads
# defaults to 0 for the entire roster and no historical weigh-in dataset
# exists to populate it. A hand-set constant that has never once changed a
# prediction is not a modelling choice, it is decoration.
#
# Kept rather than deleted because the mechanism is correct and one partial
# source is reachable: WEIGHTCLASS == "Catch Weight Bout" in
# ufc_fight_results.csv gives 82 real instances. That is a biased subset --
# it misses the common case where a fighter misses weight, forfeits purse and
# the bout proceeds under its original label -- but 82 measured instances
# against a term that has fired zero times would be the first evidence either
# way. Until then this number is unsupported and labelled as such.
MISSED_WEIGHT_PENALTY_PER_INSTANCE = 20.0
MISSED_WEIGHT_PENALTY_CAP = 80.0


def missed_weight_penalty(row: pd.Series) -> float:
    count = row.get("missed_weight_count")
    if pd.isna(count) or count <= 0:
        return 0.0
    return -min(MISSED_WEIGHT_PENALTY_PER_INSTANCE * float(count), MISSED_WEIGHT_PENALTY_CAP)


# Canonical lightest-to-heaviest ordering, used to measure how many
# divisions a fighter is jumping -- a one-division move (e.g. Welterweight
# to Middleweight) and a two-division move (e.g. Lightweight to
# Middleweight) are not the same risk, and shouldn't score identically.
# ORDERED BY WEIGHT, and the women's divisions are their own ladder rather
# than points on the men's. "Strawweight" was the only entry covering a
# women-only division and it was the label the UFC does not use; every real
# women's move -- Strawweight to Flyweight is the common one -- fell through
# the `not in DIVISION_ORDER` guard and returned 0.0 by construction.
#
# Two ladders, because a division CHANGE is a change within a fighter's own
# sex division. Comparing a women's flyweight against the men's ladder would
# either crash on a missing key or, worse, produce a number.
DIVISION_ORDER = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
]
WOMENS_DIVISION_ORDER = [
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight",
]


def _division_ladder(division: str):
    """The ordered list a division belongs to, or None if it is off-ladder."""
    if division in WOMENS_DIVISION_ORDER:
        return WOMENS_DIVISION_ORDER
    if division in DIVISION_ORDER:
        return DIVISION_ORDER
    return None

# Deliberately modest magnitudes -- this factor is NOT backtested against
# historical outcomes the way e.g. the durability calibration was (that
# would require weight-class data per historical fight, which
# fight_history.csv doesn't have). Moving up is scaled by how many
# divisions are jumped; moving down gets a flat, smaller bonus, since a
# clean cut's advantage is real but shouldn't be overstated relative to
# an unvalidated theory. Deliberately asymmetric: moving up is riskier
# than moving down is advantageous, per the reasoning discussed with the
# user (power/leverage disadvantage facing larger opponents, versus a
# capped upside from cutting into a smaller field).
WEIGHT_CLASS_UP_PENALTY_PER_DIVISION = 10.0
WEIGHT_CLASS_UP_PENALTY_CAP = 30.0
WEIGHT_CLASS_DOWN_BONUS = 8.0


def recent_weight_class(name: str, history_df: pd.DataFrame | None) -> str | None:
    """
    A fighter's settled recent division: simply their single most recent
    fight's weight class. This deliberately does NOT average over a
    window of several fights -- by the time we're predicting a fighter's
    CURRENT upcoming fight, any past one-off aberration (a short-notice
    or catchweight booking at an unusual weight) has already been
    superseded by whatever they actually fought at next, so the most
    recent entry already reflects reality. Concretely: Usman's one-off
    2023 Middleweight fight is correctly ignored because his following
    fight (2025, Welterweight) is now the most recent entry -- and
    Makhachev's genuine, deliberate Welterweight title win is correctly
    recognized immediately, rather than needing a second Welterweight
    fight to outvote two years of Lightweight history first. Returns
    None with no history at all, which the caller treats as "unknown,
    don't penalize" rather than guessing.
    """
    if history_df is None or history_df.empty:
        return None
    rows = history_df[history_df["name"] == name].sort_values("date", ascending=False)
    if rows.empty:
        return None
    return rows["weight_class"].iloc[0]


def weight_class_change_penalty(name: str, this_fight_weight_class: str | None, history_df: pd.DataFrame | None) -> float:
    """
    Penalizes a fighter moving up in weight (scaled by how many divisions),
    rewards moving down (flat, smaller bonus). Returns 0.0 -- no penalty,
    no bonus -- whenever there isn't enough information to say anything:
    no history, no current division, an unrecognized division name, or no
    settled division different from this fight's. Silence over a guess is
    deliberate here, same as elsewhere in this module.
    """
    # NORMALISED FIRST. Both sides were compared as raw strings, so
    # "Strawweight" and "Women's Strawweight" -- one real division, two
    # spellings in fighters.csv -- read as a division CHANGE, and every
    # women's division was off the ladder entirely.
    settled = normalize_division(recent_weight_class(name, history_df))
    this_fight_weight_class = normalize_division(this_fight_weight_class)
    if not settled or not this_fight_weight_class or settled == this_fight_weight_class:
        return 0.0

    ladder = _division_ladder(settled)
    if ladder is None or _division_ladder(this_fight_weight_class) is not ladder:
        return 0.0      # off-ladder, or a cross-ladder comparison that means nothing

    division_distance = ladder.index(this_fight_weight_class) - ladder.index(settled)
    if division_distance > 0:  # moving up (this fight's division is heavier)
        return -min(WEIGHT_CLASS_UP_PENALTY_PER_DIVISION * division_distance, WEIGHT_CLASS_UP_PENALTY_CAP)
    else:  # moving down
        return WEIGHT_CLASS_DOWN_BONUS


def compute_divisional_method_priors(fighters_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Divisional average method-of-victory rates. A heavyweight fight has an
    inherently higher baseline finish-by-KO rate than a strawweight fight,
    which leans heavily toward decisions -- a flat blend for every division
    ignores this real, well-documented difference between weight classes.

    COMPUTED FROM REAL UFC BOUTS, not from the roster's career totals. The
    career-total version was wrong in one direction in every single division,
    because a career record includes the regional circuit -- where finish
    rates are far higher -- and because the CURRENT roster is a survivorship
    sample of fighters good enough to still be here:

        division              source     KO   SUB   DEC      n
        Lightweight           true     0.30  0.22  0.48   1394
                              roster   0.44  0.32  0.25
        Women's Strawweight   true     0.12  0.20  0.67    348
                              roster   0.39  0.17  0.44
        Light Heavyweight     true     0.45  0.17  0.37    685
                              roster   0.57  0.21  0.22

    Decisions were understated by 13-23 points everywhere. Since these priors
    anchor every method and round-total projection on the site, the whole
    props surface was biased toward finishes.

    data/ufc_fight_results.csv carries 8,784 real UFC bouts with both a
    division and a method, so this needs no new data -- only reading the
    right table. Falls back to the roster aggregate if that file is missing,
    which keeps the function total rather than silently returning nothing.
    """
    priors = _divisional_priors_from_results()
    if priors:
        return priors

    # NORMALISE FIRST, THEN SUM -- and carry counts, not rates, until the end.
    # This loop used to group on the RAW weight_class and normalise afterwards,
    # writing straight into priors[div]. "Strawweight" and "Women's
    # Strawweight" are one real division spelled two ways in fighters.csv, so
    # whichever group pandas yielded second simply OVERWROTE the first and its
    # fighters vanished from the prior entirely -- 4 fighters and 50 wins
    # dropped, moving the division's SUB rate by 9.7 points. That is the same
    # aliasing bug eb14790 fixed on the lookup side, surviving here because
    # that commit never touched this branch. Accumulating counts under the
    # canonical label and dividing once merges the spellings instead.
    counts: dict[str, dict[str, float]] = {}
    for wc, group in fighters_df.groupby("weight_class"):
        div = normalize_division(wc)
        if div is None:
            continue
        bucket = counts.setdefault(div, {"wins": 0.0, "KO/TKO": 0.0, "SUB": 0.0, "DEC": 0.0})
        bucket["wins"] += float(group["wins"].sum())
        bucket["KO/TKO"] += float(group["ko_wins"].sum())
        bucket["SUB"] += float(group["sub_wins"].sum())
        bucket["DEC"] += float(group["dec_wins"].sum())

    priors = {}
    for div, bucket in counts.items():
        # `not (x > 0)` rather than `x <= 0` so a NaN win total is skipped
        # instead of sailing through the comparison and producing NaN rates.
        if not (bucket["wins"] > 0):
            continue
        priors[div] = {k: bucket[k] / bucket["wins"] for k in ("KO/TKO", "SUB", "DEC")}

    # "_default" IS PART OF THE CONTRACT, not a nicety. divisional_prior_for
    # resolves an unknown or unlisted division through this key and returns the
    # caller's own fallback -- the FIGHTER'S OWN RATE -- when it is absent,
    # which its docstring names as the exact defect it was written to fix. The
    # primary path guarantees it (_divisional_priors_from_results setdefaults
    # it before returning); this path did not, so falling back to the roster
    # silently also fell back to no prior at all. Computed over the whole
    # roster, INCLUDING fighters with no division, since those are precisely
    # the rows a default has to speak for.
    total_wins = float(fighters_df["wins"].sum())
    if total_wins > 0:
        priors.setdefault("_default", {
            "KO/TKO": float(fighters_df["ko_wins"].sum()) / total_wins,
            "SUB": float(fighters_df["sub_wins"].sum()) / total_wins,
            "DEC": float(fighters_df["dec_wins"].sum()) / total_wins,
        })
    return priors


_DIV_PRIOR_CACHE: dict | None = None


def divisional_finish_rate(weight_class) -> float:
    """
    Share of bouts in this division that end in a finish, from real UFC
    results. The base the durability rate shrinks toward.

    Cached: this is read per fighter per fight and the underlying table does
    not change within a build.
    """
    global _DIV_PRIOR_CACHE
    if _DIV_PRIOR_CACHE is None:
        _DIV_PRIOR_CACHE = _divisional_priors_from_results() or {}
    div = normalize_division(weight_class)
    row = _DIV_PRIOR_CACHE.get(div) if div else None
    if row is None:
        row = _DIV_PRIOR_CACHE.get("_default")
    if not row:
        return 0.5
    return float(row.get("KO/TKO", 0.0)) + float(row.get("SUB", 0.0))


def _shrunk_finish_loss_rate(row: pd.Series, base: float) -> float:
    """
    (ko_losses + sub_losses) / losses, shrunk toward `base` by
    DURABILITY_SHRINK_K pseudo-observations. At K = 0 this is the raw ratio,
    byte-identical to the pre-shrink behaviour.

    THE 0-LOSS CASE IS NOT A 0% FINISH-LOSS RATE. Returning 0.0 there was
    harmless while the estimator was unshrunk -- an undefeated fighter and a
    12-1 fighter never finished both scored 0.0, tied at the bottom. Adding
    the shrink broke that tie in the wrong direction: with k=2 and a 0.52
    base, 12-1-never-finished moves to 0.35 while 12-0 stays pinned at 0.0.
    Undefeated then scores a BETTER chin than anyone who has ever lost,
    including someone who has only ever lost a decision -- and 0.0 is a value
    the shrunk estimator can no longer produce for any real record, so the
    two corners are not even on the same scale.

    That is backwards on its own terms. No losses means no evidence about a
    fighter's chin, and the neutral answer for no evidence is the base rate,
    which is exactly what the formula already returns at losses = 0:
    (0 + k*base) / (0 + k) = base. The special case was the whole bug.

    Kept only for k = 0, where dividing by a zero denominator would raise and
    the legacy meaning of 0.0 still applies.
    """
    losses = _get(row, "losses", 0)
    finished = _get(row, "ko_losses", 0) + _get(row, "sub_losses", 0)
    k = DURABILITY_SHRINK_K
    if k <= 0:
        return finished / losses if losses else 0.0
    return (finished + k * base) / (losses + k)


def divisional_prior_for(priors: dict, weight_class, method: str, fallback: float) -> float:
    """
    A division's rate for one method, resolving aliases and thin divisions.

    Callers used to index priors with the raw weight_class string, so
    "Strawweight" and "Women's Strawweight" -- the same real division, spelled
    two ways in fighters.csv -- looked up two different priors, and anything
    unlisted silently fell back to the FIGHTER'S OWN rate, which is not a
    prior at all. Both now resolve through the canonical label and then the
    all-UFC split.
    """
    div = normalize_division(weight_class)
    row = priors.get(div) if div else None
    if row is None:
        row = priors.get("_default")
    if row is None:
        return fallback
    return row.get(method, fallback)


# ANCHORED TO THE MODULE, NOT TO THE CWD. As a bare relative path this
# resolved against whatever directory the process happened to start in, and
# os.path.exists returning False is not an error here -- it silently selects
# the roster-aggregate fallback below, which is a materially different (and
# by this function's own docstring, wrong-in-every-division) set of priors.
# A caller running from anywhere but the repo root got that substitution with
# no message. The file sits at <repo>/data/ and this module at <repo>/src/.
# Still a module-level name so scripts and tests can point it elsewhere.
UFC_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "data", "ufc_fight_results.csv")
# Below this many real bouts a division's own rate is noisier than the
# all-UFC average, so the average is used instead. Women's Featherweight has
# 29 bouts in total; a 29-fight rate is not a prior, it is an anecdote.
MIN_BOUTS_FOR_DIVISIONAL_PRIOR = 120


def _division_from_bout_label(label) -> str | None:
    """
    'Lightweight Bout' / 'UFC Women's Strawweight Title Bout' -> the division.

    Title fights carry a 'UFC ' prefix and sometimes 'Interim', and every row
    ends in ' Bout'. Stripped rather than filtered: a title fight is a normal
    fight for the purpose of how often the division goes to a decision, and
    dropping them would bias the sample toward the undercard.
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    s = str(label).strip()
    for suffix in (" Bout",):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = s.replace("Championship", "").strip()
    if s.startswith("UFC "):
        s = s[4:].strip()
    for word in ("Interim ", "Title", "Superfight"):
        s = s.replace(word, "").strip()
    # Open Weight and Catch Weight are not divisions; they get the default.
    if not s or s in ("Open Weight", "Catch Weight", "Catchweight"):
        return None
    return normalize_division(s)


def _method_bucket_from_results(method) -> str | None:
    m = str(method).strip().lower()
    if m.startswith("decision"):
        return "DEC"
    if "submission" in m:
        return "SUB"
    if "ko/tko" in m or m.startswith("tko") or m.startswith("ko"):
        return "KO/TKO"
    return None      # DQ, overturned, could not continue -- no method to attribute


def _divisional_priors_from_results() -> dict[str, dict[str, float]]:
    """Method split per division over every decided UFC bout on record."""
    if not os.path.exists(UFC_RESULTS_PATH):
        return {}
    try:
        d = pd.read_csv(UFC_RESULTS_PATH)
    except (OSError, pd.errors.ParserError):
        return {}
    if "WEIGHTCLASS" not in d.columns or "METHOD" not in d.columns:
        return {}

    d = d.assign(_div=d["WEIGHTCLASS"].map(_division_from_bout_label),
                 _m=d["METHOD"].map(_method_bucket_from_results)).dropna(subset=["_m"])
    if d.empty:
        return {}

    overall = {k: float((d["_m"] == k).mean()) for k in ("KO/TKO", "SUB", "DEC")}
    priors = {}
    for div, g in d.dropna(subset=["_div"]).groupby("_div"):
        if len(g) < MIN_BOUTS_FOR_DIVISIONAL_PRIOR:
            continue
        priors[div] = {k: float((g["_m"] == k).mean()) for k in ("KO/TKO", "SUB", "DEC")}
    # Thin and unlisted divisions resolve to the all-UFC split rather than to
    # whatever their handful of bouts happened to produce.
    priors.setdefault("_default", overall)
    return priors


def blend_method_probability(
    divisional_prior: float, fighter_own_rate: float, opponent_vulnerability: float, fighter_total_wins: int,
) -> float:
    """
    Prior-informed blend: starts at the divisional baseline, then shifts
    toward the fighter's own observed tendency -- weighted by how much
    career sample size backs it up, so a 3-fight newcomer's personal rate
    doesn't override the divisional prior as hard as a proven veteran's
    would -- then further incorporates the specific opponent's vulnerability.
    """
    experience_weight = min(1.0, fighter_total_wins / 10.0)
    fighter_adjusted = divisional_prior + (fighter_own_rate - divisional_prior) * experience_weight
    return 0.7 * fighter_adjusted + 0.3 * opponent_vulnerability


def _stat(row, field):
    """
    Prefer the RECENCY-WEIGHTED value, fall back to the all-time one.

    Career means treat a fighter's 2016 form as equal to their 2025 form.
    Measured across 1,061 fighters with 6+ tracked fights, the median
    within-career drift in strikes/min is 0.62 of the BETWEEN-fighter
    standard deviation, and 32% drift by more than a full sd -- so an
    all-time mean often describes a blend of two different fighters.

    Validated before wiring (research_recency_weighting.py): an 18-month
    half-life was swept on the tuning split with all-time included as a
    control, and the control lost MONOTONICALLY. Frozen holdout, n=1747:
    58.6% -> 59.9% accuracy, Brier 0.2345 -> 0.2329.

    The fallback matters as much as the preference: scripts/backfill_recency_stats.py
    only writes a weighted value for fighters with enough dated cage time, so
    debutants and thin records keep the all-time number rather than dropping
    out of the adjustment entirely.
    """
    v = row.get(field + "_r")
    if v is not None and pd.notna(v) and str(v) != "":
        return v
    return row.get(field)


def build_probability_waterfall(matchup: dict) -> dict | None:
    """
    Decomposes the final probability into a running journey a reader can
    follow: start from an even fight at 50%, then watch each factor move
    the number, ending exactly on the model's real pick percentage.

    WHY PERCENTAGE POINTS AND A RUNNING TOTAL. The model works internally
    in rating points, which are exactly additive -- but "+149.4" means
    nothing at a glance, which made the first version of this panel
    unreadable. Percentage points are the unit the reader already thinks
    in, and showing the running probability after every step means no
    arithmetic is required to follow it: you can see the pick move from
    50% to its final number, one factor at a time.

    The deltas still sum EXACTLY to (final - 50%), because each is
    measured as the change in probability caused by adding that factor to
    everything before it. The one honest caveat is that with a non-linear
    logistic link, a factor's share depends slightly on where in the
    order it lands. Factors are applied largest-first (after the base
    rating gap), which is both a fixed, disclosed order and the one that
    reads most naturally -- biggest movers at the top.

    Rating points are still carried on every row as a secondary number,
    for anyone who wants the model's native units.

    ORIENTATION. predict_matchup computes everything as "positive favors
    fighter A". A reader wants "why do we like the pick", so signs flip
    when the favorite is fighter B. After this, positive always means
    "helps the favorite".

    EXACTNESS. Factors below DISPLAY_THRESHOLD fold into one "other
    factors" row rather than being dropped, and the cap gets its own row
    when it bites, so the journey always lands on the true final number.
    """
    if not matchup:
        return None

    prob_a = matchup.get("prob_a")
    base_a, base_b = matchup.get("base_rating_a"), matchup.get("base_rating_b")
    if prob_a is None or base_a is None or base_b is None:
        return None

    favorite_is_a = prob_a >= 0.5
    sign = 1.0 if favorite_is_a else -1.0
    favorite = matchup["fighter_a"] if favorite_is_a else matchup["fighter_b"]
    underdog = matchup["fighter_b"] if favorite_is_a else matchup["fighter_a"]

    base_gap = (base_a - base_b) * sign
    raw_layer = matchup.get("adjustment_layer_raw", 0.0) * sign
    applied_layer = matchup.get("adjustment_layer_applied", 0.0) * sign

    # (key in matchup, display label, plain-language explanation)
    FACTORS = [
        ("wrestling_adjustment", "Wrestling", "takedowns landed per 15 minutes"),
        ("striking_adjustment", "Striking", "strike accuracy and defense"),
        ("durability_adjustment", "Durability", "how often each has been finished"),
        ("recent_form_adjustment", "Recent form", "last three fights, weighted recent"),
        ("submission_threat_adjustment", "Sub threat", "share of wins by submission"),
        ("stance_adjustment", "Stance", "orthodox vs southpaw matchup"),
        ("height_adjustment", "Height", "height difference"),
        ("layoff_adjustment", "Layoff", "time since last fight"),
        ("quick_return_adjustment", "Quick turnaround", "unusually short rest before this fight"),
        ("age_cliff_adjustment", "Age", "age vs the division's decline point"),
        ("missed_weight_adjustment", "Missed weight", "history of missing weight"),
        ("weight_class_change_adjustment", "Division change", "moving up or down in weight"),
        ("short_notice_adjustment", "Short notice", "took the fight on short notice"),
    ]

    DISPLAY_THRESHOLD = 2.0  # rating points -- below this it's noise, not a driver

    def prob_at(gap):
        return 1.0 / (1.0 + 10 ** (-gap / 400.0))

    # Collect factors, fold the negligible ones together.
    collected, folded = [], 0.0
    for key, label, why in FACTORS:
        pts = matchup.get(key, 0.0) * sign
        if abs(pts) < DISPLAY_THRESHOLD:
            folded += pts
            continue
        collected.append({"key": key, "label": label, "why": why, "points": pts})
    collected.sort(key=lambda r: abs(r["points"]), reverse=True)
    if abs(folded) >= 0.05:
        collected.append({"key": None, "label": "Other factors",
                          "why": "everything else, too small to list",
                          "points": folded})

    # Walk the journey, recording the running probability after each step.
    rows = []
    running_gap = 0.0
    running_prob = prob_at(running_gap)  # 50%

    def step(label, why, pts, kind="factor", key=None):
        nonlocal running_gap, running_prob
        before = running_prob
        running_gap += pts
        running_prob = prob_at(running_gap)
        rows.append({
            # The matchup field this row was computed from. Costs nothing to
            # carry and makes a row self-describing to any future consumer that
            # needs to join against it rather than string-match the label.
            "key": key,
            "label": label, "why": why, "kind": kind,
            "points": round(pts, 1),
            "delta_pp": round((running_prob - before) * 100, 1),
            "running_pct": round(running_prob * 100, 1),
            "favors": favorite if pts > 0 else underdog,
        })

    step("Rating gap", "career record and opposition quality", base_gap, kind="base")
    for c in collected:
        step(c["label"], c["why"], c["points"], key=c["key"])
    if matchup.get("adjustment_capped"):
        # TWO LINES, NOT A PARAGRAPH. At 103 characters this was three times
        # the longest other description and the only one the .wf-why clamp
        # ever truncated -- and it cut at "so no...", exactly where the
        # sentence was about to say WHY the cap exists, which is the only part
        # a reader could not already infer from the label. The previous note
        # in templates/site.html called it "a full sentence that cannot be
        # shortened to one line"; that was true of ONE line, not of two.
        step("Adjustment cap",
             f"capped at {ADJUSTMENT_TOTAL_CAP:.0f} points so factors "
             f"cannot outweigh the gap",
             applied_layer - raw_layer, kind="cap")

    favorite_prob = prob_a if favorite_is_a else 1.0 - prob_a
    return {
        "favorite": favorite,
        "underdog": underdog,
        "rows": rows,
        "favorite_pct": round(favorite_prob * 100, 1),
        "underdog_pct": round((1.0 - favorite_prob) * 100, 1),
        "total_points": round(base_gap + applied_layer, 1),
        # Largest single move, for scaling the bars.
        "scale": max([abs(r["delta_pp"]) for r in rows] + [1.0]),
    }


BADGE_THRESHOLD = 15.0  # rating points -- below this, a factor isn't worth calling out as a driver


def build_factor_badges(matchup: dict) -> dict:
    """
    Translates the raw adjustment numbers already computed in predict_matchup
    into small labeled badges per fighter, e.g. "+ Durability" or "- Layoff",
    so the model's reasoning is scannable at a glance instead of only living
    in the prose narrative.

    Advantage-style factors (wrestling/striking/durability) are POSITIVE
    when they favor fighter A -- badge goes on whichever fighter has the
    edge, framed as a plus for them.

    Penalty-style factors (layoff/quick-return/age-cliff/missed-weight) are
    computed as (a's own penalty - b's own penalty) -- badge goes on
    whichever fighter is actually carrying that specific risk, framed as a
    minus for them, since the badge should describe what's true about the
    fighter it's attached to.
    """
    badges_a, badges_b = [], []

    # TWO KINDS OF BADGE, AND THEY ARE NOT OPPOSITE SIGNS OF ONE SCALE.
    # An "edge" is COMPARATIVE: "Wrestling" on fighter A means A out-wrestles
    # B, a statement about the pair, and it has a magnitude -- the two were
    # measured against each other and one won by some amount.
    # A "flag" is PERSONAL: "Quick Return" on fighter A means A took this
    # fight on short rest, true of him alone with no opponent in it.
    # The UI depends on the distinction: an edge draws a rail whose length is
    # the magnitude, a flag draws no rail at all, because a rail would be
    # claiming a contest that never happened. `key` is the matchup field the
    # badge came from, so a consumer can join the magnitude from the
    # waterfall rather than deriving it a second time.

    def add_advantage(key: str, label: str):
        value = matchup.get(key, 0) or 0
        if value > BADGE_THRESHOLD:
            badges_a.append({"label": label, "direction": "+", "kind": "edge", "key": key})
        elif value < -BADGE_THRESHOLD:
            badges_b.append({"label": label, "direction": "+", "kind": "edge", "key": key})

    add_advantage("wrestling_adjustment", "Wrestling")
    add_advantage("striking_adjustment", "Striking")
    add_advantage("durability_adjustment", "Durability")
    add_advantage("stance_adjustment", "Stance")
    add_advantage("submission_threat_adjustment", "Sub Threat")
    add_advantage("height_adjustment", "Height")

    if matchup.get("short_notice_flag_a"):
        badges_a.append({"label": "Short Notice", "direction": "-", "kind": "flag"})
    if matchup.get("short_notice_flag_b"):
        badges_b.append({"label": "Short Notice", "direction": "-", "kind": "flag"})

    layoff_adj = matchup.get("layoff_adjustment", 0)
    if layoff_adj < -BADGE_THRESHOLD:
        badges_a.append({"label": "Layoff", "direction": "-", "kind": "flag"})
    elif layoff_adj > BADGE_THRESHOLD:
        badges_b.append({"label": "Layoff", "direction": "-", "kind": "flag"})

    if matchup.get("quick_return_flag_a"):
        badges_a.append({"label": "Quick Return", "direction": "-", "kind": "flag"})
    if matchup.get("quick_return_flag_b"):
        badges_b.append({"label": "Quick Return", "direction": "-", "kind": "flag"})

    if matchup.get("age_cliff_flag_a"):
        badges_a.append({"label": "Age Cliff", "direction": "-", "kind": "flag"})
    if matchup.get("age_cliff_flag_b"):
        badges_b.append({"label": "Age Cliff", "direction": "-", "kind": "flag"})

    missed_weight_adj = matchup.get("missed_weight_adjustment", 0)
    if missed_weight_adj < -BADGE_THRESHOLD / 3:  # smaller threshold - even one instance should show
        badges_a.append({"label": "Missed Weight", "direction": "-", "kind": "flag"})
    elif missed_weight_adj > BADGE_THRESHOLD / 3:
        badges_b.append({"label": "Missed Weight", "direction": "-", "kind": "flag"})

    if matchup.get("weight_class_change_flag_a"):
        direction = matchup.get("weight_class_change_direction_a")
        label = "Moving Up" if direction == "up" else "Moving Down"
        badges_a.append({"label": label, "direction": "-" if direction == "up" else "+"})
    if matchup.get("weight_class_change_flag_b"):
        direction = matchup.get("weight_class_change_direction_b")
        label = "Moving Up" if direction == "up" else "Moving Down"
        badges_b.append({"label": label, "direction": "-" if direction == "up" else "+"})

    return {"a": badges_a, "b": badges_b}


def classify_style(row: pd.Series) -> str:
    td_acc = _get(row, "td_accuracy_pct", 20)
    strike_acc = _get(row, "strike_accuracy_pct", 45)
    if td_acc >= 40:
        return "Wrestler/Grappler"
    elif strike_acc >= 47:
        return "Striker"
    return "Balanced"


def stance_matchup_adjustment(row_a: pd.Series, row_b: pd.Series) -> float:
    """
    Southpaw (or switch) gets a modest bonus against a pure-orthodox
    opponent, reflecting the real "unfamiliar look" edge -- two fighters
    sharing the same stance (including two southpaws) is neutral, since
    neither has the familiarity advantage over the other.
    """
    stance_a = str(row_a.get("stance", "Orthodox") or "Orthodox").strip()
    stance_b = str(row_b.get("stance", "Orthodox") or "Orthodox").strip()
    a_unorthodox = stance_a in ("Southpaw", "Switch")
    b_unorthodox = stance_b in ("Southpaw", "Switch")
    if a_unorthodox and not b_unorthodox:
        return STANCE_MISMATCH_BONUS
    if b_unorthodox and not a_unorthodox:
        return -STANCE_MISMATCH_BONUS
    return 0.0


def submission_threat_adjustment(row_a: pd.Series, row_b: pd.Series) -> float:
    """
    A fighter's rate of finishing wins by submission is a distinct skill
    from wrestling_adjustment's takedown-accuracy/control-time focus --
    a fighter can have modest takedown numbers but a live submission
    threat off scrambles, guard, or clinch entries, which wrestling stats
    alone don't capture. Motivated by two real misses on the same card
    where the eventual winner's submission win ended the fight despite
    wrestling_adjustment reading 0.0 for both matchups (neither fighter
    stood out on raw takedown/control numbers, even though a submission
    is exactly how each fight was actually decided) -- a gap this factor
    is meant to close, not a guess without a specific motivating case.
    """
    wins_a = int(_get(row_a, "wins", 0))
    wins_b = int(_get(row_b, "wins", 0))
    # A MISSING sub_wins IS NOT ZERO SUBMISSIONS. Defaulting it to 0 scores a
    # fighter whose method splits never backfilled as having no submission
    # game at all, and hands the difference to the opponent -- so a grappler
    # with an incomplete row is read as the LESS dangerous grappler. Both
    # corners need the split or the term says nothing.
    have_a = row_a.get("sub_wins") is not None and pd.notna(row_a.get("sub_wins"))
    have_b = row_b.get("sub_wins") is not None and pd.notna(row_b.get("sub_wins"))
    if not (have_a and have_b and wins_a and wins_b):
        return 0.0
    sub_rate_a = _get(row_a, "sub_wins", 0) / wins_a
    sub_rate_b = _get(row_b, "sub_wins", 0) / wins_b
    return (sub_rate_a - sub_rate_b) * SUBMISSION_THREAT_SCALE


def style_matchup_adjustment(
    row_a: pd.Series, row_b: pd.Series,
    weight_class_history_df: pd.DataFrame | None = None, this_fight_weight_class: str | None = None,
    reference_date: dt.date | None = None,
) -> dict:
    """
    Returns a rating-point adjustment (in favor of fighter A, can be
    negative) plus a breakdown of what drove it, for transparency.
    """
    td_acc_a = _get(row_a, "td_accuracy_pct", 20)
    td_acc_b = _get(row_b, "td_accuracy_pct", 20)
    td_def_a = _get(row_a, "td_defense_pct", 65)
    td_def_b = _get(row_b, "td_defense_pct", 65)
    strike_acc_a = _get(row_a, "strike_accuracy_pct", 45)
    strike_acc_b = _get(row_b, "strike_accuracy_pct", 45)

    def _has(row, col) -> bool:
        """True only when the column holds a real number for this fighter."""
        v = row.get(col)
        return v is not None and v != "" and pd.notna(v)

    # A DIFFERENTIAL BETWEEN A REAL NUMBER AND A DEFAULT IS NOT A DIFFERENTIAL.
    #
    # The _get defaults above exist so a missing column can't crash the model,
    # and when NEITHER fighter has the data they cancel harmlessly: 45 - 45 and
    # max(0, 20 - 65) are both zero. The dangerous case is asymmetry. An
    # established fighter at 52% accuracy against a debutant defaulted to 45%
    # produces a 7-point "striking edge" that is an artifact of who has a data
    # file, not of anything either man does in a cage -- and it always favours
    # the fighter with more history, which is a bias dressed as a signal.
    #
    # That case is not hypothetical: strike_accuracy_pct / td_accuracy_pct /
    # td_defense_pct sit at ~29% roster coverage (0 of 25 on one live card),
    # because their only source is the manual ufcstats scraper, and ufcstats
    # now sits behind a JavaScript challenge that requests+BeautifulSoup
    # cannot pass. Coverage will stay partial even if that is ever solved:
    # ufcstats records UFC bouts only, so debutants have nothing there by
    # definition, and every card has debutants.
    #
    # So each term requires BOTH fighters to have real data or it contributes
    # nothing. Today that makes these terms inert, exactly as they already
    # were in practice. If the data ever returns they light up on their own
    # for the fights that can support them, and stay correctly silent on the
    # ones that can't -- no follow-up change needed.
    striking_data_ok = _has(row_a, "strike_accuracy_pct") and _has(row_b, "strike_accuracy_pct")
    td_acc_data_ok = _has(row_a, "td_accuracy_pct") and _has(row_b, "td_accuracy_pct")
    td_def_data_ok = _has(row_a, "td_defense_pct") and _has(row_b, "td_defense_pct")

    # Striking: accuracy differential, PLUS volume differential (SLpM - SApM)
    # when that data exists. A high-output fighter who lands 45% of a high
    # volume typically outpoints a low-output 60%-accurate fighter on
    # judges' cards -- accuracy alone misses this real, well-documented
    # dynamic. Falls back to accuracy-only when strike-volume data isn't
    # populated yet (graceful no-op, not a guessed number).
    striking_adj = (((strike_acc_a - strike_acc_b) / 100.0) * STRIKING_ADVANTAGE_SCALE
                    if striking_data_ok else 0.0)
    slpm_a, sapm_a = _stat(row_a, "slpm"), _stat(row_a, "sapm")
    slpm_b, sapm_b = _stat(row_b, "slpm"), _stat(row_b, "sapm")
    volume_adj = 0.0
    if pd.notna(slpm_a) and pd.notna(sapm_a) and pd.notna(slpm_b) and pd.notna(sapm_b):
        volume_diff_a = float(slpm_a) - float(sapm_a)
        volume_diff_b = float(slpm_b) - float(sapm_b)
        volume_adj = (volume_diff_a - volume_diff_b) * VOLUME_DIFFERENTIAL_SCALE
    striking_adj += volume_adj

    # Wrestling: prefer CONTROL TIME PERCENTAGE when available -- a fighter
    # who goes 1-for-5 on takedowns but holds 4 minutes of control along the
    # fence is far more effective than raw takedown accuracy alone implies.
    # Falls back to takedown-accuracy-vs-defense when control time isn't
    # populated yet.
    ctrl_a, ctrl_b = row_a.get("control_time_pct"), row_b.get("control_time_pct")
    td_rate_a, td_rate_b = _stat(row_a, "td_per_15"), _stat(row_b, "td_per_15")
    if pd.notna(td_rate_a) and pd.notna(td_rate_b):
        # Takedown RATE differential -- validated on a frozen 2019+ holdout in
        # head_to_head_adjustment.py. Holding every other term fixed and
        # swapping only this one: 57.7% / Brier 0.2365 -> 58.2% / 0.2354, and
        # it won at every blend weight tested from 1.0 to 3.0.
        #
        # Why rate beats the accuracy-vs-defense form it replaces: accuracy is
        # a percentage with no volume in it. A fighter who goes 1-for-1 reads
        # as 100% accurate with almost no wrestling output; one who goes 6-for-12
        # reads as half as good while actually controlling where the fight
        # happens far more. The old form also clipped at zero (max(0, acc - def)),
        # discarding the sign -- so being clearly WORSE at takedowns than the
        # opponent's defense registered identically to being merely equal.
        # Testing the clipping removal on its own made things slightly worse,
        # so the clipping was never the real problem; the input was.
        wrestling_adj = (float(td_rate_a) - float(td_rate_b)) * TD_RATE_ADVANTAGE_SCALE
    elif pd.notna(ctrl_a) and pd.notna(ctrl_b) and td_def_data_ok:
        # td_def_data_ok is required as well: this branch compares each
        # fighter's REAL control time against the OTHER's takedown defense,
        # so a defaulted 65 on one side manufactures the same phantom edge
        # the striking term above guards against.
        # Fallback for fighters with no tracked cage time yet (debutants, and
        # anyone the stats backfill couldn't name-match). Unchanged prior
        # behaviour rather than a guessed rate.
        wrestling_edge_a = max(0.0, float(ctrl_a) - td_def_b) / 100.0
        wrestling_edge_b = max(0.0, float(ctrl_b) - td_def_a) / 100.0
        wrestling_adj = (wrestling_edge_a - wrestling_edge_b) * WRESTLING_ADVANTAGE_SCALE
    elif td_acc_data_ok and td_def_data_ok:
        # Wrestling: A's takedown accuracy vs. B's takedown defense, and vice versa.
        # Only counts as an "edge" if the attacker's accuracy actually exceeds
        # the defender's defense rate -- otherwise no stylistic advantage either way.
        wrestling_edge_a = max(0.0, td_acc_a - td_def_b) / 100.0
        wrestling_edge_b = max(0.0, td_acc_b - td_def_a) / 100.0
        wrestling_adj = (wrestling_edge_a - wrestling_edge_b) * WRESTLING_ADVANTAGE_SCALE
    else:
        # No wrestling data either side can support. Zero, not a guess.
        wrestling_adj = 0.0

    # Durability: how often has each been finished before (by any method)?
    # A high finish-loss rate against someone with strong finishing tools
    # is a real, specific risk -- not just "durability" in the abstract.
    losses_a = max(int(_get(row_a, "losses", 0)), 1) if _get(row_a, "losses", 0) else 1
    losses_b = max(int(_get(row_b, "losses", 0)), 1) if _get(row_b, "losses", 0) else 1
    _dur_base = divisional_finish_rate(this_fight_weight_class)
    # losses_a/losses_b are NOT passed: they carry a max(..., 1) floor that
    # would read a 0-loss fighter as (0 + k*base)/(1 + k) instead of the base
    # itself. The function reads the raw count off the row for exactly that
    # reason -- see its docstring.
    finish_loss_rate_a = _shrunk_finish_loss_rate(row_a, _dur_base)
    finish_loss_rate_b = _shrunk_finish_loss_rate(row_b, _dur_base)
    # Same guard, and this one was the most backwards of the three: with
    # ko_losses/sub_losses defaulting to 0, a fighter whose method splits are
    # simply unknown computes a finish-loss rate of zero -- a PERFECT chin --
    # and is handed a durability edge over an opponent with a real, honest
    # record of having been stopped. Unmeasured was scoring better than
    # measured.
    durability_data_ok = all(
        _has(r, c) for r in (row_a, row_b) for c in ("losses", "ko_losses", "sub_losses")
    )
    durability_adj = ((finish_loss_rate_b - finish_loss_rate_a) * DURABILITY_SCALE
                      if durability_data_ok else 0.0)

    # MEASURED CHIN -- BUILT, TESTED, AND REJECTED. CHIN_SCALE is 0.0, so this
    # is a no-op. It is kept, with its numbers, because the signal underneath
    # it is strong enough that it WILL be proposed again.
    #
    # THE IDEA. durability_adj above is built from the method split of a
    # fighter's LOSSES. data/pit_stats.csv carries kd_against and
    # fight_seconds on 17,524 fighter-bout rows and the model read neither.
    # Knockdowns absorbed per 15 minutes measures the same thing denominated
    # in cage time rather than in defeats, so unlike the proxy it sees a fight
    # the fighter survived, and its denominator is not frequently 1.
    #
    # THE SIGNAL IS REAL. On 4,813 losses from 2010 on, quintiles of this rate
    # map monotonically onto the chance the loss came by KO/TKO -- 24.9% /
    # 31.9% / 34.7% / 40.2%, point-biserial r = 0.122, p = 2e-17. It is only
    # 0.345 correlated with the existing proxy, and a model with BOTH beats
    # either alone (5-fold CV log loss 0.60985 against 0.61301 and 0.61334).
    # Every one of those numbers argues for adding this term.
    #
    # IT STILL DOES NOT WORK, because all of them answer the wrong question.
    # They predict the METHOD of a loss. The site publishes P(win). Swept
    # point-in-time (scripts/validate_chin.py), Brier against CHIN_SCALE = 0:
    #
    #     scale   recent window        prior window
    #      15     +0.00022 (p .12)     +0.00006 (p .71)
    #      30     +0.00048 (p .08)     +0.00012 (p .66)
    #      60     +0.00107 (p .04)          --
    #
    # Worse at every positive scale in the recent window, MONOTONICALLY worse
    # as the scale grows, and worse still on the subset with the widest chin
    # gap -- 0.20664 to 0.21762 at scale 60 on gap > 0.50. That gradient is
    # what makes this a finding rather than noise: the term does most damage
    # exactly where it speaks loudest. The prior window is flat in both
    # directions, and a negative scale does not replicate either (-15 and -30
    # sit at p = 0.89 and 0.92), so this is not a sign error.
    #
    # The likely reason is that being dropped is entangled with a style that
    # also wins fights. Whatever the mechanism, a signal that predicts HOW a
    # fighter loses is not thereby a signal about WHETHER they lose, and the
    # gap between those two claims is where this term died.
    #
    # kd_against_per_15 is still emitted by build_pit_stats. The natural place
    # for it is the METHOD model, which predicts the quantity it actually
    # tracks; that is untested and is not this term.
    kd_a, kd_b = _stat(row_a, "kd_against_per_15"), _stat(row_b, "kd_against_per_15")
    if pd.notna(kd_a) and pd.notna(kd_b):
        # Sign matches durability: the fighter whose OPPONENT is more
        # droppable gains. b's rate minus a's, so a high rate on b helps a.
        chin_adj = (float(kd_b) - float(kd_a)) * CHIN_SCALE
    else:
        chin_adj = 0.0

    # DATED TO THE FIGHT, NOT TO TODAY. layoff_penalty and quick_return_penalty
    # have always accepted a reference_date and nothing ever passed one, so
    # every historical fight was scored as though it happened this morning.
    # A 2015 bout then reads an eleven-year layoff for both corners: layoff
    # fires on 64-70% of backtested fights against 13% live, and quick_return
    # -- which needs a gap UNDER six months -- can never fire at all.
    #
    # This is the same defect as the recent_form one, in the two terms beside
    # it. Worth stating plainly because audit_term_coverage.py's own docstring
    # blamed the firing-rate gap on reconstruction bias in fight_history; that
    # diagnosis was wrong, and no improvement in coverage could have fixed it.
    layoff_adj_a = layoff_penalty(row_a, reference_date)
    layoff_adj_b = layoff_penalty(row_b, reference_date)
    layoff_adj = layoff_adj_a - layoff_adj_b  # penalize A if A has the longer layoff, and vice versa

    quick_return_adj_a = quick_return_penalty(row_a, reference_date)
    quick_return_adj_b = quick_return_penalty(row_b, reference_date)
    quick_return_adj = quick_return_adj_a - quick_return_adj_b

    age_cliff_adj_a = age_cliff_penalty(row_a)
    age_cliff_adj_b = age_cliff_penalty(row_b)
    age_cliff_adj = age_cliff_adj_a - age_cliff_adj_b

    missed_weight_adj_a = missed_weight_penalty(row_a)
    missed_weight_adj_b = missed_weight_penalty(row_b)
    missed_weight_adj = missed_weight_adj_a - missed_weight_adj_b

    weight_class_change_adj_a = weight_class_change_penalty(row_a.get("name"), this_fight_weight_class, weight_class_history_df)
    weight_class_change_adj_b = weight_class_change_penalty(row_b.get("name"), this_fight_weight_class, weight_class_history_df)
    weight_class_change_adj = weight_class_change_adj_a - weight_class_change_adj_b

    stance_adj = stance_matchup_adjustment(row_a, row_b)
    submission_threat_adj = submission_threat_adjustment(row_a, row_b)

    height_a = _get(row_a, "height_in", 70)
    height_b = _get(row_b, "height_in", 70)
    height_adj = (height_a - height_b) * HEIGHT_ADVANTAGE_SCALE

    short_notice_a = bool(_get(row_a, "short_notice", 0))
    short_notice_b = bool(_get(row_b, "short_notice", 0))
    short_notice_adj = (SHORT_NOTICE_PENALTY if short_notice_b else 0.0) - (SHORT_NOTICE_PENALTY if short_notice_a else 0.0)

    # SCALED BY SAMPLE SIZE -- but only the two terms actually built from rate
    # statistics. Height, age, layoff, short notice, missed weight and stance
    # are biographical facts that a debutant knows about himself as precisely
    # as a veteran does; discounting them for inexperience would be wrong.
    # Durability and submission threat come from the win/loss RECORD, which is
    # a count rather than a per-minute rate, so a short career makes them
    # noisy in the ordinary way rather than systematically extreme -- a
    # different problem, left alone here rather than swept in.
    rate_conf = rate_stat_confidence(row_a, row_b)
    striking_adj *= rate_conf
    wrestling_adj *= rate_conf
    # Chin is a per-minute rate off the same source as striking and wrestling,
    # so it takes the same sample-size discount. Durability does not, because
    # it is a count off the win/loss record -- see the note above.
    chin_adj *= rate_conf

    total_adj = (
        wrestling_adj + striking_adj + durability_adj + chin_adj + layoff_adj
        + quick_return_adj + age_cliff_adj + missed_weight_adj + weight_class_change_adj
        + stance_adj + submission_threat_adj + height_adj + short_notice_adj
    )

    return {
        "total_adjustment": total_adj,
        # Exposed so the waterfall can say WHY a style edge is small on a
        # thin-sample fight, rather than the number just quietly being
        # unimpressive with no explanation.
        "rate_stat_confidence": rate_conf,
        "wrestling_adjustment": wrestling_adj,
        "striking_adjustment": striking_adj,
        "durability_adjustment": durability_adj,
        "chin_adjustment": chin_adj,
        "submission_threat_adjustment": submission_threat_adj,
        "height_adjustment": height_adj,
        "short_notice_adjustment": short_notice_adj,
        "short_notice_flag_a": short_notice_a,
        "short_notice_flag_b": short_notice_b,
        "layoff_adjustment": layoff_adj,
        "layoff_years_a": layoff_years(row_a, reference_date),
        "layoff_years_b": layoff_years(row_b, reference_date),
        "quick_return_adjustment": quick_return_adj,
        "quick_return_flag_a": quick_return_adj_a < 0,
        "quick_return_flag_b": quick_return_adj_b < 0,
        "age_cliff_adjustment": age_cliff_adj,
        "age_cliff_flag_a": age_cliff_adj_a < 0,
        "age_cliff_flag_b": age_cliff_adj_b < 0,
        "missed_weight_adjustment": missed_weight_adj,
        "weight_class_change_adjustment": weight_class_change_adj,
        "weight_class_change_flag_a": weight_class_change_adj_a != 0,
        "weight_class_change_flag_b": weight_class_change_adj_b != 0,
        "weight_class_change_direction_a": "up" if weight_class_change_adj_a < 0 else ("down" if weight_class_change_adj_a > 0 else None),
        "weight_class_change_direction_b": "up" if weight_class_change_adj_b < 0 else ("down" if weight_class_change_adj_b > 0 else None),
        "stance_adjustment": stance_adj,
        "style_a": classify_style(row_a),
        "style_b": classify_style(row_b),
    }


def recent_form_adjustment(
    fighter_a: str, fighter_b: str, fight_history_df: pd.DataFrame | None,
    reference_date: dt.date | None = None,
) -> float:
    """
    A genuine but partial recency signal: fighters.csv only tracks
    aggregate career win/loss counts, not dated per-fight records, so a
    fully recency-weighted career rating isn't possible for the roster as
    a whole with current data. This instead looks at each fighter's most
    recent entry in fight_history.csv specifically (if they have one) --
    a small, decaying bonus for a recent win, a small penalty for a
    recent loss, weighted by how long ago it was. Fighters with no
    tracked history get exactly 0 here, same as every other graceful
    fallback in this file -- this is a real but limited signal, not a
    substitute for the full recency-weighted system a richer dataset
    would support.
    """
    if fight_history_df is None or fight_history_df.empty:
        return 0.0
    reference_date = reference_date or dt.date.today()

    def fighter_signal(name: str) -> float:
        rows = fight_history_df[
            (fight_history_df["fighter_a"] == name) | (fight_history_df["fighter_b"] == name)
        ].copy()
        if rows.empty:
            return 0.0
        rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
        rows = rows.dropna(subset=["date"]).sort_values("date")
        # STRICTLY BEFORE THE FIGHT BEING PREDICTED. Without this the tail()
        # below takes a fighter's last three fights from the WHOLE table, so
        # a caller that supplied full history plus a past reference_date read
        # bouts that had not happened yet -- and read them at FULL weight,
        # because a negative years_ago clamps the decay to 1.0. The most
        # recent fight is also the most influential, which is precisely the
        # one most likely to be in the future.
        #
        # Production was never wrong here (reference_date is today and the
        # table holds only decided fights), but the hazard is why every
        # harness omitted fight_history_df altogether, which silenced this
        # term in every backtest verdict the project has published. Filtering
        # is what makes passing history safe.
        rows = rows[rows["date"].dt.date < reference_date]
        if rows.empty:
            return 0.0
        # Last-3-fights decayed form, VALIDATED on the July 2026 walk-forward
        # backtest: formulation + scale selected on pre-2019 fights, confirmed
        # on held-out 2019+ fights, where it beat both no-form-at-all and the
        # previous single-most-recent-fight version this replaces.
        # A DRAW OR NO CONTEST IS NEITHER. `last["winner"] == name` is False
        # for a winnerless row, so an NC used to score a full -1.0 -- the same
        # penalty as a knockout defeat -- and, being the most recent fight,
        # at the heaviest decay weight. It is dropped instead: it says nothing
        # about form in either direction. (It still counts toward layoff,
        # which is read from the fight index, not from here.)
        decided = rows[rows["winner"].astype(str).str.strip().ne("")
                       & rows["winner"].notna()]
        signal = 0.0
        for _, last in decided.tail(RECENT_FORM_LOOKBACK).iterrows():
            won = last["winner"] == name
            years_ago = max((reference_date - last["date"].date()).days / 365.25, 0.0)
            decay = max(0.0, 1.0 - years_ago / RECENT_FORM_DECAY_YEARS)
            signal += (1.0 if won else -1.0) * decay * RECENT_FORM_SCALE
        return signal

    return fighter_signal(fighter_a) - fighter_signal(fighter_b)


def predict_matchup(
    fighter_a: str, fighter_b: str,
    fighters_df: pd.DataFrame,
    effective_ratings: dict[str, float],
    fight_history_df: pd.DataFrame | None = None,
    weight_class_history_df: pd.DataFrame | None = None,
    fight_weight_class: str | None = None,
    reference_date: dt.date | None = None,
) -> dict | None:
    """
    Full pairwise prediction: base rating gap + style-matchup adjustment,
    converted to a win probability, with a breakdown for the UI to explain.

    THERE IS DELIBERATELY NO scheduled_rounds PARAMETER. A main event and a
    prelim are scored identically here, which looks like an omission and has
    been raised as one, so the measurement is recorded rather than the
    argument (scripts/validate_five_round.py).

    The usual claim is that five rounds give the better fighter more time to
    express an edge, so a three-round-calibrated probability should be
    UNDERCONFIDENT on a championship-distance fight. Point-in-time, on 827
    five-round bouts across two disjoint windows, the data says the opposite:

        window        3-round gap    5-round gap
        recent          -0.0234        -0.0326      (n = 4739 / 563)
        prior           -0.0415        -0.0530      (n = 2195 / 264)

    where gap is observed hit rate minus mean confidence. Five-round bouts are
    MORE overconfident than three-round ones, consistently, by about a point.
    Sharpening -- the correction the standard story implies -- makes it worse
    in both windows and significantly so at s = 1.15 in the prior window
    (+0.00247, p = 0.046).

    Flattening is the correction the data actually points at, and it improves
    Brier in both windows at every level tried. It is NOT shipped, because it
    never approaches significance (best p = 0.15) and there is no more sample
    to find: five-round fights are ~10% of the record and that ceiling is
    structural, not a matter of waiting.

    Read the absolute gaps with the caveat established in
    validate_probability_calibration.py -- a backtest overstates
    overconfidence because departed fighters carry no height, reach or age and
    their style terms gate off. That caveat applies to both columns, which is
    why the comparison between them is the part worth trusting.

    So: the interaction is real and points the other way from the folklore,
    and correcting it is below the noise floor. If a future baseline moves it,
    flattening five-rounders is the direction to try.
    """
    # ONE SPELLING BEFORE ANY LOOKUP. The two lines below match the roster by
    # exact string equality, and base_r_a/base_r_b below read the ratings by
    # exact dict key. Both MISS SILENTLY -- an unrecognised name is an empty
    # frame and a defaulted 1500, never an error -- so a fighter arriving
    # under a second spelling is indistinguishable from an unknown fighter.
    #
    # Not hypothetical. Merging Jose Delgado's split identity canonicalised
    # fight_history and the roster, which is where elo's keys come from, but
    # the odds feed still quotes him "Jose Miguel Delgado" and nothing
    # rewrote it on the way in. That published the Noche main event at Silva
    # 77.4% -- 1713 against a defaulted 1500 -- where the model holding his
    # real 1681 says 48.7%, and left six method rows summing to 156.2%
    # because edge_finder cannot reconcile a grid against a None matchup.
    #
    # Here rather than at each caller: this function is the single door every
    # one of them goes through, and canonical_name is a no-op for every name
    # not in NAME_ALIASES, so it costs one dict lookup and closes the class
    # rather than the instance.
    fighter_a = canonical_name(fighter_a)
    fighter_b = canonical_name(fighter_b)

    match_a = fighters_df[fighters_df["name"] == fighter_a]
    match_b = fighters_df[fighters_df["name"] == fighter_b]
    if match_a.empty or match_b.empty:
        return None
    row_a, row_b = match_a.iloc[0], match_b.iloc[0]

    base_r_a = effective_ratings.get(fighter_a, 1500.0)
    base_r_b = effective_ratings.get(fighter_b, 1500.0)

    # Prefer the caller's card-specific division when they have it (the
    # actual booked weight class for this fight) over fighters_df's own
    # weight_class column, which reflects "most recently known division"
    # and could in principle drift from what a specific card says.
    this_fight_wc = fight_weight_class or row_a.get("weight_class")
    style = style_matchup_adjustment(row_a, row_b, weight_class_history_df, this_fight_wc,
                                     reference_date)
    # reference_date FORWARDED. It was accepted by recent_form_adjustment and
    # never supplied, so the term always dated itself to today. Harmless for a
    # booked fight; wrong for any historical one, and the missing parameter is
    # half of why backtests could not pass history at all.
    recent_form_adj = recent_form_adjustment(
        fighter_a, fighter_b, fight_history_df, reference_date)
    raw_layer = style["total_adjustment"] + recent_form_adj
    applied_layer = max(-ADJUSTMENT_TOTAL_CAP, min(ADJUSTMENT_TOTAL_CAP, raw_layer))
    adjusted_gap = (base_r_a - base_r_b) + applied_layer
    prob_a = 1.0 / (1.0 + 10 ** (-adjusted_gap / 400.0))

    # Uncertainty band: this is a heuristic, not a fitted confidence
    # interval (that would need a proper Bayesian treatment or bootstrap
    # over historical outcomes, which the current data doesn't support).
    # It scales down as the THINNER of the two records grows -- a matchup
    # where one side has 2 fights should visibly carry more uncertainty
    # than one where both have 25, even if the point estimate is
    # identical. Floors around ~5pp even for deep records, since MMA has
    # real irreducible variance no amount of data fully removes.
    fights_a = int(_get(row_a, "wins", 0)) + int(_get(row_a, "losses", 0))
    fights_b = int(_get(row_b, "wins", 0)) + int(_get(row_b, "losses", 0))
    thinner_record = min(fights_a, fights_b)
    # WHICH corner is the thin one. thinner_record alone can say the label was
    # capped but not who caused it, and a reader looking at "78% / Medium"
    # cannot act on the first without the second. Ties resolve to A
    # arbitrarily -- when both corners are equally thin, naming either is
    # equally true and the caller only uses this to write one name.
    thinner_corner = fighter_a if fights_a <= fights_b else fighter_b

    # IS EITHER CORNER DEBUTING? Distinct from thinner_record, which counts
    # PROFESSIONAL bouts: Anthony Wint is 7-0 as a pro and clears that floor
    # comfortably while having never fought in the UFC. This counts tracked
    # fights, the same quantity build_effective_ratings uses to decide
    # whether Elo means anything, so it identifies exactly the fighters
    # whose rating comes purely off the career-record curve.
    #
    # None when no history was supplied, so a caller that cannot know this
    # gates off rather than silently reading every fighter as a debutant.
    debut_corner = None
    debut_corner_name = None
    if fight_history_df is not None and not fight_history_df.empty:
        tracked = pd.concat([
            fight_history_df.get("fighter_a", pd.Series(dtype=str)),
            fight_history_df.get("fighter_b", pd.Series(dtype=str)),
        ]).value_counts()
        a_undebuted = int(tracked.get(fighter_a, 0)) == 0
        b_undebuted = int(tracked.get(fighter_b, 0)) == 0
        debut_corner = a_undebuted or b_undebuted
        # WHICH corner, for callers that need to name him. Same exact-string
        # lookup as the flag itself on purpose: deriving the name any other
        # way could disagree with the boolean it explains. If both corners are
        # undebuted, A is named -- the flag cannot distinguish them either.
        debut_corner_name = fighter_a if a_undebuted else (fighter_b if b_undebuted else None)
    uncertainty = UNCERTAINTY_BASE / math.sqrt(thinner_record + 1)
    prob_low = max(0.01, prob_a - uncertainty)
    prob_high = min(0.99, prob_a + uncertainty)

    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "prob_a": prob_a,
        "prob_b": 1 - prob_a,
        # Exposed so callers can gate on data depth without recomputing it.
        # The uncertainty band below is already derived from this; until now
        # nothing downstream could see the number the band was built from.
        "thinner_record": thinner_record,
        "thinner_corner": thinner_corner,
        "debut_corner": debut_corner,
        "debut_corner_name": debut_corner_name,
        "prob_low": prob_low,
        "prob_high": prob_high,
        "base_rating_a": base_r_a,
        "base_rating_b": base_r_b,
        "recent_form_adjustment": recent_form_adj,
        "adjustment_layer_raw": raw_layer,
        "adjustment_layer_applied": applied_layer,
        "adjustment_capped": applied_layer != raw_layer,
        **style,
    }
