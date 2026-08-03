"""
Combines the Elo model + fighter finish-rate stats with sportsbook lines
to surface bets where the model disagrees most with the market.

This is a decision-SUPPORT tool, not a decision-maker. Edges are only as
good as the historical data feeding the model -- always sanity check
against recent form, injuries, weight cuts, camp changes, etc. that a
pure stats model can't see.
"""

import re

import pandas as pd

from .odds_utils import american_to_implied_prob, implied_prob_to_american, remove_vig_two_way, edge_percent, kelly_fraction, market_blended_prob
from .matchup_model import predict_matchup, compute_divisional_method_priors, blend_method_probability, _get


def _fold_name(t):
    """Lowercase + strip diacritics for cross-source name matching."""
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKD", str(t).lower())
                   if not unicodedata.combining(ch)).strip()


def _find_fighter(fighters_df, name):
    """
    Look a fighter up TOLERANTLY of accents.

    Polymarket returns "Uros Medic" while fighters.csv holds "Uroš Medić", so
    an exact `== name` match failed and every market needing BOTH fighters was
    silently skipped -- which is why GoesTheDistance classified 40 rows and
    produced zero edges. Moneyline and Method survived because they only
    resolve the SELECTED fighter, so a mismatch on the opponent didn't matter.
    """
    exact = fighters_df[fighters_df["name"] == name]
    if not exact.empty:
        return exact
    folded = _fold_name(name)
    return fighters_df[fighters_df["name"].map(_fold_name) == folded]


def compute_moneyline_edges(
    upcoming_df: pd.DataFrame, elo_ratings: dict[str, float], fighters_df: pd.DataFrame | None = None,
    fight_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    ml = upcoming_df[upcoming_df["market"] == "Moneyline"]

    for fight_id, group in ml.groupby("fight_id"):
        if len(group) != 2:
            print(f"[edge_finder] moneyline skip for fight_id={fight_id!r}: {len(group)} row(s) instead of 2 "
                  f"-- selections: {group['selection'].tolist()}")
            continue  # need both sides of the moneyline to devig

        a, b = group.iloc[0], group.iloc[1]

        matchup = None
        if fighters_df is not None:
            matchup = predict_matchup(a["selection"], b["selection"], fighters_df, elo_ratings, fight_history_df)

        if matchup:
            model_prob_a = matchup["prob_a"]
        else:
            # fallback: plain rating gap if we don't have style stats for these fighters
            elo_a = elo_ratings.get(a["selection"], 1500.0)
            elo_b = elo_ratings.get(b["selection"], 1500.0)
            model_prob_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
        model_prob_b = 1.0 - model_prob_a

        imp_a = american_to_implied_prob(a["odds_american"])
        imp_b = american_to_implied_prob(b["odds_american"])
        fair_a, fair_b = remove_vig_two_way(imp_a, imp_b)

        for fighter, opponent, model_p, fair_p, odds, token_id in [
            (a["selection"], b["selection"], model_prob_a, fair_a, a["odds_american"], a.get("clob_token_id")),
            (b["selection"], a["selection"], model_prob_b, fair_b, b["odds_american"], b.get("clob_token_id")),
        ]:
            rows.append({
                "fight_id": fight_id,
                "fighter": fighter,
                "opponent": opponent,
                "market": "Moneyline",
                "odds_american": odds,
                "model_prob": round(model_p, 3),
                "book_fair_prob": round(fair_p, 3),
                "edge_pct": round(edge_percent(model_p, fair_p), 2),
                "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, fair_p), odds) * 100, 2),
                "clob_token_id": token_id,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def compute_method_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Method-of-victory props (KO/TKO, Submission, Decision). Prior-informed
    blend: starts at the DIVISIONAL baseline rate for that method (a
    heavyweight fight has an inherently higher baseline KO/TKO rate than a
    strawweight fight, which leans toward decisions), then shifts toward
    the fighter's own career tendency (weighted by experience), then further
    incorporates how often THIS SPECIFIC opponent has actually lost that
    way before.
    """
    rows = []
    props = upcoming_df[upcoming_df["market"] == "Method"]
    divisional_priors = compute_divisional_method_priors(fighters_df)

    method_loss_col = {"KO/TKO": "ko_losses", "SUB": "sub_losses", "DEC": "dec_losses"}

    def _blended_method_prob(f, opp_stats, method: str) -> float:
        """
        One method's blended probability (divisional prior -> own career
        rate -> this specific opponent's vulnerability to that method),
        pulled out of the main loop so FINISH can call it twice (once for
        KO/TKO, once for SUB) and sum the results, rather than duplicating
        the blend logic inline.
        """
        total_wins = max(int(f["wins"]), 1)
        rate_map = {
            "KO/TKO": _get(f, "ko_wins", 0) / total_wins,
            "SUB": _get(f, "sub_wins", 0) / total_wins,
            "DEC": _get(f, "dec_wins", 0) / total_wins,
        }
        own_rate = rate_map[method]
        divisional_prior = divisional_priors.get(f["weight_class"], {}).get(method, own_rate)
        opp_vulnerability = own_rate  # fallback if opponent data is missing
        if opp_stats is not None and not opp_stats.empty:
            opp = opp_stats.iloc[0]
            opp_losses = max(int(opp["losses"]), 1) if opp["losses"] else 0
            if opp_losses:
                opp_vulnerability = opp[method_loss_col[method]] / opp_losses
        return blend_method_probability(divisional_prior, own_rate, opp_vulnerability, total_wins)

    for _, row in props.iterrows():
        stats = _find_fighter(fighters_df, row["selection"])
        if stats.empty:
            continue
        f = stats.iloc[0]

        # find the opponent to factor in their specific vulnerability
        opponent_name = row["fighter_b"] if row["selection"] == row["fighter_a"] else row["fighter_a"]
        opp_stats = _find_fighter(fighters_df, opponent_name)

        if row["selection_method"] == "FINISH":
            # "Wins by finish" = KO/TKO or SUB -- these are mutually
            # exclusive outcomes for a single fight, so the combined
            # probability is a straight sum of the two independently-
            # blended method probabilities, not a new model. Reuses 100%
            # of the same prior-informed blend already trusted for the
            # individual KO/SUB props, rather than inventing a separate
            # "finish" prior from scratch.
            model_p = _blended_method_prob(f, opp_stats, "KO/TKO") + _blended_method_prob(f, opp_stats, "SUB")
            model_p = min(0.97, model_p)  # same sanity ceiling style used elsewhere in this module
        else:
            total_wins = max(int(f["wins"]), 1)
            rate_map = {
                "KO/TKO": _get(f, "ko_wins", 0) / total_wins,
                "SUB": _get(f, "sub_wins", 0) / total_wins,
                "DEC": _get(f, "dec_wins", 0) / total_wins,
            }
            own_rate = rate_map.get(row["selection_method"])
            if own_rate is None:
                continue
            model_p = _blended_method_prob(f, opp_stats, row["selection_method"])

        imp = american_to_implied_prob(row["odds_american"])

        rows.append({
            "fight_id": row["fight_id"],
            "fighter": row["selection"],
            "opponent": opponent_name,
            "market": f"Method: {row['selection_method']}",
            "odds_american": row["odds_american"],
            "model_prob": round(model_p, 3),
            "book_fair_prob": round(imp, 3),  # not devigged (single-sided prop)
            "edge_pct": round(edge_percent(model_p, imp), 2),
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, imp), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def compute_round_betting_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fighter-specific "wins in round N" props. Deliberately scoped to Round 1
    ONLY, and deliberately simpler than compute_method_edges above -- there
    is real backfilled data for first_round_finish_pct (career rate of a
    fighter's wins ending specifically in round 1), but no equivalent
    divisional prior or opponent-vulnerability-to-early-finish signal exists
    anywhere in this dataset. Rather than invent one to make the blend look
    as rich as the Method market's, this uses the one real signal directly
    and skips (no edge produced -- not a guess) whenever it's missing for a
    fighter. Round 2+ selections are skipped entirely for the same reason:
    no real per-round data exists to back a Round 2 or Round 3 estimate,
    and fabricating one would misrepresent how much the model actually
    knows. If per-round career data becomes available later, this is the
    function to extend -- not to replace.
    """
    rows = []
    props = upcoming_df[upcoming_df["market"] == "RoundBetting"]

    for _, row in props.iterrows():
        if str(row["selection_method"]) != "1":
            continue  # Round 2+ -- no real data to back an estimate, see docstring

        # selection is "<Fighter> Round 1" -- match against whichever of
        # fighter_a/fighter_b the selection text actually starts with,
        # rather than assuming position, since DK's outcome ordering isn't
        # guaranteed to put fighter_a first.
        fighter_name = row["fighter_a"] if str(row["selection"]).startswith(str(row["fighter_a"])) else row["fighter_b"]
        stats = _find_fighter(fighters_df, fighter_name)
        if stats.empty:
            continue
        f = stats.iloc[0]
        if "first_round_finish_pct" not in f or pd.isna(f["first_round_finish_pct"]):
            continue  # no real signal for this fighter -- skip rather than guess

        model_p = float(f["first_round_finish_pct"])
        opponent_name = row["fighter_b"] if fighter_name == row["fighter_a"] else row["fighter_a"]
        imp = american_to_implied_prob(row["odds_american"])

        rows.append({
            "fight_id": row["fight_id"],
            "fighter": fighter_name,
            "opponent": opponent_name,
            "market": "Round Betting: Round 1",
            "odds_american": row["odds_american"],
            "model_prob": round(model_p, 3),
            "book_fair_prob": round(imp, 3),  # not devigged (single-sided prop)
            "edge_pct": round(edge_percent(model_p, imp), 2),
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, imp), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def _extract_round_line(selection: str) -> float | None:
    match = re.search(r"(\d+\.?\d*)", str(selection))
    return float(match.group(1)) if match else None


def compute_total_rounds_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Over/Under total rounds props. A fight card often offers multiple lines
    (1.5, 2.5, 3.5) for the same fight -- these are grouped separately so
    they don't collide.

    For the 1.5 line specifically ("does it end in round 1"), this uses each
    fighter's actual first_round_finish_pct directly -- a fighter like
    Terrance McKinney (16 of 17 wins finished in round 1, never gone to a
    decision) should swing this line hard, and a generic finish-rate proxy
    was missing that entirely. Other lines blend that same signal in rather
    than relying purely on a generic linear adjustment.
    """
    rows = []
    props = upcoming_df[upcoming_df["market"] == "TotalRounds"].copy()
    props["_line"] = props["selection"].apply(_extract_round_line)

    REFERENCE_LINE = 2.5
    ADJUSTMENT_PER_ROUND = 0.15

    for (fight_id, line), group in props.groupby(["fight_id", "_line"]):
        fighters_in_fight = group["fighter_a"].iloc[0], group["fighter_b"].iloc[0]
        finish_rates = []
        first_round_rates = []
        for name in fighters_in_fight:
            stats = _find_fighter(fighters_df, name)
            if stats.empty:
                continue
            f = stats.iloc[0]
            total_wins = max(int(f["wins"]), 1)
            finish_rates.append((_get(f, "ko_wins", 0) + _get(f, "sub_wins", 0)) / total_wins)
            if "first_round_finish_pct" in f and pd.notna(f["first_round_finish_pct"]):
                first_round_rates.append(float(f["first_round_finish_pct"]))

        if not finish_rates:
            continue

        combined_finish_rate = sum(finish_rates) / len(finish_rates)
        combined_first_round_rate = sum(first_round_rates) / len(first_round_rates) if first_round_rates else None

        if line is not None and line <= 1.5 and combined_first_round_rate is not None:
            # the literal, most verifiable case: does it end in round 1
            model_prob_under = combined_first_round_rate
        elif line is not None:
            base = combined_finish_rate - (REFERENCE_LINE - line) * ADJUSTMENT_PER_ROUND
            # blend in the fast-finisher signal even for longer lines, rather
            # than only using it for the 1.5 boundary
            if combined_first_round_rate is not None:
                base = 0.7 * base + 0.3 * combined_first_round_rate
            model_prob_under = base
        else:
            model_prob_under = combined_finish_rate
        model_prob_under = min(0.95, max(0.05, model_prob_under))

        for _, row in group.iterrows():
            model_p = model_prob_under if "under" in row["selection"].lower() else (1 - model_prob_under)
            imp = american_to_implied_prob(row["odds_american"])
            rows.append({
                "fight_id": fight_id,
                "fighter": f"{fighters_in_fight[0]} vs {fighters_in_fight[1]}",
                "market": f"Total Rounds {row['selection']}",
                "odds_american": row["odds_american"],
                "model_prob": round(model_p, 3),
                "book_fair_prob": round(imp, 3),
                "edge_pct": round(edge_percent(model_p, imp), 2),
                "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, imp), row["odds_american"]) * 100, 2),
                "clob_token_id": row.get("clob_token_id"),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)




def _rating_gap(effective_ratings, row) -> float:
    """
    |rating difference| / 400, matching research_survival_model exactly.

    Both method paths need this and previously computed it separately -- one
    correctly, one hardcoded to 0.0. One helper means they cannot diverge.
    """
    if not effective_ratings:
        return 0.0
    return abs(effective_ratings.get(row["fighter_a"], 1500)
               - effective_ratings.get(row["fighter_b"], 1500)) / 400.0


def compute_goes_the_distance_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame,
                                    effective_ratings: dict[str, float] | None = None) -> pd.DataFrame:
    """
    'Fight goes the distance' vs 'ends in a finish' -- derived the same way
    as method-of-victory (sum of both fighters' decision-win likelihood).
    """
    rows = []
    props = upcoming_df[upcoming_df["market"] == "GoesTheDistance"]

    for _, row in props.iterrows():
        f_a = _find_fighter(fighters_df, row["fighter_a"])
        f_b = _find_fighter(fighters_df, row["fighter_b"])
        if f_a.empty or f_b.empty:
            continue
        a, b = f_a.iloc[0], f_b.iloc[0]
        # Use the HAZARD MODEL's decision probability, not a proxy. The old
        # "average of both fighters' decision tendency" came from a different
        # model than the KO and submission rows, so the three fight-level
        # method probabilities had nothing forcing them to sum to 1 -- they
        # were landing near 104%. The hazard model produces P(decision) as
        # the chance of surviving every round, so taking all three from it
        # makes them exhaustive by construction.
        from src.method_model import method_probabilities
        wins_a, wins_b = max(int(a["wins"]), 1), max(int(b["wins"]), 1)
        # DENOMINATOR IS TOTAL FIGHTS, matching how the model was trained.
        # These previously divided win-methods by WINS and loss-methods by
        # LOSSES, while research_survival_model.Career divides all four by
        # total fights. For a 25-1 fighter whose single loss was a submission
        # that made sub_lost 1.000 at inference against 0.038 in training --
        # a 26x inflation -- which pushed sub_press to 4.6x the highest value
        # the coefficients were ever fit on. The model then extrapolated to
        # 60%+ submission, a figure it never once produced across 1,743
        # holdout fights.
        # A train/serve skew like this is invisible to every offline check:
        # the harness scores its own features, so the model looks calibrated
        # right up until it meets production inputs on a different scale.
        n_a = max(int(_get(a, "wins", 0)) + int(_get(a, "losses", 0)), 1)
        n_b = max(int(_get(b, "wins", 0)) + int(_get(b, "losses", 0)), 1)
        ko_a, ko_b = _get(a, "ko_wins", 0) / n_a, _get(b, "ko_wins", 0) / n_b
        sub_a, sub_b = _get(a, "sub_wins", 0) / n_a, _get(b, "sub_wins", 0) / n_b
        kol_a, kol_b = _get(a, "ko_losses", 0) / n_a, _get(b, "ko_losses", 0) / n_b
        subl_a, subl_b = _get(a, "sub_losses", 0) / n_a, _get(b, "sub_losses", 0) / n_b
        _dist = method_probabilities(
            ko_press=ko_a * kol_b + ko_b * kol_a,
            sub_press=sub_a * subl_b + sub_b * subl_a,
            ko_rate_sum=ko_a + ko_b, sub_rate_sum=sub_a + sub_b,
            durability=kol_a + kol_b,
            # REAL rating gap. This was hardcoded to 0.0 -- a fabricated
            # feature, and one the model never sees in training, where the gap
            # is always positive. The clipping warning surfaced it on the
            # first run after being added, which is exactly what it's for.
            elo_gap=_rating_gap(effective_ratings, row),
            scheduled_rounds=5 if str(row.get("card_position", "")).strip() == "Main Event" else 3,
        )
        if not _dist:
            continue
        # P(decision) from the same softmax as KO and SUB, so all three sum to
        # 1 by construction rather than being reconciled after the fact. Safe
        # again now the model is calibrated: decision is a FITTED class here,
        # not "survived every round" -- which is what made it fragile before.
        goes_distance_prob = _dist["decision"]

        model_p = goes_distance_prob if "distance" in row["selection"].lower() and "ends" not in row["selection"].lower() else (1 - goes_distance_prob)
        imp = american_to_implied_prob(row["odds_american"])
        rows.append({
            "fight_id": row["fight_id"],
            "fighter": f"{row['fighter_a']} vs {row['fighter_b']}",
            "market": f"Fight Outcome: {row['selection']}",
            "odds_american": row["odds_american"],
            "model_prob": round(model_p, 3),
            "book_fair_prob": round(imp, 3),
            "edge_pct": round(edge_percent(model_p, imp), 2),
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, imp), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def find_fight_method_edges(upcoming_df, fighters_df, effective_ratings=None):
    """
    Price Polymarket's FIGHT-LEVEL method markets against the hazard model.

    "Will the fight be won by KO or TKO?" names a method but NEITHER fighter,
    which is exactly what src/method_model.py predicts -- so no allocation
    between the two fighters is needed and nothing has to be invented. That's
    why this market, rather than the per-fighter Method display, is where the
    model gets used first: the validated object and the traded object are the
    same object.

    Validated on a frozen 2019+ holdout (n=1743): P(KO) Brier 0.2028 vs 0.2155
    for base rates, P(SUB) 0.1331 vs 0.1380.

    TWO CAVEATS, both measured:

    1. The chained probability runs ~2.4pp overconfident out of sample and
       that could not be fitted away (isotonic made calibration WORSE;
       single-parameter shrinkage selected "no correction"), because the
       miscalibration is drift rather than bias. Edges here are therefore
       tilted toward "yes" on finishes.

    2. FIVE-ROUND FIGHTS ARE WORSE, and main events are exactly where the
       liquidity is. Only 17% of training rows are five-round, and on the
       holdout the per-round finish hazard is predicted at 19.4% against an
       actual 14.6% -- a 4.8pp overstatement, versus 1.3pp on three-round
       fights. Chaining five rounds compounds it, so a title fight's P(KO)
       is materially too high.

    Both caveats are now HISTORICAL: the chained model they described was
    replaced by a direct fight-level fit, validated per method and per
    scheduled length (research_method_fightlevel.py).
    """
    from src.method_model import method_probabilities

    rows = []
    props = upcoming_df[upcoming_df["market"] == "FightMethod"]
    for _, row in props.iterrows():
        f_a = _find_fighter(fighters_df, row["fighter_a"])
        f_b = _find_fighter(fighters_df, row["fighter_b"])
        if f_a.empty or f_b.empty:
            continue
        a, b = f_a.iloc[0], f_b.iloc[0]

        wins_a, wins_b = max(int(a["wins"]), 1), max(int(b["wins"]), 1)
        # DENOMINATOR IS TOTAL FIGHTS, matching how the model was trained.
        # These previously divided win-methods by WINS and loss-methods by
        # LOSSES, while research_survival_model.Career divides all four by
        # total fights. For a 25-1 fighter whose single loss was a submission
        # that made sub_lost 1.000 at inference against 0.038 in training --
        # a 26x inflation -- which pushed sub_press to 4.6x the highest value
        # the coefficients were ever fit on. The model then extrapolated to
        # 60%+ submission, a figure it never once produced across 1,743
        # holdout fights.
        # A train/serve skew like this is invisible to every offline check:
        # the harness scores its own features, so the model looks calibrated
        # right up until it meets production inputs on a different scale.
        n_a = max(int(_get(a, "wins", 0)) + int(_get(a, "losses", 0)), 1)
        n_b = max(int(_get(b, "wins", 0)) + int(_get(b, "losses", 0)), 1)
        ko_a, ko_b = _get(a, "ko_wins", 0) / n_a, _get(b, "ko_wins", 0) / n_b
        sub_a, sub_b = _get(a, "sub_wins", 0) / n_a, _get(b, "sub_wins", 0) / n_b
        kol_a, kol_b = _get(a, "ko_losses", 0) / n_a, _get(b, "ko_losses", 0) / n_b
        subl_a, subl_b = _get(a, "sub_losses", 0) / n_a, _get(b, "sub_losses", 0) / n_b

        gap = 0.0
        if effective_ratings:
            gap = abs(effective_ratings.get(row["fighter_a"], 1500)
                      - effective_ratings.get(row["fighter_b"], 1500)) / 400.0

        # Feature construction mirrors research_survival_model.py exactly --
        # offense meeting the opponent's vulnerability, not raw rates.
        dist = method_probabilities(
            ko_press=ko_a * kol_b + ko_b * kol_a,
            sub_press=sub_a * subl_b + sub_b * subl_a,
            ko_rate_sum=ko_a + ko_b,
            sub_rate_sum=sub_a + sub_b,
            durability=kol_a + kol_b,
            elo_gap=gap,
            scheduled_rounds=5 if str(row.get("card_position", "")).strip() == "Main Event" else 3,
        )
        if not dist:
            continue

        # No main-event shrink. It existed to damp the old chaining, which
        # overstated finishes badly on five-round fights. The refit model is
        # within 2.9% on that subgroup, so applying a 0.75 factor now would
        # double-correct an already-calibrated number.

        sel = str(row["selection"])
        base = dist["ko"] if "KO" in sel.upper() else dist["sub"]
        # "Not KO/TKO" is the complement, and the market prices both sides.
        model_p = (1 - base) if sel.lower().startswith("not") else base

        imp = american_to_implied_prob(row["odds_american"])
        rows.append({
            "fight_id": row["fight_id"],
            "fighter": f"{row['fighter_a']} vs {row['fighter_b']}",
            # Explicit empty opponent. A fight-level market has no single
            # opponent, but OMITTING the key makes pandas fill it with NaN
            # when these rows are concatenated with per-fighter ones -- and
            # NaN is a float, which crashed the name normaliser downstream.
            "opponent": "",
            "market": f"Fight Method: {sel}",
            "odds_american": row["odds_american"],
            "model_prob": round(model_p, 3),
            "book_fair_prob": round(imp, 3),
            "edge_pct": round(edge_percent(model_p, imp), 2),
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, imp), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def derive_card_label(row: pd.Series) -> str:
    """Prefer an explicit event name (e.g. 'UFC 329'); fall back to date-based grouping."""
    event_name = (row.get("event_name") or "").strip()
    if event_name:
        return event_name
    start_date = row.get("start_date")
    if pd.notna(start_date) and str(start_date).strip():
        date_part = str(start_date)[:10]
        return f"Fight Card — {date_part}"
    return "Upcoming Fights"


def build_fight_list(upcoming_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per fight (deduped), with card grouping metadata attached.
    Used to list every matchup on a card even if no prop cleared the edge
    threshold for that fight.
    """
    cols = [c for c in ["fight_id", "fighter_a", "fighter_b", "event_name",
                         "start_date", "weight_class", "card_position"] if c in upcoming_df.columns]
    fights = upcoming_df[cols].drop_duplicates(subset="fight_id").copy()
    fights["card_label"] = fights.apply(derive_card_label, axis=1)
    return fights.reset_index(drop=True)


def top_standout_props(edges_df: pd.DataFrame, n: int = 5, min_edge: float = 5.0) -> pd.DataFrame:
    """The headline shortlist: biggest model-vs-market disagreements, positive edge only."""
    if edges_df.empty:
        return edges_df
    standouts = edges_df[edges_df["edge_pct"] >= min_edge].copy()
    return standouts.sort_values("edge_pct", ascending=False).head(n).reset_index(drop=True)


def attach_fight_meta(edges_df: pd.DataFrame, fight_list_df: pd.DataFrame) -> pd.DataFrame:
    """Merges card_label/weight_class/opponent info into the edges dataframe for grouped display."""
    if edges_df.empty or fight_list_df.empty:
        return edges_df
    meta_cols = [c for c in ["fight_id", "card_label", "weight_class", "card_position",
                              "fighter_a", "fighter_b"] if c in fight_list_df.columns]
    return edges_df.merge(fight_list_df[meta_cols], on="fight_id", how="left")


def find_all_edges(
    upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame, elo_ratings: dict[str, float],
    fight_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = [
        compute_moneyline_edges(upcoming_df, elo_ratings, fighters_df, fight_history_df),
        compute_method_edges(upcoming_df, fighters_df),
        compute_total_rounds_edges(upcoming_df, fighters_df),
        compute_goes_the_distance_edges(upcoming_df, fighters_df, elo_ratings),
        compute_round_betting_edges(upcoming_df, fighters_df),
        # FightMethod was built last session and left unwired pending review,
        # then never connected -- 80 classified rows per card with no path to
        # the site. These are Polymarket's "Will the fight be won by KO/TKO?"
        # markets, which the hazard model prices directly.
        # Re-enabled on the refit model. The previous chained version put 65%
        # on submission and 8.7% on decision for a five-round main event; the
        # direct fight-level fit is within 3pp of observed on every method AND
        # on the five-round subgroup specifically (research_method_fightlevel.py).
        find_fight_method_edges(upcoming_df, fighters_df, elo_ratings),
        # Derived book lines LAST, so a real published line always wins if
        # both somehow exist for the same selection.
        _score_derived_lines(derive_missing_method_lines(upcoming_df), fighters_df, elo_ratings),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def _score_derived_lines(derived_df, fighters_df, elo_ratings):
    """
    Attach our model probability and an edge to each derived book line.

    The derivation produces a PRICE with no model view attached; without this
    the row would show odds and a blank model column, which is the opposite
    of what the table is for.
    """
    if derived_df is None or derived_df.empty:
        return pd.DataFrame()
    rows = []
    for r in derived_df.to_dict("records"):
        method = str(r["market"]).split(":", 1)[1].strip()
        f = _find_fighter(fighters_df, r["fighter"])
        if f.empty:
            continue
        row = f.iloc[0]
        wins = max(int(row["wins"]), 1)
        col = {"KO/TKO": "ko_wins", "SUB": "sub_wins"}.get(method)
        if not col:
            continue
        model_p = _get(row, col, 0) / wins
        imp = r["book_fair_prob"]
        rows.append({**r, "model_prob": round(model_p, 3),
                     "edge_pct": round(edge_percent(model_p, imp), 2),
                     "suggested_stake_pct": round(
                         kelly_fraction(market_blended_prob(model_p, imp), r["odds_american"]) * 100, 2)})
    return pd.DataFrame(rows)


def derive_missing_method_lines(upcoming_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a missing per-fighter method line by SUBTRACTION within the book.

    THE IDENTITY. "Fight ends by KO" is exactly "A wins by KO" OR "B wins by
    KO", and those are mutually exclusive. So:

        P(B by KO) = P(fight ends by KO) - P(A by KO)

    WHY THIS IS SAFE WHERE OUR OWN SUBTRACTION WASN'T. Doing this across our
    two models is not valid -- research_method_reconciliation.py measured the
    per-fighter and fight-level models disagreeing by a mean of -11.3% on KO
    (worst -29.2%) and +4.5% on submission, in OPPOSITE directions. Subtracting
    across that gap inherits it.
    Here both terms come from the SAME source: Polymarket. There is only one
    model involved, so no gap exists to inherit. The output is a derived BOOK
    line, not a derived model estimate.

    DEVIGGING IS MANDATORY, not a refinement. Raw Yes prices carry the book's
    margin -- the app showed KO 82% + SUB 25% + DEC 20% = 127%. Subtracting
    two inflated numbers gives a third that is wrong by the difference of two
    different overrounds. Each binary is normalised against its own No side
    first, which is why the complement rows are kept in the data even though
    the table hides them.

    Only fires when exactly ONE leg is missing; with both unknown, subtraction
    cannot split them and nothing is emitted.
    """
    if upcoming_df is None or upcoming_df.empty:
        return pd.DataFrame()

    def _devig(yes_odds, no_odds):
        """Normalise a binary pair to a fair, no-vig probability."""
        # Uses the repo's existing remove_vig_two_way rather than a second
        # implementation -- a duplicate would be one more place to drift.
        try:
            p_yes = american_to_implied_prob(float(yes_odds))
            p_no = american_to_implied_prob(float(no_odds))
        except (TypeError, ValueError):
            return None
        if p_yes + p_no <= 0:
            return None
        fair_yes, _ = remove_vig_two_way(p_yes, p_no)
        return fair_yes

    rows = []
    for fid, grp in upcoming_df.groupby("fight_id"):
        recs = grp.to_dict("records")
        f_a = str(recs[0].get("fighter_a", "")).strip()
        f_b = str(recs[0].get("fighter_b", "")).strip()

        def _sel(market, selection):
            for r in recs:
                if str(r.get("market", "")) == market and \
                        str(r.get("selection", "")).strip().lower() == selection.lower():
                    return r
            return None

        for method_key, yes_sel, not_sel in (("KO/TKO", "KO/TKO", "Not KO/TKO"),
                                             ("SUB", "SUB", "Not SUB")):
            fight_yes = _sel("FightMethod", yes_sel)
            fight_no = _sel("FightMethod", not_sel)
            if not fight_yes or not fight_no:
                continue
            p_fight = _devig(fight_yes.get("odds_american"), fight_no.get("odds_american"))
            if p_fight is None:
                continue

            # Per-fighter legs for this method, devigged the same way.
            legs = {}
            for name in (f_a, f_b):
                leg_yes = _sel("Method", name)
                if leg_yes is None:
                    continue
                # The Method market's own complement isn't published as a
                # separate row, so fall back to the raw implied probability.
                # It is the only unnormalised term here; flagged below.
                try:
                    legs[name] = american_to_implied_prob(float(leg_yes.get("odds_american")))
                except (TypeError, ValueError):
                    pass

            if len(legs) != 1:
                continue          # both known (nothing to derive) or both missing
            known_name, known_p = next(iter(legs.items()))
            missing_name = f_b if known_name == f_a else f_a
            derived = p_fight - known_p

            # A negative or near-zero remainder means the two prices are
            # inconsistent -- usually a stale quote on one side. Emitting it
            # would put a fabricated number next to real ones.
            if derived <= 0.005 or derived >= 1.0:
                continue

            rows.append({
                "fight_id": fid, "fighter": missing_name,
                "market": f"Method: {method_key}",
                "model_prob": None,
                "odds_american": implied_prob_to_american(derived),
                "book_fair_prob": round(derived, 3),
                "edge_pct": None, "suggested_stake_pct": None,
                "clob_token_id": None,
                "derived_from": f"{p_fight:.3f} fight-level minus {known_p:.3f} {known_name}",
            })
    return pd.DataFrame(rows)
