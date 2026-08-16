"""
DraftKings doesn't always return every market, and The Odds API only has
moneylines for MMA at all. So instead of method-of-victory and round-total
insight disappearing whenever live odds don't cover it, this builds a
model-only projection straight from career stats -- always available,
clearly labeled as a projection rather than a live-market edge.
"""

import pandas as pd

from src.matchup_model import predict_matchup, classify_style, compute_divisional_method_priors, blend_method_probability, build_factor_badges, build_probability_waterfall, _get
from src.radar_chart import compute_radar_metrics, build_radar_chart_svg, build_percentile_index
from src.striking_profile import (build_zone_index, zone_profile, position_profile,
                                  fight_shape)

# A chip is a claim about a fighter's habits; too few tracked fights and it is
# a claim about one night. Higher than the radar's floor because a chip states
# the number outright rather than plotting it among five others.
MIN_CHIP_FIGHTS = 5
MAX_CHIPS = 3
from src.method_model import method_probabilities, reconcile_fighter_methods, method_given_win, finish_share_before


def _fighter_row(fighters_df: pd.DataFrame, name: str) -> pd.Series | None:
    match = fighters_df[fighters_df["name"] == name]
    return match.iloc[0] if not match.empty else None


def _method_vulnerability_blend(fighter_row: pd.Series, opponent_row: pd.Series, method: str, divisional_priors: dict) -> float:
    """
    Same prior-informed blend used in edge_finder.compute_method_edges, but
    usable without a live line.

    Every count read here (the fighter's own method-breakdown wins, and
    the opponent's method-breakdown losses) is explicitly NaN-checked
    rather than directly indexed -- a fighter can have a populated wins
    total but an unresearched method breakdown (0 wins by any specific
    method isn't the same claim as "we don't know"), and direct indexing
    let that NaN silently poison this fighter's whole rate calculation,
    the same failure shape fixed in compute_stats_rating.
    """
    total_wins = max(int(fighter_row["wins"]), 1)
    rate_map = {
        "KO/TKO": _get(fighter_row, "ko_wins", 0) / total_wins,
        "Submission": _get(fighter_row, "sub_wins", 0) / total_wins,
        "Decision": _get(fighter_row, "dec_wins", 0) / total_wins,
    }
    own_rate = rate_map[method]

    # divisional_priors keys use the short form ("SUB"/"DEC") from edge_finder
    method_key_map = {"KO/TKO": "KO/TKO", "Submission": "SUB", "Decision": "DEC"}
    divisional_prior = divisional_priors.get(fighter_row["weight_class"], {}).get(method_key_map[method], own_rate)

    opp_losses_raw = _get(opponent_row, "losses", 0)
    opp_losses = max(int(opp_losses_raw), 1) if opp_losses_raw else 0
    loss_col = {"KO/TKO": "ko_losses", "Submission": "sub_losses", "Decision": "dec_losses"}[method]
    opp_loss_count = _get(opponent_row, loss_col, 0)
    opp_vulnerability = opp_loss_count / opp_losses if opp_losses else own_rate

    return blend_method_probability(divisional_prior, own_rate, opp_vulnerability, total_wins)


def build_full_market_projection(
    fighter_a: str, fighter_b: str,
    fighters_df: pd.DataFrame,
    effective_ratings: dict[str, float],
    is_five_round: bool = False,
    fight_history_df: pd.DataFrame | None = None,
) -> dict | None:
    """
    Model-only projections for method-of-victory (both fighters, all three
    methods) and total rounds -- shown even when the live book doesn't
    happen to offer that market for this particular fight, clearly labeled
    as a projection rather than an odds comparison.

    is_five_round: main events (and title fights) are scheduled for 5 rounds
    instead of 3, meaning there's simply more fight left to cover -- the
    relevant round-total lines shift up (3.5/4.5 instead of 1.5/2.5), and a
    "goes the distance" outcome takes noticeably longer to happen.
    """
    row_a, row_b = _fighter_row(fighters_df, fighter_a), _fighter_row(fighters_df, fighter_b)
    if row_a is None or row_b is None:
        return None

    # fight_history_df matters: it enables the recent-form adjustment, and
    # edge_finder passes it. Without it the win probabilities constraining THIS
    # grid differ from the ones constraining the priced grid -- and since the
    # table shows priced KO rows beside model-only SUB/DEC rows, the mixture
    # didn't sum to the moneyline even though each grid was internally
    # coherent.
    matchup = predict_matchup(fighter_a, fighter_b, fighters_df, effective_ratings, fight_history_df)
    prob_a, prob_b = matchup["prob_a"], matchup["prob_b"]
    divisional_priors = compute_divisional_method_priors(fighters_df)

    # Fight-level method distribution, so the Fight props group can always
    # show all three answers even when a market for one of them doesn't exist.
    # Denominators are TOTAL FIGHTS, matching how the model was trained --
    # dividing win-methods by wins and loss-methods by losses is the skew that
    # produced a 60% submission probability on a fight the model had no
    # business being confident about.
    _md = None
    try:
        _na = max(int(_get(row_a, "wins", 0)) + int(_get(row_a, "losses", 0)), 1)
        _nb = max(int(_get(row_b, "wins", 0)) + int(_get(row_b, "losses", 0)), 1)
        _koa, _kob = _get(row_a, "ko_wins", 0) / _na, _get(row_b, "ko_wins", 0) / _nb
        _sua, _sub = _get(row_a, "sub_wins", 0) / _na, _get(row_b, "sub_wins", 0) / _nb
        _kla, _klb = _get(row_a, "ko_losses", 0) / _na, _get(row_b, "ko_losses", 0) / _nb
        _sla, _slb = _get(row_a, "sub_losses", 0) / _na, _get(row_b, "sub_losses", 0) / _nb
        _gap = abs(effective_ratings.get(fighter_a, 1500)
                   - effective_ratings.get(fighter_b, 1500)) / 400.0 if effective_ratings else 0.0
        _md = method_probabilities(
            ko_press=_koa * _klb + _kob * _kla,
            sub_press=_sua * _slb + _sub * _sla,
            ko_rate_sum=_koa + _kob, sub_rate_sum=_sua + _sub,
            durability=_kla + _klb, elo_gap=_gap,
            scheduled_rounds=5 if is_five_round else 3,
        )
        if _md:
            _md = {k: round(v, 3) for k, v in _md.items()}
    except (TypeError, ValueError, KeyError):
        _md = None

    # PER-FIGHTER METHODS, RECONCILED.
    #
    # This was `win_prob * method_given_win` with the three method_given_win
    # values computed independently -- so they didn't sum to 1 and each
    # fighter's methods overshot his own win probability. On a real card the
    # six mutually exclusive outcomes summed to 126.6%, and a submission row
    # showed 30.6% against a market implying 15.4%: an apparent 15-point edge
    # that was arithmetic, not signal.
    #
    # Two things are known and trustworthy here:
    #   rows -- each fighter's win probability, from the validated moneyline
    #   cols -- the fight's KO/SUB/DEC split, from the validated fight-level
    #           model (research_method_fightlevel.py)
    # The per-fighter METHOD PREFERENCE is the only unvalidated part, so it is
    # used as a seed and the two known margins are imposed on it by iterative
    # proportional fitting. The result matches both by construction: each
    # row sums to that fighter's win probability, each column to the
    # fight-level method probability, and the whole grid to 1.
    # FITTED seed, replacing the hand-weighted divisional blend. Same
    # denominators as training (total fights), same signed elo gap.
    seeds = []
    for row, opp_row, own_name, opp_name in (
        (row_a, row_b, fighter_a, fighter_b), (row_b, row_a, fighter_b, fighter_a)
    ):
        n_own = max(int(_get(row, "wins", 0)) + int(_get(row, "losses", 0)), 1)
        n_opp = max(int(_get(opp_row, "wins", 0)) + int(_get(opp_row, "losses", 0)), 1)
        gap = ((effective_ratings.get(own_name, 1500) - effective_ratings.get(opp_name, 1500)) / 400.0
               if effective_ratings else 0.0)
        seeds.append(method_given_win(
            own_ko_rate=_get(row, "ko_wins", 0) / n_own,
            own_sub_rate=_get(row, "sub_wins", 0) / n_own,
            opp_ko_lost=_get(opp_row, "ko_losses", 0) / n_opp,
            opp_sub_lost=_get(opp_row, "sub_losses", 0) / n_opp,
            elo_gap=gap,
        ))
    grid = reconcile_fighter_methods(seeds[0], seeds[1], prob_a, prob_b, _md)

    method_rows = []
    for i, name in enumerate((fighter_a, fighter_b)):
        for j, method in enumerate(("KO/TKO", "Submission", "Decision")):
            method_rows.append({
                "fighter": name, "market": f"Method: {method}",
                "model_prob": round(grid[i][j], 3),
            })

    combined_finish_rate = (
        (_get(row_a, "ko_wins", 0) + _get(row_a, "sub_wins", 0)) / max(int(row_a["wins"]), 1)
        + (_get(row_b, "ko_wins", 0) + _get(row_b, "sub_wins", 0)) / max(int(row_b["wins"]), 1)
    ) / 2

    first_round_rates = [
        float(r["first_round_finish_pct"]) for r in (row_a, row_b)
        if "first_round_finish_pct" in r and pd.notna(r["first_round_finish_pct"])
    ]
    combined_first_round_rate = sum(first_round_rates) / len(first_round_rates) if first_round_rates else combined_finish_rate * 0.5

    if is_five_round:
        # More scheduled rounds means more time for a finish to still
        # happen even after an early-rounds proxy (first_round_rate) misses
        # -- shift the "mid" line up to 3.5 and add a later 4.5 checkpoint
        # instead of 3-round-fight-calibrated 1.5/2.5.
        # DERIVED FROM P(finish), so these cannot contradict the method rows.
        # The old form was `combined_finish_rate + 0.15` -- an additive proxy
        # with nothing tying it to P(decision), which produced Under 4.5 at
        # 66.0% on a fight the method model gave a 50.2% decision probability.
        # Those two claims sum to 116% and cannot both hold: for a five-round
        # fight a decision IS Over 4.5, so P(Under 4.5) <= 1 - P(decision).
        # The share-of-finishes fractions below are NOT fitted -- they encode
        # that finishes are front-loaded and that almost any finish lands
        # before the 4.5 mark. They are estimates, and the identity they
        # enforce is what actually matters here.
        _finish = (1.0 - _md["decision"]) if _md else min(combined_finish_rate, 0.95)
        rounds_mid = _finish * finish_share_before(3.5, 5)
        rounds_late = _finish * finish_share_before(4.5, 5)
        rounds_rows = [
            {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Under 3.5", "model_prob": round(rounds_mid, 3)},
            {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Over 3.5", "model_prob": round(1 - rounds_mid, 3)},
            {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Under 4.5", "model_prob": round(rounds_late, 3)},
            {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Over 4.5", "model_prob": round(1 - rounds_late, 3)},
        ]
    else:
        # Same identity for three-round fights: a decision is Over 2.5, so
        # P(Under 2.5) <= 1 - P(decision). Derived from P(finish) for the same
        # reason as the five-round branch above -- the previous blend of two
        # career-rate proxies had no relationship to the method rows and could
        # exceed the finish probability outright.
        _finish3 = (1.0 - _md["decision"]) if _md else min(combined_finish_rate, 0.95)
        rounds_2_5 = _finish3 * finish_share_before(2.5, 3)
        _under_1_5 = _finish3 * finish_share_before(1.5, 3)
        rounds_rows = [
            {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Under 1.5", "model_prob": round(_under_1_5, 3)},
            {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Over 1.5", "model_prob": round(1 - _under_1_5, 3)},
            {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Under 2.5", "model_prob": round(rounds_2_5, 3)},
            {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Total Rounds Over 2.5", "model_prob": round(1 - rounds_2_5, 3)},
        ]

    dec_rate_a = _get(row_a, "dec_wins", 0) / max(int(row_a["wins"]), 1)
    dec_rate_b = _get(row_b, "dec_wins", 0) / max(int(row_b["wins"]), 1)
    goes_distance_prob = (dec_rate_a + dec_rate_b) / 2
    distance_rows = [
        {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Fight Outcome: Goes The Distance", "model_prob": round(goes_distance_prob, 3)},
        {"fighter": f"{fighter_a} vs {fighter_b}", "market": "Fight Outcome: Ends In Finish", "model_prob": round(1 - goes_distance_prob, 3)},
    ]

    return {"method_rows": method_rows, "rounds_rows": rounds_rows, "distance_rows": distance_rows}


# Below this many professional bouts on the THINNER side, a matchup cannot
# be labelled High Confidence however wide the rating gap looks. Four is the
# same floor build_effective_ratings uses before it will trust Elo at all.
MIN_RECORD_FOR_HIGH_CONFIDENCE = 4


def _confidence_label(favorite_prob: float, thinner_record: int | None = None) -> str:
    """
    thinner_record: professional bouts on whichever fighter has fewer. A
    prediction is only as confident as its weaker side, and the label is
    what drives Lock of the Week and the confidence tally on the card.

    WHY THE CAP. predict_matchup already computes an uncertainty band that
    scales with exactly this number -- a fighter with no record maxes it out
    -- but nothing downstream read it, so a matchup could carry the model's
    widest possible error bar and still be published as High Confidence.

    Terrance Chatman debuts on 2026-08-22 with no record we can find (he is
    5-1 professionally; ESPN has no page). Against a 7-0 opponent that
    produced an 85% pick and Lock of the Week -- the single most confident
    claim the site makes -- on a fight where one corner is entirely unknown.
    method_model flagged the same matchup as far outside its training range.

    This caps the LABEL, not the probability. The point estimate is the
    model's business and changing it needs a backtest; how loudly the site
    asserts that estimate is a presentation decision, and asserting it
    loudest where the data is thinnest is simply wrong.
    """
    if favorite_prob >= 0.75:
        if thinner_record is not None and thinner_record < MIN_RECORD_FOR_HIGH_CONFIDENCE:
            # Deliberately Medium rather than Low: the gap may well be real,
            # and burying it would be its own distortion. It just must not be
            # eligible to be the week's flagship pick.
            return "Medium Confidence"
        return "High Confidence"
    elif favorite_prob >= 0.60:
        return "Medium Confidence"
    else:
        return "Low Confidence"


def _ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th ... 11th/12th/13th are the exceptions."""
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def build_spotlight_chips(row_a: dict, row_b: dict, name_a: str, name_b: str,
                          pct_index: dict) -> list[dict]:
    """
    Surface only genuinely EXTREME measured stats, percentile-scored.

    WHY CHIPS AND NOT MORE AXES. The radar shows six things for everyone,
    which is right for comparing two fighters but wrong for saying "this one
    number is remarkable". A 96th-percentile knockdown rate deserves to be
    read as a sentence, not inferred from how far one spoke reaches.

    THE TAKEDOWN-ATTEMPTS-FACED CHIP EARNS ITS PLACE SPECIFICALLY. Curtis
    Blaydes reads 35% takedown defence, which looks damning for a decorated
    wrestler -- until you see that opponents attempted only 20 takedowns
    across 22 fights. Avoidance and defence are different quantities, and the
    defence rate cannot express the first; it answers "what happens when
    someone shoots", not "does anyone dare". That distinction misled me for a
    full round of analysis, so it is worth stating outright.

    Thresholds are deliberately strict. A chip that fires on half the card is
    decoration; these should be rare enough that seeing one means something.
    """
    # A LOW VALUE IS ONLY A LIABILITY WHERE LOW MEANS WEAK. Absorbing a lot
    # of damage is a genuine vulnerability. Not scoring knockdowns is a
    # STYLE -- Gillian Robertson is a submission grappler, and zero
    # knockdowns is that working as intended, not a deficiency. Flagging it
    # red told a bettor something false about her. So the bad tone is
    # restricted to the one measure where the bad end is unambiguous; every
    # other chip fires only when a fighter is exceptional in the good
    # direction.
    BAD_TONE_ALLOWED = {"sig_strikes_absorbed_per_fight"}

    CHIPS = [
        # column, label, direction, high-threshold, low-threshold, phrasing
        # SHORT PHRASING. These now sit inside the Striking Profile panel, so
        # "significant strikes" is implied by context -- spelling it out
        # pushed every chip onto two lines and made the row read as a wall.
        ("knockdowns_per_fight", "knockdowns per fight", True, 90, 10,
         "{v:.2f} knockdowns/fight"),
        ("sig_strikes_att_per_fight", "striking volume", True, 90, 10,
         "throws {v:.0f} strikes/fight"),
        ("sig_strikes_absorbed_per_fight", "damage taken", False, 90, 10,
         "absorbs {v:.0f} strikes/fight"),
        ("td_att_faced_per_fight", "takedown attempts faced", False, 90, 10,
         "faces {v:.1f} takedowns/fight"),
    ]
    out = []
    for row, name in ((row_a, name_a), (row_b, name_b)):
        if not row.get("espn_fights") or float(row.get("espn_fights") or 0) < MIN_CHIP_FIGHTS:
            continue
        for col, label, higher_better, hi, lo, phrasing in CHIPS:
            vals = pct_index.get(col)
            v = row.get(col)
            if not vals or v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v != v:
                continue
            # MIDRANK, not strictly-below. Counting only values BELOW v gave
            # every fighter with zero knockdowns "0th percentile" -- they are
            # tied at the floor, not uniquely worst, and on a roster where a
            # third have none that reads as a damning claim about a third of
            # the division. Splitting the ties puts the whole tied block at
            # its shared midpoint instead.
            below = sum(1 for x in vals if x < v)
            equal = sum(1 for x in vals if x == v)
            raw_pct = (below + equal / 2) / len(vals) * 100
            # Percentile as "how good", so a low damage-taken number ranks high.
            score = raw_pct if higher_better else 100 - raw_pct
            if score >= hi:
                tone = "good"
            elif score <= lo and col in BAD_TONE_ALLOWED:
                tone = "bad"
            else:
                continue
            out.append({
                "fighter": name,
                "side": "a" if name == name_a else "b",
                "text": phrasing.format(v=v),
                "percentile": int(round(raw_pct)),
                "percentile_label": _ordinal(int(round(raw_pct))),
                "tone": tone,
                "label": label,
            })
    # Cap it. Two fighters x four measures could produce eight chips, which
    # would be noise; keep the most extreme few.
    out.sort(key=lambda c: abs(c["percentile"] - 50), reverse=True)
    return out[:MAX_CHIPS]


def build_fight_preview(
    fighter_a: str, fighter_b: str,
    fighters_df: pd.DataFrame,
    effective_ratings: dict[str, float],
    is_five_round: bool = False,
    weight_class_history_df: pd.DataFrame | None = None,
    fight_weight_class: str | None = None,
    fight_history_df: pd.DataFrame | None = None,
) -> dict | None:
    row_a, row_b = _fighter_row(fighters_df, fighter_a), _fighter_row(fighters_df, fighter_b)
    if row_a is None or row_b is None:
        return None

    matchup = predict_matchup(
        fighter_a, fighter_b, fighters_df, effective_ratings,
        # fight_history_df was being DROPPED here. It enables the recent-form
        # adjustment, and compute_moneyline_edges passes it -- so the headline
        # pick and the Moneyline row in the markets table were running
        # DIFFERENT models on the same fight, with the headline the less
        # informed of the two.
        fight_history_df=fight_history_df,
        weight_class_history_df=weight_class_history_df, fight_weight_class=fight_weight_class,
    )
    prob_a = matchup["prob_a"]

    favorite, favorite_prob, underdog = (
        (fighter_a, prob_a, fighter_b) if prob_a >= 0.5 else (fighter_b, 1 - prob_a, fighter_a)
    )
    favorite_row = row_a if favorite == fighter_a else row_b
    underdog_row = row_b if favorite == fighter_a else row_a

    divisional_priors = compute_divisional_method_priors(fighters_df)

    # Fight-level method distribution for the Fight props group, so it can
    # show all three answers even when a market for one is unpriced.
    # Denominators are TOTAL FIGHTS, matching how the model was trained.
    _md = None
    try:
        _na = max(int(_get(row_a, "wins", 0)) + int(_get(row_a, "losses", 0)), 1)
        _nb = max(int(_get(row_b, "wins", 0)) + int(_get(row_b, "losses", 0)), 1)
        _koa, _kob = _get(row_a, "ko_wins", 0) / _na, _get(row_b, "ko_wins", 0) / _nb
        _sua, _sub = _get(row_a, "sub_wins", 0) / _na, _get(row_b, "sub_wins", 0) / _nb
        _kla, _klb = _get(row_a, "ko_losses", 0) / _na, _get(row_b, "ko_losses", 0) / _nb
        _sla, _slb = _get(row_a, "sub_losses", 0) / _na, _get(row_b, "sub_losses", 0) / _nb
        _gap = abs(effective_ratings.get(fighter_a, 1500)
                   - effective_ratings.get(fighter_b, 1500)) / 400.0 if effective_ratings else 0.0
        _md = method_probabilities(
            ko_press=_koa * _klb + _kob * _kla,
            sub_press=_sua * _slb + _sub * _sla,
            ko_rate_sum=_koa + _kob, sub_rate_sum=_sua + _sub,
            durability=_kla + _klb, elo_gap=_gap,
            scheduled_rounds=5 if is_five_round else 3,
        )
        if _md:
            _md = {k: round(v, 3) for k, v in _md.items()}
    except (TypeError, ValueError, KeyError):
        _md = None

    # FROM THE RECONCILED GRID -- the values the table actually shows.
    #
    # Reading the fitted seed instead was still wrong: reconciliation imposes
    # the moneyline and the fight-level method split on that seed, and doing
    # so can change WHICH method comes out highest. Seven fights on one card
    # disagreed between headline and table for exactly that reason.
    # The rule is the same one this codebase keeps relearning: display the
    # number that is displayed, not an input to it.
    def _seed_for(own_row, opp_row, own_name, opp_name):
        n_o = max(int(_get(own_row, "wins", 0)) + int(_get(own_row, "losses", 0)), 1)
        n_p = max(int(_get(opp_row, "wins", 0)) + int(_get(opp_row, "losses", 0)), 1)
        g = ((effective_ratings.get(own_name, 1500) - effective_ratings.get(opp_name, 1500)) / 400.0
             if effective_ratings else 0.0)
        return method_given_win(
            own_ko_rate=_get(own_row, "ko_wins", 0) / n_o,
            own_sub_rate=_get(own_row, "sub_wins", 0) / n_o,
            opp_ko_lost=_get(opp_row, "ko_losses", 0) / n_p,
            opp_sub_lost=_get(opp_row, "sub_losses", 0) / n_p,
            elo_gap=g,
        )

    _grid = reconcile_fighter_methods(
        _seed_for(row_a, row_b, fighter_a, fighter_b),
        _seed_for(row_b, row_a, fighter_b, fighter_a),
        prob_a, 1.0 - prob_a, _md,
    )
    _fav_idx = 0 if favorite == fighter_a else 1
    method_rates = dict(zip(["KO/TKO", "Submission", "Decision"], _grid[_fav_idx]))
    likely_method = max(method_rates, key=method_rates.get)

    combined_finish_rate = (
        (_get(row_a, "ko_wins", 0) + _get(row_a, "sub_wins", 0)) / max(int(row_a["wins"]), 1)
        + (_get(row_b, "ko_wins", 0) + _get(row_b, "sub_wins", 0)) / max(int(row_b["wins"]), 1)
    ) / 2
    rounds_lean = "Under" if combined_finish_rate >= 0.5 else "Over"

    style_note = ""
    if abs(matchup["wrestling_adjustment"]) > 15:
        stronger_wrestler = fighter_a if matchup["wrestling_adjustment"] > 0 else fighter_b
        style_note = (
            f" {stronger_wrestler}'s takedown accuracy against the opponent's takedown defense "
            f"gives a real wrestling-based edge here."
        )
    elif abs(matchup["durability_adjustment"]) > 15:
        more_durable = fighter_a if matchup["durability_adjustment"] > 0 else fighter_b
        style_note = f" {more_durable} has been finished less often historically, a durability factor working in their favor."
    elif abs(matchup.get("submission_threat_adjustment", 0)) > 15:
        submission_threat = fighter_a if matchup["submission_threat_adjustment"] > 0 else fighter_b
        style_note = f" {submission_threat} finishes a real share of wins by submission, a live threat the opponent has to respect everywhere the fight goes to the mat."

    layoff_note = ""
    for name, yrs in [(fighter_a, matchup["layoff_years_a"]), (fighter_b, matchup["layoff_years_b"])]:
        if yrs and yrs > 1.0:
            layoff_note += (
                f" {name} is returning from a {yrs:.1f}-year layoff, which carries real ring-rust risk "
                f"regardless of what their career numbers say."
            )

    reach_diff = _get(row_a, "reach_in", 70) - _get(row_b, "reach_in", 70)
    reach_note = ""
    if abs(reach_diff) >= 4:
        longer = fighter_a if reach_diff > 0 else fighter_b
        reach_note = f" {longer} also holds a notable reach advantage ({abs(reach_diff):.0f} inches)."

    fast_finisher_note = ""
    for name, row in [(fighter_a, row_a), (fighter_b, row_b)]:
        rate = row.get("first_round_finish_pct")
        if pd.notna(rate) and rate >= 0.6:
            fast_finisher_note += (
                f" {name} is a genuine round-1 threat — {rate*100:.0f}% of their career wins have come "
                f"before the first round even ends, which should pull any rounds/distance line lower "
                f"regardless of who's favored to win outright."
            )

    quick_return_note = ""
    for name, row, flagged in [(fighter_a, row_a, matchup["quick_return_flag_a"]), (fighter_b, row_b, matchup["quick_return_flag_b"])]:
        if flagged:
            method_label = row.get("last_fight_method", "finish")
            quick_return_note += (
                f" {name} is coming back quickly after being finished by {method_label} in their last fight — "
                f"a short turnaround from a finish carries real risk that career numbers alone won't show."
            )

    age_cliff_note = ""
    for name, row, flagged in [(fighter_a, row_a, matchup.get("age_cliff_flag_a")), (fighter_b, row_b, matchup.get("age_cliff_flag_b"))]:
        if flagged:
            age_cliff_note += (
                f" {name} ({int(row['age'])}) is past the typical age cliff for {row['weight_class']} — "
                f"this division tends to see a real decline past that point, independent of career record."
            )

    missed_weight_note = ""
    for name, row in [(fighter_a, row_a), (fighter_b, row_b)]:
        count = row.get("missed_weight_count")
        if pd.notna(count) and count > 0:
            missed_weight_note += f" {name} has missed weight {int(count)} time(s) before — a documented red flag for camp issues."

    five_round_note = " This is scheduled for 5 rounds, not the usual 3 — cardio and championship rounds matter here." if is_five_round else ""

    narrative = (
        f"Model favors {favorite} at {favorite_prob*100:.0f}% over {underdog} "
        f"({matchup['style_a']} vs. {matchup['style_b']} stylistically). "
        f"Path to victory most likely runs through {likely_method.lower()} "
        f"(projected at {method_rates[likely_method]*100:.0f}%, weighing {favorite.split()[-1]}'s own tendencies "
        f"against {underdog.split()[-1]}'s specific vulnerability profile). "
        f"Combined finish rate between both fighters sits at {combined_finish_rate*100:.0f}%, "
        f"leaning {rounds_lean.lower()} on total rounds."
        f"{style_note}{reach_note}{layoff_note}{quick_return_note}{age_cliff_note}{missed_weight_note}{five_round_note}{fast_finisher_note}"
    )

    if matchup.get("adjustment_capped"):
        narrative += (
            " Note: the situational factors here stack unusually high, hitting the model's "
            "sanity cap -- the final number is deliberately more conservative than the raw "
            "factor pile would suggest."
        )

    def _format_height(height_in):
        """72.0 -> '6\\'0\"' ; 75.0 -> '6\\'3\"'. Rounds to the nearest inch since
        ESPN's height field is whole inches in practice; None stays None so the
        template's existing '—' fallback keeps working unchanged."""
        if height_in is None:
            return None
        total_inches = round(height_in)
        feet, inches = divmod(total_inches, 12)
        return f"{feet}'{inches}\""

    def _fighter_card(name: str, row: pd.Series) -> dict:
        height_in_val = float(row["height_in"]) if pd.notna(row.get("height_in")) else None
        return {
            "name": name,
            "age": int(row["age"]) if pd.notna(row.get("age")) else None,
            "height_in": height_in_val,
            "height_display": _format_height(height_in_val),
            "reach_in": float(row["reach_in"]) if pd.notna(row.get("reach_in")) else None,
            "stance": row.get("stance") if pd.notna(row.get("stance")) else None,
            "style": classify_style(row),
            "record": f"{int(row['wins'])}-{int(row['losses'])}",
            "ko_wins": int(row["ko_wins"]) if pd.notna(row.get("ko_wins")) else None,
            "sub_wins": int(row["sub_wins"]) if pd.notna(row.get("sub_wins")) else None,
            "dec_wins": int(row["dec_wins"]) if pd.notna(row.get("dec_wins")) else None,
            "ko_losses": int(row["ko_losses"]) if pd.notna(row.get("ko_losses")) else None,
            "sub_losses": int(row["sub_losses"]) if pd.notna(row.get("sub_losses")) else None,
            "dec_losses": int(row["dec_losses"]) if pd.notna(row.get("dec_losses")) else None,
            "last_fight_date": row.get("last_fight_date") if pd.notna(row.get("last_fight_date")) else None,
            "last_fight_result": row.get("last_fight_result") if pd.notna(row.get("last_fight_result")) else None,
            "last_fight_method": row.get("last_fight_method") if pd.notna(row.get("last_fight_method")) else None,
            "last_fight_opponent": row.get("last_fight_opponent") if pd.notna(row.get("last_fight_opponent")) else None,
            "strike_accuracy_pct": float(row["strike_accuracy_pct"]) if pd.notna(row.get("strike_accuracy_pct")) else None,
            "td_defense_pct": float(row["td_defense_pct"]) if pd.notna(row.get("td_defense_pct")) else None,
            "td_accuracy_pct": float(row["td_accuracy_pct"]) if pd.notna(row.get("td_accuracy_pct")) else None,
        }

    comparison = {"a": _fighter_card(fighter_a, row_a), "b": _fighter_card(fighter_b, row_b)}

    # Percentile population = the WHOLE roster, not this card. Ranking a
    # fighter against the handful of people they happen to share a card with
    # would make the same fighter's chart change shape from week to week.
    # Cached on the DataFrame so the roster is scanned once per build rather
    # than twice per fight.
    pct_index = getattr(fighters_df, "_radar_pct_index", None)
    if pct_index is None:
        pct_index = build_percentile_index(fighters_df)
        try:
            fighters_df._radar_pct_index = pct_index
        except Exception:
            pass
    # Zone index, cached on the DataFrame like the radar's. Ranks a fighter
    # against the WHOLE roster, not the card -- otherwise the same fighter's
    # silhouette would change shade depending on who he happens to face.
    zone_index = getattr(fighters_df, "_zone_pct_index", None)
    if zone_index is None:
        zone_index = build_zone_index(fighters_df)
        try:
            fighters_df._zone_pct_index = zone_index
        except Exception:
            pass

    dict_a, dict_b = row_a.to_dict(), row_b.to_dict()
    pos_a = position_profile(dict_a, zone_index)
    pos_b = position_profile(dict_b, zone_index)
    striking_profile = {
        "a_lands": zone_profile(dict_a, zone_index, "strikes"),
        "a_absorbs": zone_profile(dict_a, zone_index, "absorbed"),
        "b_lands": zone_profile(dict_b, zone_index, "strikes"),
        "b_absorbs": zone_profile(dict_b, zone_index, "absorbed"),
        "shape": fight_shape(pos_a, pos_b, dict_a, dict_b),
    }
    # Omit the whole panel unless at least one fighter has a profile -- an
    # empty pair of silhouettes invites the reader to wonder what broke.
    if not (striking_profile["a_lands"] or striking_profile["b_lands"]):
        striking_profile = None

    radar_metrics_a = compute_radar_metrics(row_a.to_dict(), pct_index)
    radar_metrics_b = compute_radar_metrics(row_b.to_dict(), pct_index)
    radar_svg = build_radar_chart_svg(radar_metrics_a, radar_metrics_b, fighter_a, fighter_b)
    factor_badges = build_factor_badges(matchup)
    comparison["a"]["badges"] = factor_badges["a"]
    comparison["b"]["badges"] = factor_badges["b"]
    waterfall = build_probability_waterfall(matchup)

    return {
        "favorite": favorite,
        "favorite_prob": round(favorite_prob, 3),
        "method_distribution": _md,
        "underdog": underdog,
        "likely_method": likely_method,
        "likely_method_rate": round(method_rates[likely_method], 3),
        # Thinner record passed through so the label can refuse "High
        # Confidence" on a matchup where one corner has no career to read.
        # Taken from the matchup rather than recomputed, so the label and the
        # uncertainty band can never disagree about how thin the data is.
        "confidence_label": _confidence_label(favorite_prob, matchup.get("thinner_record")),
        "rounds_lean": rounds_lean,
        "combined_finish_rate": round(combined_finish_rate, 3),
        "style_a": matchup["style_a"],
        "style_b": matchup["style_b"],
        "narrative": narrative,
        "comparison": comparison,
        "radar_svg": radar_svg,
        "striking_profile": striking_profile,
        "spotlight_chips": build_spotlight_chips(row_a.to_dict(), row_b.to_dict(),
                                                 fighter_a, fighter_b, pct_index),
        "waterfall": waterfall,
    }
