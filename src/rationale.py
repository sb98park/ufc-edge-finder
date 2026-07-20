"""
Turns a computed edge row into a plain-English explanation of what's
actually driving the model's number -- so a flagged prop isn't just a
mystery percentage, you can see the stat behind it and judge for yourself
whether it's a real signal or a model blind spot.
"""

import hashlib
import re

import pandas as pd

from src.odds_utils import format_american_odds


def _pick_variant(key: str, variants: list[str]) -> str:
    """
    Deterministically picks one of several phrasings for the same
    underlying fact, keyed on something stable (fighter names + market)
    rather than random -- the same fight/market always reads the same
    way on every site regeneration, but different fights or different
    lines land on different phrasing instead of all sharing one fixed
    template sentence with only the numbers swapped in.
    """
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]
from src.matchup_model import _get


def _fighter_stats(fighters_df: pd.DataFrame, name: str) -> dict | None:
    row = fighters_df[fighters_df["name"] == name]
    if row.empty:
        return None
    r = row.iloc[0]
    total_wins = max(int(r["wins"]), 1)
    total_losses = max(int(r["losses"]), 1) if r["losses"] else 0
    total_fights = max(int(r["wins"]) + int(r["losses"]), 1)
    method_data_known = any(
        col in r and pd.notna(r[col]) for col in ("ko_wins", "sub_wins", "dec_wins")
    )
    return {
        "win_pct": r["wins"] / total_fights,
        "finish_rate": (_get(r, "ko_wins", 0) + _get(r, "sub_wins", 0)) / total_wins,
        "method_data_known": method_data_known,
        "ko_rate": _get(r, "ko_wins", 0) / total_wins,
        "sub_rate": _get(r, "sub_wins", 0) / total_wins,
        "dec_rate": _get(r, "dec_wins", 0) / total_wins,
        "ko_loss_rate": (_get(r, "ko_losses", 0) / total_losses) if total_losses else 0.0,
        "sub_loss_rate": (_get(r, "sub_losses", 0) / total_losses) if total_losses else 0.0,
        "dec_loss_rate": (_get(r, "dec_losses", 0) / total_losses) if total_losses else 0.0,
        "reach_in": _get(r, "reach_in", 70),
        "wins": int(r["wins"]),
        "losses": int(r["losses"]),
        "weight_class": r["weight_class"],
        "first_round_finish_pct": float(r["first_round_finish_pct"]) if "first_round_finish_pct" in r and pd.notna(r["first_round_finish_pct"]) else None,
    }


from src.matchup_model import predict_matchup


def explain_moneyline(row: dict, fighters_df: pd.DataFrame) -> str:
    stats = _fighter_stats(fighters_df, row["fighter"])
    edge_dir = "higher" if row["edge_pct"] > 0 else "lower"
    base = (
        f"The model puts {row['fighter']}'s win probability at {row['model_prob']*100:.0f}%, "
        f"{edge_dir} than the market's {row['book_fair_prob']*100:.0f}% implied probability "
        f"at {format_american_odds(row['odds_american'])} ({row['edge_pct']:+.1f}% edge)."
    )

    opponent = row.get("opponent")
    if opponent:
        matchup = predict_matchup(row["fighter"], opponent, fighters_df, {})
        # predict_matchup needs effective_ratings for the base gap, but we don't
        # have that here -- just the style breakdown, which doesn't depend on it
        if matchup:
            drivers = []
            if abs(matchup["wrestling_adjustment"]) > 15:
                who = row["fighter"] if matchup["wrestling_adjustment"] > 0 else opponent
                drivers.append(f"{who}'s takedown accuracy vs. the opponent's takedown defense")
            if abs(matchup["striking_adjustment"]) > 10:
                who = row["fighter"] if matchup["striking_adjustment"] > 0 else opponent
                drivers.append(f"{who}'s striking accuracy edge")
            if abs(matchup["durability_adjustment"]) > 15:
                who = row["fighter"] if matchup["durability_adjustment"] > 0 else opponent
                drivers.append(f"{who} having been finished less often historically")
            if abs(matchup.get("submission_threat_adjustment", 0)) > 15:
                who = row["fighter"] if matchup["submission_threat_adjustment"] > 0 else opponent
                drivers.append(f"{who}'s real submission-finish rate")
            layoff_a = matchup.get("layoff_years_a")
            layoff_b = matchup.get("layoff_years_b")
            if layoff_a and layoff_a > 1.0:
                drivers.append(f"{row['fighter']} coming off a {layoff_a:.1f}-year layoff (ring rust risk)")
            if layoff_b and layoff_b > 1.0:
                drivers.append(f"{opponent} coming off a {layoff_b:.1f}-year layoff (ring rust risk)")

            if drivers:
                base += f" Biggest factors in that number: {', '.join(drivers)}."
            elif stats and stats["method_data_known"]:
                base += (
                    f" That's built on a {stats['wins']}-{stats['losses']} record "
                    f"({stats['win_pct']*100:.0f}% win rate) and a {stats['finish_rate']*100:.0f}% finish rate, "
                    f"with no major style, durability, or layoff mismatch pulling the number further."
                )
            elif stats:
                base += (
                    f" That's built on a {stats['wins']}-{stats['losses']} record "
                    f"({stats['win_pct']*100:.0f}% win rate) -- no tracked method-of-victory breakdown yet for a "
                    f"finish-rate read, with no major style, durability, or layoff mismatch pulling the number further."
                )
            return base

    if stats and stats["method_data_known"]:
        base += (
            f" That's built on a {stats['wins']}-{stats['losses']} record "
            f"({stats['win_pct']*100:.0f}% win rate) and a {stats['finish_rate']*100:.0f}% finish rate."
        )
    elif stats:
        base += (
            f" That's built on a {stats['wins']}-{stats['losses']} record "
            f"({stats['win_pct']*100:.0f}% win rate) -- no tracked method-of-victory breakdown yet for a finish-rate read."
        )
    return base


def explain_method(row: dict, fighters_df: pd.DataFrame) -> str:
    stats = _fighter_stats(fighters_df, row["fighter"])
    method = row["market"].replace("Method: ", "")
    base = (
        f"{row['fighter']} to win by {method} is priced at {format_american_odds(row['odds_american'])} "
        f"({row['book_fair_prob']*100:.0f}% implied), while the model estimates {row['model_prob']*100:.0f}% "
        f"({row['edge_pct']:+.1f}% edge)."
    )
    if not stats or not stats["method_data_known"]:
        return base

    rate_key = {"KO/TKO": "ko_rate", "SUB": "sub_rate", "DEC": "dec_rate"}.get(method)
    loss_key = {"KO/TKO": "ko_loss_rate", "SUB": "sub_loss_rate", "DEC": "dec_loss_rate"}.get(method)
    win_col = {"KO/TKO": "ko_wins", "SUB": "sub_wins", "DEC": "dec_wins"}.get(method)
    if not rate_key:
        return base
    own_rate = stats[rate_key]

    opponent = row.get("opponent")
    opp_stats = _fighter_stats(fighters_df, opponent) if opponent else None
    opp_vulnerability = opp_stats[loss_key] if opp_stats and opp_stats["method_data_known"] else None

    weight_class = stats.get("weight_class")
    divisional_rate = own_rate
    if weight_class is not None and win_col:
        div_group = fighters_df[fighters_df["weight_class"] == weight_class]
        total_div_wins = div_group["wins"].sum()
        if total_div_wins > 0:
            divisional_rate = div_group[win_col].sum() / total_div_wins

    div_gap = own_rate - divisional_rate
    method_lower = method.lower().replace("ko/tko", "KO/TKO").replace("sub", "submission").replace("dec", "decision")

    # Pick whichever angle is actually most distinctive about THIS matchup,
    # rather than always leading with the same blended-factors sentence --
    # different fights genuinely have different "why" depending on the data.

    if opp_vulnerability is not None and opp_vulnerability < 0.08 and opp_stats["losses"] >= 2:
        # opponent has essentially never lost this way -- worth naming directly as the tension
        detail = (
            f" Worth flagging directly: {opponent} has never lost by {method_lower} across "
            f"{opp_stats['losses']} career loss(es), even though {row['fighter']} has finished "
            f"{own_rate*100:.0f}% of wins that way -- the model still leans toward it, but this "
            f"specific matchup history is a real headwind on the pick."
        )
    elif opp_vulnerability is not None and opp_vulnerability >= 0.45:
        # opponent is genuinely vulnerable to this specific method -- lead with that
        detail = (
            f" {opponent} has gone down by {method_lower} in {opp_vulnerability*100:.0f}% of their "
            f"career losses -- a real, specific vulnerability this matchup plays into, on top of "
            f"{row['fighter']}'s own {own_rate*100:.0f}% career rate finishing fights that way."
        )
    elif abs(div_gap) >= 0.15 and weight_class:
        # fighter's own rate is well off the divisional norm -- that's the interesting part
        comparison = "well above" if div_gap > 0 else "well below"
        detail = (
            f" {row['fighter']}'s {own_rate*100:.0f}% career rate by {method_lower} runs {comparison} "
            f"the {divisional_rate*100:.0f}% baseline for {weight_class} -- a real outlier for the "
            f"division, not just a generic tendency."
        )
    elif stats["wins"] < 6:
        # small sample -- worth being upfront that this leans on limited data
        detail = (
            f" Built on a smaller sample ({stats['wins']} career wins), so {row['fighter']}'s "
            f"{own_rate*100:.0f}% rate by {method_lower} carries more uncertainty than a longer "
            f"track record would."
        )
    else:
        # nothing sharply distinctive -- fall back to the blended explanation, but vary the wording
        detail = (
            f" No single factor dominates here -- it's a blend of {row['fighter']}'s own "
            f"{own_rate*100:.0f}% career rate by {method_lower} and how {opponent or 'their opponent'} "
            f"has historically fared against that specific type of finish."
        )

    return base + detail


def explain_total_rounds(row: dict, fighters_df: pd.DataFrame) -> str:
    names = [n.strip() for n in row["fighter"].split(" vs ")]
    fighter_stats = []
    fast_finishers = []
    for name in names:
        s = _fighter_stats(fighters_df, name)
        if s:
            fighter_stats.append((name, s))
            if s["first_round_finish_pct"] and s["first_round_finish_pct"] >= 0.6:
                fast_finishers.append((name, s["first_round_finish_pct"]))

    base = (
        f"{row['market']} at {format_american_odds(row['odds_american'])} implies {row['book_fair_prob']*100:.0f}%, "
        f"vs. the model's {row['model_prob']*100:.0f}% ({row['edge_pct']:+.1f}% edge)."
    )

    is_over = "Over" in row["market"]
    line_match = re.search(r"(\d+\.\d+)", row["market"])
    line_value = float(line_match.group(1)) if line_match else None
    variant_key = f"{row['fighter']}|{row['market']}"

    if len(fighter_stats) == 2:
        (name_a, s_a), (name_b, s_b) = fighter_stats
        if not (s_a["method_data_known"] and s_b["method_data_known"]):
            variants = [
                " Not enough tracked method-of-victory history for one or both fighters here to say much "
                "about finish tendencies -- this one leans on the line itself more than either fighter's profile.",
                " Method-of-victory data is thin for at least one side of this matchup, so there's not a real "
                "finish-rate read to lean on here.",
            ]
            base += _pick_variant(variant_key, variants)
        else:
            rate_a, rate_b = s_a["finish_rate"], s_b["finish_rate"]
            avg_finish = (rate_a + rate_b) / 2
            gap = abs(rate_a - rate_b)

            if gap >= 0.30:
                higher_name, higher_rate = (name_a, rate_a) if rate_a > rate_b else (name_b, rate_b)
                lower_name, lower_rate = (name_b, rate_b) if rate_a > rate_b else (name_a, rate_a)
                hi_pct, lo_pct = f"{higher_rate*100:.0f}%", f"{lower_rate*100:.0f}%"
                if is_over:
                    variants = [
                        f" This one's lopsided on paper -- {higher_name} finishes {hi_pct} of wins, while {lower_name} "
                        f"sits at just {lo_pct}. For the Over, the hope is {lower_name} gets the win, or {higher_name} "
                        f"wins in a way that isn't their usual game.",
                        f" {higher_name} finishes {hi_pct} of career wins against {lower_name}'s {lo_pct} -- a real gap. "
                        f"Backing the Over here means betting against {higher_name}'s own history if they're the one who wins.",
                        f" The finish-rate split is stark: {hi_pct} for {higher_name}, {lo_pct} for {lower_name}. "
                        f"The Over needs either {lower_name} in the winner's circle, or {higher_name} to go off-script.",
                    ]
                else:
                    variants = [
                        f" This one's lopsided on paper -- {higher_name} finishes {hi_pct} of wins, while {lower_name} "
                        f"sits at just {lo_pct}. The Under really just needs {higher_name}'s normal finishing instinct "
                        f"to show up if they're the one who wins.",
                        f" {higher_name} closes the show {hi_pct} of the time, well clear of {lower_name}'s {lo_pct}. "
                        f"That gap is exactly what the Under is leaning on -- {higher_name} doing what {higher_name} usually does.",
                        f" A wide finish-rate gap here: {hi_pct} for {higher_name} vs. {lo_pct} for {lower_name}. "
                        f"If {higher_name} wins, history says this doesn't see the judges.",
                    ]
                base += _pick_variant(variant_key, variants)
            elif avg_finish >= 0.65:
                a_pct, b_pct = f"{rate_a*100:.0f}%", f"{rate_b*100:.0f}%"
                if is_over:
                    variants = [
                        f" Both fighters finish often ({a_pct} and {b_pct} of their wins) -- real risk for anyone leaning Over here.",
                        f" {a_pct} and {b_pct} finish rates between them -- two fighters who both like to end things early, "
                        f"which cuts hard against the Over.",
                        f" Neither side is shy about finishing -- {a_pct} and {b_pct} of career wins by stoppage. "
                        f"That's not a great backdrop for betting this one goes long.",
                    ]
                else:
                    variants = [
                        f" Both fighters finish often ({a_pct} and {b_pct} of their wins), which is exactly what the Under is pricing in.",
                        f" {a_pct} and {b_pct} finish rates -- two natural finishers in the same cage. "
                        f"The Under is the side that matches how these two usually fight.",
                        f" This is a finisher-vs-finisher matchup on paper ({a_pct} and {b_pct}), and that combination "
                        f"tends to end before the cards matter.",
                    ]
                base += _pick_variant(variant_key, variants)
            elif avg_finish <= 0.30:
                a_pct, b_pct = f"{rate_a*100:.0f}%", f"{rate_b*100:.0f}%"
                if is_over:
                    variants = [
                        f" Neither fighter finishes much ({a_pct} and {b_pct} of wins) -- this leans toward distance "
                        f"almost by default, favoring the Over.",
                        f" Low finish rates across the board here ({a_pct} and {b_pct}) -- there's no obvious source "
                        f"of an early stoppage, which is the Over's whole case.",
                        f" {a_pct} and {b_pct} -- neither fighter has much of a finishing history. Absent that, "
                        f"distance is the default outcome, and the Over is built for exactly that.",
                    ]
                else:
                    variants = [
                        f" Neither fighter finishes much ({a_pct} and {b_pct} of wins) -- the Under is fighting the tape here.",
                        f" Low finish rates on both sides ({a_pct} and {b_pct}) make the Under a tougher sell -- "
                        f"there's little history suggesting an early ending.",
                        f" {a_pct} and {b_pct} finish rates aren't promising for an early stoppage, which is exactly "
                        f"the case the Under needs to make.",
                    ]
                base += _pick_variant(variant_key, variants)
            else:
                avg_pct = f"{avg_finish*100:.0f}%"
                variants = [
                    f" A fairly even {avg_pct} combined finish rate between the two, nothing lopsided pushing this line either way.",
                    f" {avg_pct} combined finish rate, roughly split down the middle -- no clear stylistic push toward either side of this line.",
                    f" Nothing dramatic in the finish-rate profile here ({avg_pct} combined) -- this line comes down "
                    f"to the fight itself more than either fighter's general tendencies.",
                ]
                base += _pick_variant(variant_key, variants)

    if fast_finishers:
        at_the_line = line_value is not None and line_value <= 1.5
        for name, rate in fast_finishers:
            pct = f"{rate*100:.0f}%"
            if at_the_line:
                variants = [
                    f" Worth flagging: {pct} of {name}'s career wins have come in round 1 specifically -- directly on point at this line.",
                    f" {name} has finished {pct} of career wins in round 1 alone, which matters a lot at this exact number.",
                    f" One more thing: {pct} of {name}'s wins are first-round finishes -- about as relevant as a stat can be to this specific line.",
                ]
            else:
                variants = [
                    f" Worth flagging: {pct} of {name}'s career wins have come in round 1 -- part of a broader "
                    f"early-finish pattern, even if this specific line isn't about round 1 alone.",
                    f" {name} finishes {pct} of wins in round 1 alone -- a fast-starting fighter, though this "
                    f"line is asking about more than just the opening round.",
                    f" Also notable: {pct} of {name}'s career wins are first-round finishes, one piece of a "
                    f"broader tendency to end things early.",
                ]
            base += _pick_variant(f"{variant_key}|{name}", variants)
    return base


def explain_goes_the_distance(row: dict, fighters_df: pd.DataFrame) -> str:
    names = row["fighter"].split(" vs ")
    fighter_dec_info = []
    unknown_count = 0
    for name in names:
        s = _fighter_stats(fighters_df, name.strip())
        if s:
            if s["method_data_known"]:
                fighter_dec_info.append((name.strip(), s["dec_rate"]))
            else:
                unknown_count += 1

    base = (
        f"{row['market']} at {format_american_odds(row['odds_american'])} implies {row['book_fair_prob']*100:.0f}%, "
        f"vs. the model's {row['model_prob']*100:.0f}% ({row['edge_pct']:+.1f}% edge)."
    )
    is_distance = "Goes The Distance" in row["market"]
    variant_key = f"{row['fighter']}|{row['market']}"

    if unknown_count > 0:
        variants = [
            " Not enough tracked method-of-victory history for one or both fighters here to say much "
            "about decision tendencies -- this one leans on the line itself more than either fighter's profile.",
            " Method-of-victory data is thin for at least one side of this matchup, so there's not a real "
            "decision-rate read to lean on here.",
        ]
        base += _pick_variant(variant_key, variants)
    elif fighter_dec_info:
        avg_dec = sum(r for _, r in fighter_dec_info) / len(fighter_dec_info)
        gap = abs(fighter_dec_info[0][1] - fighter_dec_info[1][1]) if len(fighter_dec_info) == 2 else 0
        avg_pct = f"{avg_dec*100:.0f}%"

        if len(fighter_dec_info) == 2 and gap >= 0.30:
            higher_name, higher_rate = max(fighter_dec_info, key=lambda x: x[1])
            lower_name, lower_rate = min(fighter_dec_info, key=lambda x: x[1])
            hi_pct, lo_pct = f"{higher_rate*100:.0f}%", f"{lower_rate*100:.0f}%"
            if is_distance:
                variants = [
                    f" Split profile here -- {higher_name} goes to the cards {hi_pct} of the time, but {lower_name} "
                    f"only {lo_pct}. Going the distance really hinges on {lower_name}'s usual finishing instinct not showing up.",
                    f" {higher_name}'s decision rate ({hi_pct}) dwarfs {lower_name}'s ({lo_pct}). For this to go "
                    f"the distance, {lower_name} likely has to be the one doing the winning, and not in their usual style.",
                    f" There's a real gap in how these two get to a decision -- {hi_pct} for {higher_name}, {lo_pct} "
                    f"for {lower_name}. The distance case rests almost entirely on {lower_name} staying out of finishing mode.",
                ]
            else:
                variants = [
                    f" Split profile here -- {higher_name} goes to the cards {hi_pct} of the time, but {lower_name} "
                    f"only {lo_pct}. If {lower_name}'s normal game shows up, this ends before the scorecards matter.",
                    f" {higher_name}'s decision rate sits at {hi_pct}, well above {lower_name}'s {lo_pct}. That gap "
                    f"is the whole case for an early finish here, assuming {lower_name} is the one who wins.",
                    f" {lower_name} rarely sees a decision ({lo_pct} of career wins), a sharp contrast to "
                    f"{higher_name}'s {hi_pct}. That's the engine behind an early-finish lean.",
                ]
            base += _pick_variant(variant_key, variants)
        elif is_distance:
            variants = [
                f" Based on both fighters' career decision rate averaging {avg_pct}, which directly supports this going to the cards.",
                f" A combined {avg_pct} decision rate between them -- fighters who tend to end up in front of the "
                f"judges, which lines up with going the distance.",
                f" {avg_pct} combined decision rate says these are two fighters who usually get to hear the scorecards read.",
            ]
            base += _pick_variant(variant_key, variants)
        else:
            variants = [
                f" Based on both fighters' career decision rate averaging {avg_pct} -- the lower that number, the more room there is for an early finish.",
                f" A modest {avg_pct} combined decision rate leaves real room for this to end before the final bell.",
                f" With decisions averaging just {avg_pct} between them, there's more history pointing toward an "
                f"early finish than a trip to the scorecards.",
            ]
            base += _pick_variant(variant_key, variants)
    return base


def explain_favorite_pick(row: dict, fighters_df: pd.DataFrame) -> str:
    """
    A different voice than explain_edge on purpose. That function answers
    "what stat is driving the model's number" -- useful for auditing a
    prop, but it reads like a data citation, not a reason to actually put
    real money on something. This answers a different question: "why is
    THIS specific pick something worth sizing up on," which means
    weighing the opponent's exploitable weaknesses as much as the
    fighter's own strengths, and explicitly addressing why the current
    price still represents value rather than just restating the edge.
    Only fires for Moneyline, since that's what favorite picks are.
    """
    fighter = row["fighter"]
    opponent = row.get("opponent")
    stats = _fighter_stats(fighters_df, fighter)
    opp_stats = _fighter_stats(fighters_df, opponent) if opponent else None

    signals = []  # (magnitude, sentence)

    if opponent and stats and opp_stats:
        matchup = predict_matchup(fighter, opponent, fighters_df, {})
        if matchup:
            wrestling = matchup.get("wrestling_adjustment", 0)
            if abs(wrestling) > 8:
                if wrestling > 0:
                    signals.append((abs(wrestling), f"{fighter} has a real path to control the fight positionally -- {opponent}'s takedown defense doesn't match up well against it, and fights that go where {fighter} wants them tend to stay safe and one-sided"))
                else:
                    signals.append((abs(wrestling), f"{opponent} is genuinely live on the mat against {fighter}, which tempers the confidence here even with the number where it is"))

            striking = matchup.get("striking_adjustment", 0)
            if abs(striking) > 6:
                if striking > 0:
                    signals.append((abs(striking), f"on the feet, {fighter} lands at a clip {opponent} hasn't shown much answer for -- that's the kind of advantage that tends to compound over a full fight rather than fade"))
                else:
                    signals.append((abs(striking), f"{opponent} actually has the sharper striking profile here, which is a real headwind worth weighing against the pick"))

            durability = matchup.get("durability_adjustment", 0)
            # Finish-loss rate from a thin loss record is noise, not a
            # pattern -- an elite fighter with just 1-2 career losses can
            # have that rate swing to 0% or 100% purely from small-sample
            # variance, which would misleadingly read as a real signal.
            durability_sample_ok = stats["losses"] >= 3 and opp_stats["losses"] >= 3
            if abs(durability) > 8 and durability_sample_ok:
                if durability > 0:
                    signals.append((abs(durability), f"{opponent} has been finished at a notably higher rate than {fighter}, and durability gaps like that are exactly what tends to hold up bet after bet -- it's not a one-fight fluke, it's a pattern"))
                else:
                    signals.append((abs(durability), f"{fighter}'s own durability history is a genuine soft spot, which is worth knowing even if the model still leans this way"))

            submission_threat = matchup.get("submission_threat_adjustment", 0)
            # Same small-sample risk as durability above -- a fighter with
            # 2 career wins and 1 submission reads as a "50% sub rate"
            # that isn't a real pattern yet.
            sub_sample_ok = stats["wins"] >= 3 and opp_stats["wins"] >= 3
            if abs(submission_threat) > 8 and sub_sample_ok:
                if submission_threat > 0:
                    signals.append((abs(submission_threat), f"{fighter} finishes a real share of wins by submission, a live threat {opponent} has to respect anywhere the fight touches the mat"))
                else:
                    signals.append((abs(submission_threat), f"{opponent} carries a real submission-finish rate of their own, which is a live risk for {fighter} if this fight goes to the ground"))

            layoff_a, layoff_b = matchup.get("layoff_years_a") or 0, matchup.get("layoff_years_b") or 0
            layoff_gap = layoff_b - layoff_a
            # Compare relatively, not independently -- citing "opponent's
            # layoff hurts them" AND "fighter's own layoff hurts them" in
            # the same breath is contradictory when both are similar, and
            # only means something when there's a real gap between the two.
            if layoff_gap > 0.75 and layoff_b > 1.0:
                signals.append((layoff_gap * 8, f"{opponent} is coming off a {layoff_b:.1f}-year layoff, and ring rust after time away is one of the more reliable soft edges in this sport -- sharpness doesn't always come back on schedule"))
            elif layoff_gap < -0.75 and layoff_a > 1.0:
                signals.append((abs(layoff_gap) * 6, f"{fighter}'s own {layoff_a:.1f}-year layoff is a real variable working against this pick, not for it"))

            if matchup.get("age_cliff_flag_b"):
                signals.append((12, f"{opponent} is at the stage of their career where physical decline shows up fast in this sport -- age isn't just a number here, it's a fight-specific liability"))
            if matchup.get("age_cliff_flag_a"):
                signals.append((12, f"{fighter}'s own age curve is working against this pick, which tempers how much size makes sense even at a good price"))

    # Fallback / supplementary signal: raw finish-resistance if nothing
    # matchup-specific stood out, or to add a second data point alongside
    # a matchup-specific one.
    if stats and stats["losses"] >= 3:
        finish_resistance = 1 - (stats["ko_loss_rate"] + stats["sub_loss_rate"])
        if finish_resistance >= 0.75:
            signals.append((finish_resistance * 15, f"{fighter} simply doesn't get finished -- {int(finish_resistance*100)}% of their career losses have gone the distance, which caps the downside even on an off night"))

    signals.sort(key=lambda s: s[0], reverse=True)
    top = [s[1] for s in signals[:2]]

    odds_display = format_american_odds(row["odds_american"])
    prob_pct = round(row["model_prob"] * 100)

    if top:
        body = ". ".join(s[0].upper() + s[1:] for s in top) + "."
    else:
        # No sharp matchup-specific signal -- be honest that this is a
        # cleaner, less dramatic case rather than forcing a narrative.
        body = f"Nothing dramatic separates this matchup on paper -- it's a cleaner, lower-variance read on {fighter} rather than one built on a single standout factor."

    return (
        f"{body} At {odds_display}, that's real, bettable value on a pick the model has at {prob_pct}% -- "
        f"the kind of number worth sizing up on rather than treating as a coinflip."
    )


def explain_edge(row: dict, fighters_df: pd.DataFrame) -> str:
    if row["market"] == "Moneyline":
        return explain_moneyline(row, fighters_df)
    elif row["market"].startswith("Method"):
        return explain_method(row, fighters_df)
    elif row["market"].startswith("Total Rounds"):
        return explain_total_rounds(row, fighters_df)
    elif row["market"].startswith("Fight Outcome"):
        return explain_goes_the_distance(row, fighters_df)
    return f"{row['fighter']} — {row['market']}: {row['edge_pct']:+.1f}% edge vs. the market."
