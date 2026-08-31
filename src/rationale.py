"""
Turns a computed edge row into a plain-English explanation of what's
actually driving the model's number -- so a flagged prop isn't just a
mystery percentage, you can see the stat behind it and judge for yourself
whether it's a real signal or a model blind spot.
"""

import hashlib
import re

import pandas as pd

from src.odds_utils import format_american_odds, add_estimated_vig


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
from src.matchup_model import _get, divisional_prior_for, compute_divisional_method_priors, normalize_division


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
        # Prefer the imputed reach over the bare 70 so this agrees with
        # compute_stats_rating. Nothing consumes this key today; the point is
        # that if something starts to, it cannot silently disagree with the
        # rating it is supposed to be explaining.
        "reach_in": _get(r, "reach_in", None) if pd.notna(_get(r, "reach_in", None))
                    else _get(r, "reach_in_imputed", 70),
        "wins": int(r["wins"]),
        "losses": int(r["losses"]),
        "weight_class": r["weight_class"],
        "first_round_finish_pct": float(r["first_round_finish_pct"]) if "first_round_finish_pct" in r and pd.notna(r["first_round_finish_pct"]) else None,
        # THE PER-FIGHT RATE COLUMNS, which this function did not expose and
        # so no sentence in the file could ever name. They are the raw
        # material for saying WHY a factor fired instead of that it did:
        # strike accuracy, output and absorption, takedown volume, takedown
        # defense, control share, age. None is None-checked here rather than
        # defaulted, because a default silently becomes a comparison -- the
        # failure matchup_model documents at length.
        **{k: (float(r[k]) if k in r and pd.notna(r[k]) else None) for k in (
            "strike_accuracy_pct", "td_accuracy_pct", "td_defense_pct",
            "control_time_pct", "slpm", "sapm", "td_per_15", "age")},
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
            # THE SAME MECHANISM RENDERING explain_favorite_pick uses. This
            # built a comma-joined list of factor NAMES -- "X's striking
            # accuracy edge, Y having been finished less often historically"
            # -- which is the construction the owner quoted back as generic.
            # It could not be otherwise: there are 13 factors, so there were
            # 13 possible list items, and any two fights sharing a top factor
            # produced the same clause verbatim.
            #
            # Naming the two numbers behind the factor instead draws on
            # hundreds of stat pairs rather than 13 labels, and is checkable.
            opp_stats = _fighter_stats(fighters_df, opponent)
            drivers = []
            if stats and opp_stats:
                if abs(matchup["wrestling_adjustment"]) > 15:
                    a, b = (row["fighter"], opponent) if matchup["wrestling_adjustment"] > 0 else (opponent, row["fighter"])
                    sa, sb = (stats, opp_stats) if matchup["wrestling_adjustment"] > 0 else (opp_stats, stats)
                    drivers.append(_grappling_clause(a, b, sa, sb))
                if abs(matchup["striking_adjustment"]) > 10:
                    a, b = (row["fighter"], opponent) if matchup["striking_adjustment"] > 0 else (opponent, row["fighter"])
                    sa, sb = (stats, opp_stats) if matchup["striking_adjustment"] > 0 else (opp_stats, stats)
                    drivers.append(_striking_clause(a, b, sa, sb))
                if abs(matchup["durability_adjustment"]) > 15 and stats["losses"] >= 3 and opp_stats["losses"] >= 3:
                    a, b = (row["fighter"], opponent) if matchup["durability_adjustment"] > 0 else (opponent, row["fighter"])
                    sa, sb = (stats, opp_stats) if matchup["durability_adjustment"] > 0 else (opp_stats, stats)
                    drivers.append(_durability_clause(a, b, sa, sb))
                if abs(matchup.get("submission_threat_adjustment", 0)) > 15 and stats["wins"] >= 3 and opp_stats["wins"] >= 3:
                    a, b = (row["fighter"], opponent) if matchup["submission_threat_adjustment"] > 0 else (opponent, row["fighter"])
                    sa = stats if matchup["submission_threat_adjustment"] > 0 else opp_stats
                    drivers.append(_submission_clause(a, b, sa))
            layoff_a = matchup.get("layoff_years_a")
            layoff_b = matchup.get("layoff_years_b")
            if layoff_a and layoff_a > 1.0 and not (layoff_b and abs(layoff_a - layoff_b) < 0.75):
                drivers.append(f"{row['fighter']} has not fought in {layoff_a:.1f} years")
            if layoff_b and layoff_b > 1.0 and not (layoff_a and abs(layoff_a - layoff_b) < 0.75):
                drivers.append(f"{opponent} has not fought in {layoff_b:.1f} years")

            if drivers:
                # At most two, as complete sentences. Three stat pairs in one
                # muted 0.75rem block on a phone reads as a stat dump, and the
                # panel's own warning was that number-stuffing costs more
                # trust than it buys.
                base += " " + ". ".join(d[0].upper() + d[1:] for d in drivers[:2]) + "."
            # THE SAME PRICE ARITHMETIC THE FAVOURITE PICKS GET. This path
            # opened by restating the model number against the implied one and
            # then never said what the price actually requires -- so a reader
            # got the two figures and no statement of the gap between them.
            price = _price_clause(row)
            if price:
                base += f" {price[0].upper() + price[1:]}."
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

    if method == "FINISH":
        # "Wins by finish" = KO/TKO or SUB combined -- own_rate here is the
        # SAME finish_rate _fighter_stats already computes elsewhere (used
        # for the moneyline write-up too), not a new stat. Written
        # separately from the KO/SUB/DEC branch below since neither
        # rate_key/loss_key/win_col has a FINISH entry -- this is a genuine
        # different shape of prop (combined method), not just a 4th value
        # slotting into the same per-method lookup tables.
        own_rate = stats["finish_rate"]
        opponent = row.get("opponent")
        opp_stats = _fighter_stats(fighters_df, opponent) if opponent else None
        opp_vulnerability = (
            opp_stats["ko_loss_rate"] + opp_stats["sub_loss_rate"]
            if opp_stats and opp_stats["method_data_known"] else None
        )
        if opp_vulnerability is not None and opp_vulnerability >= 0.6:
            detail = (
                f" {opponent} has been finished (by any method) in {opp_vulnerability*100:.0f}% of "
                f"career losses -- combined with {row['fighter']}'s own {own_rate*100:.0f}% career "
                f"finish rate, this is a matchup where the fight ending early is the more likely shape, "
                f"not the exception."
            )
        elif own_rate >= 0.7:
            detail = (
                f" {row['fighter']} finishes {own_rate*100:.0f}% of career wins -- a genuine finisher, "
                f"which is what a safer 'wins by finish' pick leans on instead of betting a single "
                f"method (KO or Sub alone) and being wrong about which one."
            )
        else:
            detail = (
                f" A blend of {row['fighter']}'s {own_rate*100:.0f}% career finish rate and "
                f"{f'{opponent}' if opponent else 'the opponent'}'s own durability -- covering both "
                f"finish methods here is the safer read than picking KO or Submission alone."
            )
        return base + detail

    rate_key = {"KO/TKO": "ko_rate", "SUB": "sub_rate", "DEC": "dec_rate"}.get(method)
    loss_key = {"KO/TKO": "ko_loss_rate", "SUB": "sub_loss_rate", "DEC": "dec_loss_rate"}.get(method)
    win_col = {"KO/TKO": "ko_wins", "SUB": "sub_wins", "DEC": "dec_wins"}.get(method)
    if not rate_key:
        return base
    own_rate = stats[rate_key]

    opponent = row.get("opponent")
    opp_stats = _fighter_stats(fighters_df, opponent) if opponent else None
    opp_vulnerability = opp_stats[loss_key] if opp_stats and opp_stats["method_data_known"] else None

    # SAME DIVISIONAL BASELINE THE MODEL USES. This grouped on the RAW
    # weight_class string, so "Strawweight" and "Women's Strawweight" produced
    # two different baselines for one real division -- and it computed the
    # rate from the current roster's CAREER win totals, the survivorship-
    # biased source matchup_model documents as overstating finishes by 13-23
    # points. The model layer was converted to divisional_prior_for; the copy
    # layer that explains the model to the reader was not, so the prose could
    # contradict the number beside it.
    # THE CARD'S DIVISION, NOT THE FIGHTER'S. stats["weight_class"] comes from
    # fighters.csv, which is null for 114 of 310 roster entries -- and because
    # bool(float('nan')) is True, an `and weight_class` guard let NaN straight
    # through and rendered "baseline for nan" on 28 of 152 method calls. The
    # row carries the BOOKED division for this fight, which is populated, and
    # is also the more correct quantity: it is the division being fought in.
    #
    # Career rates are still the fighter's own, accumulated across whatever
    # divisions they fought in, so the prose says "the baseline for
    # Lightweight" rather than attributing the fighter's rate to that cohort.
    weight_class = normalize_division(row.get("weight_class")) or normalize_division(stats.get("weight_class"))
    divisional_rate = own_rate
    _prior_key = {"ko_wins": "KO/TKO", "sub_wins": "SUB", "dec_wins": "DEC"}.get(win_col)
    if _prior_key:
        divisional_rate = divisional_prior_for(
            compute_divisional_method_priors(fighters_df), weight_class, _prior_key, own_rate)

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
        # OPPONENT IS VULNERABLE TO THIS SPECIFIC METHOD. Rendered as counts,
        # because this branch produced the file's single most overclaiming
        # sentence: "has gone down by KO/TKO in 100% of their career losses"
        # fired on 31 of 124 method outputs, and the fighters it fired on
        # mostly had two or three losses. A rate of 100% on three defeats and
        # a rate of 100% on twelve are the same sentence and completely
        # different evidence; the count carries its own sample.
        opp_losses = opp_stats["losses"]
        opp_n = int(round(opp_vulnerability * opp_losses))
        own_n = int(round(own_rate * stats["wins"]))
        detail = (
            f" {opp_n} of {opponent}'s {_count(opp_losses, 'defeat', 'defeats')} have come by "
            f"{method_lower}, against {own_n} of {row['fighter']}'s "
            f"{_count(stats['wins'], 'win', 'wins')} finished that way."
        )
    elif abs(div_gap) >= 0.15 and weight_class:
        # fighter's own rate is well off the divisional norm -- that's the interesting part
        comparison = "well above" if div_gap > 0 else "well below"
        detail = (
            f" {int(round(own_rate * stats['wins']))} of {row['fighter']}'s "
            f"{_count(stats['wins'], 'win', 'wins')} have come by {method_lower}, which runs {comparison} "
            f"the {divisional_rate*100:.0f}% baseline for {weight_class}."
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


def explain_round_betting(row: dict, fighters_df: pd.DataFrame) -> str:
    """
    Round 1 only (see compute_round_betting_edges' docstring for why) --
    built on the one real signal available, first_round_finish_pct, rather
    than a full multi-factor blend like explain_method has. Explicitly
    frames the case this prop exists for: a heavy favorite whose moneyline
    price offers no real value, but who has a genuine fast-finish tendency
    worth a separate look.
    """
    stats = _fighter_stats(fighters_df, row["fighter"])
    base = (
        f"{row['fighter']} to win in Round 1 is priced at {format_american_odds(row['odds_american'])} "
        f"({row['book_fair_prob']*100:.0f}% implied), while the model estimates {row['model_prob']*100:.0f}% "
        f"({row['edge_pct']:+.1f}% edge)."
    )
    # HOISTED ABOVE THE GATE BELOW. first_round_finish_pct is populated for
    # well under half the roster, and the early return meant that whenever it
    # was missing this market shipped nothing but a restatement of the three
    # numbers already printed above the blurb -- one skeleton on 101 of 124
    # outputs. The tracked-bout count needs no such column.
    prof = _length_profile(row["fighter"])
    opp_prof = _length_profile(row.get("opponent")) if row.get("opponent") else None
    if prof:
        past, n = _past_line(prof, 300)
        inside = n - past
        detail = f" {inside} of {row['fighter']}'s {n} tracked fights have ended inside the first round"
        if opp_prof:
            o_past, o_n = _past_line(opp_prof, 300)
            detail += f", against {o_n - o_past} of {o_n} for {row['opponent']}"
        return base + detail + "."

    if not stats or stats["first_round_finish_pct"] is None:
        return base

    rate = stats["first_round_finish_pct"]
    # NOTE: deliberately NOT trying to detect "is this fighter a heavy
    # moneyline favorite" here -- row["book_fair_prob"] is this ROUND-1
    # PROP's own implied probability (always long odds, since winning
    # specifically in round 1 is inherently rarer than winning at all), not
    # the fighter's overall fight-win probability. This function doesn't
    # have clean access to that separate number without threading extra
    # context through explain_edge's otherwise-uniform (row, fighters_df)
    # dispatcher signature, so rather than guess with the wrong number,
    # this sticks to the one real signal that's actually available.
    # THE COUNT, FROM TRACKED BOUTS, BEFORE THE CAREER RATE. This market asks
    # whether a fight ends inside round one, and pit_stats holds the clock time
    # of every tracked bout, so the frequency is directly countable for both
    # corners. The career rate is a proxy and a percentage of an unstated
    # denominator; "three of his eleven tracked fights" is the same claim with
    # its sample attached.
    #
    # This branch previously emitted one skeleton on 101 of 124 outputs.
    if rate >= 0.5:
        detail = (
            f" {row['fighter']} has ended {rate*100:.0f}% of career wins in round 1 -- a genuine "
            f"fast-finish tendency worth a look here, especially for a fighter whose moneyline price "
            f"alone may not offer much value."
        )
    else:
        detail = (
            f" {row['fighter']}'s {rate*100:.0f}% career rate of round-1 finishes is the only real "
            f"signal behind this one -- no opponent-specific early-finish vulnerability is tracked yet, "
            f"so treat this as a lighter-conviction read than the method-of-victory props."
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

    # EMPIRICAL FIRST, PROXY SECOND. Everything below this block explains a
    # rounds line in terms of career FINISH RATE, which is a proxy for fight
    # length. When both fighters have enough tracked bouts, the actual
    # frequency of passing this line's clock time is available and is the
    # thing being priced -- so it leads, and the finish-rate branches become
    # the fallback they should always have been.
    #
    # It also breaks the shape monoculture on its own: the old branches
    # produced ONE sentence skeleton on 123 of 124 outputs, because every
    # variant said the same thing about the same statistic. Two counts and
    # two denominators differ for every pairing and every line.
    if len(names) == 2 and line_value is not None:
        emp = _length_clause(names[0], names[1], line_value, is_over)
        if emp:
            out = f"{base} {emp[0].upper() + emp[1:]}."
            dur = _duration_clause(names[0], names[1])
            if dur:
                out += f" {dur[0].upper() + dur[1:]}."
            return out

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

    # THE DISTANCE MARKET IS A LENGTH MARKET. Explaining it by career decision
    # RATE is a proxy for the question "do this pair's fights reach the cards";
    # the tracked bouts answer it directly. A three-round fight goes the
    # distance at 900 seconds and a five-rounder at 1500, and this market is
    # overwhelmingly the former -- so the count below is against the round-3
    # bell, and the clause says "past that mark" rather than naming a round,
    # which keeps it true for either format.
    #
    # Leads for the same reason as in explain_total_rounds: the old branches
    # emitted one skeleton on 46 of 124 outputs and three in total.
    if len(names) == 2:
        emp = _length_clause(names[0].strip(), names[1].strip(), 2.5, is_distance)
        if emp:
            out = f"{base} {emp[0].upper() + emp[1:]}."
            dur = _duration_clause(names[0].strip(), names[1].strip())
            if dur:
                out += f" {dur[0].upper() + dur[1:]}."
            return out

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

    # (magnitude, sentence, sign) -- sign is +1 when the fact argues FOR the
    # pick and -1 when it argues against. Previously untracked, which is how
    # 9 of 76 blurbs came to consist entirely of reasons the pick might lose
    # and then close by calling it bettable value.
    signals = []
    # Facts that cleared the magnitude threshold but were blocked by a
    # sample-size guard. Kept so the no-signal fallback can say WHY it is
    # quiet instead of claiming the fight is featureless.
    suppressed = []

    if opponent and stats and opp_stats:
        matchup = predict_matchup(fighter, opponent, fighters_df, {})
        if matchup:
            wrestling = matchup.get("wrestling_adjustment", 0)
            if abs(wrestling) > 8:
                if wrestling > 0:
                    signals.append((abs(wrestling), _grappling_clause(fighter, opponent, stats, opp_stats), +1, "own_grappling"))
                else:
                    signals.append((abs(wrestling), f"{opponent} is genuinely live on the mat against {fighter}, which tempers the confidence here even with the number where it is", -1, "opp_grappling"))

            striking = matchup.get("striking_adjustment", 0)
            if abs(striking) > 6:
                if striking > 0:
                    signals.append((abs(striking), _striking_clause(fighter, opponent, stats, opp_stats), +1, "own_striking"))
                else:
                    signals.append((abs(striking), f"{opponent} actually has the sharper striking profile here, which is a real headwind worth weighing against the pick", -1, "opp_striking"))

            durability = matchup.get("durability_adjustment", 0)
            # Finish-loss rate from a thin loss record is noise, not a
            # pattern -- an elite fighter with just 1-2 career losses can
            # have that rate swing to 0% or 100% purely from small-sample
            # variance, which would misleadingly read as a real signal.
            durability_sample_ok = stats["losses"] >= 3 and opp_stats["losses"] >= 3
            if abs(durability) > 8:
                if not durability_sample_ok:
                    thin = fighter if stats["losses"] < 3 else opponent
                    thin_n = min(stats["losses"], opp_stats["losses"])
                    suppressed.append(f"the model sees a durability gap here, but {thin} has only {_count(thin_n, 'career loss', 'career losses')} on record, which is too thin to read as a pattern")
                elif durability > 0:
                    signals.append((abs(durability), _durability_clause(fighter, opponent, stats, opp_stats), +1, "opp_chin"))
                else:
                    signals.append((abs(durability), f"{fighter}'s own durability history is a genuine soft spot, which is worth knowing even if the model still leans this way", -1, "own_chin"))

            submission_threat = matchup.get("submission_threat_adjustment", 0)
            # Same small-sample risk as durability above -- a fighter with
            # 2 career wins and 1 submission reads as a "50% sub rate"
            # that isn't a real pattern yet.
            sub_sample_ok = stats["wins"] >= 3 and opp_stats["wins"] >= 3
            if abs(submission_threat) > 8:
                if not sub_sample_ok:
                    suppressed.append(f"there is a submission-threat gap in the numbers, but on {_count(min(stats['wins'], opp_stats['wins']), 'career win', 'career wins')} it is not yet a pattern")
                elif submission_threat > 0:
                    signals.append((abs(submission_threat), _submission_clause(fighter, opponent, stats), +1, "own_subs"))
                else:
                    signals.append((abs(submission_threat), f"{opponent} carries a real submission-finish rate of their own, which is a live risk for {fighter} if this fight goes to the ground", -1, "opp_subs"))

            layoff_a, layoff_b = matchup.get("layoff_years_a") or 0, matchup.get("layoff_years_b") or 0
            layoff_gap = layoff_b - layoff_a
            # Compare relatively, not independently -- citing "opponent's
            # layoff hurts them" AND "fighter's own layoff hurts them" in
            # the same breath is contradictory when both are similar, and
            # only means something when there's a real gap between the two.
            if layoff_gap > 0.75 and layoff_b > 1.0:
                signals.append((layoff_gap * 8, f"{opponent} has not fought in {layoff_b:.1f} years, against {layoff_a:.1f} for {fighter}", +1, "opp_layoff"))
            elif layoff_gap < -0.75 and layoff_a > 1.0:
                signals.append((abs(layoff_gap) * 6, f"{fighter}'s own {layoff_a:.1f}-year layoff is a real variable working against this pick, not for it", -1, "own_layoff"))

            if matchup.get("age_cliff_flag_b"):
                age_b = opp_stats.get("age")
                signals.append((12, f"{opponent} is {int(age_b)} and past the age this division's fighters typically start declining" if age_b else f"{opponent} is past the age this division's fighters typically start declining", +1, "opp_age"))
            if matchup.get("age_cliff_flag_a"):
                age_a = stats.get("age")
                signals.append((12, f"{fighter}'s own age curve is working against this pick" + (f" -- {int(age_a)}, past the divisional decline point" if age_a else ""), -1, "own_age"))

    # Supplementary: raw finish-resistance, as a COUNT rather than a rate.
    # "100% of their career losses" on four losses is the single most
    # overclaiming construction this file produced; the honest version is
    # also the more concrete one.
    if stats and stats["losses"] >= 3:
        finish_resistance = 1 - (stats["ko_loss_rate"] + stats["sub_loss_rate"])
        if finish_resistance >= 0.75:
            went_long = int(round(finish_resistance * stats["losses"]))
            signals.append((finish_resistance * 15, f"{fighter} has been stopped {_count(stats['losses'] - went_long, 'time', 'times')} in {_count(stats['losses'], 'defeat', 'defeats')}", +1, "own_chin_good"))

    # SIGN-PARTITIONED, not top-2-by-magnitude. Taking the two largest
    # regardless of direction is what produced blurbs whose every sentence
    # argued against the pick. One reason for, one reason against, and the
    # counterargument is included precisely BECAUSE the product's credibility
    # is the asset -- a pick that names what would beat it cannot read as a tout.
    fors = sorted([s for s in signals if s[2] > 0], key=lambda s: s[0], reverse=True)
    againsts = sorted([s for s in signals if s[2] < 0], key=lambda s: s[0], reverse=True)
    # THE NAMED RISK, exported so it can be stored with the pick. Recovering
    # it after the fight would mean regenerating this blurb from ratings that
    # already absorbed the result, which is the contamination the frozen-pick
    # restore exists to prevent -- so it has to be captured now or not at all.
    row["pick_falsifier"] = againsts[0][3] if againsts else ""
    top = [fors[0][1]] if fors else []
    if againsts:
        top.append(againsts[0][1])
    elif len(fors) > 1:
        top.append(fors[1][1])

    odds_display = format_american_odds(row["odds_american"])
    prob_pct = round(row["model_prob"] * 100)

    if top:
        body = ". ".join(s[0].upper() + s[1:] for s in top) + "."
    elif suppressed:
        # A factor cleared threshold and a sample guard blocked it. Saying
        # "nothing dramatic separates this matchup" over a suppressed signal
        # is false: missing evidence is MORE uncertainty, not less.
        body = suppressed[0][0].upper() + suppressed[0][1:] + "."
    elif opp_stats is None or _thin_profile(opp_stats):
        body = f"There is not much tracked history on {opponent} to model against, so this number leans on {fighter}'s own record more than on the matchup."
    else:
        body = f"No single factor separates these two by much -- the model's edge here is the accumulation of small ones rather than anything that stands out."

    close = _close(row, prob_pct, odds_display, bool(againsts) and not fors)
    price = _price_clause(row)
    if price:
        close += f" {price[0].upper() + price[1:]}."
    return f"{body} {close}"


# Below this many observations, a percentage is a story about a small number
# and a count is the honest form. "100% of their losses" on four losses reads as
# a law; "stopped once in four defeats" reads as what it is.
_COUNT_WORDS = {0: "no", 1: "once", 2: "twice"}


def _count(n: int, singular: str, plural: str) -> str:
    """'3 career losses', but 'once' / 'twice' where English prefers it."""
    n = int(n)
    if singular in ("time",):
        return _COUNT_WORDS.get(n, f"{n} times")
    return f"{n} {singular if n == 1 else plural}"


def _thin_profile(st: dict) -> bool:
    """True when three or more advanced stats are missing for this fighter."""
    return sum(1 for k in ("strike_accuracy_pct", "td_accuracy_pct", "td_defense_pct",
                           "control_time_pct", "slpm") if st.get(k) is None) >= 3


# CLOSERS, BANDED ON THE DISPLAYED INTEGER PERCENT. Banding on the raw float
# would let a blurb rewrite itself between two rebuilds that show the reader
# the same number; banding on what is printed means the text can only change
# when the visible figure does.
#
# The single previous closer asserted "real, bettable value... worth sizing up
# on rather than treating as a coinflip" on all 76 picks -- including 25 under
# 55% and one at 50.2%, where it is simply false. Confidence now scales with
# the number, and the sub-55 band is not permitted to use the vocabulary of
# conviction at all.
_CLOSERS = {
    "coinflip": [
        "At {odds} the model has this at {pct}%, which is a lean rather than a read.",
        "The model lands on {pct}% here. That is close enough to even that the price matters more than the pick.",
        "{pct}% at {odds} -- the model has a side, not a strong one.",
        "This is one of the model's closest calls: {pct}%, priced at {odds}.",
        "At {pct}% the model is barely off a coin flip, and {odds} should be read in that light.",
    ],
    "lean": [
        "That puts the model at {pct}% against a price of {odds}.",
        "The model reads it {pct}%; {odds} is the number to weigh that against.",
        "{pct}% at {odds}, which is a real lean without being a standout.",
        "At {odds}, the model's {pct}% is a modest but genuine disagreement with the market.",
    ],
    "solid": [
        "The model settles at {pct}%, priced {odds}.",
        "{pct}% at {odds} -- a clear side, and the reasoning above is most of it.",
        "That reasoning gets the model to {pct}% against {odds}.",
        "At {odds} the model's {pct}% reflects a fight it thinks it understands.",
    ],
    "strong": [
        "The model has this at {pct}%, one of its firmer reads on the card, at {odds}.",
        "{pct}% at {odds}. The model is not close to neutral here.",
        "That combination gets the model to {pct}% -- priced at {odds}.",
        "At {odds} the model's {pct}% is about as far from a coin flip as it gets on this card.",
    ],
}

# When every rendered signal argues AGAINST the pick, no band may claim
# conviction. This is asserted rather than left to the band, because the
# failure it prevents -- a blurb reasoning its way to a loss and then calling
# itself value -- fired on 32 of 76 picks.
_CLOSERS_HEDGED = [
    "The model still lands on {fighter} at {pct}%, priced {odds}, but the case above is the case against.",
    "Those are reasons for caution; the model comes out at {pct}% anyway, at {odds}.",
    "That is the argument against, and the model still reads {pct}% at {odds}.",
]


def _close(row: dict, prob_pct: int, odds_display: str, all_against: bool) -> str:
    band = ("coinflip" if prob_pct < 55 else
            "lean" if prob_pct < 65 else
            "solid" if prob_pct < 75 else "strong")
    pool = _CLOSERS_HEDGED if all_against else _CLOSERS[band]
    key = f"{row.get('fighter')}|{row.get('opponent')}|{row.get('market')}|{band}|{prob_pct}"
    return _pick_variant(key, pool).format(pct=prob_pct, odds=odds_display, fighter=row.get("fighter"))



# ---------------------------------------------------------------------------
# MECHANISM CLAUSES.
#
# The file shipped 13 body templates because the model has 13 factors, and
# each template named its factor: "has a real path to control the fight
# positionally", "lands at a clip the opponent hasn't shown much answer for".
# Two fights whose top factor matched therefore read identically, and no
# amount of extra phrasings fixes that -- three paraphrases of a sentence
# about the same abstraction is one monoculture at a third the density.
#
# These render the TWO OPPOSING NUMBERS that produced the factor instead. The
# combinatorics change completely: there are 13 factors and hundreds of stat
# pairs, so specificity comes from the fight's own data rather than from a
# thesaurus. It is also more useful -- "3.26 takedowns per fifteen minutes
# against 0.00" is checkable, and "a real path to control the fight" is not.
#
# Every clause degrades to its qualitative form when the numbers are absent.
# That path is not a fallback to be tolerated: 55 of 310 roster fighters carry
# a scraped rate rather than a computed one, and some carry neither.
# ---------------------------------------------------------------------------

# NO GENDERED PRONOUNS. Roughly a fifth of the roster is women, the copy is
# generated from one set of templates for everyone, and the first draft of
# these clauses shipped "he is the more accurate of the two" onto Denise
# Gomes. Name the fighter, or use "their" -- never infer a pronoun from a
# name or from the division.
def _num(v, fmt="{:.0f}"):
    return None if v is None else fmt.format(v)


def _grappling_clause(fighter, opponent, st, opp) -> str:
    td_f, tdd_o = st.get("td_per_15"), opp.get("td_defense_pct")
    ctrl_f, ctrl_o = st.get("control_time_pct"), opp.get("control_time_pct")
    if td_f is not None and tdd_o is not None and td_f >= 0.5:
        base = f"{fighter} lands {td_f:.1f} takedowns per fifteen minutes and {opponent} stops {tdd_o:.0f}% of what comes at them"
        rank = (_cohort_note(fighter, "td_per_15", True,
                             "the most active takedown threat on this card",
                             "one of the most active takedown threats on this card")
                or _cohort_note(opponent, "td_defense_pct", False,
                                "the leakiest takedown defense on this card",
                                "one of the leakier takedown defenses on this card"))
        if rank:
            base += f" -- {rank}"
        if ctrl_f is not None and ctrl_o is not None and ctrl_f - ctrl_o > 8:
            out = base + f", and the control time runs the same way -- {ctrl_f:.0f}% of fight minutes to {ctrl_o:.0f}%"
            rank = _cohort_note(fighter, "control_time_pct", True,
                                "more than anyone else booked on this card",
                                "one of the highest control shares on this card")
            return out + (f", {rank}" if rank else "")
        return base
    if ctrl_f is not None and ctrl_o is not None and ctrl_f - ctrl_o > 8:
        return f"{fighter} has held position for {ctrl_f:.0f}% of all fight minutes against {ctrl_o:.0f}% for {opponent}, and this is a matchup where that tends to decide things"
    return f"{fighter} has a real path to control where this fight happens, and {opponent}'s takedown defense is the weaker half of that exchange"


def _striking_clause(fighter, opponent, st, opp) -> str:
    acc_f, acc_o = st.get("strike_accuracy_pct"), opp.get("strike_accuracy_pct")
    slpm_f, sapm_o = st.get("slpm"), opp.get("sapm")
    net_f = None if st.get("slpm") is None or st.get("sapm") is None else st["slpm"] - st["sapm"]
    net_o = None if opp.get("slpm") is None or opp.get("sapm") is None else opp["slpm"] - opp["sapm"]
    if slpm_f is not None and sapm_o is not None and slpm_f > 2.0:
        base = f"{fighter} lands {slpm_f:.1f} significant strikes a minute into an opponent who absorbs {sapm_o:.1f}"
        rank = (_cohort_note(opponent, "sapm", True,
                             "nobody booked on this card takes more",
                             "one of the most-hit fighters on this card")
                or _cohort_note(fighter, "slpm", True,
                                "the busiest striker on this card",
                                "one of the busiest strikers on this card"))
        if rank:
            return base + f" -- {rank}"
        if acc_f is not None and acc_o is not None and acc_f - acc_o > 4:
            out = base + f", and is the more accurate of the two at {acc_f:.0f}% to {acc_o:.0f}%"
            rank = _cohort_note(fighter, "strike_accuracy_pct", True,
                                "the best accuracy on this card",
                                "one of the best accuracy figures on this card")
            return out + (f", {rank}" if rank else "")
        return base
    if net_f is not None and net_o is not None and net_f - net_o > 0.8:
        return f"{fighter} is net {net_f:+.1f} strikes a minute, {opponent} {net_o:+.1f} -- the exchange has gone one way for a while"
    if acc_f is not None and acc_o is not None and acc_f - acc_o > 4:
        return f"{fighter} connects on {acc_f:.0f}% of their significant strikes to {opponent}'s {acc_o:.0f}%"
    return f"{fighter} has the sharper striking profile of the two, and it is the clearest gap in the matchup"


def _durability_clause(fighter, opponent, st, opp) -> str:
    fl_o = int(round((opp["ko_loss_rate"] + opp["sub_loss_rate"]) * opp["losses"]))
    fl_f = int(round((st["ko_loss_rate"] + st["sub_loss_rate"]) * st["losses"]))
    return (f"{opponent} has been stopped {_count(fl_o, 'time', 'times')} in "
            f"{_count(opp['losses'], 'defeat', 'defeats')}, {fighter} "
            f"{_count(fl_f, 'time', 'times')} in {_count(st['losses'], 'defeat', 'defeats')}")


def _submission_clause(fighter, opponent, st) -> str:
    subs = int(round(st["sub_rate"] * st["wins"]))
    return (f"{subs} of {fighter}'s {_count(st['wins'], 'win', 'wins')} have come by submission, "
            f"a threat {opponent} has to carry into every exchange on the mat")


# ---------------------------------------------------------------------------
# DID THE NAMED RISK HAPPEN?
#
# Every pick now names the strongest thing arguing against it. Once the fight
# resolves, the honest question is whether that specific thing is what beat it
# -- but only some of those risks are checkable from a result. A pick that
# warned about the opponent's striking and then lost by knockout is a clean
# yes; the same pick losing a decision is not evidence either way, and a
# layoff or age warning cannot be adjudicated by a method at all.
#
# So this returns True only where the method corresponds directly, False only
# where the pick lost to something the warning specifically did not describe,
# and None everywhere else. None prints nothing. Guessing here would turn a
# credibility feature into a worse version of the touting it replaced.
# ---------------------------------------------------------------------------

# risk kind -> the loss methods that would confirm it
_FALSIFIER_METHODS = {
    "opp_striking": {"KO/TKO"},
    "opp_grappling": {"SUB"},
    "opp_subs": {"SUB"},
    "own_chin": {"KO/TKO", "SUB"},
}
# Risks a fight result cannot adjudicate. Listed rather than omitted so the
# distinction is deliberate and survives the next edit.
_FALSIFIER_UNRESOLVABLE = {"own_layoff", "own_age", ""}


def falsifier_fired(kind: str | None, won: bool, actual_method: str | None) -> bool | None:
    """True / False / None -- see the note above on why None is common."""
    if won or not kind or kind in _FALSIFIER_UNRESOLVABLE:
        return None
    expected = _FALSIFIER_METHODS.get(kind)
    if not expected:
        return None
    m = _method_bucket(actual_method)
    if m is None:
        return None
    return m in expected


def _method_bucket(method) -> str | None:
    """ESPN and fight_results spell methods several ways; normalise to three."""
    if not method:
        return None
    t = str(method).strip().lower()
    if t.startswith("dec") or "decision" in t:
        return "DEC"
    if "sub" in t:
        return "SUB"
    if "ko" in t or "tko" in t or "knockout" in t:
        return "KO/TKO"
    return None


# ---------------------------------------------------------------------------
# THE MORNING AFTER.
#
# Roughly two of every five picks on this site lose. Every blurb above will
# eventually sit under a red L, be screenshotted, and be read back by somebody
# who did not take the bet. Nothing in this file was written for that moment.
#
# This appends one flat line to the ORIGINAL blurb, which is kept verbatim --
# the pre-fight reasoning is the record and rewriting it after the fact is the
# exact failure generate_site.py's frozen-pick restore exists to prevent.
#
# THE TONE IS IDENTICAL FOR WINS AND LOSSES, and that is the whole design. A
# product whose differentiator is a public ledger including the misses cannot
# have a losing note that sounds different from a winning one. There is a hard
# ban below on the vocabulary of excuse -- "unlucky", "variance", "robbed",
# "should have" -- because those words are how a track record becomes
# marketing, and this file already shipped five sentences that had to be
# deleted for exactly that reason.
#
# It also closes the loop the sign-partitioned signals opened: every pick now
# names what would beat it, so once the fight resolves the honest question is
# simply whether that thing happened.
# ---------------------------------------------------------------------------

_EXCUSE_WORDS = ("unlucky", "unfortunate", "variance", "robbed", "should have",
                 "deserved", "close call", "bad luck", "hosed", "screwed")


def explain_settled(original: str, won: bool, method: str | None = None,
                    falsifier_fired: bool | None = None,
                    band_record: tuple[int, int] | None = None) -> str:
    """
    The original blurb, unchanged, plus one line of result.

    falsifier_fired: whether the counterargument the blurb named is what
        actually happened. None when the blurb named none, or when nobody has
        judged it -- in which case it is simply not mentioned rather than
        guessed at.
    band_record: (won, total) for picks in this confidence band this season.
        Published whatever it is. A band that reads 3-11 prints 3-11; the
        moment it is filtered it becomes the thing it was built to prevent.
    """
    verdict = "Won" if won else "Lost"
    parts = [f"{verdict}{f' by {method}' if method else ''}."]

    if falsifier_fired is True:
        parts.append("The risk named above is what happened.")
    elif falsifier_fired is False and not won:
        parts.append("Not for the reason named above.")

    if band_record:
        w, n = band_record
        if n >= 10:
            parts.append(f"Picks in this confidence band are {w}-{n - w} this season.")

    line = " ".join(parts)
    # Belt and braces: the ban is asserted, not merely intended. A future
    # edit that reaches for "unlucky" trips this rather than shipping it.
    low = line.lower()
    for w in _EXCUSE_WORDS:
        if w in low:
            raise AssertionError(f"explain_settled must not editorialise: {w!r} in {line!r}")
    return f"{original.rstrip()} {line}"


# ---------------------------------------------------------------------------
# CARD COHORT.
#
# The one axis that produces a claim no template can reproduce, because it
# does not exist inside a single fight: "the least of anyone booked here" is a
# fact about the whole card. It is also free -- the card is already loaded --
# and it is a CENSUS rather than a sample, so a superlative over it is exactly
# true rather than an inference.
#
# Two guards, both learned the hard way elsewhere in this project:
#
#   MARGIN, NOT A BARE COMPARISON. A rank flips when a fight is cancelled or
#   an opponent is replaced, and the site rebuilds ~48x a day, so a blurb that
#   claims "the highest on the card" by 0.1 will contradict itself by
#   Thursday. A superlative requires clear daylight over second place.
#
#   POPULATED ONLY. A fighter missing the stat is not last, they are absent.
#   Ranking nulls is how "0.0 control time" becomes "the least on the card".
# ---------------------------------------------------------------------------

_COHORT: dict | None = None
COHORT_MIN_MARGIN = {"control_time_pct": 6.0, "sapm": 0.6, "slpm": 0.8,
                     "td_per_15": 1.0, "td_defense_pct": 6.0, "strike_accuracy_pct": 4.0}


def set_card_cohort(fighters_df, names) -> None:
    """
    Register the fighters booked on the card being rendered.

    Called once per build by the caller that knows the card. Absent this, all
    cohort clauses return None and the copy simply does not make card claims
    -- which is the correct failure, not a silent fallback to a roster-wide
    comparison the reader was never promised.
    """
    global _COHORT
    vals = {}
    for stat in COHORT_MIN_MARGIN:
        pairs = []
        for n in names:
            st = _fighter_stats(fighters_df, n)
            if st and st.get(stat) is not None:
                pairs.append((n, float(st[stat])))
        if len(pairs) >= 8:      # too small a card and "on this card" means little
            vals[stat] = pairs
    _COHORT = vals or None


# Rank 1 is a superlative and needs daylight; ranks 2-3 get the softer form,
# which needs no margin because "among the highest" survives a reshuffle that
# "the highest" would not.
COHORT_TOP_N = 3
COHORT_MIN_POOL = 20


def _cohort_note(name: str, stat: str, want_high: bool,
                 solo: str, among: str) -> str | None:
    """
    Where this fighter sits among everyone booked on the card, when it is
    worth saying and safe to say.

    THE FIRST VERSION OF THIS FIRED ZERO TIMES on a 123-fighter card, and the
    reason is worth keeping: it accepted only rank 1, so at most one fighter
    per statistic could ever qualify, and only if that fighter also happened
    to be the favourite in a blurb that reached the one branch calling it.
    Three of the six statistics could not clear their margin at all.
    A claim that can only be true once per card is not a feature.

    Rank 1 still requires clear daylight over rank 2 -- the site rebuilds ~48
    times a day and a card changes through the week, so a bare "the highest"
    won by 0.1 contradicts itself by Thursday. Ranks 2 and 3 get "among the",
    which stays true through exactly that reshuffle.
    """
    if not _COHORT or stat not in _COHORT:
        return None
    pairs = sorted(_COHORT[stat], key=lambda p: p[1], reverse=want_high)
    if len(pairs) < COHORT_MIN_POOL:
        return None
    idx = next((i for i, (n, _) in enumerate(pairs) if n == name), None)
    if idx is None or idx >= COHORT_TOP_N:
        return None
    # The caller supplies both phrasings ready-made rather than an adjective
    # this function bolts "the"/"among the" onto. Assembling them here produced
    # "among the most absorbed on this card", which is not English -- the
    # superlative and the plural need different words, and only the caller
    # knows which statistic reads as what.
    if idx == 0:
        margin = abs(pairs[0][1] - pairs[1][1])
        if margin >= COHORT_MIN_MARGIN[stat]:
            return solo
    return among


# ---------------------------------------------------------------------------
# PRICE CLAUSE.
#
# The closer took odds_display and prob_pct and did no arithmetic between
# them, so it could assert "real, bettable value" on a row where the model was
# BELOW the break-even the price demands. It also meant a 79% pick with 7
# points of cushion and a 61% pick with 11 read identically -- and the 61% is
# the better bet.
#
# The cushion is the number that makes the claim falsifiable: it states what
# the price requires and what the model actually says, so a reader can check
# it against the block printed directly above. That is the opposite of touting
# and it is why this prints the cushion and NOT a stake. A stake figure is a
# recommendation about the reader's money, issued identically to every reader
# regardless of bankroll, and belongs nowhere in prose.
#
# THE VIG DISCLOSURE. Every edge on this site is measured against a de-vigged
# Polymarket line, so a 7-point edge is smaller at a real book. Volunteering
# that costs a little of the number and buys more than any adjective.
# ---------------------------------------------------------------------------

def _price_clause(row: dict) -> str | None:
    """Break-even, the model's number, and the gap between them."""
    try:
        model_p = float(row["model_prob"])
        fair_p = float(row["book_fair_prob"])
    except (TypeError, ValueError, KeyError):
        return None
    if model_p != model_p or fair_p != fair_p or not (0 < fair_p < 1):
        return None

    cushion = (model_p - fair_p) * 100
    odds = format_american_odds(row["odds_american"])

    # NEVER claim value against the arithmetic. The old closer asserted it
    # unconditionally; this refuses to.
    if cushion <= 0:
        return (f"{odds} needs {fair_p*100:.1f}% to break even and the model is at "
                f"{model_p*100:.1f}% -- the price is ahead of the model here")

    base = (f"{odds} needs {fair_p*100:.1f}% to break even; the model says "
            f"{model_p*100:.1f}%")
    if cushion < 8:
        # Thin enough that the vig matters to whether it survives at a book.
        try:
            vig_p, _ = add_estimated_vig(fair_p, 1 - fair_p)
            shrunk = (model_p - vig_p) * 100
            if shrunk < cushion:
                return base + (f", a {cushion:.1f}-point cushion that is nearer "
                               f"{shrunk:.1f} once a real book's margin is priced in")
        except (ValueError, ZeroDivisionError):
            pass
        return base + f", a cushion of {cushion:.1f} points"
    return base + f", a cushion of {cushion:.1f} points"


# ---------------------------------------------------------------------------
# FIGHT-LENGTH PROFILES.
#
# The rounds and distance markets were explained entirely in terms of career
# FINISH RATE, which is a proxy. The quantity those markets actually price is
# how long a fighter's fights last, and data/pit_stats.csv carries the real
# duration of all 8,626 tracked bouts. Nothing in this file was reading it.
#
# A round-total line maps to an exact clock time -- "Over 2.5" is the fight
# passing 2:30 of round three, which is 750 seconds -- so the honest sentence
# is the empirical frequency of precisely the event being priced, from that
# fighter's own history. That is not a template: it is a different pair of
# counts for every fighter and every line.
# ---------------------------------------------------------------------------

_LENGTH_CACHE: dict | None = None


def _length_profile(name: str) -> dict | None:
    """{n, mean_secs, past: fn(secs) -> count} from a fighter's tracked bouts."""
    global _LENGTH_CACHE
    if _LENGTH_CACHE is None:
        try:
            from scripts.build_pit_stats import load_pit_stats
            _LENGTH_CACHE = load_pit_stats()
        except Exception:
            _LENGTH_CACHE = {}
    rows = _LENGTH_CACHE.get(str(name).strip().lower(), [])
    secs = [float(r["fight_seconds"]) for r in rows
            if r.get("fight_seconds") not in (None, "") and float(r["fight_seconds"]) > 0]
    if len(secs) < 4:
        return None      # too few bouts for a frequency to mean anything
    return {"n": len(secs), "mean": sum(secs) / len(secs), "secs": secs}


def _line_seconds(line: float) -> int:
    """
    'Over 2.5 rounds' -> 750s. A .5 line is the midpoint of the next round, so
    2.5 is two full five-minute rounds plus 2:30.
    """
    return int(line) * 300 + 150


def _past_line(prof: dict, secs_threshold: int) -> tuple[int, int]:
    """(fights that passed the mark, total tracked)."""
    return sum(1 for s in prof["secs"] if s > secs_threshold), prof["n"]


def _mmss_label(secs: float) -> str:
    m, s = divmod(int(round(secs)), 60)
    return f"{m}:{s:02d}"


def _length_clause(name_a, name_b, line: float, is_over: bool) -> str | None:
    """
    The empirical frequency of the exact event the line prices, for both men.

    Returns None when either side lacks enough tracked bouts, so the caller
    falls back rather than reporting a fraction of three fights as a rate.
    """
    pa, pb = _length_profile(name_a), _length_profile(name_b)
    if not pa or not pb:
        return None
    mark = _line_seconds(line)
    ca, na = _past_line(pa, mark)
    cb, nb = _past_line(pb, mark)
    side = "past" if is_over else "short of"
    if not is_over:
        ca, cb = na - ca, nb - cb
    return (f"{ca} of {name_a}'s {na} tracked fights have finished {side} that mark, "
            f"and {cb} of {name_b}'s {nb}")


def _duration_clause(name_a, name_b) -> str | None:
    """Average fight length, side by side -- the plainest length fact there is."""
    pa, pb = _length_profile(name_a), _length_profile(name_b)
    if not pa or not pb:
        return None
    if abs(pa["mean"] - pb["mean"]) < 90:
        return (f"both men average close to the same fight length -- "
                f"{_mmss_label(pa['mean'])} for {name_a}, {_mmss_label(pb['mean'])} for {name_b}")
    longer, shorter = (name_a, name_b) if pa["mean"] > pb["mean"] else (name_b, name_a)
    pl, ps = (pa, pb) if pa["mean"] > pb["mean"] else (pb, pa)
    return (f"{longer}'s fights have averaged {_mmss_label(pl['mean'])} against "
            f"{_mmss_label(ps['mean'])} for {shorter}")


# Anchored on the closing "(+8.3% edge)." -- note the decimal point inside the
# percentage, which is why this can't be written as [^.]*: the first attempt
# used that and never matched anything with a fractional edge.
_RESTATEMENT = re.compile(r"^(.*?edge\)\.)\s*", re.IGNORECASE)
_RESTATEMENT_SIGNS = ("model puts", "model estimates", "is priced at", "implied")


def _strip_restatement(text: str) -> str:
    """
    Drop the opening sentence, which repeats the numbers shown directly above.

    Every explainer opens by restating model probability, implied probability,
    the price and the edge -- all four of which are already in the Book Price /
    Model / Edge block on the same card, in a form that's easier to compare.
    What follows it is the only synthesis on a standout card: WHY the model
    disagrees. That part stays.

    The pattern is anchored on "...(+8.3% edge)." so it can only ever match the
    generated preamble; anything that doesn't match is returned untouched
    rather than guessed at.
    """
    if not text:
        return text
    m = _RESTATEMENT.match(text)
    if not m:
        return text
    # Only strip a prefix that actually looks like the generated preamble.
    # Matching on the "edge)." anchor alone would happily eat a sentence that
    # merely ended that way.
    if not any(sign in m.group(1).lower() for sign in _RESTATEMENT_SIGNS):
        return text
    trimmed = text[m.end():].strip()
    if not trimmed:
        return text            # the whole string was preamble -- keep it
    # Follow-ons were written to continue a sentence ("That's built on...").
    trimmed = re.sub(r"^That's built on", "Built on", trimmed)
    return trimmed[0].upper() + trimmed[1:] if trimmed else trimmed


def explain_edge(row: dict, fighters_df: pd.DataFrame) -> str:
    if row["market"] == "Moneyline":
        out = explain_moneyline(row, fighters_df)
    elif row["market"].startswith("Method"):
        out = explain_method(row, fighters_df)
    elif row["market"].startswith("Total Rounds"):
        out = explain_total_rounds(row, fighters_df)
    elif row["market"].startswith("Fight Outcome"):
        out = explain_goes_the_distance(row, fighters_df)
    elif row["market"].startswith("Round Betting"):
        out = explain_round_betting(row, fighters_df)
    else:
        return f"{row['fighter']} — {row['market']}: {row['edge_pct']:+.1f}% edge vs. the market."
    return _strip_restatement(out)
