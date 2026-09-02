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

from src.fight_format import is_five_round as _is_five_round, scheduled_rounds as _scheduled_rounds

from .odds_utils import american_to_implied_prob, implied_prob_to_american, remove_vig_two_way, edge_percent, kelly_fraction, market_blended_prob, devig_single_sided, american_to_decimal
from .method_model import method_probabilities, reconcile_fighter_methods, method_given_win, finish_share_before
from .ufc_method_rates import rates_or_prior
from .names import canonical_name
from .matchup_model import (predict_matchup, compute_divisional_method_priors,
                            blend_method_probability, divisional_prior_for, _get,
                            normalize_division)


def _fold_name(t):
    """Lowercase + strip diacritics for cross-source name matching.

    canonical_name FIRST, because folding cannot reach a middle name. This
    helper is one of the several folds in the project and it silently did not
    resolve aliases, so _find_fighter below returned empty for a fighter who
    IS on the roster under his canonical spelling -- see _reconciled.
    """
    import unicodedata
    t = canonical_name(t)
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


def _devig_and_shop(by_source, has_source: bool):
    """
    Turn every source's quote on a TWO-WAY market into one fair line and one
    bettable price per side.

    This is the shared spine of the moneyline and total-rounds builders, and
    it exists because both broke the same way when the second feed arrived.
    Three things have to happen in this order and none of them are optional:

    1. DE-VIG WITHIN A SOURCE. Pairing a DraftKings side against a FanDuel
       side yields a "fair" number belonging to neither book, and against a
       vig-free Polymarket side it is not even dimensionally consistent.
    2. AVERAGE INTO A CONSENSUS. A consensus is sharper than any single book
       and it stops the number the model is judged against from swinging when
       one book moves alone.
    3. SHOP THE PRICE, but only among VIG-BEARING sources. Polymarket's
       midpoint is the reference, not a bet; treating it as shoppable would
       hand the reader a price no sportsbook will honour -- which is the exact
       defect this whole rework exists to remove.

    Returns None when no single source quoted both sides, since there is then
    nothing to de-vig and a one-sided "fair" line would be an invention.
    """
    fairs_a, fairs_b, quotes = [], [], []
    a = b = None
    for src_name, g in by_source:
        if len(g) != 2:
            continue
        ra, rb = g.iloc[0], g.iloc[1]
        if a is None:
            a, b = ra, rb           # first complete pair fixes the ordering
        elif ra["selection"] != a["selection"]:
            ra, rb = rb, ra         # keep every source in the same order
        ia = american_to_implied_prob(ra["odds_american"])
        ib = american_to_implied_prob(rb["odds_american"])
        fa, fb = remove_vig_two_way(ia, ib)
        fairs_a.append(fa)
        fairs_b.append(fb)
        if has_source and not bool(ra.get("source_is_vig_free")):
            quotes.append((src_name, ra, rb))

    if a is None:
        return None

    fair_a = sum(fairs_a) / len(fairs_a)
    fair_b = sum(fairs_b) / len(fairs_b)
    _tot = fair_a + fair_b
    if _tot > 0:
        fair_a, fair_b = fair_a / _tot, fair_b / _tot

    def _best(idx: int, fallback):
        # No book quoting -> fall back to the reference price so nothing
        # regresses on a Polymarket-only build.
        if not quotes:
            return fallback, (fallback.get("source") if has_source else None), 1, {}
        side = [(nm, (pa, pb)[idx]) for nm, pa, pb in quotes]
        nm, r = min(side, key=lambda t: american_to_implied_prob(t[1]["odds_american"]))
        # EVERY BOOK'S PRICE, NOT ONLY THE WINNER'S.
        #
        # Shopping is right for a single bet and wrong for a parlay, which is
        # one ticket at one book. Keeping only the best price fragments the
        # per-book boards: DraftKings may quote all four legs a slip wants,
        # but if FanDuel beats it on one, that leg leaves the DraftKings pool
        # entirely and the slip can no longer be built anywhere. Measured on
        # Nurmagomedov vs. Song, best_book split 150 two-way edges into
        # DraftKings 32 / FanDuel 26 / Polymarket 92, and no parlay could be
        # formed from any single book's 32.
        #
        # The shopped price above is unchanged and still drives everything
        # else. This is an additional field that nothing is forced to read.
        prices = {nm2: float(row2["odds_american"]) for nm2, row2 in side}
        return r, nm, len(side), prices

    return a, b, fair_a, fair_b, _best(0, a), _best(1, b)


def _two_numbers(model_p: float, fair_p: float, odds: float) -> dict:
    """
    THE TWO NUMBERS, and they answer different questions.

      edge_pct  raw model against the consensus fair line. "Does the model
                disagree with the market?" -- a validation question, and the
                one the site used to lead with.
      ev_pct    the BLENDED probability against the price you can actually
                take. "Does this bet return money?" -- the only one that
                decides anything, and so the one that now leads.

    They disagree usefully. A high edge with negative EV is the near-miss:
    the model found something real and the vig ate it. A low edge with
    negative EV is simply nothing there. Reported alone, those two look
    identical, and the site spent its whole life showing only the first.

    EV IS COMPUTED ON THE BLENDED PROBABILITY, never the raw model. Raw-model
    EV is flattering by construction -- it is built from the same disagreement
    the edge measures, so it would report the model's own optimism as profit
    and reintroduce, one layer down, exactly the bias the blend exists to
    remove.

    vig_cost_pct is the bridge between the two, in points of implied
    probability: what the book charges on this side over the fair line.
    """
    blended_p = market_blended_prob(model_p, fair_p)
    return {
        "blended_prob": round(blended_p, 4),
        "edge_pct": round(edge_percent(model_p, fair_p), 2),
        "ev_pct": round((blended_p * american_to_decimal(odds) - 1.0) * 100.0, 2),
        "vig_cost_pct": round((american_to_implied_prob(odds) - fair_p) * 100.0, 2),
    }


def compute_moneyline_edges(
    upcoming_df: pd.DataFrame, elo_ratings: dict[str, float], fighters_df: pd.DataFrame | None = None,
    fight_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    ml = upcoming_df[upcoming_df["market"] == "Moneyline"]
    has_source = "source" in ml.columns

    for fight_id, fight_rows in ml.groupby("fight_id"):
        # ONE FIGHT NOW HAS MORE THAN TWO ROWS. With Polymarket, DraftKings
        # and FanDuel all quoting, a bout arrives as six moneyline rows, and
        # the old `len(group) != 2` guard skipped every one of them -- which
        # would have deleted every moneyline edge the moment the second feed
        # was switched on.
        #
        # De-vigging has to happen WITHIN a source. Pairing a DraftKings side
        # against a FanDuel side produces a "fair" number belonging to neither
        # book, and with a vig-free Polymarket side in the mix it is not even
        # dimensionally consistent.
        by_source = (list(fight_rows.groupby("source")) if has_source
                     else [(None, fight_rows)])
        shopped = _devig_and_shop(by_source, has_source)
        if shopped is None:
            print(f"[edge_finder] moneyline skip for fight_id={fight_id!r}: no source "
                  f"quoted both sides -- rows: {len(fight_rows)}")
            continue
        a, b, fair_a, fair_b, (best_a, book_a, n_a, px_a), (best_b, book_b, n_b, px_b) = shopped

        # CATCH IT WHERE IT IS MADE, WITH ITS INPUTS. book_fair_prob is a
        # cross-source de-vigged consensus and odds_american is the single
        # best bettable quote, so they differ by the vig and a little line
        # shopping -- a couple of points. Twenty rows in predictions_log
        # disagree by more than fifty (Rei Tsuruya: fair 0.449 against a -800
        # price implying 0.889), all on cards from 2026-08-18 onward, and the
        # bad pair was only found months later by regrading the ledger.
        # track_record now refuses to grade CLV on such a pair, but that is a
        # guard; this is the only place the SOURCES are still in scope. It
        # does not reproduce on a single-source card, so the next time it
        # fires the log has to carry enough to diagnose it.
        for _side, _fair, _px in ((a, fair_a, best_a), (b, fair_b, best_b)):
            _imp = american_to_implied_prob(_px["odds_american"])
            if _imp is not None and abs(_fair - _imp) > 0.10:
                print(f"[edge_finder] INCOHERENT fair vs price on "
                      f"{_side['selection']!r}: fair={_fair:.3f} "
                      f"price={_px['odds_american']} implied={_imp:.3f} "
                      f"price_source={_px.get('source')!r} "
                      f"sources_quoting={[nm for nm, _ in by_source]}")

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

        for fighter, opponent, model_p, fair_p, priced, book, n_books, ref, book_px in [
            (a["selection"], b["selection"], model_prob_a, fair_a, best_a, book_a, n_a, a, px_a),
            (b["selection"], a["selection"], model_prob_b, fair_b, best_b, book_b, n_b, b, px_b),
        ]:
            odds = priced["odds_american"]
            nums = _two_numbers(model_p, fair_p, odds)
            rows.append({
                "fight_id": fight_id,
                "fighter": fighter,
                "opponent": opponent,
                "market": "Moneyline",
                # THE BEST BETTABLE PRICE, not the reference midpoint. This is
                # the number the reader can actually take, and every figure
                # derived from it below now refers to a real bet.
                "odds_american": odds,
                "model_prob": round(model_p, 3),
                # Consensus of every source that quoted both sides, de-vigged
                # within each source first. This is the FAIR line -- what the
                # market thinks, cleanly -- and it is what edge is measured
                # against.
                "book_fair_prob": round(fair_p, 3),
                **nums,
                "suggested_stake_pct": round(kelly_fraction(nums["blended_prob"], odds) * 100, 2),
                "clob_token_id": priced.get("clob_token_id") or ref.get("clob_token_id"),
                # Provenance of the PRICE. `src` was written as a bare `row`
                # here once, which does not exist in this scope, and the
                # resulting NameError deleted every moneyline edge for several
                # builds while looking like a thin market.
                "source": priced.get("source"),
                "source_is_vig_free": priced.get("source_is_vig_free"),
                # Line shopping. books_quoting = 1 means nothing was shopped,
                # which the reader should be told rather than left to assume.
                "best_book": book,
                "books_quoting": n_books,
                # See _best: the whole board, so a parlay can be built at ONE
                # book instead of from the best-price union of several.
                "book_prices": book_px,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def compute_method_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame,
                         elo_ratings: dict[str, float] | None = None,
                         fight_history_df: pd.DataFrame | None = None) -> pd.DataFrame:
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
        total_wins = max(int(_get(f, "wins", 0)), 1)
        rate_map = {
            "KO/TKO": _get(f, "ko_wins", 0) / total_wins,
            "SUB": _get(f, "sub_wins", 0) / total_wins,
            "DEC": _get(f, "dec_wins", 0) / total_wins,
        }
        own_rate = rate_map[method]
        divisional_prior = divisional_prior_for(divisional_priors, f["weight_class"], method, own_rate)
        opp_vulnerability = own_rate  # fallback if opponent data is missing
        if opp_stats is not None and not opp_stats.empty:
            opp = opp_stats.iloc[0]
            opp_losses = max(int(opp["losses"]), 1) if opp["losses"] else 0
            if opp_losses:
                opp_vulnerability = opp[method_loss_col[method]] / opp_losses
        return blend_method_probability(divisional_prior, own_rate, opp_vulnerability, total_wins)

    # RECONCILED GRID, cached per fight. These rows previously came from an
    # independent blend, so the priced KO rows and the model-only SUB/DEC rows
    # disagreed: each fighter's methods overshot his win probability and the
    # six summed to 119.9%. Fixing only the projection path left the priced
    # rows untouched, which is why the KO numbers didn't move.
    # One computation, shared with model_preview via method_model.
    _grid_cache = {}

    def _reconciled(fight_id, name_a, name_b):
        if fight_id in _grid_cache:
            return _grid_cache[fight_id]
        # TWO SPELLINGS, EACH WITH ONE JOB. The odds feed's spelling KEYS the
        # result, because every caller below looks the grid up by
        # row["selection"], which is the feed's. The canonical spelling does
        # every LOOKUP -- roster, ratings, method rates -- because those are
        # keyed off fight_history and fighters.csv.
        #
        # Conflating them is what published the Noche main event wrong: the
        # feed quotes "Jose Miguel Delgado", the roster and elo hold "Jose
        # Delgado", so ra came back empty, `out` stayed None, and every method
        # row silently fell through to the unreconciled per-fighter blend --
        # six rows summing to 156.2% instead of 100%.
        canon_a, canon_b = canonical_name(name_a), canonical_name(name_b)
        ra, rb = _find_fighter(fighters_df, canon_a), _find_fighter(fighters_df, canon_b)
        out = None
        if not ra.empty and not rb.empty:
            a, b = ra.iloc[0], rb.iloc[0]
            # SAME ARGUMENTS as compute_moneyline_edges. Omitting
            # fight_history_df drops the recent-form adjustment, so the win
            # probabilities used to constrain this grid differed from the
            # moneyline shown two rows above -- each fighter's methods missed
            # his own win probability by ~2.5pp.
            # This is the third time two predict_matchup calls with different
            # arguments have produced disagreeing numbers on the same page.
            matchup = predict_matchup(canon_a, canon_b, fighters_df, elo_ratings, fight_history_df)
            if matchup:
                # FITTED seed, matching model_preview exactly -- one source
                # for the shape, as the totals already share one reconciler.
                # UFC-ONLY RATES, AND THIS FILE MUST MATCH model_preview.
                # reconcile_fighter_methods runs iterative proportional
                # fitting -- it forces the grid's rows to the win
                # probabilities and its columns to the fight-level method
                # split. IPF converges only when both margins describe the
                # same object, so seeding the PRICED path from career rates
                # while the model-only path seeds from UFC rates makes the two
                # margins disagree and the grid stops summing to 1.
                # Not a theory: fixing model_preview alone took the linter
                # from 1 failure to 17, with six method rows summing to 116.2%
                # on Mederos/Jones. The reconciler's own docstring records a
                # previous fix missing this exact caller.
                def _wcls(row):
                    # NOT _get(): that coerces to float, and a weight class is
                    # a string.
                    try:
                        v = row.get("weight_class")
                    except AttributeError:
                        return None
                    if v is None or (isinstance(v, float) and v != v):
                        return None
                    return str(v).strip() or None

                def _seed(own, opp, own_name, opp_name):
                    o_ko, o_sub, _okl, _osl = rates_or_prior(own_name, divisional_priors, _wcls(own))
                    _pko, _psub, p_kl, p_sl = rates_or_prior(opp_name, divisional_priors, _wcls(opp))
                    g = ((elo_ratings.get(own_name, 1500) - elo_ratings.get(opp_name, 1500)) / 400.0
                         if elo_ratings else 0.0)
                    return method_given_win(
                        own_ko_rate=o_ko,
                        own_sub_rate=o_sub,
                        opp_ko_lost=p_kl,
                        opp_sub_lost=p_sl,
                        elo_gap=g,
                    )
                seeds = [_seed(a, b, canon_a, canon_b), _seed(b, a, canon_b, canon_a)]
                koa, sua, kla, sla = rates_or_prior(canon_a, divisional_priors, _wcls(a))
                kob, sub_, klb, slb = rates_or_prior(canon_b, divisional_priors, _wcls(b))
                gap = abs(elo_ratings.get(canon_a, 1500) - elo_ratings.get(canon_b, 1500)) / 400.0 if elo_ratings else 0.0
                dist = method_probabilities(
                    ko_press=koa * klb + kob * kla, sub_press=sua * slb + sub_ * sla,
                    ko_rate_sum=koa + kob, sub_rate_sum=sua + sub_,
                    durability=kla + klb, elo_gap=gap,
                )
                out = {
                    name_a: dict(zip(("KO/TKO", "SUB", "DEC"),
                                     reconcile_fighter_methods(seeds[0], seeds[1],
                                                               matchup["prob_a"], matchup["prob_b"], dist)[0])),
                    name_b: dict(zip(("KO/TKO", "SUB", "DEC"),
                                     reconcile_fighter_methods(seeds[0], seeds[1],
                                                               matchup["prob_a"], matchup["prob_b"], dist)[1])),
                }
        _grid_cache[fight_id] = out
        return out

    for _, row in props.iterrows():
        stats = _find_fighter(fighters_df, row["selection"])
        if stats.empty:
            continue
        f = stats.iloc[0]

        # find the opponent to factor in their specific vulnerability
        opponent_name = row["fighter_b"] if row["selection"] == row["fighter_a"] else row["fighter_a"]
        opp_stats = _find_fighter(fighters_df, opponent_name)
        _grid = _reconciled(row["fight_id"], row["fighter_a"], row["fighter_b"])

        if row["selection_method"] == "FINISH":
            # "Wins by finish" = KO/TKO or SUB -- these are mutually
            # exclusive outcomes for a single fight, so the combined
            # probability is a straight sum of the two independently-
            # blended method probabilities, not a new model. Reuses 100%
            # of the same prior-informed blend already trusted for the
            # individual KO/SUB props, rather than inventing a separate
            # "finish" prior from scratch.
            if _grid and row["selection"] in _grid:
                model_p = _grid[row["selection"]]["KO/TKO"] + _grid[row["selection"]]["SUB"]
            else:
                model_p = _blended_method_prob(f, opp_stats, "KO/TKO") + _blended_method_prob(f, opp_stats, "SUB")
            model_p = min(0.97, model_p)  # same sanity ceiling style used elsewhere in this module
        else:
            total_wins = max(int(_get(f, "wins", 0)), 1)
            rate_map = {
                "KO/TKO": _get(f, "ko_wins", 0) / total_wins,
                "SUB": _get(f, "sub_wins", 0) / total_wins,
                "DEC": _get(f, "dec_wins", 0) / total_wins,
            }
            own_rate = rate_map.get(row["selection_method"])
            if own_rate is None:
                continue
            # Reconciled value when available, so this row agrees with the
            # moneyline and with the fight-level split. The raw blend stays as
            # a fallback for fights the reconciler can't build a grid for.
            if _grid and row["selection"] in _grid:
                model_p = _grid[row["selection"]][row["selection_method"]]
            else:
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
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, devig_single_sided(imp, f"Method: {row['selection_method']}")), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
            # PROVENANCE TRAVELS WITH THE PRICE. Without it "Book" means
            # whichever feed happened to carry that market, and a vig-free
            # peer-to-peer quote gets compared against a vigged one.
            "source": row.get("source"),
            "source_is_vig_free": row.get("source_is_vig_free"),
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
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, devig_single_sided(imp, "Round Betting: Round 1")), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
            # PROVENANCE TRAVELS WITH THE PRICE. Without it "Book" means
            # whichever feed happened to carry that market, and a vig-free
            # peer-to-peer quote gets compared against a vigged one.
            "source": row.get("source"),
            "source_is_vig_free": row.get("source_is_vig_free"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def _extract_round_line(selection: str) -> float | None:
    match = re.search(r"(\d+\.?\d*)", str(selection))
    return float(match.group(1)) if match else None


def compute_total_rounds_edges(upcoming_df: pd.DataFrame, fighters_df: pd.DataFrame,
                               effective_ratings: dict[str, float] | None = None) -> pd.DataFrame:
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

    # Scheduled length PER FIGHT, derived once. It was inferred from the LINE
    # (`5 if line > 3 else 3`), which is wrong: a five-round fight is offered
    # 2.5 lines too, so its 2.5 got the three-round share (0.865) while its
    # 3.5 got the five-round one (0.805) -- and Under 2.5 came out HIGHER than
    # Under 3.5 on the same fight.
    # card_position is authoritative; the widest line offered is the fallback,
    # since a 3.5 or 4.5 line only exists on a five-round bout.
    _sched_by_fight = {}
    for fid, g in props.groupby("fight_id"):
        pos = str(g.get("card_position", pd.Series([""])).iloc[0] or "").strip()
        widest = g["_line"].dropna().max() if g["_line"].notna().any() else 0
        # The widest priced line is an INDEPENDENT signal -- a book offering
        # Over 4.5 has told us this is a five-rounder regardless of how the
        # card_position column reads -- so it stays alongside the shared rule
        # rather than being folded into it.
        _sched_by_fight[fid] = 5 if (_is_five_round(g.iloc[0]) or (widest or 0) > 3) else 3

    for (fight_id, line), group in props.groupby(["fight_id", "_line"]):
        scheduled = _sched_by_fight.get(fight_id, 3)
        fighters_in_fight = group["fighter_a"].iloc[0], group["fighter_b"].iloc[0]
        finish_rates = []
        first_round_rates = []
        # THE DIVISION COMES FROM THE BOUT, NOT THE ROSTER ROW. This read
        # weight_class off fighters.csv, and NOT ONE of the 27 booked fighters
        # carries one there -- so `division` was None on every call and the
        # division-conditioned finish curve, which exists precisely to split
        # the round shares by weight, has never once run in production. The
        # props row carries the booked weight class for the fight actually
        # being priced, which is also the more correct source: it is the
        # weight THIS bout is at, not the last one the fighter was listed at.
        _division = None
        if "weight_class" in group.columns:
            _division = normalize_division(group["weight_class"].iloc[0])
        for name in fighters_in_fight:
            stats = _find_fighter(fighters_df, name)
            if stats.empty:
                continue
            f = stats.iloc[0]
            # Either corner's listed division answers the question -- they are
            # fighting each other, so it is one bout at one weight. Taking the
            # first non-null rather than requiring both keeps the conditioning
            # alive when one fighter carries no division, which is the case for
            # 114 of 310 roster entries.
            if _division is None:
                _division = normalize_division(f.get("weight_class"))
            total_wins = max(int(_get(f, "wins", 0)), 1)
            finish_rates.append((_get(f, "ko_wins", 0) + _get(f, "sub_wins", 0)) / total_wins)
            if "first_round_finish_pct" in f and pd.notna(f["first_round_finish_pct"]):
                first_round_rates.append(float(f["first_round_finish_pct"]))

        if not finish_rates:
            continue

        combined_finish_rate = sum(finish_rates) / len(finish_rates)
        combined_first_round_rate = sum(first_round_rates) / len(first_round_rates) if first_round_rates else None

        # P(finish) FROM THE METHOD MODEL, so a round total cannot contradict
        # the method rows shown above it. The heuristic below produced Under
        # 4.5 at 66.0% on a fight the method model gave a 50.2% decision
        # probability -- and a decision IS Over 4.5 in a five-round fight, so
        # those two summed to 116%.
        # The identity is what matters; the share-of-finishes fractions are
        # estimates encoding that finishes are front-loaded.
        a_row = _find_fighter(fighters_df, fighters_in_fight[0])
        b_row = _find_fighter(fighters_df, fighters_in_fight[1])
        _md = None
        if not a_row.empty and not b_row.empty:
            a, b = a_row.iloc[0], b_row.iloc[0]
            n_a = max(int(_get(a, "wins", 0)) + int(_get(a, "losses", 0)), 1)
            n_b = max(int(_get(b, "wins", 0)) + int(_get(b, "losses", 0)), 1)
            koa, kob = _get(a, "ko_wins", 0) / n_a, _get(b, "ko_wins", 0) / n_b
            sua, sub_ = _get(a, "sub_wins", 0) / n_a, _get(b, "sub_wins", 0) / n_b
            kla, klb = _get(a, "ko_losses", 0) / n_a, _get(b, "ko_losses", 0) / n_b
            sla, slb = _get(a, "sub_losses", 0) / n_a, _get(b, "sub_losses", 0) / n_b
            gap = 0.0
            if effective_ratings:
                gap = abs(effective_ratings.get(fighters_in_fight[0], 1500)
                          - effective_ratings.get(fighters_in_fight[1], 1500)) / 400.0
            _md = method_probabilities(
                ko_press=koa * klb + kob * kla, sub_press=sua * slb + sub_ * sla,
                ko_rate_sum=koa + kob, sub_rate_sum=sua + sub_,
                durability=kla + klb, elo_gap=gap,
                scheduled_rounds=scheduled,
            )

        if _md is not None:
            finish = 1.0 - _md["decision"]
            # Fraction of finishes landing before each line. Higher lines
            # capture nearly all of them; the 1.5 line only the early ones.
            # Conditioned on division on three-round bouts: the split between
            # lines moves with weight even when P(finish) does not.
            share = finish_share_before(line, scheduled, _division)
            model_prob_under = finish * share
        elif line is not None and line <= 1.5 and combined_first_round_rate is not None:
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

        # TOTALS ARE A TWO-WAY MARKET AND ARE NOW QUOTED BY REAL BOOKS.
        # TheRundown serves market_id 3, so Over/Under arrives from DraftKings
        # and FanDuel carrying vig, alongside the vig-free Polymarket pair.
        # This loop used to emit one row per source with `book_fair_prob` set
        # to the RAW implied probability -- which silently relabelled a
        # DraftKings -135 as a 57.4% fair line when fair is 55.9%, and emitted
        # the same bet three times over, once per feed.
        #
        # Same treatment as the moneyline: de-vig inside each source, average
        # to a consensus, then shop the price among the books only.
        has_src = "source" in group.columns
        by_src = (list(group.groupby("source")) if has_src else [(None, group)])
        shopped = _devig_and_shop(by_src, has_src)

        if shopped is not None:
            a, b, fair_a, fair_b, (best_a, book_a, n_a, px_a), (best_b, book_b, n_b, px_b) = shopped
            emit = [(a, fair_a, best_a, book_a, n_a, px_a),
                    (b, fair_b, best_b, book_b, n_b, px_b)]
        else:
            # NO SOURCE QUOTED BOTH SIDES. A lone vig-free quote is already a
            # fair price and can be published as one. A lone VIGGED quote
            # cannot -- its raw implied probability carries the book's whole
            # margin, and it is exactly that number being labelled "fair" that
            # this rework exists to stop. Fall back to the proportional
            # single-sided de-vig rather than dropping the row or lying about
            # it, and say so in the provenance.
            emit = []
            for _, r in group.iterrows():
                imp_r = american_to_implied_prob(r["odds_american"])
                fair_r = (imp_r if bool(r.get("source_is_vig_free"))
                          else devig_single_sided(imp_r, f"Total Rounds {r['selection']}"))
                emit.append((r, fair_r, r, r.get("source") if has_src else None, 1, {}))

        for ref, fair_p, priced, book, n_books, book_px in emit:
            model_p = (model_prob_under if "under" in str(ref["selection"]).lower()
                       else 1 - model_prob_under)
            odds = priced["odds_american"]
            nums = _two_numbers(model_p, fair_p, odds)
            rows.append({
                "fight_id": fight_id,
                "fighter": f"{fighters_in_fight[0]} vs {fighters_in_fight[1]}",
                "market": f"Total Rounds {ref['selection']}",
                "odds_american": odds,
                "model_prob": round(model_p, 3),
                "book_fair_prob": round(fair_p, 3),
                **nums,
                "suggested_stake_pct": round(kelly_fraction(nums["blended_prob"], odds) * 100, 2),
                "clob_token_id": priced.get("clob_token_id") or ref.get("clob_token_id"),
                # PROVENANCE TRAVELS WITH THE PRICE. Without it "Book" means
                # whichever feed happened to carry that market, and a vig-free
                # peer-to-peer quote gets compared against a vigged one.
                "source": priced.get("source"),
                "source_is_vig_free": priced.get("source_is_vig_free"),
                "best_book": book,
                "books_quoting": n_books,
                "book_prices": book_px,
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
            scheduled_rounds=_scheduled_rounds(row),
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
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, devig_single_sided(imp, f"Fight Outcome: {row['selection']}")), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
            # PROVENANCE TRAVELS WITH THE PRICE. Without it "Book" means
            # whichever feed happened to carry that market, and a vig-free
            # peer-to-peer quote gets compared against a vigged one.
            "source": row.get("source"),
            "source_is_vig_free": row.get("source_is_vig_free"),
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
            scheduled_rounds=_scheduled_rounds(row),
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
            "suggested_stake_pct": round(kelly_fraction(market_blended_prob(model_p, devig_single_sided(imp, f"Fight Method: {sel}")), row["odds_american"]) * 100, 2),
            "clob_token_id": row.get("clob_token_id"),
            # PROVENANCE TRAVELS WITH THE PRICE. Without it "Book" means
            # whichever feed happened to carry that market, and a vig-free
            # peer-to-peer quote gets compared against a vigged one.
            "source": row.get("source"),
            "source_is_vig_free": row.get("source_is_vig_free"),
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
        compute_method_edges(upcoming_df, fighters_df, elo_ratings, fight_history_df),
        compute_total_rounds_edges(upcoming_df, fighters_df, elo_ratings),
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
    out = pd.concat(frames, ignore_index=True)
    return _finalise_two_numbers(out).sort_values("edge_pct", ascending=False).reset_index(drop=True)


def _finalise_two_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee every published row carries EV, and that no VIGGED price ever
    reaches the page with its raw implied probability labelled "fair".

    Two of the seven builders -- moneyline and total rounds -- price two-way
    markets and do this properly for themselves. The other five handle genuine
    single-sided props (method cells, goes-the-distance, round betting) that
    only Polymarket quotes, and a vig-free quote IS its own fair price, so
    their raw implied probability is correct TODAY.

    That is a fact about the current feed, not about the code. TheRundown's
    catalogue lists method and round markets for sport 7; the day it starts
    serving them, those five builders would each relabel a vigged price as
    fair and overstate the blend, silently, exactly as the totals builder did
    the day market_id 3 arrived. This pass makes the invariant hold by
    construction instead of by coincidence:

      vig-free source  -> implied is already fair, nothing to do
      vigged source    -> proportional single-sided de-vig, since there is no
                          complement to pair against

    EV is then computed for every row from the price actually carried, so the
    headline number exists on the whole table rather than only where a book
    happens to quote both sides.
    """
    if df.empty or ("ev_pct" in df.columns and df["ev_pct"].notna().all()):
        return df

    need = df["ev_pct"].isna() if "ev_pct" in df.columns else pd.Series(True, index=df.index)
    for i in df.index[need]:
        odds = df.at[i, "odds_american"]
        model_p = df.at[i, "model_prob"]
        fair_p = df.at[i, "book_fair_prob"]
        if pd.isna(odds) or pd.isna(model_p) or pd.isna(fair_p):
            continue
        flag = df.at[i, "source_is_vig_free"] if "source_is_vig_free" in df.columns else True
        # MISSING PROVENANCE MEANS UNKNOWN, and unknown takes the conservative
        # branch: assume the price may carry vig. Note `bool(flag)` rather
        # than `flag is True` -- pandas hands back numpy.bool_, and
        # `numpy.bool_(True) is True` is False, which would have de-vigged
        # every vig-free Polymarket row in the table.
        vig_free = False if (flag is None or pd.isna(flag)) else bool(flag)
        if not vig_free:
            fair_p = devig_single_sided(float(fair_p), str(df.at[i, "market"]))
            df.at[i, "book_fair_prob"] = round(fair_p, 3)
        nums = _two_numbers(float(model_p), float(fair_p), float(odds))
        for k, v in nums.items():
            df.at[i, k] = v
        # AND SIZE THE BET OFF THE SAME BLEND THE EV IS QUOTED FROM.
        #
        # These five builders each computed their own stake as
        # market_blended_prob(model, devig_single_sided(implied)), applied
        # unconditionally. On a VIG-FREE price that divides out an overround
        # nobody charged: the fair probability comes back too low, the blend
        # with it comes back too low, and Kelly stakes under what its own
        # inputs justify. Erring small is the safe direction, which is why it
        # survived unnoticed, but it also meant the stake and the displayed EV
        # were derived from two different numbers for the same bet.
        #
        # One blend per row now, computed once above with provenance taken
        # into account, and both the headline and the stake read from it.
        df.at[i, "suggested_stake_pct"] = round(
            kelly_fraction(nums["blended_prob"], float(odds)) * 100, 2)
    return df


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
        wins = max(int(_get(row, "wins", 0)), 1)
        col = {"KO/TKO": "ko_wins", "SUB": "sub_wins"}.get(method)
        if not col:
            continue
        model_p = _get(row, col, 0) / wins
        imp = r["book_fair_prob"]
        # NOT PUT THROUGH devig_single_sided, unlike the five priced prop paths
        # above. Those receive a raw Yes price that still carries the book's
        # whole margin. This one does not: `imp` is the REMAINDER of a
        # subtraction inside the book (derive_missing_method_lines), and its
        # market string is "Method: KO/TKO", so overround_for_market sent it to
        # the 20% six-cell margin and shaved a further 17% off a number that
        # was never a quoted Yes price at all. The derived rows are exactly the
        # thin ones, so that haircut is what decided whether they got a stake:
        # it zeroed most of them outright.
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

    BUT THE NORMALISATION BELOW IS CURRENTLY A NO-OP, AND SAYING SO IS THE
    POINT. The paragraph above describes the design, not what this source of
    data delivers. The "No" rows _devig normalises against are not independent
    quotes: polymarket_source builds them as implied_prob_to_american(1 -
    price_a) from the same midpoint as the Yes row, so p_yes + p_no is 1.000000
    for every FightMethod pair in the feed (measured: n=108, mean 1.0000, min
    1.0000, max 1.0000) and remove_vig_two_way returns its input unchanged.
    `derived` is therefore still two effectively-raw midpoints subtracted, and
    known_p is raw by the admission below. Left in place rather than deleted:
    it is correct, it costs nothing, and it starts doing real work the moment
    a genuinely two-sided book is added. What must not happen is a reader
    trusting this docstring's claim and treating the output as fair --
    _score_derived_lines does not, and says why.

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
                # NOT A QUOTED PRICE. This side was never posted by anyone --
                # it is the remainder of a subtraction between two prices that
                # were. Labelling it with the feed's name would claim a quote
                # that does not exist, which is exactly the ambiguity that made
                # 16 implausible method legs unattributable in the builder.
                "source": "derived",
                "source_is_vig_free": None,
            })
    return pd.DataFrame(rows)
