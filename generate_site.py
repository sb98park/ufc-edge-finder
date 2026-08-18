"""
Generates docs/index.html: live odds/props grouped by real upcoming fight
cards, with a standout-props section flagging the biggest model-vs-market
disagreements. Run by GitHub Actions on a schedule; can also run locally:

    ODDS_API_KEY=your_key python generate_site.py
"""

import datetime as dt
import json
import os
import re
import unicodedata
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.build_pit_stats import enrich_roster
from src.rationale import set_card_cohort
from jinja2 import Environment, FileSystemLoader

from src.elo import EloRatingSystem
from src.edge_finder import find_all_edges
from src.live_props import get_live_props, record_edge_health
from src.odds_utils import measure_overrounds, set_measured_overrounds
from src.card_matcher import (
    load_fight_cards, group_edges_by_card, top_standout_props, top_disagreement_props, top_favorite_picks,
    assign_canonical_fight_ids, group_unmatched_by_fight,
    is_pickable_market, price_is_fragile,
)
from src.power_rating import build_effective_ratings
from src.odds_utils import implied_prob_to_american, format_american_odds
from src.parlay_builder import build_bankroll_builder_parlays, build_lotto_parlays
from src.parlay_ledger import record_slips
from src.recommendations import build_recommendations
from src.line_movement import (
    load_snapshot, save_snapshot, annotate_movement, attach_charts_to_fight,
    load_token_cache, save_token_cache, update_token_cache,
)
from src.track_record import (
    log_predictions, compute_track_record, load_momentum_by_key,
    load_logged_predictions_by_key, _pair_key,
    LOCK_OF_WEEK_MAX, LOCK_OF_WEEK_MIN_PROB,
)
from src.schedule import build_fight_schedule, apply_live_corrections, promote_card_if_stale
from src.results_fetcher import fetch_and_log_new_results, fetch_espn_live_fight_key
from src.card_discovery import discover_and_append_new_cards, normalize_existing_card_order, resync_tracked_card_order, deduplicate_tracked_fights
from src.fighter_backfill import backfill_fighters, fill_missing_last_fights, ensure_roster_rows, fill_last_fight_methods
from src.calibration_chart import build_calibration_svg
from src.sparkline_chart import build_sparkline_svg
from src.units_chart import build_units_timeseries_svg
from src.donut_chart import build_donut_svg
from src.damage_silhouette import build_damage_silhouette_svg
from src.fun_facts import compute_fun_facts

DATA_DIR = "data"
OUTPUT_PATH = "docs/index.html"


def _format_friendly_date(date_str: str) -> str:
    """
    "2026-07-18" -> "Sat, Jul 18". Falls back to the raw string on a
    malformed value rather than crashing the whole page over one bad date.
    """
    try:
        parsed = dt.datetime.strptime(date_str, "%Y-%m-%d")
        return parsed.strftime("%a, %b %-d")
    except (ValueError, TypeError):
        return date_str


def _method_display(value):
    """
    Dataset shorthand to the words used everywhere else on the page.

    fight_history.csv stores "DEC" and "SUB" because that is how the raw
    UFC dataset encodes them, so a last-fight line read "W by DEC against
    Max Holloway" while the market table two inches below said "Decision".
    Mapped at DISPLAY time rather than in the data: the stored codes are
    what the model reads, and rewriting them in place would mean every
    consumer needed to know about both forms.
    Anything unrecognised passes through -- ESPN returns full phrases
    like "Submission (rear-naked choke)" and those are already readable.

    MODULE LEVEL, not a closure inside main(): fight_results.csv carries the
    same shorthand (7 rows of "DEC"/"SUB" alongside rows of "Decision -
    Unanimous"), so the concluded-fight labels built ~350 lines above the
    Jinja environment need it too. It was registered as a template filter
    only, which meant the two server-side result strings were the one place
    on the page where a raw "DEC" could reach a reader.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    return {
        "dec": "Decision", "decision": "Decision",
        "sub": "Submission", "submission": "Submission",
        "ko": "KO/TKO", "tko": "KO/TKO", "ko/tko": "KO/TKO",
    }.get(value.strip().lower(), value.strip())


def _result_label(winner: str, method) -> str:
    """
    The one-line "who won and how" used where there is NO arrow beside the
    fighter's name -- the what's-new feed and the countdown banner's
    just-concluded line.

    ONE PRODUCER, because there were two: this string was built inline both
    on the fight object and again on the just_concluded payload, in the same
    shape by coincidence rather than by construction. Two copies of a display
    format drift, and the drift is silent.

    Surname only, deliberately: the banner already prints "Islam Makhachev
    vs. Ian Machado Garry" immediately before it, so the full name would be
    the third time the winner's name appears in one sentence.

    NOT SHOUTED. The surname was uppercased and the connective was "BY",
    which is fine for the caps-styled result slot on the fight card but not
    for this string, which is dropped raw into the middle of a prose line:
    "Islam Makhachev vs. Ian Machado Garry — MAKHACHEV BY Decision -
    Unanimous". Casing belongs to the slot that renders the text, and the
    slot that wants caps (result_method) applies its own.
    """
    return f"{str(winner).strip().split()[-1]} by {_method_display(str(method).strip())}".strip()


def build_ratings(fighters_df: pd.DataFrame, history_df: pd.DataFrame) -> dict[str, float]:
    elo = EloRatingSystem()
    elo.build_from_history(history_df)
    return build_effective_ratings(fighters_df, elo.ratings, history_df)


def _soonest(event_list):
    """
    The chronologically next event, not merely the first in the list.

    future_events inherits future_cards.csv's row order, which is
    discovery/append order -- the dedupe and resync helpers rewrite that
    file by concatenating groups without ever sorting by date. Trusting
    position here is what put a card 19 days out in front of one happening
    that same weekend.
    """
    if not event_list:
        return None
    dated = [e for e in event_list if e.get("event_date")]
    if not dated:
        return event_list[0]
    return min(dated, key=lambda e: str(e["event_date"]))


def main():
    cards_df = load_fight_cards(f"{DATA_DIR}/fight_cards.csv")

    try:
        current_event_name = cards_df["event_name"].iloc[0] if not cards_df.empty else None
        discover_and_append_new_cards(f"{DATA_DIR}/future_cards.csv", current_event_name=current_event_name)
    except Exception as e:
        print(f"[generate_site] card discovery failed unexpectedly, continuing without it: {e}")

    try:
        deduplicate_tracked_fights(f"{DATA_DIR}/future_cards.csv")
    except Exception as e:
        print(f"[generate_site] fight deduplication failed unexpectedly, continuing without it: {e}")

    try:
        resync_tracked_card_order(f"{DATA_DIR}/future_cards.csv")
    except Exception as e:
        print(f"[generate_site] card order resync against ESPN failed unexpectedly, continuing without it: {e}")

    # The active card gets NONE of the above once promoted out of
    # future_cards.csv -- confirmed this was a real, total gap, not a
    # partial one: no self-healing function ever targeted fight_cards.csv
    # at all, so a real fighter replacement on the CURRENT event (the
    # exact scenario this project's own code already anticipated in a
    # comment: "Rountree Jr." replaced by "Guskov") could never be picked
    # up, no matter how many runs passed. Same functions, same ESPN-is-
    # ground-truth resync, just also pointed at the currently-active card.
    try:
        deduplicate_tracked_fights(f"{DATA_DIR}/fight_cards.csv")
    except Exception as e:
        print(f"[generate_site] active-card deduplication failed unexpectedly, continuing without it: {e}")

    try:
        resync_tracked_card_order(f"{DATA_DIR}/fight_cards.csv")
    except Exception as e:
        print(f"[generate_site] active-card resync against ESPN failed unexpectedly, continuing without it: {e}")

    # fight_cards.csv is meant to hold exactly ONE active event, unlike
    # future_cards.csv where several legitimately coexist -- so unlike
    # that file, the resync above finding more than one distinct
    # event_name here means a real fighter replacement just got detected
    # (the old name's rows kept conservatively as "maybe just a transient
    # ESPN gap," appended after the fresh ones rather than removed).
    # resync_tracked_card_order always places freshly-confirmed ESPN
    # matches first and appends orphaned/stale rows last, so the first
    # event_name present is the one just confirmed live -- verified this
    # ordering directly rather than assumed it. Keep only that one, drop
    # the stale leftover entirely rather than leave the active card
    # representing two different events at once.
    try:
        active_df = pd.read_csv(f"{DATA_DIR}/fight_cards.csv")
        distinct_names = active_df["event_name"].unique()
        if len(distinct_names) > 1:
            keep_name = active_df["event_name"].iloc[0]
            print(f"[generate_site] active card had {len(distinct_names)} distinct event names after resync "
                  f"({list(distinct_names)}) -- keeping only '{keep_name}', dropping the rest")
            active_df[active_df["event_name"] == keep_name].to_csv(f"{DATA_DIR}/fight_cards.csv", index=False)
    except Exception as e:
        print(f"[generate_site] active-card single-event cleanup failed unexpectedly, continuing without it: {e}")

    cards_df = load_fight_cards(f"{DATA_DIR}/fight_cards.csv")

    try:
        normalize_existing_card_order(f"{DATA_DIR}/future_cards.csv")
    except Exception as e:
        print(f"[generate_site] card order normalization failed unexpectedly, continuing without it: {e}")

    # Backfill against BOTH card files. Previously only future_cards.csv was
    # passed, so any fighter who appeared on the CURRENT card without first
    # having been on a tracked future card never got a roster row at all --
    # and predict_matchup returns None when either fighter is missing, which
    # silently strips the ENTIRE preview (confidence, tale of the tape,
    # reasoning, waterfall) leaving only the moneyline chart. That's how a
    # roster gap shows up as a rendering bug. Real case: four prelim fighters
    # on a live card, three fights rendered bare.
    for _cards in (f"{DATA_DIR}/fight_cards.csv", f"{DATA_DIR}/future_cards.csv"):
        try:
            backfill_fighters(f"{DATA_DIR}/fighters.csv", _cards)
        except Exception as e:
            print(f"[generate_site] fighter backfill failed for {_cards}, continuing: {e}")

    # Runs regardless of whether backfill_fighters took its early return --
    # that early exit is precisely why fighters on future cards kept ending
    # up with no last fight at all.
    # Roster rows FIRST: a fighter with no row at all produces no model
    # preview, no tale of the tape and no radar -- the entire fight renders
    # empty. The main backfill matches by event name and silently drops what
    # it misses; this catches those by date instead.
    try:
        ensure_roster_rows(f"{DATA_DIR}/fighters.csv",
                           (f"{DATA_DIR}/fight_cards.csv", f"{DATA_DIR}/future_cards.csv"))
    except Exception as e:
        print(f"[generate_site] roster top-up failed, continuing: {e}")

    try:
        fill_missing_last_fights(f"{DATA_DIR}/fighters.csv",
                                 (f"{DATA_DIR}/fight_cards.csv", f"{DATA_DIR}/future_cards.csv"))
    except Exception as e:
        print(f"[generate_site] last-fight fill failed, continuing: {e}")

    # LAST, after fill_missing_last_fights -- that can set a new last fight,
    # and running the method fill before it would leave those blank. Reads
    # local history rather than ESPN: the eventsMap reports status as "Final",
    # a completion state rather than a method, which is why this field was
    # empty and rendered as "L by None".
    try:
        fill_last_fight_methods(f"{DATA_DIR}/fighters.csv", f"{DATA_DIR}/fight_history.csv")
    except Exception as e:
        print(f"[generate_site] last-fight method fill failed, continuing: {e}")


    fighters_df = pd.read_csv(f"{DATA_DIR}/fighters.csv")
    # RATE COLUMNS FROM REAL PER-BOUT HISTORY. fighters.csv's rates are a
    # partial ufcstats scrape -- control_time_pct sits at 0% and slpm/sapm at
    # ~1% on the booked card -- so the style layer's control-time and volume
    # branches were unreachable in production while the harnesses could reach
    # them from data/pit_stats.csv. Same source, same estimator, both sides.
    # Falls through to whatever fighters.csv has for anyone unmatched.
    fighters_df = enrich_roster(fighters_df)
    print(f"[pit_stats] rate columns filled for "
          f"{fighters_df.attrs.get('pit_stats_filled', 0)} of {len(fighters_df)} fighters")
    history_df = pd.read_csv(f"{DATA_DIR}/fight_history.csv")
    elo_ratings = build_ratings(fighters_df, history_df)

    future_cards_df = load_fight_cards(f"{DATA_DIR}/future_cards.csv")

    # Register the booked cohort so the copy can make card-level claims
    # ("the least of anyone booked here"). AFTER the load, obviously -- the
    # first draft of this referenced future_cards_df one line above its own
    # assignment. A census of the card, not a sample; see set_card_cohort.
    try:
        _booked = pd.concat([future_cards_df["fighter_a"], future_cards_df["fighter_b"]]).dropna().unique()
        set_card_cohort(fighters_df, list(_booked))
        print(f"[rationale] card cohort registered: {len(_booked)} booked fighters")
    except Exception as _e:
        print(f"[rationale] card cohort unavailable ({_e}) -- card-level claims disabled")
    pre_promotion_event_name = cards_df["event_name"].iloc[0] if not cards_df.empty else None
    cards_df, future_cards_df, days_since_event = promote_card_if_stale(cards_df, future_cards_df)

    if not cards_df.empty and cards_df["event_name"].iloc[0] != pre_promotion_event_name:
        # A promotion actually happened this run -- persist it. Without
        # this, fight_cards.csv's on-disk "current" event never advances
        # past whatever it was the very first time this ever fired, since
        # nothing else writes the result back -- every future run would
        # silently re-derive the same stale promotion from the same
        # frozen starting point forever, never able to progress to
        # whatever's genuinely next. Confirmed this was already happening
        # live: fight_cards.csv still held the very first card this
        # project ever tracked, even after the site had already moved on
        # (in-memory only, every run) to a later one.
        try:
            cards_df.to_csv(f"{DATA_DIR}/fight_cards.csv", index=False)
            future_cards_df.to_csv(f"{DATA_DIR}/future_cards.csv", index=False)
            print(f"[generate_site] promoted '{pre_promotion_event_name}' -> '{cards_df['event_name'].iloc[0]}', persisted to disk")
        except Exception as e:
            print(f"[generate_site] promotion persistence failed unexpectedly, this run's HTML is still correct "
                  f"but the promotion may need to re-happen next run: {e}")

    try:
        weight_class_history_df = pd.read_csv(f"{DATA_DIR}/fighter_weight_class_history.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        weight_class_history_df = pd.DataFrame(columns=["name", "date", "weight_class"])

    live_error = None
    edges_df = pd.DataFrame()
    source = None
    previous_snapshot = load_snapshot()

    try:
        # Hand the market scrapers the fighters we're actually tracking, so
        # Polymarket can recognise a fight event by its participants instead
        # of relying on their title format staying stable.
        _tracked = []
        for _df in (cards_df, future_cards_df):
            if _df is not None and not _df.empty and {"fighter_a", "fighter_b"} <= set(_df.columns):
                _dates = _df["event_date"] if "event_date" in _df.columns else [None] * len(_df)
                for _a, _b, _d in zip(_df["fighter_a"], _df["fighter_b"], _dates):
                    if pd.notna(_a) and pd.notna(_b):
                        # The event date is part of Polymarket's slug format,
                        # so it has to travel with the bout.
                        _tracked.append((str(_a), str(_b), str(_d) if pd.notna(_d) else None))
        upcoming_df, source = get_live_props(known_fighters=_tracked)
        all_known_cards = pd.concat([cards_df, future_cards_df], ignore_index=True)
        upcoming_df = assign_canonical_fight_ids(upcoming_df, all_known_cards)
        # WHAT THE BOOKS ARE REALLY CHARGING, measured before anything is
        # priced. The overround constants were averaged off 5,646 historical
        # bouts; DraftKings and FanDuel now quote this card live, so the real
        # margin is available and every consumer -- staking, the slip builder,
        # the rationale copy -- reads it through overround_for_market.
        # Falls back to the constants when too few book quotes were seen.
        _measured = measure_overrounds(upcoming_df.to_dict("records"))
        set_measured_overrounds(_measured)
        print(f"[vig] measured overrounds {_measured}")

        edges_df = find_all_edges(upcoming_df, fighters_df, elo_ratings, history_df)
        record_edge_health(edges_df)

        if not edges_df.empty:
            edge_records = edges_df.to_dict("records")
            annotate_movement(edge_records, previous_snapshot)
            edges_df = pd.DataFrame(edge_records)

        if edges_df.empty:
            live_error = f"No usable live odds returned right now (source: {source})."
    except Exception as exc:
        # NAME THE FAILURE HONESTLY. This block covers the fetch AND every
        # edge computation after it, and it used to report all of them as
        # "live odds fetch failed". A NameError in compute_moneyline_edges
        # therefore looked identical to a network outage: the site rendered
        # with no edges, no standout props and no parlays, and the empty
        # sections were read as a thin market rather than a code bug. That
        # cost several builds.
        #
        # A network or HTTP error is a fetch problem; anything else is ours.
        import traceback as _tb
        _is_network = isinstance(exc, (OSError, TimeoutError)) or \
            exc.__class__.__module__.startswith(("requests", "urllib", "http"))
        if _is_network:
            print(f"[generate_site] live odds FETCH failed: {type(exc).__name__}: {exc}")
            live_error = "Couldn't fetch live odds right now — will retry on the next update."
        else:
            print(f"[generate_site] EDGE COMPUTATION FAILED (this is a bug, not the feed): "
                  f"{type(exc).__name__}: {exc}")
            _tb.print_exc()
            live_error = "Live odds are temporarily unavailable."

    # history_df threaded through so the PREVIEW runs the same model as the
    # moneyline edge row -- without it the headline pick omitted the
    # recent-form adjustment and the weight-class penalty, and the two
    # disagreed on screen for the same fight.
    events, unmatched_df = group_edges_by_card(edges_df, cards_df, fighters_df, elo_ratings, weight_class_history_df, history_df)
    future_events, still_unmatched_df = group_edges_by_card(unmatched_df, future_cards_df, fighters_df, elo_ratings, weight_class_history_df, history_df)

    # Event display order must be chronological (soonest first), independent
    # of whatever order their rows happen to sit in the source CSV -- that
    # order reflects when each card was discovered or re-discovered (e.g.
    # after a lineup-change replacement), not the event's actual date. This
    # is a separate concern from fight order WITHIN one event (billing
    # order, Main Event first), which group_edges_by_card already handles.
    events.sort(key=lambda e: e["event_date"])
    future_events.sort(key=lambda e: e["event_date"])

    # CANCELLED FIGHTS ARE NOT VALUE. `cancelled` was checked in exactly two
    # places in the template and nowhere here, so a void bout kept feeding
    # Standout Props, Favorite Picks and Parlays -- every one of which is a
    # recommendation to bet. A parlay is worse than a prop: it prices a
    # combined payout and a combined hit probability that both include a leg
    # that cannot settle.
    #
    # Filtered at the source rather than in each of the three consumers, so a
    # fourth consumer added later inherits the guard instead of re-earning it.
    # fight_key is STAMPED HERE because this is the only place both the edge
    # and the fight it belongs to are in scope. Edge rows carry fighter and
    # opponent, but fight-level markets (Total Rounds, Fight Outcome) set no
    # opponent at all, so the pair cannot be reconstructed downstream. Live
    # parlay grading matches on this key against the same canonicalisation the
    # ESPN poller uses -- see canonicalKey in the template.
    tracked_edges = pd.DataFrame(
        [dict(edge, fight_key=f"{fight.get('fighter_a')}|{fight.get('fighter_b')}")
         for event in events for fight in event["fights"]
         if not fight.get("cancelled") for edge in fight["edges"]]
    )

    # Standout Props / Favorite Picks / Parlays are meant to answer "where
    # does the model see value RIGHT NOW" -- once the current card's own
    # markets have closed (fight's over, nothing left to price), that
    # question has no honest answer for THIS card anymore, even though
    # "This Weekend" correctly keeps showing its result for a day per the
    # days-since-event display logic elsewhere. Rather than showing these
    # sections empty for a full day, fall back to the next tracked event's
    # edges once the current card's own pool is genuinely thin -- with an
    # explicit flag so the template can label which event is actually
    # being shown, since silently swapping the underlying event without
    # saying so would be confusing, not helpful.
    analytics_source_event = None
    MIN_EDGES_FOR_CURRENT_CARD = 3  # below this, the current card's pool is too thin to be a real signal
    # Only ever fall back once the current card has ACTUALLY happened.
    # Thin edges alone aren't proof a card is over -- a card promoted in
    # from future_cards.csv can legitimately have few or no odds posted
    # yet, and falling back there would claim "this weekend's card has
    # concluded" about a card that hasn't happened. Anchoring on the
    # event date rather than on confirmed results is deliberate: results
    # can genuinely fail to auto-confirm (ESPN publishes no usable
    # method-of-victory text, see results_fetcher), and the fallback
    # shouldn't depend on a source that might never land. This is also
    # exactly what retires the banner on the Monday handoff: once
    # promote_card_if_stale swaps the next card in as current, its date
    # is in the future, so the fallback stops applying on its own.
    current_card_has_happened = False
    if not cards_df.empty:
        try:
            _current_event_date = dt.date.fromisoformat(str(cards_df["event_date"].iloc[0]))
            current_card_has_happened = _current_event_date <= dt.datetime.now(
                dt.timezone(dt.timedelta(hours=-4))
            ).date()
        except (ValueError, TypeError):
            current_card_has_happened = False
    if current_card_has_happened and len(tracked_edges) < MIN_EDGES_FOR_CURRENT_CARD and future_events:
        next_event = _soonest(future_events)
        next_tracked_edges = pd.DataFrame(
            [edge for fight in next_event["fights"] for edge in fight["edges"]]
        )
        if len(next_tracked_edges) > len(tracked_edges):
            tracked_edges = next_tracked_edges
            analytics_source_event = next_event["event_name"]
            events_for_model_only = [next_event]
        else:
            events_for_model_only = events
    else:
        events_for_model_only = events

    def _is_complement_row(e):
        """
        "Not KO/TKO" carries nothing its counterpart doesn't -- it's the same
        market mirrored, so it renders as a second line saying -239 -> -144
        beside +239 -> +144. The markets table already strips these; movement
        and standout props never got the same filter, which doubled the length
        of both sections for no added information.
        """
        market = str(e.get("market", "") or "").lower()
        selection = str(e.get("selection", "") or "").lower()
        tail = market.split(":")[-1]
        if "not" in tail or selection.startswith("not"):
            return True
        return "ends in finish" in f"{market} {selection}"

    # Same complement filter before standout props are chosen, so a "Not X"
    # row can't take one of the five slots from a real market.
    #
    # AND THE SAME MARKET-QUALITY FENCE THE OTHER TWO PICK SURFACES USE.
    # Favorite Picks applies it (card_matcher.top_favorite_picks) and the
    # parlay builder applies it (parlay_builder, whose comment records this
    # exact bug being found there: 13 of 32 legs on one card were Over 0.5
    # rounds). Standout Props never did, and it is the section at the TOP of
    # the page -- so the flagship "worth a look" list shipped as four of five
    # "Total Rounds Over 0.5", each rendered at the hottest heat level with a
    # 32-38 point divergence bar.
    #
    # Those are not edges. A 0.5-round line asks whether the fight passes
    # 2:30 of round one; Polymarket had all four sitting at even money
    # (-104/+102/-106/-102) because nobody is pricing them. The gap between a
    # ~50% quote and a ~99% outcome is the book not making a market, and
    # presenting it as the model's best find of the week is the single most
    # misleading thing on the page. card_matcher's fence comment argues this
    # at length; the argument was never in question here, the filter was just
    # never wired up on this third surface.
    #
    # The Edges tab still shows every one of these, deliberately -- it exists
    # to show every disagreement, and it flags fragile prices rather than
    # hiding them. A headline pick list is a recommendation, which is a
    # different claim.
    _sp_source = tracked_edges
    if not tracked_edges.empty:
        _keep = [
            not _is_complement_row(r) and is_pickable_market(r) and not price_is_fragile(r)
            for r in tracked_edges.to_dict("records")
        ]
        _sp_source = tracked_edges[_keep]
    standout_props = top_standout_props(_sp_source, fighters_df, n=5)
    # The old headline list, kept behind a disclosure with its record attached
    # rather than deleted -- see top_disagreement_props.
    disagreement_props = top_disagreement_props(_sp_source, fighters_df, n=5, min_edge=5.0)

    # Fun facts: genuinely notable patterns for fighters on the current
    # card (active method streaks, career purity, win streaks) -- gated
    # in src/fun_facts.py so an empty list on a quiet week is the
    # correct outcome, and the whole hub card/section simply doesn't
    # render then (see template).
    #
    # The SECTION follows the same event the other analytics sections
    # follow (events_for_model_only), so it falls back to next weekend's
    # card in step with Standout Props / Favorite Picks / Parlays rather
    # than sitting on a concluded card alone. Facts are still computed
    # for BOTH cards' fighters, though, so the per-fight chips keep
    # rendering on whichever fight cards are actually on screen -- the
    # chips live on the fight cards in This Weekend, which keeps showing
    # the concluded card for a day after the fallback kicks in.
    def _fighter_names(event_list):
        return {
            n for event in event_list for fight in event["fights"]
            for n in (fight.get("fighter_a"), fight.get("fighter_b")) if n
        }

    section_fighter_names = _fighter_names(events_for_model_only)
    all_fact_fighter_names = sorted(_fighter_names(events) | section_fighter_names)
    all_fun_facts = compute_fun_facts(all_fact_fighter_names, f"{DATA_DIR}/fight_history.csv", fighters_df)
    fun_facts_by_fighter = {f["fighter"]: f for f in all_fun_facts}
    # compute_fun_facts returns rarity-sorted, so filtering preserves that order.
    fun_facts = [f for f in all_fun_facts if f["fighter"] in section_fighter_names]
    favorite_picks = top_favorite_picks(tracked_edges, fighters_df, n=5)

    # CARRY THE NAMED RISK ONTO THE FIGHT so log_predictions can store it.
    # explain_favorite_pick writes pick_falsifier onto the row it is given, so
    # taking it from there -- rather than recomputing it -- guarantees the risk
    # we log is the one the reader was actually shown. Recomputing later is not
    # an option: after the fight, predict_matchup runs against ratings that
    # already absorbed the result.
    _fals = {}
    for _p in favorite_picks:
        if _p.get("pick_falsifier"):
            _fals[_pair_key(_p.get("fighter"), _p.get("opponent"))] = _p["pick_falsifier"]
    if _fals:
        for _ev in events:
            for _ft in _ev["fights"]:
                _k = _pair_key(_ft.get("fighter_a"), _ft.get("fighter_b"))
                if _k in _fals and _ft.get("preview"):
                    _ft["preview"]["pick_falsifier"] = _fals[_k]

    tracked_edges_list = tracked_edges.to_dict("records") if not tracked_edges.empty else []
    for e in tracked_edges_list:
        # Fight-level rows (GoesTheDistance, "Fight Outcome") never set an
        # "opponent" field. Building a DataFrame from a mix of rows that
        # do and don't have that key fills the gap with NaN, which is
        # truthy in Python -- so a template check like {% if row.opponent %}
        # doesn't filter it out, it just prints the literal word "nan".
        if pd.isna(e.get("opponent")):
            e["opponent"] = None

    # THE CANCELLED FILTER HAS TO BE REPEATED HERE, and the comment on
    # tracked_edges above is the reason it was missed: it says the guard lives
    # "at the source ... so a fourth consumer added later inherits the guard",
    # and then this dict -- built twelve lines below it, feeding the same three
    # parlay builders -- inherited nothing, because it is a second source
    # rather than a consumer of the first.
    #
    # Live consequence, confirmed in a published build: the Moonshot slate
    # carried "Kody Steele vs Gauge Young Over 1.5 rounds" on a bout marked
    # cancelled=True in fight_cards.csv. A cancelled fight cannot settle, so
    # that slip could never win and never lose.
    #
    # fight_key is stamped for the same reason it is stamped on tracked_edges:
    # model-only rows carry a canonical fight_id with no "|" in it whenever
    # the fight has any real edges, so parlay_builder._fight_key fell through
    # to None and the live grader -- which matches legs on fight_key -- could
    # never settle a slip containing one.
    model_only_by_fight = {}
    for event in events_for_model_only:
        for fight in event["fights"]:
            if fight.get("cancelled") or not fight.get("model_only_rows"):
                continue
            fid = fight["edges"][0]["fight_id"] if fight["edges"] else None
            if fid is None:
                fid = f"{fight.get('fighter_a')}|{fight.get('fighter_b')}"
            key = f"{fight.get('fighter_a')}|{fight.get('fighter_b')}"
            model_only_by_fight[fid] = [dict(r, fight_key=key)
                                        for r in fight["model_only_rows"]]

    try:
        record_edge_health(edges_df, tracked_edges_list)
        bankroll_parlays = build_bankroll_builder_parlays(tracked_edges_list, model_only_by_fight)
        lotto_parlays = build_lotto_parlays(tracked_edges_list, model_only_by_fight)
    except Exception as e:
        # Never let a parlay-building bug take the whole site down with it --
        # confirmed live: a single fighter with a NaN power rating (missing
        # reach_in, silently un-defaulted) corrupted a projected price deep
        # in this pipeline and crashed the ENTIRE generate_site.py run before
        # it ever reached the line that writes docs/index.html, freezing the
        # whole site on stale data. The actual data-completeness bugs are
        # fixed at the source now, but this stays as a second line of
        # defense against whatever the next one turns out to be.
        print(f"[parlays] build failed unexpectedly, continuing without parlay sections: {e}")
        bankroll_parlays, lotto_parlays = [], []

    # WRITE DOWN WHAT WE PUBLISHED. Nine slips a week have been going out
    # ungraded while single picks are scored to three decimals on the same
    # page; nothing recorded them, so the record was not merely bad, it did
    # not exist. Merged on slip_id, so the 5-minute rebuild cycle leaves nine
    # rows per card rather than nine per render.
    record_slips(
        {"bankroll": bankroll_parlays, "lotto": lotto_parlays},
        event_name=(events[0].get("event_name") if events else None),
    )

    # THE LEG INVENTORY for the slip builder. Every leg the card offers,
    # priced at book-equivalent odds, for the reader to combine themselves --
    # see src/slip_builder for why that is the product rather than another
    # generated slate.
    # LEGS THE MODEL RECOMMENDS, each with the price it needs to be worth
    # taking. Double Chance and round-start markets are derived from the
    # method grid and the round curve -- the feed does not quote them, so the
    # output is a threshold rather than an edge. See src/recommendations.
    try:
        model_legs = build_recommendations(events, tracked_edges_list)
    except Exception as e:
        print(f"[recommendations] failed, section will be empty: {e}")
        model_legs = []

    # Notable line movement, SPLIT BY CARD rather than pooled. Sorting one
    # combined list purely by pct_change let a big move on a fight three weeks
    # out push this weekend's movement off an 8-row cut entirely -- the card
    # you can actually act on losing to one you can't, for no reason beyond
    # magnitude. Movement on a distant fight is still worth seeing; it just
    # shouldn't compete for the same slots.
    def _notable(edges, limit):
        return sorted(
            [e for e in edges
             if e.get("movement") and e["movement"].get("notable") and not _is_complement_row(e)],
            key=lambda e: e["movement"]["pct_change"], reverse=True,
        )[:limit]

    notable_movements = _notable(tracked_edges_list, 8)
    # Derive "later cards" BY EXCLUSION, not from a separate source. Once the
    # current card has finished, tracked_edges is REPLACED with the next
    # future event -- so that event sat in both lists and every one of its
    # movements rendered twice, once in each section.
    _promoted = {e["event_name"] for e in events_for_model_only}
    notable_movements_upcoming = _notable(
        [edge for event in future_events if event["event_name"] not in _promoted
         for fight in event["fights"] for edge in fight["edges"]], 6
    )

    if not edges_df.empty:
        updated_snapshot = save_snapshot(edges_df.to_dict("records"), previous_snapshot)
    else:
        updated_snapshot = previous_snapshot

    token_cache = load_token_cache()

    for event in events + future_events:
        for fight in event["fights"]:
            attach_charts_to_fight(fight, updated_snapshot, token_cache)

    if not edges_df.empty:
        token_cache = update_token_cache(edges_df.to_dict("records"), token_cache)
        save_token_cache(token_cache)

    generated_at_str = dt.datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p ET")
    # Separate, compact format for the top-of-page display only -- the
    # full generated_at_str above stays untouched since it's also stored
    # in predictions_log.csv and other logic may depend on that exact
    # format; this is purely a second, shorter rendering of the same
    # moment for a spot where space is tight.
    # %-I not %I: the padded form rendered "07:19 PM" while the client-side
    # live-prices chip renders "7:19 PM", so the two timestamps sitting side
    # by side in the same meta row disagreed on format.
    _now_et = dt.datetime.now(ZoneInfo("America/New_York"))
    generated_at_short = _now_et.strftime("%b %-d, %-I:%M %p ET")
    # Time only when the build is from TODAY, which it almost always is -- the
    # date is noise in that case. It's emitted alongside the full form and the
    # client picks, because "today" is the READER's today, not the builder's.
    generated_at_time_only = _now_et.strftime("%-I:%M %p ET")
    generated_at_date = _now_et.strftime("%Y-%m-%d")
    momentum_by_key = load_momentum_by_key()
    for event in events:
        for fight in event["fights"]:
            key = frozenset({fight["fighter_a"].strip().lower(), fight["fighter_b"].strip().lower()})
            fight["momentum"] = momentum_by_key.get(key)
    track_record = compute_track_record()
    calibration_svg = None
    units_sparkline_svg = None
    units_timeseries_svg = None
    if track_record and track_record.get("calibration", {}).get("ready"):
        calibration_svg = build_calibration_svg(track_record["calibration"]["points"])
    if track_record and track_record.get("units_stats") and len(track_record["units_stats"]["running_total"]) >= 2:
        units_sparkline_svg = build_sparkline_svg(track_record["units_stats"]["running_total"])
        units_timeseries_svg = build_units_timeseries_svg(track_record["units_stats"]["running_total"])

    event_short_name = (
        analytics_source_event.split(":")[0].strip() if analytics_source_event
        else events[0]["event_name"].split(":")[0].strip() if events
        else "This Weekend"
    )
    # FULL name, matchup included. event_short_name cuts at the colon, so the
    # Standout Props heading read "Standout Props for UFC Fight Night" -- true
    # of most cards and therefore useless for telling one from another. The
    # matchup is the part that identifies the event.
    event_full_name = (
        analytics_source_event.strip() if analytics_source_event
        else events[0]["event_name"].strip() if events
        else "This Weekend"
    )
    # Just the matchup. The prefix ("UFC Fight Night", "UFC 330") is either
    # generic or already implied, and including it pushed the Standout Props
    # heading onto two lines. The names are what identify the card.
    event_matchup = (
        event_full_name.split(":", 1)[1].strip()
        if ":" in event_full_name else event_full_name
    )

    # Countdown target: this weekend's tracked event if we have one, otherwise
    # the nearest future card. ET is UTC-4 (EDT) for all currently tracked
    # events (July-August) -- would need adjusting for events during EST months.
    countdown_target_iso = None
    countdown_label = None
    next_event = events[0] if events else _soonest(future_events)
    if next_event:
        countdown_target_iso = f"{next_event['event_date']}T{next_event.get('event_start_time_et', '19:00')}:00-04:00"
        countdown_label = next_event["event_name"]

    # Attempt to auto-fetch any results not yet in fight_results.csv,
    # before matching results to fights below. Best-effort and silent on
    # failure by design (see results_fetcher.py's own docstring for the
    # honest caveat on how confident to be in this) -- manual entry via
    # fight_results.csv remains the reliable fallback regardless of
    # whether this succeeds.
    if events:
        try:
            added = fetch_and_log_new_results(events[0]["event_name"], cards_df)
            if added:
                print(f"[generate_site] results_fetcher added {added} new result(s)")
        except Exception as e:
            print(f"[generate_site] results_fetcher failed unexpectedly, continuing without it: {e}")

    # Results already recorded (if any) -- used to mark fights as FINISHED
    # server-side, which is more reliable than a time-based estimate once
    # the user has actually told us the outcome.
    STAT_COLS = [
        "fa_sig_landed", "fa_sig_att", "fb_sig_landed", "fb_sig_att",
        "fa_total_landed", "fa_total_att", "fb_total_landed", "fb_total_att",
        "fa_td_landed", "fa_td_att", "fb_td_landed", "fb_td_att",
        "fa_kd", "fb_kd", "fa_head", "fa_body", "fa_leg", "fb_head", "fb_body", "fb_leg",
    ]
    finished_results = {}
    if os.path.exists("data/fight_results.csv"):
        results_df = pd.read_csv("data/fight_results.csv")
        for _, r in results_df.iterrows():
            if pd.notna(r.get("winner")):
                key = frozenset({str(r["fighter_a"]).strip().lower(), str(r["fighter_b"]).strip().lower()})
                # Decisions always run the full final round (5:00 in modern
                # UFC, every round) -- Google's own convention, and the only
                # honest value when nobody logged a stoppage clock. Finishes
                # use the exact round/time as entered.
                method = str(r.get("method", "")).strip()
                is_decision = method.upper().startswith("DEC")
                end_round = r.get("end_round")
                end_round = int(end_round) if pd.notna(end_round) else None
                end_time = "5:00" if is_decision else (str(r.get("end_time")).strip() if pd.notna(r.get("end_time")) else None)

                stats_present = all(pd.notna(r.get(c)) for c in STAT_COLS)
                stats = None
                if stats_present:
                    stats = {c: int(r[c]) for c in STAT_COLS}

                finished_results[key] = {
                    "winner": r["winner"], "method": method,
                    "end_round": end_round, "end_time": end_time,
                    "stats": stats,
                    "stats_fighter_a": r["fighter_a"] if stats_present else None,
                    "stats_fighter_b": r["fighter_b"] if stats_present else None,
                }

    # Loaded once, outside the loop -- it is the whole log and does not vary
    # per fight. Used to put back the pick that was actually made on any
    # fight that has since been decided; see the block below.
    logged_predictions = load_logged_predictions_by_key()

    for event in events:
        for fight in event["fights"]:
            key = frozenset({fight["fighter_a"].strip().lower(), fight["fighter_b"].strip().lower()})
            result = finished_results.get(key)
            if result:
                fight["winner"] = result["winner"]
                fight["result_label"] = _result_label(result["winner"], result["method"])
                # Method on its own, for the card's result line. The winner is
                # already carried by the arrow beside their name, so repeating
                # it in the middle spends the widest slot on the card
                # restating what the arrow just said. result_label keeps the
                # winner because it's used where there IS no arrow (what's-new
                # feed, countdown ticker).
                #
                # Through _method_display BEFORE the caps, not after: the caps
                # are this slot's own styling, but "DEC" uppercased is still
                # "DEC". fight_results.csv holds both the shorthand and the
                # long form, so without the mapping the card printed "DEC"
                # while the comparison table two sections down printed
                # "Decision" for the very same fight.
                fight["result_method"] = _method_display(str(result["method"]).strip()).upper()
                fight["result_round_time"] = (
                    f"R{result['end_round']} {result['end_time']}"
                    if result["end_round"] and result["end_time"] else None
                )
                fight["result_stats"] = None
                if result["stats"]:
                    # fight_results.csv's fa_/fb_ columns are keyed to
                    # whichever order THAT row was entered in, which may not
                    # match this card's fighter_a/fighter_b order -- swap if
                    # needed so the stats always land on the right side.
                    same_order = (
                        str(result["stats_fighter_a"]).strip().lower() == fight["fighter_a"].strip().lower()
                    )
                    s = result["stats"]
                    fight["result_stats"] = {
                        "a": {k[3:]: s[k] for k in s if k.startswith("fa_" if same_order else "fb_")},
                        "b": {k[3:]: s[k] for k in s if k.startswith("fb_" if same_order else "fa_")},
                    }
                # RESTORE THE PICK WE ACTUALLY MADE. Everything above this
                # line has just told the page how the fight ended -- and the
                # preview attached to it was rebuilt minutes ago from data
                # that already knew. fighter_backfill had rewritten both
                # records from ESPN and merge_results_into_history had fed
                # the bout into the ratings, so re-running the model was a
                # lookup wearing a prediction's clothes: it returns whoever
                # won.
                #
                # Live during UFC 330: the card called Mansur Abdul-Malik at
                # 51%, he was submitted, and two builds later the same card
                # showed Dustin Stoltzfus at 67% as "the pick" -- while the
                # track record section, which reads the log, still correctly
                # showed Mansur. The page was disagreeing with itself, and
                # always in the model's favour.
                #
                # Only the four fields the card presents as the CALL are
                # restored. Everything else in the preview (narrative,
                # method distribution, radar, spotlight chips) is descriptive
                # rather than a claim about the outcome, and a logged row
                # doesn't carry it anyway.
                logged = logged_predictions.get(
                    _pair_key(fight["fighter_a"], fight["fighter_b"])
                )
                if logged and fight.get("preview"):
                    for field in ("favorite", "favorite_prob",
                                  "confidence_label", "likely_method"):
                        if logged.get(field) is not None:
                            fight["preview"][field] = logged[field]
                    fight["preview"]["pick_is_logged"] = True

                    # THE WATERFALL IS NOT DESCRIPTIVE. The category above
                    # drew the line in the wrong place. A block headed "Why
                    # the model likes Mackenzie Dern" is a CLAIM about the
                    # outcome, and it is rebuilt from ratings that already
                    # absorbed the result -- so on a fight we called wrong it
                    # reads as the reasoning for the fighter who won, sitting
                    # directly under a badge naming the one we actually
                    # picked. Six fights on UFC 330 rendered exactly that.
                    #
                    # Restoring wf.favorite from the log would not fix it.
                    # Every number in the block -- the rating gap, each
                    # factor's contribution, the final percentage -- was
                    # recomputed with post-fight knowledge, so a matching
                    # name would just hide the contamination behind a
                    # correct-looking header. The honest options are to
                    # persist the pre-fight object or to stop showing it.
                    #
                    # Dropped rather than persisted because the log has no
                    # column for it and adding one is a schema change; the
                    # frozen call above still renders (fighter, probability,
                    # confidence), so what the card loses is the breakdown,
                    # not the claim. Upcoming fights are untouched -- this
                    # branch only runs once a result exists.
                    fight["preview"]["waterfall"] = None
            else:
                fight["winner"] = None
                fight["result_label"] = None
                fight["result_round_time"] = None
                fight["result_stats"] = None

    # Log predictions AFTER results are matched, not before -- so
    # finished_results.keys() (the set of fights that already have a
    # confirmed result) can be passed through and those predictions
    # locked in, rather than a fight's logged "prediction" silently
    # drifting after the outcome is already known just because the site
    # keeps regenerating while the card sits in "This Weekend."
    log_predictions(events, generated_at_str, decided_keys=set(finished_results.keys()))

    # Attach lock-of-week status back onto each fight for the This
    # Weekend display -- log_predictions() just computed and persisted
    # it, this just reads it back rather than recomputing the same
    # ranking a second time.
    lock_keys = set()
    if os.path.exists("data/predictions_log.csv"):
        lock_df = pd.read_csv("data/predictions_log.csv")
        for _, r in lock_df.iterrows():
            if str(r.get("is_lock_of_week")).strip().lower() == "true":
                lock_keys.add(frozenset({str(r["fighter_a"]).strip().lower(), str(r["fighter_b"]).strip().lower()}))
    for event in events:
        for fight in event["fights"]:
            fkey = frozenset({fight["fighter_a"].strip().lower(), fight["fighter_b"].strip().lower()})
            fight["is_lock_of_week"] = fkey in lock_keys

    # When the analytics sections have fallen back to next weekend's card,
    # that card has no logged predictions yet (log_predictions only runs
    # for the current card), so none of its fights carry a lock
    # designation and the Locks section would sit empty -- or, worse,
    # keep showing the concluded card's locks while every neighbouring
    # section had already moved on. Designate them in memory here using
    # the SAME rule log_predictions uses (top N High Confidence picks by
    # probability), purely for display.
    #
    # Deliberately NOT written to predictions_log.csv: logging a pick
    # this early would freeze its pick_odds for CLV against a market
    # that hasn't settled yet, quietly corrupting the honesty of the
    # track record for the sake of a display detail. These early locks
    # get logged for real, at real odds, on the normal schedule once the
    # card becomes current.
    if analytics_source_event:
        for event in events_for_model_only:
            ranked_high_conf = sorted(
                [f for f in event["fights"]
                 if f.get("preview") and f["preview"].get("confidence_label") == "High Confidence"
                 and f["preview"].get("favorite_prob", 0) >= LOCK_OF_WEEK_MIN_PROB],
                key=lambda f: f["preview"]["favorite_prob"], reverse=True,
            )
            early_lock_ids = {id(f) for f in ranked_high_conf[:LOCK_OF_WEEK_MAX]}
            for fight in event["fights"]:
                fight["is_lock_of_week"] = id(fight) in early_lock_ids

    # Locks of the Week, pulled out into their own flat list for a dedicated
    # section -- previously only visible as a badge on each fight card, so
    # seeing all of them meant checking every fight individually. A lock is
    # about the model's conviction on the fight itself, independent of
    # market price (unlike favorite_picks, which is specifically about
    # favorable odds) -- kept as its own section rather than folded into
    # Favorite Picks, since blending those two different concepts together
    # would blur what each one actually means.
    lock_picks = [
        {
            "fighter_a": fight["fighter_a"], "fighter_b": fight["fighter_b"],
            "weight_class": fight.get("weight_class"), "card_position": fight.get("card_position"),
            "cancelled": bool(fight.get("cancelled")),
            "favorite": fight["preview"]["favorite"], "favorite_prob": fight["preview"]["favorite_prob"],
            "underdog": fight["preview"]["underdog"], "likely_method": fight["preview"]["likely_method"],
            "narrative": fight["preview"]["narrative"],
        }
        for event in events_for_model_only for fight in event["fights"]
        if fight.get("is_lock_of_week") and fight.get("preview")
    ]

    # Results coverage, for This Weekend's card specifically -- surfaced
    # both as a step summary (visible directly in the GitHub Actions run
    # UI, not buried in console logs someone has to think to check) and
    # passed to the template so a gap is visible on the site itself,
    # rather than something only noticed by manually cross-referencing
    # against another source after the fact.
    results_coverage = None
    if events:
        this_weekend_fights = events[0]["fights"]
        # CANCELLED fights are excluded from BOTH sides of this count. They
        # can never produce a result, so counting one as outstanding leaves
        # the banner stuck at "12/13 results confirmed -- some may still be
        # pending" permanently, for a fight that is not pending and never
        # will be. Worse, it makes a genuinely complete card look incomplete,
        # which is exactly the alarm this banner exists to raise and trains
        # the reader to ignore it. Seen live on the Johns vs Rosas
        # cancellation.
        def _is_cancelled(v):
            # `not v` is WRONG here and was wrong in the first version of this
            # fix: read straight from CSV the column is the STRING "False",
            # which is truthy, so every non-cancelled fight was dropped and
            # the banner read 0/0. Test the value, not its presence -- the
            # same trap already documented in recompute_prediction.py for
            # is_lock_of_week.
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() == "true"

        scoreable_fights = [f for f in this_weekend_fights if not _is_cancelled(f.get("cancelled"))]
        total_fights = len(scoreable_fights)
        confirmed_fights = sum(1 for f in scoreable_fights if f.get("result_label"))
        if total_fights:
            results_coverage = {"confirmed": confirmed_fights, "total": total_fights}
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path and total_fights:
            missing = [
                f"{f['fighter_a']} vs {f['fighter_b']}"
                for f in scoreable_fights if not f.get("result_label")
            ]
            with open(summary_path, "a") as f:
                f.write(f"### Results coverage: {confirmed_fights}/{total_fights} — {events[0]['event_name']}\n")
                if missing:
                    f.write(f"**Missing ({len(missing)}):**\n")
                    for m in missing:
                        f.write(f"- {m}\n")
                else:
                    f.write("All fights on this card have results. ✅\n")

    # Fight-by-fight schedule for live-state tracking -- only for THIS
    # WEEKEND's tracked card, since future cards are weeks out and this
    # estimate only matters once a card is imminent/underway. Consumed
    # entirely client-side (compared against the visitor's own clock), so
    # this doesn't need a faster server refresh cadence to stay useful.
    #
    # apply_live_corrections re-anchors the remaining schedule using real
    # confirmed results as ground truth (see schedule.py) instead of
    # trusting the static pre-card estimate as the night actually plays
    # out -- the fix for the reported "feels inaccurate" drift. It also
    # strips confirmed fights out of the schedule entirely, so the
    # client only ever estimates fights that genuinely haven't happened.
    fight_schedule = []
    just_concluded = None
    if events:
        raw_card_rows = cards_df.to_dict("records")
        # Prefer ESPN's PUBLISHED main-card time over the schedule's offset
        # estimate. The offset assumes a constant-length prelim block, which
        # doesn't hold: a US card runs 19:00 -> 21:00 while a short
        # international card runs 10:00 -> 13:00. An explicit segment_starts
        # entry wins over the derived value inside build_fight_schedule, so
        # passing it here is all that's needed; omitting it (ESPN hasn't
        # published times yet) leaves the estimate in place rather than
        # inventing a number.
        _published_main = (events[0].get("event_main_card_time_et")
                           or (raw_card_rows[0].get("event_main_card_time_et") if raw_card_rows else None))
        fight_schedule = build_fight_schedule(
            raw_card_rows, events[0]["event_date"],
            events[0].get("event_start_time_et", "17:00"),
            segment_starts={"Main Card": _published_main} if _published_main else None,
        )
        finished_keys = {
            frozenset({str(r["fighter_a"]).strip().lower(), str(r["fighter_b"]).strip().lower()})
            for r in raw_card_rows
            if frozenset({str(r["fighter_a"]).strip().lower(), str(r["fighter_b"]).strip().lower()}) in finished_results
        }
        # The last CHRONOLOGICALLY concluded fight, for the just-concluded
        # display -- found by walking the schedule (already true fight
        # order) and taking the last one that's confirmed, not by
        # date_added, which doesn't reliably reflect fight order.
        for f in fight_schedule:
            key = frozenset({f["fighter_a"].strip().lower(), f["fighter_b"].strip().lower()})
            if key in finished_keys:
                r = finished_results[key]
                just_concluded = {
                    "fighter_a": f["fighter_a"], "fighter_b": f["fighter_b"],
                    "winner": r["winner"],
                    "result_label": _result_label(r["winner"], r["method"]),
                    "result_round_time": f"R{r['end_round']} {r['end_time']}" if r["end_round"] and r["end_time"] else None,
                }
        fight_schedule, last_confirmed_at = apply_live_corrections(fight_schedule, finished_keys)
        if just_concluded:
            just_concluded["last_confirmed_at"] = last_confirmed_at

    # Countdown banner flanking info (desktop only, see templates/site.html).
    # Location and main-event weight class work off next_event directly, so
    # they're available whether we're counting down to this weekend's fully-
    # analyzed card or a further-out future one. Main Card start time, edge
    # count, and confidence breakdown all specifically need data that only
    # exists for the CURRENTLY TRACKED event (fight_schedule with real
    # estimated times, tracked_edges, per-fight model previews) -- none of
    # that exists yet for a distant future card, so those three stay None
    # rather than fabricate a number for something not actually computed.
    countdown_location = next_event.get("event_location") if next_event else None
    # Short form for the simplified countdown-banner display: venue + city
    # only, dropping the trailing state/country segment. event_location is
    # already built as "Venue, City, State/Country" (see card_discovery.py) --
    # taking the first two comma-separated parts is a safe way to get this
    # without needing a second, separately-plumbed field.
    countdown_location_short = (
        " · ".join(countdown_location.split(", ")[:2]) if countdown_location else None
    )
    countdown_weight_class = None
    if next_event:
        main_event_fight = next(
            (f for f in next_event.get("fights", []) if f.get("card_position") == "Main Event"), None
        )
        countdown_weight_class = main_event_fight.get("weight_class") if main_event_fight else None

    countdown_main_card_time = None
    countdown_edge_count = None
    countdown_confidence_counts = None
    if events and next_event is events[0]:
        main_card_starts = [
            f["estimated_start_iso"] for f in fight_schedule
            if f.get("card_position") == "Main Card" and f.get("estimated_start_iso")
        ]
        if main_card_starts:
            earliest = min(main_card_starts)
            # estimated_start_iso carries the -04:00 ET offset already
            # baked in (see schedule.py) -- parse just the wall-clock time
            # portion rather than re-deriving timezone math here.
            hh, mm = int(earliest[11:13]), int(earliest[14:16])
            period = "AM" if hh < 12 else "PM"
            hh12 = hh % 12 or 12
            countdown_main_card_time = f"{hh12}:{mm:02d} {period} ET"
        countdown_edge_count = len(standout_props)
        confidence_tally = {"High Confidence": 0, "Medium Confidence": 0, "Low Confidence": 0}
        for fight in next_event.get("fights", []):
            if fight.get("cancelled"):
                continue  # a cancelled fight's pick is void -- not part of this card's confidence story
            preview = fight.get("preview")
            # READ the label, do not recompute it. Rebuilding it from the
            # probability alone skips every gate _confidence_label applies --
            # the thin-record cap and the debut cap both need matchup fields
            # this loop does not have -- so the countdown tally would report
            # a Medium the fight card itself shows as Low.
            if preview and preview.get("confidence_label") in confidence_tally:
                confidence_tally[preview["confidence_label"]] += 1
        if sum(confidence_tally.values()) > 0:
            countdown_confidence_counts = confidence_tally


    # ESPN's live-fight signal is only meaningful on the actual event day --
    # deliberately checking the event's own date directly rather than
    # days_since_event, which has different semantics (0 for the entire
    # window from card promotion through the day after the event, not
    # specifically "today is fight day" -- see promote_card_if_stale).
    # Getting this gate wrong would mean a wasted call every 5 minutes for
    # days before the event actually happens.
    espn_live_fight_key = None
    if events:
        try:
            today_et = dt.datetime.now(dt.timezone(dt.timedelta(hours=-4))).date()
            event_date_actual = dt.date.fromisoformat(str(events[0]["event_date"]))
            if today_et == event_date_actual:
                known_fighters_lower = {
                    str(n).strip().lower() for n in pd.concat([cards_df["fighter_a"], cards_df["fighter_b"]])
                }
                espn_live_fight_key = fetch_espn_live_fight_key(
                    events[0]["event_name"], events[0]["event_date"], known_fighters_lower
                )
        except Exception as e:
            print(f"[generate_site] ESPN live-status lookup failed unexpectedly, continuing without it: {e}")
            espn_live_fight_key = None

    env = Environment(loader=FileSystemLoader("templates"))
    env.filters["american"] = format_american_odds
    # Probability -> the price at which a bet on it breaks even, i.e. the
    # model's own fair line. Both existing helpers already exist; this just
    # composes them so a template can render the Model column in the same
    # unit as the Odds column beside it.
    env.filters["fair_odds"] = lambda p: (
        format_american_odds(implied_prob_to_american(float(p))) if p is not None else "")
    env.filters["friendly_date"] = _format_friendly_date

    # Defined at module level (see above) so the concluded-fight result
    # strings built earlier in this function can use the same mapping the
    # templates do; this just exposes it to Jinja under its filter name.
    env.filters["method_display"] = _method_display
    env.globals["donut_svg"] = build_donut_svg
    env.globals["damage_svg"] = build_damage_silhouette_svg

    env.filters["tojson"] = lambda obj: json.dumps(obj, default=str)
    # NaN is truthy in Python, so a plain {% if x %} check doesn't catch a
    # pandas-filled missing value -- it just prints the literal word
    # "nan". This test explicitly excludes both None and NaN (the classic
    # "x != x" is only ever true for NaN) so templates can check
    # "is real_value" instead of relying on Jinja's default truthiness.
    # Also reject EMPTY strings. This tested only for None and NaN, so when
    # fight-level rows started carrying `opponent: ""` -- deliberately, to
    # stop pandas filling the gap with NaN and crashing the name normaliser --
    # every "{{ fighter }}{% if opponent is real_value %} vs ..." rendered a
    # trailing "vs" with nothing after it.
    # A blank string is not a real value in any of the places this test is
    # used, so the fix belongs here rather than at each of the four call sites.
    env.tests["real_value"] = lambda x: (
        x is not None and x == x and not (isinstance(x, str) and not x.strip())
    )

    def clear_market_label(market, fighter):
        """
        "Method: KO/TKO" alone doesn't say WHICH fighter -- shown next to
        a "Fighter A vs Fighter B" line, a reader has no way to tell if
        it's A or B winning by KO/TKO. Rewriting it to explicitly name
        the fighter removes the ambiguity instead of relying on the
        reader to correctly guess which name it's attached to.
        """
        if not market or not fighter:
            return market
        if market.startswith("Method: "):
            return f"{fighter} by {market[len('Method: '):]}"
        if market.startswith("Round Betting: "):
            return f"{fighter} — {market[len('Round Betting: '):]}"
        return market

    def short_market_label(market):
        """
        Strips the "Method: " / "Fight Outcome: " prefix for tables that
        already show the fighter or selection in their own column right
        next to it -- unlike clear_market_label above, there's no
        ambiguity to resolve here, so the prefix is pure redundancy that
        costs real width on narrow screens for no added clarity.
        """
        if not market:
            return market
        for prefix in ("Method: ", "Fight Outcome: ", "Round Betting: "):
            if market.startswith(prefix):
                return market[len(prefix):]
        return market

    env.filters["clear_market"] = clear_market_label
    env.filters["short_market"] = short_market_label
    template = env.get_template("site.html")

    # Lightweight snapshot for the "what's new since your last visit" strip --
    # deliberately minimal (just enough to diff against) rather than dumping
    # full row objects, since this gets embedded directly in the page and
    # compared client-side via localStorage.
    whats_new_snapshot = {
        "standout": [
            {"key": f"{p['fighter']}|{p['market']}", "label": f"{p['fighter']} {p['market']}", "edge_pct": p["edge_pct"]}
            for p in standout_props
        ],
        "movements": [
            {"key": f"{m['fighter']}|{m['market']}", "label": f"{m['fighter']} {m['market']}", "pct_change": m["movement"]["pct_change"]}
            for m in notable_movements
        ],
        "results": [
            {"key": f"{r['fighter_a']}|{r['fighter_b']}", "label": f"{r['fighter_a']} vs. {r['fighter_b']}"}
            for r in (track_record["results"] if track_record else [])
        ],
    }

    # Stable id for the deferred movement fragments, assigned BEFORE render so
    # the page and the files on disk use the same value by construction.
    #
    # fight_id looked like the obvious key and isn't present on every fight --
    # in the template that would have rendered an empty URL and failed
    # silently at fetch time rather than at build time. Derived from the two
    # names instead, which every fight has.
    def _movements_id(fight):
        raw = f"{fight.get('fighter_a', '')}-vs-{fight.get('fighter_b', '')}"
        slug = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-") or "fight"

    for _ev in list(events) + list(future_events):
        for _f in _ev.get("fights", []):
            _f["movements_id"] = _movements_id(_f)

    html = template.render(
        events=events,
        future_events=future_events,
        unmatched=unmatched_df.to_dict("records") if not unmatched_df.empty else [],
        standout_props=standout_props,
        disagreement_props=disagreement_props,
        fun_facts=fun_facts,
        fun_facts_by_fighter=fun_facts_by_fighter,
        favorite_picks=favorite_picks,
        lock_picks=lock_picks,
        event_short_name=event_short_name,
        event_full_name=event_full_name,
        event_matchup=event_matchup,
        countdown_target_iso=countdown_target_iso,
        fight_schedule_json=json.dumps(fight_schedule),
        just_concluded_json=json.dumps(just_concluded),
        espn_live_fight_key_json=json.dumps(espn_live_fight_key),
        days_since_event=days_since_event,
        results_coverage=results_coverage,
        analytics_source_event=analytics_source_event,
        countdown_label=countdown_label,
        countdown_location=countdown_location,
        countdown_location_short=countdown_location_short,
        countdown_weight_class=countdown_weight_class,
        countdown_main_card_time=countdown_main_card_time,
        countdown_edge_count=countdown_edge_count,
        countdown_confidence_counts=countdown_confidence_counts,
        whats_new_snapshot=whats_new_snapshot,
        track_record=track_record,
        calibration_svg=calibration_svg,
        units_sparkline_svg=units_sparkline_svg,
        units_timeseries_svg=units_timeseries_svg,
        bankroll_parlays=bankroll_parlays,
        lotto_parlays=lotto_parlays,
        model_legs=model_legs,
        notable_movements=notable_movements,
        notable_movements_upcoming=notable_movements_upcoming,
        live_error=live_error,
        source=source,
        generated_at=generated_at_str,
        generated_at_short=generated_at_short,
        generated_at_time_only=generated_at_time_only,
        generated_at_date=generated_at_date,
    )

    os.makedirs("docs", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    # Secondary movement charts, one fragment per fight, fetched when the
    # reader opens that fight's "other line movements".
    #
    # These were 2.76MB of a 4.26MB page -- 93% of it -- for 11-15 charts per
    # fight sitting inside a collapsed <details>. Written as the SAME
    # Python-rendered SVG rather than re-implemented in JS, so the two can't
    # diverge the way every other duplicated calculation here has.
    #
    # Stale fragments are cleared first: a fight dropping off the card would
    # otherwise leave its file behind forever, and the directory only ever
    # grows in a repo that already has a size problem.
    mv_dir = os.path.join("docs", "movements")
    if os.path.isdir(mv_dir):
        for old in os.listdir(mv_dir):
            if old.endswith(".html"):
                os.remove(os.path.join(mv_dir, old))
    os.makedirs(mv_dir, exist_ok=True)

    written = total = 0
    for ev in list(events) + list(future_events):
        for fight in ev.get("fights", []):
            charts = fight.get("other_charts") or []
            if not charts:
                continue
            frag = "".join(
                f'<div class="chart-block">'
                f'<div class="chart-label">{c["label"]}</div>{c["svg"]}</div>'
                for c in charts
            )
            path = os.path.join(mv_dir, f'{fight["movements_id"]}.html')
            with open(path, "w") as f:
                f.write(frag)
            written += 1
            total += len(frag)

    print(f"Wrote {OUTPUT_PATH} ({len(events)} events, {len(future_events)} future events, {len(standout_props)} agreed reads, {len(disagreement_props)} disagreements)")
    print(f"Wrote {written} movement fragment(s), {total/1e6:.2f}MB deferred out of the page")


if __name__ == "__main__":
    main()
