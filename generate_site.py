"""
Generates docs/index.html: live odds/props grouped by real upcoming fight
cards, with a standout-props section flagging the biggest model-vs-market
disagreements. Run by GitHub Actions on a schedule; can also run locally:

    ODDS_API_KEY=your_key python generate_site.py
"""

import datetime as dt
import json
import argparse
import os
import re
import unicodedata
from zoneinfo import ZoneInfo


# REAL EASTERN, not a fixed -04:00. Eastern is -04:00 only between the second
# Sunday in March and the first Sunday in November; the rest of the year it is
# -05:00. Every site that stamped or read ET with a hardcoded -4 was an hour
# early from November through mid-March: the countdown reached zero an hour
# before first bell, every estimated fight window opened an hour early (pushing
# "LIVE NOW" about one fight ahead of reality for the whole night), and the
# date used at ~line 1413 rolled over between 23:00 and 00:00 EST, so the
# server stopped fetching the live-fight key during the main event of every
# winter card.
#
# Not hypothetical -- UFC Fight Night: Bonfim vs. Brady is already tracked for
# 2026-11-07, six days after DST ends. event_start_time_et was always correct
# (card_discovery converts through ZoneInfo); the bug was re-stamping that
# correct wall-clock with a fixed offset.
_ET_ZONE = ZoneInfo("America/New_York")


def _et_now() -> dt.datetime:
    return dt.datetime.now(_ET_ZONE)


def _et_stamp(date_str, hhmm) -> str:
    """ISO string for an ET wall-clock time, carrying that date's true offset."""
    h, m = (int(x) for x in str(hhmm).split(":")[:2])
    d = dt.datetime.fromisoformat(str(date_str)).replace(
        hour=h, minute=m, second=0, microsecond=0, tzinfo=_ET_ZONE)
    return d.isoformat()


import pandas as pd

from scripts.build_pit_stats import enrich_roster
from src.matchup_model import attach_history_coverage, reconcile_last_fight_from_history
from src.rationale import set_card_cohort
from jinja2 import Environment, FileSystemLoader

from src import tiering

from src.elo import EloRatingSystem
from src.fighter_history import build_fighter_history, fold_name as fh_fold, summarise as fh_summarise
from src.radar_chart import build_category_radar_svg
from src.fighter_profile import (build_profiles, summarise as fp_summarise,
                                 RAIL_LABELS, CATEGORIES as PROFILE_CATEGORIES,
                                 ATTRIBUTES as PROFILE_ATTRIBUTES, DRAWER_RANKS, tier as profile_tier)
from src.fighter_history import build_fighter_history, fold_name as fh_fold_name
from src.display_names import surname as display_surname, PARTICLES as NAME_PARTICLES, SUFFIXES as NAME_SUFFIXES
from src.edge_finder import find_all_edges
from src.live_props import get_live_props, record_edge_health
from src.odds_utils import measure_overrounds, set_measured_overrounds
from src.prop_ledger import record_prop_prices
from src.card_matcher import (
    load_fight_cards, group_edges_by_card, top_standout_props, top_disagreement_props, top_favorite_picks,
    assign_canonical_fight_ids, group_unmatched_by_fight,
    is_pickable_market, price_is_fragile, fight_key,
)
from src.power_rating import attach_imputed_reach, build_effective_ratings
from src.odds_utils import (implied_prob_to_american, format_american_odds,
                            decimal_to_american)
from src.parlay_builder import build_bankroll_builder_parlays
from src.parlay_builder import _build_candidate_pieces as _candidate_pieces
from src import parlay_pin
from src import parlay_grader
from src.parlay_ledger import load as parlay_load, record_slips
from src.recommendations import build_recommendations
from src.card_plays import build_card_plays
from src import bankroll as bankroll_state
from src.plays_ledger import (
    load as plays_load, record_plays, committed_for, play_id,
    grade_rows as grade_plays, summarise as summarise_plays, write_graded, void_stale,
    summarise_by_event as plays_by_event,
    FIELDNAMES as PLAYS_FIELDNAMES, LEDGER_PATH as PLAYS_LEDGER_PATH,
)
from src.line_movement import (
    build_snapshot_chart,
    load_snapshot, save_snapshot, annotate_movement, attach_charts_to_fight,
    load_token_cache, save_token_cache, update_token_cache,
)
from src.track_record import (
    STAKE_SCHEDULE,
    _is_settled_price,
    log_predictions, compute_track_record, load_momentum_by_key,
    load_logged_predictions_by_key, _pair_key,
    LOCK_OF_WEEK_MAX, LOCK_OF_WEEK_MIN_PROB,
    UNITS_BY_CONFIDENCE, LOCK_OF_WEEK_UNITS,
)
from src.schedule import build_fight_schedule, apply_live_corrections, promote_card_if_stale
from src.results_fetcher import fetch_and_log_new_results, fetch_espn_live_fight_key
from src.card_discovery import discover_and_append_new_cards, normalize_existing_card_order, resync_tracked_card_order, deduplicate_tracked_fights
from src.fighter_backfill import backfill_fighters, fill_missing_last_fights, ensure_roster_rows, fill_last_fight_methods, fill_from_espn_id_map
from src.calibration_chart import build_calibration_svg
from src.sparkline_chart import build_sparkline_svg
from src.units_chart import build_units_timeseries_svg
from src.donut_chart import build_donut_svg, build_split_donut_svg
from src.damage_silhouette import build_damage_silhouette_svg
from src.fun_facts import compute_fun_facts

DATA_DIR = "data"
OUTPUT_PATH = "docs/index.html"
FREE_OUTPUT_PATH = "build/free.html"


# One reader-facing string for "this card is real but has no headliner yet".
# Defined once so the app and the landing page cannot drift apart.
MAIN_EVENT_TBD = "Main event TBD"


def split_event_name(name) -> tuple[str, str | None]:
    """"UFC 331: Van vs. Pantoja 2" -> ("UFC 331", "Van vs. Pantoja 2").

    Returns (series, matchup), with matchup None when no main event has been
    announced. That is a NORMAL STATE, not missing data: the UFC puts a
    numbered card on sale months before it names a headliner, and UFC 332 sits
    on the schedule today with seven bouts booked -- including a co-main --
    and no main event. Five places in this file and the two templates each
    split the name themselves and each disagreed about what to do when there
    was nothing after the colon: one printed the full name, one dropped the
    eyebrow, one substituted a placeholder, and the app's hub card emitted a
    literally empty div. Hence one function.

    A MATCHUP CONTAINS "vs". "UFC Fight Night: Las Vegas" has a colon and no
    headliner -- the tail is the venue, which the card already displays
    separately from event_location, so reading it as a matchup would put the
    city where two names belong. Treating a vs-less tail as unannounced can in
    principle misfire on some future name that really is a matchup without the
    word, and the failure is deliberately in the safe direction: we would show
    "main event to be announced" for a card that has one, which under-claims,
    rather than presenting a venue as a fight, which would be false.
    """
    # rstrip the separator too: a name that arrives as "UFC 333: " strips to
    # "UFC 333:", which no longer contains ": " and would otherwise be handed
    # back as a series with a dangling colon.
    text = str(name or "").strip()
    if ": " not in text:
        return text.rstrip(": ").strip() or text, None
    series, tail = text.split(": ", 1)
    series, tail = series.strip(), tail.strip()
    if not tail or " vs" not in tail.lower():
        return (series if series else text), None
    return series, tail


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


def card_result_coverage(cards_df, results_path: str = "data/fight_results.csv") -> tuple[int, int]:
    """
    (confirmed, total) for the card in `cards_df`, counting only bouts that
    occupy a slot on the night.

    MODULE LEVEL SO IT CAN BE REHEARSED. This was inline in main(), which meant
    the only way to test the handover decision was to re-implement it -- and a
    test that re-implements the logic drifts from it silently, which is the
    failure this repo has already had with two copies of a display string.
    scripts/rehearse_fight_night.py imports this exact function.

    Cancelled bouts are excluded: they occupy no slot, so counting them would
    mean a card could never read complete.
    """
    if cards_df is None or cards_df.empty:
        return 0, 0
    try:
        res = pd.read_csv(results_path)
        done = {
            frozenset({str(r["fighter_a"]).strip().lower(),
                       str(r["fighter_b"]).strip().lower()})
            for _, r in res.iterrows()
            if pd.notna(r.get("winner")) and str(r.get("winner")).strip()
        }
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError, KeyError):
        done = set()
    total = confirmed = 0
    for _, row in cards_df.iterrows():
        if str(row.get("cancelled", "")).strip().lower() == "true":
            continue
        total += 1
        if frozenset({str(row["fighter_a"]).strip().lower(),
                      str(row["fighter_b"]).strip().lower()}) in done:
            confirmed += 1
    return confirmed, total


def card_is_over(confirmed: int, total: int, days_since_event: int,
                 current_card_has_happened: bool) -> bool:
    """
    Whether the forward-looking sections should move to the next event.

    Results all in OR the calendar backstop. Neither is safe alone: results can
    genuinely never confirm (see results_fetcher on draws), and the calendar
    alone costs a day and a half of forward content every week. See the comment
    at the call site for the full argument.
    """
    results_all_in = total > 0 and confirmed >= total
    return current_card_has_happened and (results_all_in or days_since_event >= 2)


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


def main(tier: str = "member", output_path: str | None = None):
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

    # LAST RESORT, after the scoreboard passes have had their go. A late
    # replacement is added once the card has stopped being a future card, so
    # ESPN's scoreboard entry for that event may never carry him -- which is
    # how Pavel Andrusca published as 0-0 on 2026-09-05 while ESPN had him
    # 8-0 and his id sat in data/espn_athlete_ids.csv the whole time.
    try:
        fill_from_espn_id_map(f"{DATA_DIR}/fighters.csv",
                              (f"{DATA_DIR}/fight_cards.csv", f"{DATA_DIR}/future_cards.csv"))
    except Exception as e:
        print(f"[generate_site] ESPN id-map backfill failed, continuing: {e}")

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
    # HOW MUCH OF EACH FIGHTER DO WE ACTUALLY HOLD. Compared against their own
    # claimed record, so the model can tell a genuine five-year layoff from a
    # career we only have one bout of -- see matchup_model.layoff_years.
    fighters_df = attach_imputed_reach(fighters_df)
    fighters_df = reconcile_last_fight_from_history(fighters_df, history_df)
    fighters_df = attach_history_coverage(fighters_df, history_df)
    _thin = fighters_df["history_coverage"] < 0.60
    if _thin.any():
        print(f"[coverage] {int(_thin.sum())} fighter(s) hold under 60% of their "
              f"claimed bouts; layoff is not read for them")
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
        # RECORD THE DERIVABLE-MARKET QUOTES, so the one open question left
        # -- whether a two-way Double Chance / goes-the-distance market is
        # loose enough to beat -- becomes answerable in a few months instead
        # of never. See src/prop_ledger.
        try:
            _ev = cards_df["event_name"].iloc[0] if not cards_df.empty else None
            _ed = str(cards_df["event_date"].iloc[0]) if not cards_df.empty else None
            record_prop_prices(upcoming_df.to_dict("records"), _ev, _ed)
        except Exception as _e:
            print(f"[prop_ledger] skipped ({_e})")

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

    # DEV ESCAPE HATCH. A full build is minutes of network and model work, and
    # anything downstream of this line -- the plays selector, the templates --
    # is a pure function of `events`. Set OCTANE_DUMP_EVENTS to a path to
    # snapshot them and iterate against the snapshot instead of the pipeline.
    if os.environ.get("OCTANE_DUMP_EVENTS"):
        with open(os.environ["OCTANE_DUMP_EVENTS"], "w") as _f:
            json.dump({"events": events, "future_events": future_events}, _f, default=str)
        print(f"[dev] dumped events to {os.environ['OCTANE_DUMP_EVENTS']}")

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
    # ONLY EVER FALL BACK ONCE THE CURRENT CARD HAS ACTUALLY HAPPENED. A card
    # promoted in from future_cards.csv can legitimately have few or no odds
    # posted yet, and moving on from it would claim "this card has concluded"
    # about one that has not. This also retires the fallback on its own at the
    # Monday handoff: promote_card_if_stale swaps the next card in as current,
    # its date is in the future, and the guard stops applying.
    #
    # (MIN_EDGES_FOR_CURRENT_CARD lived here and is gone. Counting remaining
    # edges made a bookmaker's odds pool the thing that decided when the site
    # moved on -- see the card_is_over comment below.)
    current_card_has_happened = False
    if not cards_df.empty:
        try:
            _current_event_date = dt.date.fromisoformat(str(cards_df["event_date"].iloc[0]))
            current_card_has_happened = _current_event_date <= dt.datetime.now(
                _ET_ZONE
            ).date()
        except (ValueError, TypeError):
            current_card_has_happened = False
    # HOW MANY OF THIS CARD'S FIGHTS HAVE A RECORDED RESULT. Counted here
    # rather than reusing results_coverage, which is not built until ~800 lines
    # further down -- and this decision has to be made before tracked_edges is
    # finalised, since it is what tracked_edges gets repointed at.
    _card_confirmed, _card_total = card_result_coverage(cards_df)

    # THE CARD IS OVER WHEN ITS RESULTS ARE IN, OR WHEN THE CALENDAR SAYS SO.
    #
    # This used to be "the date has passed AND fewer than 3 edges remain",
    # which put the handover in a bookmaker's hands: the sections moved on when
    # the odds pool happened to thin, not when the fights finished. Worse, the
    # date half is true ON FIGHT DAY, so the condition was armed for the whole
    # card -- a thinning feed mid-event could repoint Reads, Locks, the Parlay
    # AND the plays list at next week while money was still live on this one.
    #
    # NEITHER CONDITION IS SAFE ALONE, which is why it is an OR. Results can
    # genuinely never confirm -- results_fetcher's own comment says so, for a
    # bout ESPN publishes no usable method text for -- and waiting on
    # confirmed == total would strand these sections on a dead card forever.
    # The calendar backstop is the same day-2 rule promote_card_if_stale uses,
    # so in the worst case everything hands over together on Monday. It
    # degrades INTO agreement rather than away from it.
    _results_all_in = _card_total > 0 and _card_confirmed >= _card_total
    _card_is_over = card_is_over(_card_confirmed, _card_total,
                                 days_since_event, current_card_has_happened)
    if _card_is_over and future_events:
        print(f"[analytics] current card is over "
              f"({'results ' + str(_card_confirmed) + '/' + str(_card_total) if _results_all_in else 'day ' + str(days_since_event)})"
              f" -- forward-looking sections move to the next event")
        next_event = _soonest(future_events)
        # SAME SHAPE AS THE PRIMARY PATH ABOVE -- fight_key stamped on, and
        # cancelled fights excluded. This built bare edges, so every leg it
        # produced reached parlay_builder._fight_key with no fight_key and a
        # fight_id ("card_19") containing no "|", which returns None. A leg
        # with fight_key None can never be resolved by parlay_grader.grade_leg
        # (it needs two names from that key), so it stays 'unresolved' forever
        # and its slip stays 'open'. 99 of 815 legs on file carry None today.
        next_tracked_edges = pd.DataFrame(
            [dict(edge, fight_key=f"{fight.get('fighter_a')}|{fight.get('fighter_b')}")
             for fight in next_event["fights"]
             if not fight.get("cancelled") for edge in fight["edges"]]
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

    # ---------------------------------------------------------------------
    # THE LANDING PAGE'S FACT STRIP is picked from the WHOLE tracked roster,
    # not from this week's card. The section it feeds is demonstrating the
    # rarity ladder, and the ladder is only visible if the three cards are
    # actually three different tiers -- the fighters booked on any one
    # weekend rarely supply a legendary, so sampling the card would show the
    # feature while hiding the idea. Every fact is still literally true and
    # still appears in the app on that fighter's own card.
    #
    # RECENCY GUARD. "Riding an N-fight win streak" is present tense, and a
    # fighter who has not competed in two years is not riding anything. The
    # roster file is the model's tracked set rather than an all-time index so
    # the risk is small, but a stale streak on the marketing page is the kind
    # of error a reader who knows the sport spots instantly.
    _hist_dates = pd.read_csv(f"{DATA_DIR}/fight_history.csv", usecols=["date", "fighter_a", "fighter_b"])
    _last_bout = {}
    for _col in ("fighter_a", "fighter_b"):
        for _n, _d in _hist_dates.groupby(_col)["date"].max().items():
            if _d and (_n not in _last_bout or _d > _last_bout[_n]):
                _last_bout[_n] = _d
    _cutoff = (dt.date.today() - dt.timedelta(days=730)).isoformat()
    _active = [n for n in fighters_df["name"].dropna().unique().tolist()
               if _last_bout.get(n, "") >= _cutoff]
    _roster_facts = compute_fun_facts(_active, f"{DATA_DIR}/fight_history.csv", fighters_df)

    def _fact_kind(f):
        """What SHAPE of anomaly this is, so the strip is not three of one."""
        t = f.get("text", "")
        if t.startswith("Riding"):
            return "streak"
        if t.startswith("Has never won by decision"):
            return "purity"
        if t.startswith("All "):
            return "purity-move"
        if t.startswith("Has never been finished"):
            return "chin"
        if t.startswith("Last "):
            return "recent"
        return t[:16]

    # SCARCEST TIER FIRST. Greedy in tier order picks the top legendary, which
    # is usually the same shape as the top gold, and gold has by far the
    # fewest distinct shapes to fall back on -- so it ends up repeating. Let
    # the most constrained tier choose while it still has a free choice and
    # the other two, which have alternatives, work around it.
    _by_tier = {t: [f for f in _roster_facts if f.get("tier") == t]
                for t in ("legendary", "gold", "hot")}
    _order = sorted(_by_tier, key=lambda t: len({_fact_kind(f) for f in _by_tier[t]}))
    landing_facts, _seen_kind = [], set()
    for _tier in _order:
        _pool = _by_tier[_tier]                       # already rarity-sorted
        _pick = next((f for f in _pool if _fact_kind(f) not in _seen_kind), None)
        # A tier with nothing new to say still beats an empty slot: better a
        # repeated shape at a different rarity than a strip of two.
        _pick = _pick or (_pool[0] if _pool else None)
        if _pick:
            landing_facts.append(_pick)
            _seen_kind.add(_fact_kind(_pick))
    landing_facts.sort(key=lambda f: {"legendary": 0, "gold": 1}.get(f.get("tier"), 2))
    print("[landing] fight facts: " + (", ".join(
        f"{f['tier']}/{_fact_kind(f)}/{f['fighter']}" for f in landing_facts) or "none"))
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
        # CHOSEN ONCE PER CARD, THEN HELD. The builders run every render, and
        # letting them re-pick each time produced 51 bankroll and 93 lotto
        # slips on a single card -- see src/parlay_pin.py. The pin keeps the
        # legs and lets the prices move, which is what makes the slip a thing
        # that can be graded rather than a stream.
        # THE SAME KEY record_slips USES, twenty lines below. event_full_name
        # is not in scope yet -- it is built further down the function -- and
        # reaching for it here would raise a NameError straight into the
        # catch-all below, which would drop the parlay sections from the site
        # silently rather than loudly.
        # THE EVENT THE SLIP IS ACTUALLY BUILT FROM, which is not always
        # events[0]. Once the current card has happened and its edge pool
        # thins, the block above swaps tracked_edges to next_event -- but this
        # kept keying on events[0], so the NEXT card's slip was pinned into
        # the CONCLUDED card's slot, destroying the pin that card was
        # committed to before its results ever landed. parlay_grader.grade_pinned
        # builds its wanted-set only from the live pin file, so there is no
        # backstop: the overwritten commitment is simply gone.
        #
        # Observed on file: the pin under "Nurmagomedov vs. Song" (fought
        # 2026-08-29) held Felipe Lima and Mario Pinto, both of whom fight on
        # the Paris card a week later. Across the whole ledger 164 of 196
        # slips would settle cleanly against recorded results and ZERO are
        # graded -- the published parlay record has never been able to fill.
        _pin_event = analytics_source_event or (events[0].get("event_name") if events else None)
        _pin_pieces = _candidate_pieces(tracked_edges_list, model_only_by_fight)
        bankroll_parlays = parlay_pin.hold(
            _pin_event, "bankroll",
            build_bankroll_builder_parlays(tracked_edges_list, model_only_by_fight),
            _pin_pieces)
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
        bankroll_parlays = []

    # WRITE DOWN WHAT WE PUBLISHED. Nine slips a week have been going out
    # ungraded while single picks are scored to three decimals on the same
    # page; nothing recorded them, so the record was not merely bad, it did
    # not exist. Merged on slip_id, so the 5-minute rebuild cycle leaves nine
    # rows per card rather than nine per render.
    record_slips(
        {"bankroll": bankroll_parlays},
        # Must be the SAME event the pin above used, or the ledger row and the
        # pin disagree about which card the slip belongs to and the grader
        # matches neither.
        event_name=(analytics_source_event or (events[0].get("event_name") if events else None)),
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
    # The build is the ONE caller that should extend the accuracy history.
    track_record = compute_track_record(log_snapshot=True)
    calibration_svg = None
    units_sparkline_svg = None
    units_timeseries_svg = None
    units_shortlist_svg = None
    if track_record and track_record.get("calibration", {}).get("ready"):
        calibration_svg = build_calibration_svg(track_record["calibration"]["points"])
    if track_record and track_record.get("units_stats") and len(track_record["units_stats"]["running_total"]) >= 2:
        units_sparkline_svg = build_sparkline_svg(track_record["units_stats"]["running_total"])
        units_timeseries_svg = build_units_timeseries_svg(track_record["units_stats"]["running_total"])
        # A SECOND CURVE FOR THE LANDING PAGE, plotting only the two tiers a
        # reader would actually back. See track_record for why the all-picks
        # line was the wrong evidence to put under the shortlist table.
        _sl = track_record["units_stats"].get("shortlist_running") or []
        if len(_sl) >= 2:
            units_shortlist_svg = build_units_timeseries_svg(_sl)

    event_short_name = (
        split_event_name(analytics_source_event)[0] if analytics_source_event
        else split_event_name(events[0]["event_name"])[0] if events
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
    # Falls back to the series, not to a placeholder: this lands in
    # "Clearest Reads . X", where "UFC 332" names the card perfectly well and
    # "Clearest Reads . Main event TBD" would not.
    _series, _match = split_event_name(event_full_name)
    event_matchup = _match or _series or event_full_name

    # Countdown target: this weekend's tracked event if we have one, otherwise
    # the nearest future card. ET is UTC-4 (EDT) for all currently tracked
    # events (July-August) -- would need adjusting for events during EST months.
    countdown_target_iso = None
    countdown_label = None
    countdown_series = None
    countdown_matchup = None
    next_event = events[0] if events else _soonest(future_events)
    if next_event:
        # True ET offset for that date -- see _et_stamp. Hardcoding -04:00 here
        # made the countdown reach zero an hour before first bell on every card
        # between November and mid-March.
        countdown_target_iso = _et_stamp(next_event['event_date'],
                                         next_event.get('event_start_time_et', '19:00'))
        countdown_label = next_event["event_name"]
        # The banner sets the series as a gold eyebrow ABOVE the matchup
        # rather than running both into one line, so the matchup -- the only
        # part a reader is actually scanning for -- gets the full width and
        # the largest type. "UFC Fight Night: Hernandez vs. Rodrigues" splits
        # into "UFC FIGHT NIGHT" / "Hernandez vs. Rodrigues"; numbered cards
        # split into "UFC 333" / "Volkanovski vs. Lopes". An event with no
        # colon (rare, but it happens on newly-announced cards where ESPN has
        # only the series name) keeps the whole string as the matchup and
        # renders no eyebrow -- better than an eyebrow with nothing under it.
        # THE LAST RAW SPLIT. split_event_name is the one definition; this
        # copy still read a venue as a matchup, so "UFC Fight Night: Las
        # Vegas" would print the CITY in the slot reserved for two names,
        # with "Meta APEX, Las Vegas" repeated directly beneath it.
        countdown_series, countdown_matchup = split_event_name(countdown_label)
        countdown_matchup = countdown_matchup or MAIN_EVENT_TBD

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

                # A REMATCH IS A DIFFERENT FIGHT, and this key carries no
                # event -- so the second meeting of a pair overwrites the
                # first, last row winning. Before the rematch happens that
                # renders the PREVIOUS bout's winner and method on a fight
                # that has not occurred, marks it finished so it is stripped
                # from the live schedule, and lets just_concluded announce it.
                # On a site whose entire claim is "published before the
                # fights", showing a result for an unfought bout is the worst
                # shape of wrong.
                #
                # RESOLVED THE SAME CONSERVATIVE WAY results_fetcher does: a
                # pair recorded once keeps the plain key, so every pairing
                # that works today is untouched -- including across an event
                # rename, where requiring the name to match would BREAK
                # matching. Only a pair already present must prove it belongs
                # to this row's event before it is allowed to overwrite.
                _prev = finished_results.get(key)
                if _prev is not None:
                    _this_ev = str(r.get("event_name") or "").strip().lower()
                    _prev_ev = str(_prev.get("event_name") or "").strip().lower()
                    if _this_ev != _prev_ev:
                        _live_evs = {str(e).strip().lower()
                                     for e in (cards_df["event_name"].tolist() if not cards_df.empty else [])}
                        # Keep whichever row belongs to the card being rendered;
                        # if neither does, keep the one already held rather than
                        # letting file order decide.
                        if _this_ev not in _live_evs:
                            continue
                finished_results[key] = {
                    "event_name": str(r.get("event_name") or "").strip(),
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
            # THE RESULT HAS TO BELONG TO THIS BOUT, not merely to this pair.
            # finished_results is keyed on the fighter pair with no event
            # because grade_plays (below) needs every historical result in one
            # map -- but for DISPLAY that means a rematch inherits the previous
            # meeting's outcome. Before UFC 331: Van vs. Pantoja 2 is fought,
            # the pair already resolves to UFC 317, so the card would print
            # Pantoja as the winner of a fight that has not happened, mark it
            # finished, strip it from the live schedule, and let
            # just_concluded announce it.
            #
            # Filtered HERE rather than by narrowing finished_results, which
            # would break grading: line ~1293 passes that same map to
            # grade_plays for plays on cards from weeks ago, and those results
            # are correctly not on this card.
            #
            # A result with no recorded event predates the column and is
            # accepted, so nothing that renders today changes.
            if result and result.get("event_name"):
                if str(result["event_name"]).strip().lower() != str(event.get("event_name") or "").strip().lower():
                    result = None
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

    # ---- THE PLAYS CARD -------------------------------------------------
    # Sits here because it needs is_lock_of_week, which the block above just
    # read back off predictions_log, and because it must see the SAME fight
    # objects the rest of the page renders -- a plays section computed from a
    # different snapshot than the card beside it is worse than no plays
    # section.
    #
    # Committed plays go in first. A play published on Tuesday is money down;
    # this render may only decide what to ADD to it. See src/plays_ledger for
    # why the card budget has to be spent as it is committed rather than
    # granted afresh on every one of a week's worth of renders.
    plays_card = {"event_name": None, "plays": [], "passed": [], "dropped": [],
                  "total_units": 0.0, "new_units": 0.0, "fights_considered": 0}
    plays_rows, plays_record, bankroll = [], None, None
    plays_events = {}
    # PUBLISHED, NOT PLAYED. Graded slips only -- summarise drops everything
    # ungraded -- so this carries no read on a fight that has not happened and
    # is free by the same rule the plays record is. Empty until the first
    # pinned slip settles, which is what the template's guard is for.
    parlay_record, parlay_events = None, {}
    try:
        _plays_event = events_for_model_only[0] if events_for_model_only else None
        _ledger = plays_load()
        plays_card = build_card_plays(
            _plays_event,
            # event_date as well as the name -- see committed_for. Without it a
            # renamed event hands select_card a fresh budget and double-books
            # every play on the card.
            committed=committed_for(_plays_event.get("event_name") if _plays_event else None,
                                    _ledger,
                                    event_date=(_plays_event.get("event_date") if _plays_event else None)),
        )

        # The closing line for everything already on the board, including
        # plays this render is no longer selecting: a bet placed on Tuesday
        # still has a closing price, and CLV is the one number on this site
        # that claims to show edge independently of results.
        _live = {}
        for _f in (_plays_event or {}).get("fights", []):
            for _e in _f.get("edges") or []:
                _live[play_id(_plays_event.get("event_name"), _f["fighter_a"], _f["fighter_b"],
                              _e.get("market"), _e.get("fighter"))] = _e.get("odds_american")

        plays_rows = record_plays(plays_card, generated_at_str, live_prices=_live)

        # GRADED FROM RESULTS, NEVER FROM THE RENDER. finished_results is the
        # same map the rest of this build settles picks from, so the plays
        # ledger cannot drift into a different opinion about one night.
        _all = plays_load()
        _cancelled = {frozenset({f["fighter_a"].strip().lower(), f["fighter_b"].strip().lower()}): {"cancelled": True}
                      for _ev in events for f in _ev["fights"] if f.get("cancelled")}
        _n = grade_plays(_all, {**_cancelled, **finished_results}, generated_at_str)

        # A PLAY THAT NEVER GRADES IS WORSE THAN ONE THAT LOSES. It stays open
        # forever, never reaches the bankroll, and -- since summarise_by_event
        # is settled-only -- vanishes from the card rather than showing as a
        # hole. The cause is almost always a late opponent change, which a book
        # settles by voiding the whole market, moneyline included: the bet was
        # on a matchup and the matchup no longer exists.
        _voided = void_stale(_all, generated_at_str)
        for _v in _voided:
            print(f"[plays] VOID {_v['units']}U {_v['label']!r} on "
                  f"{_v['fighter_a']} vs {_v['fighter_b']} -- {_v['void_reason']}")
        _n += len(_voided)
        if _n:
            write_graded(_all)
            plays_rows = [r for r in plays_load()
                          if r.get("event_name") == plays_card.get("event_name")]
        # THE PARLAYS, settled from the same results map. Only the PINNED
        # slip for a card is graded -- before src/parlay_pin the builder
        # re-picked every render, so one card holds dozens of variants and
        # grading them all would answer a question nobody acted on.
        #
        # Nothing here is staked. The figure is what 1U flat on a published
        # slip would have returned, and the site says so wherever it appears.
        try:
            parlay_grader.grade_pinned(
                parlay_pin.load(), generated_at_str,
                results_path=parlay_grader.RESULTS_PATH)
            _parlay_rows = parlay_load()
            parlay_record = parlay_grader.summarise(_parlay_rows)
            # Keyed on the slip's `event`, which is the same string
            # track_record groups by -- so the template looks it up with
            # group.event_name and needs no mapping of its own.
            parlay_events = parlay_grader.summarise_by_event(_parlay_rows)
            if parlay_record["n"]:
                print(f"[parlay_grader] record {parlay_record['cashed']}/"
                      f"{parlay_record['n']} cashed, "
                      f"{parlay_record['units_flat']:+.2f}U at 1U flat "
                      f"across {parlay_record['events']} card(s)")
        except Exception as _exc:
            # A bookkeeping bug must not take the build down.
            print(f"[parlay_grader] not run ({_exc}) -- continuing")

        # THE BANKROLL, folded forward from whatever has just settled. Only
        # newly graded plays move it, and only once each -- see src/bankroll.
        _all_now = plays_load()
        _bank = bankroll_state.apply_settled(bankroll_state.load(), _all_now)
        bankroll_state.save(_bank)
        bankroll = bankroll_state.summarise(_bank)
        # Per-card, for the track record's Bets tab. A card missing from this
        # dict was graded before the ledger existed and keeps its old view.
        plays_events = plays_by_event(_all_now)
        # THIS CARD'S RECORD, NOT THE RUNNING ONE. This was
        # summarise_plays(_all_now) -- every play ever written -- rendered
        # under a heading that says "This Week's Plays", beside a list that IS
        # scoped to one event. It read 1-0 only because two plays existed in
        # total; it would have become 3-1, then 6-2: a cumulative figure
        # wearing a weekly label. The running view already has a home in the
        # Record tab, which is where it belongs.
        _pe = plays_events.get(plays_card.get("event_name")) or {}
        plays_record = {
            "won": _pe.get("won", 0), "lost": _pe.get("lost", 0),
            "settled": _pe.get("settled", 0), "units": _pe.get("units", 0.0),
            "staked": _pe.get("staked", 0.0), "roi_pct": _pe.get("roi_pct"),
        } if _pe else None
        _shelved = len(plays_card.get("shelved") or [])
        _note = "" if plays_card["discretionary_on"] else f", {_shelved} shelved"
        print(f"[plays] {len(plays_card['plays'])} new, {len(plays_rows)} on the card, "
              f"{plays_card['total_units']}U committed{_note}; record "
              f"{plays_record['won']}-{plays_record['lost']} "
              f"({plays_record['units']:+.2f}U), bankroll {bankroll['multiple']:.4f}x")
    except Exception as e:
        # A broken plays section must not take the site down with it. Every
        # other section on this page is older and has a record behind it.
        print(f"[plays] failed, section will be empty: {e}")

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
    # City alone, for the banner's meta line ("Sacramento · Aug 22 · 8:00 PM").
    # The venue is dropped there because the line has to share a 221px rail
    # with the date and the start time, and "Golden 1 Center" is the least
    # useful of the three to somebody deciding whether to watch.
    # location_parts is built as [venue, city, state-or-country] with empties
    # dropped (see card_discovery.py), so index 1 is the city whenever all
    # three are present. With only two parts the pair is almost always
    # [city, state] -- ESPN supplies `country` when `state` is missing, so
    # losing BOTH is far likelier than losing the venue name -- and index 0
    # is the city there.
    countdown_city = None
    countdown_venue = None
    if countdown_location:
        _loc_parts = [p.strip() for p in countdown_location.split(", ") if p.strip()]
        if _loc_parts:
            countdown_city = _loc_parts[1] if len(_loc_parts) >= 3 else _loc_parts[0]
            # Venue is desktop-only (CSS hides it below the breakpoint). It is
            # only trustworthy when all three parts are present -- with two,
            # index 0 is being read as the city above, and it cannot be both.
            countdown_venue = _loc_parts[0] if len(_loc_parts) >= 3 else None
    countdown_confidence_counts = None
    if events and next_event is events[0]:
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
            today_et = _et_now().date()
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
    # A GLOBAL, not a context variable. Jinja macros do not see the render
    # context, and every fight card is rendered through render_fight_card --
    # so a context-passed `tier` would read as Undefined inside exactly the
    # markup that needs to consult it, silently taking the member branch.
    env.globals["tier"] = tier
    # ONE event-name split for both templates. See split_event_name: five
    # copies of this logic disagreed about the unannounced case, and one of
    # them rendered an empty div.
    env.globals["MAIN_EVENT_TBD"] = MAIN_EVENT_TBD
    env.filters["event_series"] = lambda n: split_event_name(n)[0]
    # For a dedicated matchup line, which must never be blank.
    env.filters["event_matchup"] = lambda n: split_event_name(n)[1] or MAIN_EVENT_TBD
    # For "Section heading . X", where naming the card is the job and the
    # series alone still does it.
    env.filters["event_label"] = lambda n: (split_event_name(n)[1]
                                            or split_event_name(n)[0])
    # The stake ladder in force TODAY, so the units caption states the
    # real numbers instead of a hardcoded copy that goes stale the first
    # time a stake changes -- which it now has.
    # HOW MANY DIGITS THE SIGN-IN CODE HAS. Supabase makes this a project
    # setting (6-10), and the templates had hardcoded 6 while the project
    # issues 8 -- so the input capped at six characters and slice(0, 6) threw
    # the last two away, submitting a truncated code every time. One
    # definition, shared by the landing page and the app, and the input still
    # ACCEPTS up to 10 so a future settings change degrades to "tap the
    # button" rather than to "cannot type your code".
    # WHAT SUPABASE ACTUALLY ISSUES. Changed in the project settings from 8 to
    # 6; this drives the placeholder and the aria-label, and while it said 8 the
    # code box invited a reader to type two digits that were never sent. The
    # SUBMIT path stays length-agnostic on purpose -- it fires when the field
    # settles rather than at a fixed count -- so this being stale can only ever
    # mislabel the box, never refuse a valid code.
    env.globals["otp_length"] = 6
    env.globals["otp_max_length"] = 10
    env.globals["stake_ladder"] = UNITS_BY_CONFIDENCE
    # The day the current ladder took effect, for the one-line note under the
    # units table. Read off STAKE_SCHEDULE rather than typed, so the date and
    # the stakes can never drift apart.
    _eff = next((d for d, _ in STAKE_SCHEDULE if d), "")
    try:
        env.globals["stake_change_date"] = dt.datetime.strptime(
            _eff, "%Y-%m-%d").strftime("%b %-d")
    except ValueError:
        env.globals["stake_change_date"] = _eff
    env.globals["lock_units"] = LOCK_OF_WEEK_UNITS
    env.filters["american"] = format_american_odds
    # THE PRICE A SETTLED SLIP ACTUALLY PAID, from the decimal the units
    # were computed from. Printing combined_american instead would let the
    # displayed price drift from the number beside it the moment a leg
    # voided and the slip shortened -- the row would then disagree with
    # itself, which is the exact defect class the staking guard exists for.
    # Uncapped: this is a whole slip, not one market.
    env.filters["american_from_decimal"] = lambda d: format_american_odds(
        decimal_to_american(float(d)), cap=None)
    # Spelled out, because the sentence it lands in is prose. Falls back to
    # the digits above ninety-nine, where a word would be worse than a number.
    _ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
             "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
    _TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
             6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}

    def _spell(n):
        try:
            n = int(n)
        except (TypeError, ValueError):
            return str(n)
        if n < 20:
            return _ONES[n] if 0 <= n < 20 else str(n)
        if n < 100:
            t, o = divmod(n, 10)
            return _TENS[t] + ("-" + _ONES[o] if o else "")
        return str(n)

    env.filters["spell"] = _spell
    # Probability -> the price at which a bet on it breaks even, i.e. the
    # model's own fair line. Both existing helpers already exist; this just
    # composes them so a template can render the Model column in the same
    # unit as the Odds column beside it.
    # A PROBABILITY OF 0 OR 1 HAS NO FAIR PRICE, and this filter used to take
    # the whole build down when handed one. implied_prob_to_american raises on
    # purpose -- its docstring says so, and says the point is to let "the
    # caller's existing try/except around this function actually catch it".
    # This caller had no try/except. It guarded None and nothing else, so a
    # method row whose probability rounded to 0.0000 (a submission chance for
    # a fighter who has never been near one) reached it and raised inside
    # Jinja, aborting the render after every page had already been computed.
    #
    # Hit on a local build 2026-08-28 with two such rows. CI had not tripped
    # it, which is the whole problem with leaving it: nothing about a card
    # night makes a 0.0 less likely, and the failure mode is the entire site
    # rather than one empty cell.
    def _fair_odds(p):
        if p is None:
            return ""
        try:
            return format_american_odds(implied_prob_to_american(float(p)))
        except (TypeError, ValueError):
            # An unrenderable probability leaves the cell blank. The number
            # beside it still prints, so the row degrades rather than lying.
            return ""
    env.filters["fair_odds"] = _fair_odds
    env.filters["friendly_date"] = _format_friendly_date

    # Defined at module level (see above) so the concluded-fight result
    # strings built earlier in this function can use the same mapping the
    # templates do; this just exposes it to Jinja under its filter name.
    env.filters["method_display"] = _method_display
    # One rule for shortening a name, shared by the five places on the card
    # that print one. See src/display_names.py for why it runs backwards.
    env.filters["surname"] = display_surname
    # The drawer builds opponent names in the browser, so the JS needs the
    # same word lists. Shipped as data rather than retyped in JavaScript --
    # the algorithm is four lines, but a list that drifts is a silent bug.
    env.globals["name_particles_json"] = json.dumps(
        {"p": sorted(NAME_PARTICLES), "s": sorted(NAME_SUFFIXES)})
    env.globals["donut_svg"] = build_donut_svg
    env.globals["split_donut_svg"] = build_split_donut_svg
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

    # ===== Scouting drawer: per-fighter bout history =====
    # Built from the per-ROUND stats file, which until now nothing on the page
    # consumed. Scoped to fighters who actually appear on a card here -- the
    # whole database is 2,724 fighters and 136 KB gzipped, which is a
    # deferred-fragment problem rather than an inline one, and nobody needs
    # Georges St-Pierre's 2007 rounds to read Saturday's card.
    _booked = set()
    for _ev in list(events) + list(future_events):
        for _f in _ev.get("fights", []):
            for _side in ("fighter_a", "fighter_b"):
                if _f.get(_side):
                    _booked.add(str(_f[_side]))
    fighter_history = build_fighter_history(_booked)
    _fh = fh_summarise(fighter_history)
    print(f"[history] {_fh['fighters']} of {len(_booked)} booked fighters, "
          f"{_fh['bouts']} bouts, {_fh['rounds']} rounds"
          + (f", {_fh['bouts_without_control']} without control time"
             if _fh['bouts_without_control'] else ", control time complete"))
    fighter_history_json = json.dumps(fighter_history, separators=(",", ":"))

    # ===== Scout row: per-fighter attribute percentiles =====
    # Replaces the matchup-factor badges, which repeated the waterfall printed
    # directly below them on 91% of labels. See src/fighter_profile.
    _profiles = build_profiles(_booked)
    _fp = fp_summarise(_profiles)
    print(f"[profile] {_fp['profiled']} of {_fp['total']} booked fighters have a "
          f"{3}+ bout profile, {_fp['no_bouts']} have never fought in the UFC")
    # Six rails, not all nine -- see RAIL_LABELS. The radar below the card
    # plots the other three inside their categories.
    _plabels = list(RAIL_LABELS)
    for _ev in list(events) + list(future_events):
        for _f in _ev.get("fights", []):
            _pa = _profiles.get(fh_fold(_f.get("fighter_a", "")), {"bouts": 0, "pct": None})
            _pb = _profiles.get(fh_fold(_f.get("fighter_b", "")), {"bouts": 0, "pct": None})
            # Rows are built here rather than in the template so the two sides
            # stay locked to one attribute order and a missing side is an
            # explicit None instead of a silently absent key.
            _f["profile"] = {
                "a_bouts": _pa["bouts"], "b_bouts": _pb["bouts"],
                "a_ok": bool(_pa["pct"]), "b_ok": bool(_pb["pct"]),
                "rows": [{"label": lab,
                          "a": (_pa["pct"] or {}).get(lab),
                          "b": (_pb["pct"] or {}).get(lab)} for lab in _plabels]
                        if (_pa["pct"] or _pb["pct"]) else [],
                # Who is short of bouts, and by how much. Built here so the
                # sentence reads the same whether one side is missing or both
                # -- a fight with neither man profiled used to render the row
                # silently, which looks like breakage rather than like a card
                # full of debutants.
                # Category scores for the Tale of the Tape radar, plus the
                # attribute members so a tapped category can open its parts.
                "cats": [{"label": c,
                          "a": (_pa.get("cats") or {}).get(c),
                          "b": (_pb.get("cats") or {}).get(c),
                          "members": [{"label": l,
                                       "a": (_pa["pct"] or {}).get(l),
                                       "b": (_pb["pct"] or {}).get(l)}
                                      for l, _f, _h, cc in PROFILE_ATTRIBUTES if cc == c]}
                         for c in PROFILE_CATEGORIES]
                        if (_pa.get("cats") or _pb.get("cats")) else [],
                "thin": [
                    {"name": display_surname(_f.get(_side, "")), "bouts": _p["bouts"]}
                    for _side, _p in (("fighter_a", _pa), ("fighter_b", _pb))
                    if not _p["pct"] and _f.get(_side)
                ],
            }
            _f["profile"]["radar_svg"] = build_category_radar_svg(_f["profile"]["cats"])

    # Career rates for the drawer header, from the ENRICHED roster -- these are
    # the columns that were unreachable in production until control time was
    # reconnected (control_time_pct was populated on 0 of 323 fighters).
    # None rather than 0 where a rate is genuinely unknown: a fighter with no
    # matched timeline should show a dash, not a confident zero.
    _rate_cols = ("control_time_pct", "slpm", "td_defense_pct", "strike_accuracy_pct")
    fighter_rates = {}
    for _row in fighters_df.to_dict("records"):
        _nm = str(_row.get("name") or "").strip()
        if not _nm or fh_fold(_nm) not in fighter_history:
            continue
        _vals = {}
        for _c in _rate_cols:
            _v = _row.get(_c)
            if _v is None or (isinstance(_v, float) and _v != _v):
                continue
            _vals[_c] = round(float(_v), 1)
        _wc = _row.get("weight_class")
        _wc = "" if _wc is None or (isinstance(_wc, float) and _wc != _wc) else str(_wc).strip()
        if _wc.lower() in ("nan", "none"):
            _wc = ""
        # A tier word per tile, so "52.3% control" says whether that is good.
        # Ranked against the same pit_stats population the value itself comes
        # from -- verified identical on 131 fighters, so the number and its
        # label cannot contradict each other.
        _pr = (_profiles.get(fh_fold(_nm)) or {}).get("pct") or {}
        _tiers = {}
        for _col, _attr in DRAWER_RANKS.items():
            _t = profile_tier(_pr.get(_attr))
            if _t and _col in _vals:
                _tiers[_col] = _t
        fighter_rates[fh_fold(_nm)] = {"n": _nm, "wc": _wc, "r": _vals, "t": _tiers}
    fighter_rates_json = json.dumps(fighter_rates, separators=(",", ":"))

    # The key the drawer looks a fighter up by. Emitted onto the markup so the
    # template never has to reimplement the fold -- the last time two folds
    # disagreed in this repo, every accented fighter silently lost their data.
    for _ev in list(events) + list(future_events):
        for _f in _ev.get("fights", []):
            for _side in ("fighter_a", "fighter_b"):
                if _f.get(_side):
                    _f[f"{_side}_key"] = fh_fold(_f[_side])

    context = dict(
        events=events,
        future_events=future_events,
        unmatched=unmatched_df.to_dict("records") if not unmatched_df.empty else [],
        standout_props=standout_props,
        disagreement_props=disagreement_props,
        fun_facts=fun_facts,
        fun_facts_by_fighter=fun_facts_by_fighter,
        favorite_picks=favorite_picks,
        lock_picks=lock_picks,
        plays_card=plays_card, plays_rows=plays_rows, plays_record=plays_record,
        bankroll=bankroll, plays_events=plays_events,
        parlay_record=parlay_record, parlay_events=parlay_events,
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
        countdown_series=countdown_series,
        countdown_matchup=countdown_matchup,
        countdown_city=countdown_city,
        countdown_venue=countdown_venue,
        countdown_confidence_counts=countdown_confidence_counts,
        fighter_history_json=fighter_history_json,
        fighter_rates_json=fighter_rates_json,
        whats_new_snapshot=whats_new_snapshot,
        track_record=track_record,
        calibration_svg=calibration_svg,
        units_sparkline_svg=units_sparkline_svg,
        units_timeseries_svg=units_timeseries_svg,
        bankroll_parlays=bankroll_parlays,
        model_legs=model_legs,
        notable_movements=notable_movements,
        notable_movements_upcoming=notable_movements_upcoming,
        live_error=live_error,
        source=source,
        generated_at=generated_at_str,
        generated_at_short=generated_at_short,
        generated_at_time_only=generated_at_time_only,
        generated_at_date=generated_at_date,
        tier="member",
    )

    # THE PAYWALL PARTITION. Redaction happens here, on the data, rather than
    # as conditionals inside the template -- see src/tiering.py for why sixty
    # `{% if tier %}` branches would have been the more dangerous design.
    # BOTH PAYLOADS FROM ONE DATA SNAPSHOT. Running the build twice -- once
    # per tier -- would fetch live odds twice, minutes apart, and the two
    # payloads could disagree about the same fight. Rendering the redacted
    # context from the context already in memory costs a second render and no
    # network at all, and guarantees the free build is the member build minus
    # exactly the model layer rather than a different snapshot of the card.
    if tier == "member":
        free_ctx, _redacted, _assertable = tiering.redact_context(dict(context))
        os.makedirs("build", exist_ok=True)
        with open("build/redacted-manifest.json", "w") as _mf:
            json.dump({"count": len(_redacted), "removed": _redacted,
                       "values": _assertable}, _mf, indent=1)
        with open(FREE_OUTPUT_PATH, "w") as _ff:
            _ff.write(template.render(**free_ctx))
        print(f"[tier] also wrote {FREE_OUTPUT_PATH} -- redacted {len(_redacted)} model "
              f"value(s), {len(_assertable)} distinctive enough to assert on")

    if tier == "free":
        context, _redacted, _assertable = tiering.redact_context(context)
        os.makedirs("build", exist_ok=True)
        with open("build/redacted-manifest.json", "w") as _mf:
            json.dump({"count": len(_redacted), "removed": _redacted,
                       "values": _assertable}, _mf, indent=1)
        print(f"[tier] redacted {len(_redacted)} model value(s); "
              f"{len(_assertable)} distinctive enough to assert on")

    html = template.render(**context)

    # ---------------------------------------------------------------------
    # THE LANDING PAGE. Rendered from the SAME context as the app, because its
    # entire argument is the live track record -- a hardcoded "+66U" would be
    # wrong within a week, and a marketing page that overstates the record is
    # worse than no marketing page.
    # ---------------------------------------------------------------------
    if tier != "free" and track_record:
        try:
            _write_landing(env, track_record, units_shortlist_svg or units_timeseries_svg,
                           events, future_events, generated_at_short,
                           countdown_target_iso, landing_facts, updated_snapshot,
                           countdown_series, countdown_matchup)
        except Exception as exc:                      # never break the main build
            print(f"[landing] skipped: {exc}")

    # LEGAL PAGES. Rendered every build so the "last updated" date is honest
    # rather than a hardcoded string that quietly ages, and so the three pages
    # cannot drift apart in styling or footer links.
    if tier != "free":
        try:
            _write_legal(env, generated_at_date)
        except Exception as exc:
            print(f"[legal] skipped: {exc}")

    out_path = output_path or OUTPUT_PATH
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
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
    # Only the canonical build owns this directory. The fragments are line
    # movements -- market data, free in both tiers -- so a free build
    # regenerating them would clear and rewrite files the member build is
    # serving, for no difference in content.
    if out_path != OUTPUT_PATH:
        print(f"Wrote {out_path} (tier={tier}); movement fragments left to the canonical build")
        return
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

    print(f"Wrote {out_path} ({len(events)} events, {len(future_events)} future events, {len(standout_props)} agreed reads, {len(disagreement_props)} disagreements)")
    print(f"Wrote {written} movement fragment(s), {total/1e6:.2f}MB deferred out of the page")





# TWO VALUES THAT MUST BE REAL BEFORE LAUNCH. They are placeholders on
# purpose and deliberately conspicuous: a terms page naming the wrong legal
# entity, or no entity at all, is worse than one that is obviously unfinished.
# THE FULL LEGAL NAME, not the everyday one. "Octane Alpha" on its own is a
# trade name and not a party that exists, so the terms name the person who is
# actually contracting -- which, with no LLC formed, is the operator. Customers
# still see the brand everywhere else.
#
# Revisit when the LLC is formed: the entity becomes the party, and the
# operator's name comes off this public page.
LEGAL_ENTITY = "Sung Beom Park, doing business as Octane Alpha"
GOVERNING_LAW = "the State of New Jersey"


def _write_legal(env, updated):
    """docs/terms.html, docs/privacy.html, docs/refunds.html."""
    pages = {
        "terms.html": "legal_terms.html",
        "privacy.html": "legal_privacy.html",
        "refunds.html": "legal_refunds.html",
    }
    os.makedirs("docs", exist_ok=True)
    for out_name, template_name in pages.items():
        html = env.get_template(template_name).render(
            updated=updated,
            legal_entity=LEGAL_ENTITY,
            governing_law=GOVERNING_LAW,
        )
        with open(os.path.join("docs", out_name), "w") as f:
            f.write(html)
    print(f"Wrote {len(pages)} legal page(s)")


def _display_name(folded: str, names: list[str]) -> str:
    """
    build_fighter_history keys on a folded name (case and accents stripped).
    This maps one back to the roster spelling, because "kelvin gastelum" is
    not what goes on the page.
    """
    for n in names:
        if fh_fold_name(n) == folded:
            return n
    return folded.title()


def _write_landing(env, track_record, units_svg, events, future_events, generated_at_short,
                   countdown_target_iso=None, landing_facts=None, odds_snapshot=None,
                   countdown_series=None, countdown_matchup=None):
    """
    docs/welcome.html -- the marketing page.

    The paired card is the whole idea: one fight that has happened with the
    read revealed and its result stamped on, and one that has not with the
    read sealed. Both show the same analytics. Picking the MOST RECENT graded
    fight rather than the best-ever one is deliberate -- "here is last
    Saturday" is a more honest demonstration than "here is our best day", and
    the aggregate record sits directly above it either way.
    """
    conf_short = {"High Confidence": "HIGH", "Medium Confidence": "MED",
                  "Low Confidence": "LOW", "Lock of the Week": "LOCK"}

    graded = None
    for ev in (track_record.get("results_by_event") or []):
        for r in ev.get("results", []):
            if r.get("correct") and r.get("units_result") is not None:
                graded = dict(r)
                graded["event_short"] = _format_friendly_date(r.get("date_added")) or ev.get("event_name", "")
                graded["division"] = r.get("card_position") or ""
                graded["conf_short"] = conf_short.get(r.get("confidence_label"), "")
                graded["pick_odds_label"] = format_american_odds(r.get("pick_odds"))
                break
        if graded:
            break
    if not graded:
        raise ValueError("no graded result to showcase")

    upcoming = None
    for source in (future_events or []), (events or []):
        for ev in source:
            for f in ev.get("fights", []):
                if f.get("winner") or f.get("cancelled"):
                    continue
                price = None
                for e in (f.get("edges") or []):
                    if e.get("market") == "Moneyline" and e.get("odds_american"):
                        price = format_american_odds(e["odds_american"])
                        break
                upcoming = {
                    "fighter_a": f.get("fighter_a", ""),
                    "fighter_b": f.get("fighter_b", ""),
                    "division": f.get("weight_class") or "",
                    "when": ev.get("event_name", "Next card"),
                    "price_label": price or "--",
                }
                break
            if upcoming:
                break
        if upcoming:
            break
    if not upcoming:
        raise ValueError("no upcoming fight to seal")

    # ---- THE TAPE ----------------------------------------------------------
    # Real fighters at their real moneyline, with the move since the first
    # price we ever recorded for that bout. A ticker carrying invented numbers,
    # on a page whose entire argument is a public ledger, would be the one
    # dishonest element on it -- so it reads the same odds snapshot every other
    # module reads, and a fighter with no recorded price is DROPPED rather than
    # filled in with a plausible-looking number.
    try:
        with open("data/odds_snapshot.json") as _f:
            _snap = json.load(_f)
    except (OSError, ValueError):
        _snap = {}

    # ---- THE COUNTDOWN -----------------------------------------------------
    # main() computes countdown_target_iso for the APP's banner, where the card
    # that just happened is still "the current card". On a landing page that
    # reads "Next card sealed in", the same value is a countdown that has
    # already expired -- it pointed three days into the past on the first
    # build. So the target is validated as future, and re-derived from the
    # soonest upcoming card when it isn't. If nothing is ahead of us the line
    # is dropped rather than shown stale.
    _now = dt.datetime.now(dt.timezone.utc)

    def _future_or_none(iso):
        if not iso:
            return None
        try:
            when = dt.datetime.fromisoformat(iso)
        except ValueError:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        return iso if when > _now else None

    countdown_target_iso = _future_or_none(countdown_target_iso)
    if not countdown_target_iso:
        # THE NAME MOVES WITH THE CLOCK. The landing page now prints the event
        # beside the countdown, so re-deriving the target without re-deriving
        # the label would leave the hero naming last Saturday's card against
        # next Saturday's timer -- a wrong fact stated confidently, which is
        # worse than the stale timer this block was written to prevent.
        countdown_series = countdown_matchup = None
        for ev in sorted((future_events or []), key=lambda e: e.get("event_date") or "9999"):
            candidate = _et_stamp(ev.get('event_date'),
                                  ev.get('event_start_time_et', '19:00'))
            countdown_target_iso = _future_or_none(candidate)
            if countdown_target_iso:
                # The eyebrow is the series and the line below it is the
                # matchup. Previously an unannounced card put its whole name
                # in the matchup slot and left the eyebrow to fall back to a
                # generic "Next card", so UFC 332 lost its own name from the
                # one place a reader looks for it.
                countdown_series, countdown_matchup = split_event_name(
                    ev.get("event_name"))
                countdown_matchup = countdown_matchup or MAIN_EVENT_TBD
                break
    if not countdown_target_iso:
        countdown_series = countdown_matchup = None

    # ---- THE CLV STORY -----------------------------------------------------
    # DYNAMIC, not pinned to one fight. A hardcoded "we took McMillen at +160"
    # is true forever but ages badly; computing the best line move in the
    # ledger each build means the page always shows its strongest real example
    # and can never drift from the data. Only WINS are eligible -- a big line
    # move on a loser is still a genuine CLV result, but as the single headline
    # story on a landing page it invites exactly the argument you do not want.
    # RANKED ON THE STORY, NOT ON THE PERCENTAGE. Sorting by raw CLV% picked a
    # favourite that got more favoured -- true, but "-147 closed -567, won
    # +0.68U" is an argument only a quant hears. A pick that CROSSES the line,
    # published as an underdog and closing as a favourite, is the market
    # completely reversing its view, which is the same claim in a form anyone
    # can read. So crossing wins first, then size of move.
    def _clv_rank(m):
        clv = m.get("clv") or {}
        try:
            pick = float(clv.get("pick_odds"))
            close = float(clv.get("closing_odds"))
        except (TypeError, ValueError):
            return (0, 0.0)
        crossed = 1 if (pick > 0 and close < 0) else 0
        return (crossed, float(clv.get("clv_pct") or 0))

    best_clv = None
    _best_rank = (-1, -1.0)
    for m in (track_record.get("results") or []):
        clv = m.get("clv") or {}
        pct = clv.get("clv_pct")
        if not m.get("correct") or pct is None or not clv.get("beat_clv"):
            continue
        rank = _clv_rank(m)
        if rank > _best_rank:
            _best_rank = rank
            best_clv = {
                "fighter": m.get("predicted_favorite", ""),
                "surname": (m.get("predicted_favorite") or " ").split()[-1],
                "opponent": (m.get("fighter_b") if m.get("fighter_a") == m.get("predicted_favorite")
                             else m.get("fighter_a")) or "",
                "pick_odds": format_american_odds(clv.get("pick_odds")),
                "closing_odds": format_american_odds(clv.get("closing_odds")),
                "swing": abs(int(round(float(clv.get("closing_odds", 0))
                                       - float(clv.get("pick_odds", 0))))),
                "clv_pct": pct,
                "units": m.get("units_result"),
                # "Decision - Unanimous" reads as a database field, not a sentence
                "method": (m.get("actual_method") or "decision").split(" - ")[0].lower()
                          .replace("ko/tko", "KO/TKO").replace("submission", "submission"),
                "when": _format_friendly_date(m.get("date_added")) or "",
            }

    tape = []
    _seen = set()
    for source in (future_events or []), (events or []):
        for ev in source:
            for f in ev.get("fights", []):
                if f.get("winner") or f.get("cancelled"):
                    continue
                for name in (f.get("fighter_a"), f.get("fighter_b")):
                    if not name or name in _seen:
                        continue
                    hist = ((_snap.get(f"{name}|Moneyline") or {}).get("history")) or []
                    if not hist:
                        continue
                    # THE LAST QUOTE IS NOT ALWAYS A QUOTE. A market that has
                    # resolved prints its outcome as a price: -199900 is
                    # 99.95%, a settlement tick rather than a line anyone
                    # could have bet. Four fighters were riding the tape at
                    # -199900 with a "+75.0%" move next to them that was
                    # nothing but the market closing.
                    # The charts have always stripped these (_drop_settled in
                    # line_movement). Reusing track_record's threshold rather
                    # than picking a third one: 0.97 is loose enough that a
                    # genuine -3000 blowout favourite still rides the tape.
                    hist = [h for h in hist if not _is_settled_price(h.get("odds"))]
                    if not hist:
                        continue
                    try:
                        now = float(hist[-1]["odds"])
                        opened = float(hist[0]["odds"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    _seen.add(name)
                    # THE MOVE IS IN IMPLIED PROBABILITY, NOT IN THE NUMBER.
                    # A percent change of the American price is not a real
                    # quantity: +110 -> -110 is a 4.8-point move that reads as
                    # -200%, and the sign flips arbitrarily as the price crosses
                    # +/-100. Converting both ends to implied probability and
                    # differencing gives percentage POINTS, which is what a line
                    # move actually is and what every book quotes internally.
                    def _implied(o):
                        return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)

                    delta = None
                    if len(hist) > 1:
                        try:
                            delta = round((_implied(now) - _implied(opened)) * 100, 1)
                        except ZeroDivisionError:
                            delta = None
                    tape.append({
                        "name": name.split()[-1],
                        "odds": int(round(now)),
                        # None renders as a neutral marker, not a fake zero move
                        "delta": delta,
                    })
                if len(tape) >= 18:
                    break
            if len(tape) >= 18:
                break
        if len(tape) >= 18:
            break


    # --------------------------------------------------------------------
    # TWO REAL FIGHTS, ONE PER CARD.
    #
    # This was a single fight required to carry the price chart, the scouting
    # rails AND the radar together, and to be graded. That combination never
    # occurs: events and future_events only ever hold the current and the
    # upcoming cards, and a graded fight is by definition on neither. So the
    # page served the illustration fallback on every build from the day the
    # filter was written, and the diagnostic reporting it read as a data
    # problem rather than as the logical impossibility it actually was.
    #
    # THE GRADED FILTER WAS THE WRONG GUARD. Its stated job was to stop the
    # preview advertising something the free tier redacts. But redact_fight
    # strips the MODEL layer and nothing else -- the pick, the confidence, the
    # waterfall, the per-row opinions. profile.rows, profile.radar_svg and
    # moneyline_chart all pass through it untouched, and the free block on the
    # page names the rails, the radar and the movement out loud. All three are
    # free on an upcoming fight too, so the honest constraint was never
    # "graded", it is "carries no model opinion" -- which these cards do not.
    #
    # CHOSEN INDEPENDENTLY, AND RANKED FOR CONTRAST. One fight rarely has both
    # a dramatic price path and two sharply divergent fighters; demanding one
    # supply both is how the old rule ended up asking for four things at once.
    # A flat line, or two near-identical scouting profiles, would show the
    # feature working while making it look like it had found nothing.
    # --------------------------------------------------------------------
    _all_fights = [f for ev in (list(events or []) + list(future_events or []))
                   for f in (ev.get("fights") or [])]

    # WHAT MAKES A GOOD PRICE CHART is not the biggest number. Ranking on
    # swing alone picked a 72-point "move" that was a thin book with a single
    # 38-point discontinuity in it and 37 hours of history -- dramatic,
    # erratic, and a flat contradiction of the copy above it promising every
    # price since the fight was booked.
    #
    # THOSE PREFERENCES ARE A SCORE, NOT A GATE. They were hard rejects
    # first, and that shipped the illustration to production: a minimum span,
    # a maximum step and a minimum move together left exactly ONE qualifying
    # fight on my snapshot, and none at all on the build CI ran an hour later.
    # A filter that admits one candidate is a filter that admits zero on the
    # next set of data. Preferences belong in the ranking, where a mediocre
    # real chart still beats a drawing; only a genuinely flat line is
    # rejected, because that is the one case where the illustration is
    # honestly the better picture.
    # FIGHTS THAT ARE ALREADY OVER ARE THE BETTER PICTURE, and they were not
    # in the pool at all. An upcoming fight's chart stops at today, halfway
    # through the story; a graded one runs the whole way to the bell, which is
    # the arc this card's headline actually promises. Built from the snapshot
    # alone -- see build_snapshot_chart -- so 89 extra candidates cost no
    # network at all.
    _past = []
    for _r in (track_record.get("results") or []):
        _c = build_snapshot_chart(_r.get("fighter_a", ""), _r.get("fighter_b", ""),
                                  odds_snapshot or {})
        if _c:
            _c["_when"] = _r.get("date_added", "")
            _c["_settled"] = True
            _past.append(_c)

    _MIN_NET_PP = 3.0
    _cands = []
    for _f in _all_fights + _past:
        if not _f.get("moneyline_chart"):
            continue
        _net  = abs(_f.get("moneyline_net_pp") or 0)
        _span = _f.get("moneyline_span_days") or 0
        _step  = _f.get("moneyline_max_step_pp") or 0
        _jumps = _f.get("moneyline_jumps") or 0
        _jump_pp = _f.get("moneyline_jump_pp") or 0.0
        if _net < _MIN_NET_PP:
            continue
        # RANK ON THE PART OF THE MOVE THAT HAPPENED GRADUALLY. Counting
        # jumps and subtracting a fixed cost per jump was still the wrong
        # shape: it let a chart win on a big net move that arrived entirely
        # as one cliff. Production picked exactly that -- Keita/Naimov, +28pp
        # net across 1.6 days, flat for half its width and then a single
        # 30-point vertical step. Every number in the ranking said it was the
        # best chart available and it was the one picture the card must not
        # show, because a card headlined "watch the market move" needs a
        # market that visibly moved rather than teleported.
        # Net minus the travel spent in jumps is that quantity directly, and
        # it needs no tuned constant to express.
        # NOT clamped at zero. Flooring it made every jumpy chart tie at
        # nothing, which handed the decision to span and the flip bonus --
        # and the runner-up on a build where the good chart was missing was
        # Xiong/Polastri, thirteen jumps totalling 373 points of travel. A
        # clean three-point drift over eight days is a better picture than
        # that, and only an unclamped cost says so.
        _smooth = _net - _jump_pp
        _score = (
            _smooth * 1.5
            # Span still carries real weight: "open to bell" is the claim on
            # the card, and a fortnight tells that story where an evening
            # cannot. Capped so age alone cannot win it.
            + min(_span, 21) * 2.0
            + (8 if _f.get("moneyline_flipped") else 0)
            # A whisper of raw net, purely to break ties on a bad week when
            # every candidate is jumps and _smooth collapses to zero for all
            # of them. Too small to reorder anything that is not already tied.
            + _net * 0.15
        )
        _cands.append((_score, _net, _span, _step, _jumps, _f))
    _cands.sort(key=lambda c: -c[0])
    if _cands:
        print(f"[landing] market card: {len(_cands)} candidates (net / span / worst step)")
        for _sc, _net, _span, _step, _jumps, _f in _cands[:5]:
            print(f"           {_sc:6.1f}  {_f.get('fighter_a','?')[:16]:16s} vs "
                  f"{_f.get('fighter_b','?')[:16]:16s} net {_net:5.1f}pp  "
                  f"span {_span:5.1f}d  smooth {_net - (_f.get('moneyline_jump_pp') or 0):6.1f}pp  "
                  f"jumps {_jumps} ({_f.get('moneyline_jump_pp') or 0:.0f}pp)"
                  + ("  FLIP" if _f.get("moneyline_flipped") else ""))
    market_preview = None
    if _cands:
        _f = _cands[0][5]
        market_preview = {
            "fighter_a": _f.get("fighter_a", ""), "fighter_b": _f.get("fighter_b", ""),
            "chart": _f["moneyline_chart"],
            "net": _f.get("moneyline_net_pp"),
            "span_days": _f.get("moneyline_span_days"),
            "flipped": _f.get("moneyline_flipped"),
            "settled": bool(_f.get("_settled")),
            "score": round(_cands[0][0], 1),
        }
        print(f"[landing] market card best this build: {market_preview['fighter_a']} vs "
              f"{market_preview['fighter_b']}, net {market_preview['net']:+}pp over "
              f"{market_preview['span_days']}d (score {market_preview['score']})")

    # ------------------------------------------------------------------
    # THE CHOSEN CHART IS REMEMBERED BETWEEN BUILDS.
    #
    # Re-deciding every 30 minutes is what put a bad chart in front of
    # readers. The pool is not stable: CLOB history is available for some
    # fights on some runs and not others, so one build sees a 13-day 1,241
    # point series and the next sees the same fight with 30 snapshot points.
    # Picking the best of whatever happens to be present means the page is
    # only ever as good as its unluckiest build -- and the card that shipped
    # was a 30-point vertical cliff chosen on a run where the good series
    # simply was not there.
    #
    # A marketing illustration has no reason to churn at that cadence. The
    # winner is written to disk with its score and only replaced when a new
    # candidate is CLEARLY better, so a good chart survives every thin build
    # after it. Two escape valves stop it fossilising: anything materially
    # better takes the slot immediately, and the stored chart is retired once
    # it is old enough that "recent" stops being true of it.
    # ------------------------------------------------------------------
    _held_path = f"{DATA_DIR}/landing_chart.json"
    _today = dt.date.today().isoformat()
    _held = None
    try:
        with open(_held_path) as _fh:
            _held = json.load(_fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        _held = None

    _HOLD_MARGIN = 1.15      # a challenger must be 15% better, not merely different
    _HOLD_MAX_AGE_DAYS = 45  # after this the held chart is no longer "recent"

    def _age_days(iso):
        try:
            return (dt.date.today() - dt.date.fromisoformat(iso)).days
        except (TypeError, ValueError):
            return 10**6

    if _held and _age_days(_held.get("chosen_on", "")) > _HOLD_MAX_AGE_DAYS:
        print(f"[landing] held chart retired at {_age_days(_held.get('chosen_on',''))}d old")
        _held = None

    # A FINISHED FIGHT IS NOT A LIVE MARKET. Age was the only escape valve, so
    # a chart chosen on 2026-08-24 was still being published under "Watch the
    # market move, from open to bell" on 2026-08-31 -- two days AFTER Tsuruya
    # beat Borjas, with the axis stopping a week short. Its replacement scored
    # 7.1 against a 57.73 hold threshold, so nothing would have displaced it
    # until the 45-day timer expired on 2026-10-08. This is the one card a
    # prospective subscriber judges the live-odds feature by.
    if _held:
        try:
            _res = pd.read_csv(f"{DATA_DIR}/fight_results.csv")
            _settled = {fight_key(r["fighter_a"], r["fighter_b"], "")[0]
                        for _, r in _res.iterrows()}
            if fight_key(_held.get("fighter_a"), _held.get("fighter_b"), "")[0] in _settled:
                print(f"[landing] held chart retired -- {_held.get('fighter_a')} vs "
                      f"{_held.get('fighter_b')} has a recorded result")
                _held = None
        except (OSError, KeyError, pd.errors.EmptyDataError):
            pass    # no results file yet is not a reason to drop a good chart

    if market_preview and _held:
        if market_preview["score"] > _held.get("score", 0) * _HOLD_MARGIN:
            print(f"[landing] new chart beats the held one "
                  f"({market_preview['score']} vs {_held.get('score')}) -- replacing")
            _held = None
        else:
            print(f"[landing] keeping the held chart: {_held.get('fighter_a')} vs "
                  f"{_held.get('fighter_b')} (score {_held.get('score')}, "
                  f"chosen {_held.get('chosen_on')}); this build's best was "
                  f"{market_preview['score']}")

    if _held:
        market_preview = _held
    elif market_preview:
        market_preview["chosen_on"] = _today
        try:
            with open(_held_path, "w") as _fh:
                json.dump(market_preview, _fh)
        except OSError as _exc:
            print(f"[landing] could not persist the chart choice: {_exc}")
    else:
        # Now genuinely says something: every chart on the page is flat.
        print(f"[landing] no chart moved {_MIN_NET_PP}pp or more across "
              f"{sum(1 for f in _all_fights if f.get('moneyline_chart'))} charts "
              f"-- market card falls back to an illustration")

    scout_preview, _best = None, None
    for _f in _all_fights:
        _prof = _f.get("profile") or {}
        # Both men profiled, or one side's rails come out empty and the card
        # reads as a rendering failure rather than as a thin record.
        if not (_prof.get("radar_svg") and _prof.get("a_ok") and _prof.get("b_ok")):
            continue
        # RANK ON EXACTLY THE RAILS THE CARD SHOWS, in their canonical order.
        # Picking whichever subset diverges most would overstate what is on
        # screen: the reader would be looking at a curated sample presented as
        # an ordinary one. What is measured here is what renders.
        #
        # ALL SIX NOW, matching fighter_profile.RAIL_LABELS and therefore the
        # product. The card showed four and the app has always drawn six, so
        # the landing page was quietly advertising less than it ships --
        # Control and Chin, two of the more separating measures, were missing
        # from the pitch and present in the thing being pitched.
        _rows = (_prof.get("rows") or [])[:len(RAIL_LABELS)]
        if len(_rows) < len(RAIL_LABELS) or any(r.get("a") is None or r.get("b") is None for r in _rows):
            continue
        _contrast = sum(abs(r["a"] - r["b"]) for r in _rows) / len(_rows)
        if _best is None or _contrast > _best[0]:
            _best = (_contrast, _f, _rows, _prof)
    if _best:
        _contrast, _f, _rows, _prof = _best
        scout_preview = {
            "fighter_a": _f.get("fighter_a", ""), "fighter_b": _f.get("fighter_b", ""),
            "rows": _rows, "radar_svg": _prof["radar_svg"],
        }
        print(f"[landing] scouting card: {scout_preview['fighter_a']} vs "
              f"{scout_preview['fighter_b']}, {_contrast:.0f}pt mean rail gap")
    else:
        print("[landing] no fight with both sides profiled -- scouting card falls back")

    # HISTORY CARD. The scouting card ranks a fighter; this one is the tape
    # behind the ranking, and until now the landing page never mentioned it
    # existed -- which made it the biggest thing in the product with no line
    # of copy anywhere.
    #
    # Picked for DEPTH, not for a good night. The claim is "all of it, not the
    # last five", so the card has to be carried by someone with enough career
    # for that to mean something; a two-bout fighter would illustrate the
    # opposite. Ties break toward the fighter on the nearest card, so the name
    # is one a reader recognises from the fights above.
    history_preview = None
    try:
        # NEAREST CARD FIRST. In the app this history is one tap from a fighter
        # on Saturday's card, so the name here should be one the reader has
        # just scrolled past rather than someone booked in six weeks.
        _near = {n for ev in (events or []) for f in (ev.get("fights") or [])
                 for n in (f.get("fighter_a"), f.get("fighter_b")) if n}
        _booked = [n for _f in _all_fights
                   for n in (_f.get("fighter_a"), _f.get("fighter_b")) if n]
        _hist = build_fighter_history(_booked) if _booked else {}
        _best = None
        for _name in _booked:
            _bouts = _hist.get(fh_fold_name(_name)) or []
            # Enough career that "all of them, not the last five" means
            # something -- a three-bout fighter illustrates the opposite.
            if len(_bouts) < 6:
                continue
            _shown = _bouts[:4]
            _drawn = sum(len(b.get("rs") or []) for b in _shown)
            # A strip that is mostly hatching is a picture of missing data.
            if _drawn < 8:
                continue
            # RESULT-BLIND, deliberately. Ranking on rounds that render and
            # career depth says nothing about whether the man won them, so
            # this cannot quietly become a highlight reel; whoever it lands
            # on, his record is drawn as it happened.
            _rank = (_name in _near, _drawn, len(_bouts))
            if _best is None or _rank > _best[0]:
                _best = (_rank, _name, _bouts)
        if _best:
            _rank, _name, _bouts = _best
            _covered = {k: v for k, v in _hist.items() if v}
            history_preview = {
                "fighter": _name,
                "total_bouts": len(_bouts),
                # Four fits the canvas without this becoming the tallest card
                # on the page. The copy carries the "all of them" claim and
                # the counts underneath prove it.
                "bouts": _bouts[:4],
                "fighters_covered": len(_covered),
                "bouts_covered": sum(len(v) for v in _covered.values()),
                "rounds_covered": sum(len(b.get("rs") or []) for v in _covered.values() for b in v),
                # THE TWO DEEPEST CAREERS ON THE CURRENT ROSTER, generated.
                # The copy underneath used to name Gastelum and Vera with
                # their bout counts written out by hand. Both were true the
                # day they were typed and both stop being true the moment
                # those two are not booked -- build_fighter_history only
                # covers booked fighters, so the sentence would go on naming
                # someone the page no longer carries, and nothing would warn
                # us because it is marketing copy rather than a figure.
                # Whoever actually tops the list now wins the sentence.
                "deepest": [
                    {"name": _display_name(k, _booked), "bouts": len(v)}
                    for k, v in sorted(_covered.items(), key=lambda kv: -len(kv[1]))[:2]
                ],
            }
            print(f"[landing] history card: {_name}, {len(_bouts)} bouts, "
                  f"{_rank[1]} rounds drawn ({history_preview['bouts_covered']} bouts / "
                  f"{history_preview['rounds_covered']} rounds across "
                  f"{history_preview['fighters_covered']} booked fighters)")
        else:
            print("[landing] no booked fighter with a deep enough history -- card falls back")
    except Exception as _exc:
        # Marketing copy must never take the build down.
        print(f"[landing] history card unavailable ({_exc})")

    # ---------------------------------------------------------------------
    # FORWARD COVERAGE. Every competitor's landing page is about this
    # Saturday, because this Saturday is all most of them have. The model
    # prices a fight the moment it is announced, so on any given day there are
    # two months of cards already read -- and that is a capability claim no
    # screenshot of a single event can make.
    # Rendered as the ACTUAL SCHEDULE rather than a number in a sentence:
    # "we cover 8 upcoming cards" is a claim, the list of eight named cards
    # with their dates is evidence, and it costs the same space.
    # ---------------------------------------------------------------------
    coverage = None
    _cards = []
    for ev in sorted((future_events or []), key=lambda e: e.get("event_date") or "9999"):
        _fights = ev.get("fights") or []
        if not _fights:
            continue                       # an announced card with no bouts read yet proves nothing
        _name = (ev.get("event_name") or "").strip()
        # "UFC 331: Van vs. Pantoja 2" -> badge "UFC 331", matchup the rest.
        # A numbered PPV and a Fight Night are different things to a reader and
        # the split is what makes the list scannable at a glance.
        # LEFT AS None WHEN UNANNOUNCED, deliberately. This value feeds two
        # slots: a list row, which substitutes MAIN_EVENT_TBD because it must
        # never be blank, and a prose sentence, which reads "<label>, <matchup>,
        # is N weeks away" and omits the clause entirely when there is nothing
        # to name. Defaulting it here would have written "Fight Night, Main
        # event TBD, is 10 weeks away". Truthful data, presentation decides.
        _label, _match = split_event_name(_name)
        _label = _label.replace("UFC Fight Night", "Fight Night").replace("Noche UFC", "Noche")
        try:
            _d = dt.datetime.strptime(ev["event_date"], "%Y-%m-%d")
            _date = f"{_d.strftime('%b')} {_d.day}"
        except (KeyError, ValueError, TypeError):
            continue                       # no parseable date, no row -- never guess one
        _cards.append({"label": _label, "matchup": _match, "date": _date,
                       "iso": ev["event_date"], "fights": len(_fights)})

    if _cards:
        _last = dt.datetime.strptime(_cards[-1]["iso"], "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc)
        coverage = {
            "cards": _cards,
            "card_count": len(_cards),
            "fight_count": sum(c["fights"] for c in _cards),
            "last": _cards[-1],
            # Floor, so the headline number is never larger than the truth.
            "weeks_out": max(1, (_last - dt.datetime.now(dt.timezone.utc)).days // 7),
        }
        print(f"[landing] forward coverage: {coverage['card_count']} cards, "
              f"{coverage['fight_count']} fights, out to {_cards[-1]['date']} "
              f"({coverage['weeks_out']}w)")
    else:
        print("[landing] no future cards with fights -- coverage section omitted")

    # ------------------------------------------------------------------
    # THE SHORT LIST. Two tiers, disjoint, each with its own stake.
    #
    # DISJOINT ON PURPOSE. Every Lock of the Week is also a High Confidence
    # pick, so showing "9-0" beside the tier's full "19-1" invites a reader to
    # add 46.96 and 58.76 into +105.72U -- against a record card 200px below
    # saying +63.44U. Splitting the locks out leaves two sets that genuinely
    # do not overlap: 9 + 10 = 19, 46.96 + 11.80 = 58.76, and nothing has to
    # be disclaimed in a footnote.
    #
    # The stake sits under each label because without it the second row reads
    # as the weaker tier. It is not: 11 picks at 5U returned +21.5% on stake
    # against the locks' +52.2%. The gap is the stake, not the quality.
    # ------------------------------------------------------------------
    shortlist = None
    _lock = track_record.get("lock_record") or {}
    _hi = (track_record.get("by_confidence") or {}).get("High Confidence") or {}
    _bt = (track_record.get("units_stats") or {}).get("by_tier") or {}
    if _lock.get("total") and _hi.get("total") and "Lock of the Week" in _bt:
        _nl_total = _hi["total"] - _lock["total"]
        _nl_ok = _hi["correct"] - _lock["correct"]
        shortlist = {
            "lock": {"won": _lock["correct"], "lost": _lock["total"] - _lock["correct"],
                     "units": _bt["Lock of the Week"]["units"], "stake": LOCK_OF_WEEK_UNITS},
            "high": {"won": _nl_ok, "lost": _nl_total - _nl_ok,
                     "units": _bt.get("High Confidence", {}).get("units", 0.0),
                     "stake": UNITS_BY_CONFIDENCE["High Confidence"]},
        }
        print(f"[landing] short list: lock {shortlist['lock']['won']}-{shortlist['lock']['lost']} "
              f"{shortlist['lock']['units']:+.2f}U · high {shortlist['high']['won']}-"
              f"{shortlist['high']['lost']} {shortlist['high']['units']:+.2f}U")

    html = env.get_template("landing.html").render(
        market_preview=market_preview,
        scout_preview=scout_preview,
        history_preview=history_preview,
        tape=tape,
        coverage=coverage,
        landing_facts=landing_facts or [],
        best_clv=best_clv,
        clv_stats=track_record.get("clv_stats"),
        countdown_target_iso=countdown_target_iso or "",
        countdown_series=countdown_series,
        countdown_matchup=countdown_matchup,
        # Both public by design. The anon key carries no privileges of its
        # own -- everything it can do is what the RLS policies in
        # supabase/migrations allow -- which is why it can sit in page source.
        supabase_url="https://kmifrsmgypghjwmzoffd.supabase.co",
        supabase_anon_key=(
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImttaWZyc21neXBnaGp3bXpvZmZkIiwicm9sZSI6"
            "ImFub24iLCJpYXQiOjE3ODc0MzU3OTEsImV4cCI6MjEwMzAxMTc5MX0."
            "iKjw9pcpFQPfCd8H0_pt3EUJl3svd29aPMH2CWlGBco"
        ),
        tr=track_record,
        shortlist=shortlist,
        units_timeseries_svg=units_svg,
        demo_graded=graded,
        demo_upcoming=upcoming,
        generated_at_short=generated_at_short,
    )
    os.makedirs("docs", exist_ok=True)
    with open("docs/welcome.html", "w") as f:
        f.write(html)
    print(f"Wrote docs/welcome.html ({len(html)/1024:.0f}KB)")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Build the Octane Alpha site.")
    _ap.add_argument("--tier", choices=("member", "free"), default="member",
                     help="member (default) writes the full payload to docs/index.html; "
                          "free writes a model-redacted payload to build/free.html")
    _ap.add_argument("--out", default=None, help="override the output path")
    _args = _ap.parse_args()
    _out = _args.out or (FREE_OUTPUT_PATH if _args.tier == "free" else None)
    main(tier=_args.tier, output_path=_out)
