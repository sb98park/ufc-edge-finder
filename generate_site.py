"""
Generates docs/index.html: live odds/props grouped by real upcoming fight
cards, with a standout-props section flagging the biggest model-vs-market
disagreements. Run by GitHub Actions on a schedule; can also run locally:

    ODDS_API_KEY=your_key python generate_site.py
"""

import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.elo import EloRatingSystem
from src.edge_finder import find_all_edges
from src.live_props import get_live_props
from src.card_matcher import (
    load_fight_cards, group_edges_by_card, top_standout_props, top_favorite_picks,
    assign_canonical_fight_ids, group_unmatched_by_fight,
)
from src.power_rating import build_effective_ratings
from src.odds_utils import format_american_odds
from src.parlay_builder import build_bankroll_builder_parlays, build_lotto_parlays, build_moonshot_parlays
from src.line_movement import (
    load_snapshot, save_snapshot, annotate_movement, attach_charts_to_fight,
    load_token_cache, save_token_cache, update_token_cache,
)
from src.track_record import log_predictions, compute_track_record, load_momentum_by_key
from src.schedule import build_fight_schedule, apply_live_corrections, promote_card_if_stale
from src.results_fetcher import fetch_and_log_new_results, fetch_espn_live_fight_key
from src.card_discovery import discover_and_append_new_cards, normalize_existing_card_order, resync_tracked_card_order, deduplicate_tracked_fights
from src.fighter_backfill import backfill_fighters
from src.calibration_chart import build_calibration_svg
from src.sparkline_chart import build_sparkline_svg
from src.units_chart import build_units_timeseries_svg
from src.donut_chart import build_donut_svg
from src.damage_silhouette import build_damage_silhouette_svg

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


def build_ratings(fighters_df: pd.DataFrame, history_df: pd.DataFrame) -> dict[str, float]:
    elo = EloRatingSystem()
    elo.build_from_history(history_df)
    return build_effective_ratings(fighters_df, elo.ratings, history_df)


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

    try:
        normalize_existing_card_order(f"{DATA_DIR}/future_cards.csv")
    except Exception as e:
        print(f"[generate_site] card order normalization failed unexpectedly, continuing without it: {e}")

    try:
        backfill_fighters(f"{DATA_DIR}/fighters.csv", f"{DATA_DIR}/future_cards.csv")
    except Exception as e:
        print(f"[generate_site] fighter backfill failed unexpectedly, continuing without it: {e}")

    fighters_df = pd.read_csv(f"{DATA_DIR}/fighters.csv")
    history_df = pd.read_csv(f"{DATA_DIR}/fight_history.csv")
    elo_ratings = build_ratings(fighters_df, history_df)

    future_cards_df = load_fight_cards(f"{DATA_DIR}/future_cards.csv")
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
        upcoming_df, source = get_live_props()
        all_known_cards = pd.concat([cards_df, future_cards_df], ignore_index=True)
        upcoming_df = assign_canonical_fight_ids(upcoming_df, all_known_cards)
        edges_df = find_all_edges(upcoming_df, fighters_df, elo_ratings, history_df)

        if not edges_df.empty:
            edge_records = edges_df.to_dict("records")
            annotate_movement(edge_records, previous_snapshot)
            edges_df = pd.DataFrame(edge_records)

        if edges_df.empty:
            live_error = f"No usable live odds returned right now (source: {source})."
    except Exception as exc:
        live_error = f"Couldn't fetch live odds: {exc}"

    events, unmatched_df = group_edges_by_card(edges_df, cards_df, fighters_df, elo_ratings, weight_class_history_df)
    future_events, still_unmatched_df = group_edges_by_card(unmatched_df, future_cards_df, fighters_df, elo_ratings, weight_class_history_df)

    # Event display order must be chronological (soonest first), independent
    # of whatever order their rows happen to sit in the source CSV -- that
    # order reflects when each card was discovered or re-discovered (e.g.
    # after a lineup-change replacement), not the event's actual date. This
    # is a separate concern from fight order WITHIN one event (billing
    # order, Main Event first), which group_edges_by_card already handles.
    events.sort(key=lambda e: e["event_date"])
    future_events.sort(key=lambda e: e["event_date"])

    tracked_edges = pd.DataFrame(
        [edge for event in events for fight in event["fights"] for edge in fight["edges"]]
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
    if len(tracked_edges) < MIN_EDGES_FOR_CURRENT_CARD and future_events:
        next_event = future_events[0]
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

    standout_props = top_standout_props(tracked_edges, fighters_df, n=5, min_edge=5.0)
    favorite_picks = top_favorite_picks(tracked_edges, fighters_df, n=5)

    tracked_edges_list = tracked_edges.to_dict("records") if not tracked_edges.empty else []
    for e in tracked_edges_list:
        # Fight-level rows (GoesTheDistance, "Fight Outcome") never set an
        # "opponent" field. Building a DataFrame from a mix of rows that
        # do and don't have that key fills the gap with NaN, which is
        # truthy in Python -- so a template check like {% if row.opponent %}
        # doesn't filter it out, it just prints the literal word "nan".
        if pd.isna(e.get("opponent")):
            e["opponent"] = None

    model_only_by_fight = {}
    for event in events_for_model_only:
        for fight in event["fights"]:
            fid = fight["edges"][0]["fight_id"] if fight["edges"] else None
            if fid is None and fight.get("model_only_rows"):
                fid = f"{fight['fighter_a']}|{fight['fighter_b']}"
            if fid is not None and fight.get("model_only_rows"):
                model_only_by_fight[fid] = fight["model_only_rows"]

    try:
        bankroll_parlays = build_bankroll_builder_parlays(tracked_edges_list, model_only_by_fight)
        lotto_parlays = build_lotto_parlays(tracked_edges_list, model_only_by_fight)
        moonshot_parlays = build_moonshot_parlays(tracked_edges_list, model_only_by_fight)
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
        bankroll_parlays, lotto_parlays, moonshot_parlays = [], [], []

    # Notable line movement across everything we track, for its own section
    all_display_edges = tracked_edges_list + [
        edge for event in future_events for fight in event["fights"] for edge in fight["edges"]
    ]
    notable_movements = sorted(
        [e for e in all_display_edges if e.get("movement") and e["movement"].get("notable")],
        key=lambda e: e["movement"]["pct_change"], reverse=True,
    )[:8]

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
    generated_at_short = dt.datetime.now(ZoneInfo("America/New_York")).strftime("%b %-d, %I:%M %p ET")
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

    # Countdown target: this weekend's tracked event if we have one, otherwise
    # the nearest future card. ET is UTC-4 (EDT) for all currently tracked
    # events (July-August) -- would need adjusting for events during EST months.
    countdown_target_iso = None
    countdown_label = None
    next_event = events[0] if events else (future_events[0] if future_events else None)
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

    for event in events:
        for fight in event["fights"]:
            key = frozenset({fight["fighter_a"].strip().lower(), fight["fighter_b"].strip().lower()})
            result = finished_results.get(key)
            if result:
                winner_last = result["winner"].strip().split()[-1].upper()
                fight["winner"] = result["winner"]
                fight["result_label"] = f"{winner_last} BY {result['method']}".strip()
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
            "favorite": fight["preview"]["favorite"], "favorite_prob": fight["preview"]["favorite_prob"],
            "underdog": fight["preview"]["underdog"], "likely_method": fight["preview"]["likely_method"],
            "narrative": fight["preview"]["narrative"],
        }
        for event in events for fight in event["fights"]
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
        total_fights = len(this_weekend_fights)
        confirmed_fights = sum(1 for f in this_weekend_fights if f.get("result_label"))
        if total_fights:
            results_coverage = {"confirmed": confirmed_fights, "total": total_fights}
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path and total_fights:
            missing = [
                f"{f['fighter_a']} vs {f['fighter_b']}"
                for f in this_weekend_fights if not f.get("result_label")
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
        fight_schedule = build_fight_schedule(
            raw_card_rows, events[0]["event_date"], events[0].get("event_start_time_et", "17:00")
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
                    "result_label": f"{r['winner'].strip().split()[-1].upper()} BY {r['method']}".strip(),
                    "result_round_time": f"R{r['end_round']} {r['end_time']}" if r["end_round"] and r["end_time"] else None,
                }
        fight_schedule, last_confirmed_at = apply_live_corrections(fight_schedule, finished_keys)
        if just_concluded:
            just_concluded["last_confirmed_at"] = last_confirmed_at

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
    env.filters["friendly_date"] = _format_friendly_date
    env.globals["donut_svg"] = build_donut_svg
    env.globals["damage_svg"] = build_damage_silhouette_svg

    env.filters["tojson"] = lambda obj: json.dumps(obj, default=str)
    # NaN is truthy in Python, so a plain {% if x %} check doesn't catch a
    # pandas-filled missing value -- it just prints the literal word
    # "nan". This test explicitly excludes both None and NaN (the classic
    # "x != x" is only ever true for NaN) so templates can check
    # "is real_value" instead of relying on Jinja's default truthiness.
    env.tests["real_value"] = lambda x: x is not None and x == x

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
        for prefix in ("Method: ", "Fight Outcome: "):
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

    html = template.render(
        events=events,
        future_events=future_events,
        unmatched=unmatched_df.to_dict("records") if not unmatched_df.empty else [],
        standout_props=standout_props,
        favorite_picks=favorite_picks,
        lock_picks=lock_picks,
        event_short_name=event_short_name,
        countdown_target_iso=countdown_target_iso,
        fight_schedule_json=json.dumps(fight_schedule),
        just_concluded_json=json.dumps(just_concluded),
        espn_live_fight_key_json=json.dumps(espn_live_fight_key),
        days_since_event=days_since_event,
        results_coverage=results_coverage,
        analytics_source_event=analytics_source_event,
        countdown_label=countdown_label,
        whats_new_snapshot=whats_new_snapshot,
        track_record=track_record,
        calibration_svg=calibration_svg,
        units_sparkline_svg=units_sparkline_svg,
        units_timeseries_svg=units_timeseries_svg,
        bankroll_parlays=bankroll_parlays,
        lotto_parlays=lotto_parlays,
        moonshot_parlays=moonshot_parlays,
        notable_movements=notable_movements,
        live_error=live_error,
        source=source,
        generated_at=generated_at_str,
        generated_at_short=generated_at_short,
    )

    os.makedirs("docs", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Wrote {OUTPUT_PATH} ({len(events)} events, {len(future_events)} future events, {len(standout_props)} standout props flagged)")


if __name__ == "__main__":
    main()
