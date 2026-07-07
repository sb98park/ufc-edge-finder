"""
Generates docs/index.html: live odds/props grouped by real upcoming fight
cards, with a standout-props section flagging the biggest model-vs-market
disagreements. Run by GitHub Actions on a schedule; can also run locally:

    ODDS_API_KEY=your_key python generate_site.py
"""

import datetime as dt
import os
from zoneinfo import ZoneInfo

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.elo import EloRatingSystem
from src.edge_finder import find_all_edges
from src.live_props import get_live_props
from src.card_matcher import (
    load_fight_cards, group_edges_by_card, top_standout_props,
    assign_canonical_fight_ids, group_unmatched_by_fight,
)
from src.power_rating import build_effective_ratings
from src.odds_utils import format_american_odds
from src.parlay_builder import build_bankroll_builder_parlays, build_lotto_parlays
from src.line_movement import load_snapshot, save_snapshot, annotate_movement

DATA_DIR = "data"
OUTPUT_PATH = "docs/index.html"


def build_ratings(fighters_df: pd.DataFrame) -> dict[str, float]:
    history_df = pd.read_csv(f"{DATA_DIR}/fight_history.csv")
    elo = EloRatingSystem()
    elo.build_from_history(history_df)
    return build_effective_ratings(fighters_df, elo.ratings, history_df)


def main():
    fighters_df = pd.read_csv(f"{DATA_DIR}/fighters.csv")
    elo_ratings = build_ratings(fighters_df)
    cards_df = load_fight_cards(f"{DATA_DIR}/fight_cards.csv")
    future_cards_df = load_fight_cards(f"{DATA_DIR}/future_cards.csv")

    live_error = None
    edges_df = pd.DataFrame()
    source = None

    try:
        upcoming_df, source = get_live_props()
        all_known_cards = pd.concat([cards_df, future_cards_df], ignore_index=True)
        upcoming_df = assign_canonical_fight_ids(upcoming_df, all_known_cards)
        edges_df = find_all_edges(upcoming_df, fighters_df, elo_ratings)

        if not edges_df.empty:
            previous_snapshot = load_snapshot()
            edge_records = edges_df.to_dict("records")
            annotate_movement(edge_records, previous_snapshot)
            edges_df = pd.DataFrame(edge_records)

        if edges_df.empty:
            live_error = f"No usable live odds returned right now (source: {source})."
    except Exception as exc:
        live_error = f"Couldn't fetch live odds: {exc}"

    events, unmatched_df = group_edges_by_card(edges_df, cards_df, fighters_df, elo_ratings)
    future_events, still_unmatched_df = group_edges_by_card(unmatched_df, future_cards_df, fighters_df, elo_ratings)
    upcoming_other_fights = group_unmatched_by_fight(still_unmatched_df)

    tracked_edges = pd.DataFrame(
        [edge for event in events for fight in event["fights"] for edge in fight["edges"]]
    )
    standout_props = top_standout_props(tracked_edges, fighters_df, n=5, min_edge=5.0)

    tracked_edges_list = tracked_edges.to_dict("records") if not tracked_edges.empty else []

    model_only_by_fight = {}
    for event in events:
        for fight in event["fights"]:
            fid = fight["edges"][0]["fight_id"] if fight["edges"] else None
            if fid is None and fight.get("model_only_rows"):
                fid = f"{fight['fighter_a']}|{fight['fighter_b']}"
            if fid is not None and fight.get("model_only_rows"):
                model_only_by_fight[fid] = fight["model_only_rows"]

    bankroll_parlays = build_bankroll_builder_parlays(tracked_edges_list, model_only_by_fight)
    lotto_parlays = build_lotto_parlays(tracked_edges_list, model_only_by_fight)

    # Notable line movement across everything we track, for its own section
    all_display_edges = tracked_edges_list + [
        edge for event in future_events for fight in event["fights"] for edge in fight["edges"]
    ]
    notable_movements = sorted(
        [e for e in all_display_edges if e.get("movement") and e["movement"].get("notable")],
        key=lambda e: e["movement"]["pct_change"], reverse=True,
    )[:8]

    if not edges_df.empty:
        save_snapshot(edges_df.to_dict("records"))

    event_short_name = events[0]["event_name"].split(":")[0].strip() if events else "This Weekend"

    env = Environment(loader=FileSystemLoader("templates"))
    env.filters["american"] = format_american_odds
    template = env.get_template("site.html")

    html = template.render(
        events=events,
        future_events=future_events,
        unmatched=unmatched_df.to_dict("records") if not unmatched_df.empty else [],
        standout_props=standout_props,
        event_short_name=event_short_name,
        bankroll_parlays=bankroll_parlays,
        lotto_parlays=lotto_parlays,
        notable_movements=notable_movements,
        upcoming_other_fights=upcoming_other_fights,
        live_error=live_error,
        source=source,
        generated_at=dt.datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M %p ET"),
    )

    os.makedirs("docs", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Wrote {OUTPUT_PATH} ({len(events)} events, {len(future_events)} future events, {len(standout_props)} standout props flagged)")


if __name__ == "__main__":
    main()
