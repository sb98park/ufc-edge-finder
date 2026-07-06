"""
Generates docs/index.html: live odds/props grouped by real upcoming fight
cards, with a standout-props section flagging the biggest model-vs-market
disagreements. Run by GitHub Actions on a schedule; can also run locally:

    ODDS_API_KEY=your_key python generate_site.py
"""

import datetime as dt
import os

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.elo import EloRatingSystem
from src.edge_finder import find_all_edges
from src.live_props import get_live_props
from src.card_matcher import load_fight_cards, group_edges_by_card, top_standout_props
from src.power_rating import build_effective_ratings
from src.odds_utils import format_american_odds

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

    live_error = None
    edges_df = pd.DataFrame()
    source = None

    try:
        upcoming_df, source = get_live_props()
        edges_df = find_all_edges(upcoming_df, fighters_df, elo_ratings)
        if edges_df.empty:
            live_error = f"No usable live odds returned right now (source: {source})."
    except Exception as exc:
        live_error = f"Couldn't fetch live odds: {exc}"

    events, unmatched_df = group_edges_by_card(edges_df, cards_df, fighters_df, elo_ratings)
    standout_props = top_standout_props(edges_df, fighters_df, n=5, min_edge=5.0)

    rankings_df = pd.DataFrame(
        [{"fighter": f, "elo": r} for f, r in elo_ratings.items()]
    ).sort_values("elo", ascending=False).reset_index(drop=True)

    env = Environment(loader=FileSystemLoader("templates"))
    env.filters["american"] = format_american_odds
    template = env.get_template("site.html")

    html = template.render(
        events=events,
        unmatched=unmatched_df.to_dict("records") if not unmatched_df.empty else [],
        standout_props=standout_props,
        rankings=rankings_df.to_dict("records"),
        live_error=live_error,
        source=source,
        generated_at=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )

    os.makedirs("docs", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Wrote {OUTPUT_PATH} ({len(events)} events, {len(standout_props)} standout props flagged)")


if __name__ == "__main__":
    main()
