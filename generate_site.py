"""
Generates a static HTML page (docs/index.html) from live odds + the Elo
model, for publishing via GitHub Pages. This is what the GitHub Actions
workflow runs on a schedule.

Run manually with:
    ODDS_API_KEY=your_key python generate_site.py
"""

import datetime as dt
import os

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from src.elo import EloRatingSystem
from src.edge_finder import find_all_edges
from src.live_props import get_live_props

DATA_DIR = "data"
OUTPUT_PATH = "docs/index.html"


def build_ratings() -> dict[str, float]:
    history_df = pd.read_csv(f"{DATA_DIR}/fight_history.csv")
    elo = EloRatingSystem()
    elo.build_from_history(history_df)
    return elo.ratings


def main():
    elo_ratings = build_ratings()
    fighters_df = pd.read_csv(f"{DATA_DIR}/fighters.csv")

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

    rankings_df = pd.DataFrame(
        [{"fighter": f, "elo": r} for f, r in elo_ratings.items()]
    ).sort_values("elo", ascending=False).reset_index(drop=True)

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("site.html")

    html = template.render(
        edges=edges_df.to_dict("records"),
        rankings=rankings_df.to_dict("records"),
        live_error=live_error,
        generated_at=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )

    os.makedirs("docs", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Wrote {OUTPUT_PATH} ({len(edges_df)} edges, {len(rankings_df)} ranked fighters)")


if __name__ == "__main__":
    main()
