"""
UFC Edge Finder - run with: python app.py
Then open http://127.0.0.1:5000

Reads data/fighters.csv, data/fight_history.csv, data/upcoming_props.csv,
builds Elo ratings from history, and surfaces where the model disagrees
most with the current sportsbook lines.

Swap the CSVs in data/ with real, current data to use this for real.
"""

import pandas as pd
from flask import Flask, render_template

from src.elo import EloRatingSystem
from src.edge_finder import find_all_edges
from src.power_rating import build_effective_ratings
from src.odds_utils import format_american_odds

app = Flask(__name__)

DATA_DIR = "data"


def load_data():
    fighters_df = pd.read_csv(f"{DATA_DIR}/fighters.csv")
    history_df = pd.read_csv(f"{DATA_DIR}/fight_history.csv")
    upcoming_df = pd.read_csv(f"{DATA_DIR}/upcoming_props.csv")
    return fighters_df, history_df, upcoming_df


@app.route("/")
def index():
    fighters_df, history_df, upcoming_df = load_data()

    elo = EloRatingSystem()
    elo.build_from_history(history_df)
    effective_ratings = build_effective_ratings(fighters_df, elo.ratings, history_df)

    edges_df = find_all_edges(upcoming_df, fighters_df, effective_ratings)
    if not edges_df.empty:
        edges_df["odds_american"] = edges_df["odds_american"].apply(format_american_odds)

    rankings_df = pd.DataFrame(
        [{"fighter": f, "elo": r} for f, r in effective_ratings.items()]
    ).sort_values("elo", ascending=False).reset_index(drop=True)

    edges = edges_df.to_dict("records")
    rankings = rankings_df.to_dict("records")

    return render_template("index.html", edges=edges, rankings=rankings)


if __name__ == "__main__":
    app.run(debug=True)
