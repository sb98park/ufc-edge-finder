"""
The main event: pulls live DraftKings odds (moneyline + method of victory +
round totals), compares every single line against what the Elo/finish-rate
model thinks it SHOULD be priced at, and flags anything where the gap is
big enough to be worth a second look.

Run:
    python find_ev_bets.py
    python find_ev_bets.py --min-edge 8   # only show edges >= 8%

This is a screening tool, not a betting signal. A flagged prop means "the
model disagrees with the market by more than X%" -- it does NOT mean
"this will win." Model error, injuries, and fight-specific context the
model can't see are all real risks. Treat this as a shortlist to research
further, not a final answer.
"""

import argparse

import pandas as pd

from src.elo import EloRatingSystem
from src.edge_finder import find_all_edges
from src.live_props import get_live_props

DATA_DIR = "data"


def build_ratings() -> dict[str, float]:
    history_df = pd.read_csv(f"{DATA_DIR}/fight_history.csv")
    elo = EloRatingSystem()
    elo.build_from_history(history_df)
    return elo.ratings



def main():
    parser = argparse.ArgumentParser(description="Find mispriced UFC props vs. the model")
    parser.add_argument("--min-edge", type=float, default=5.0, help="Minimum |edge %%| to flag (default 5)")
    args = parser.parse_args()

    fighters_df = pd.read_csv(f"{DATA_DIR}/fighters.csv")
    elo_ratings = build_ratings()

    upcoming_df, source = get_live_props()
    if upcoming_df.empty:
        print("No live props available right now.")
        return

    edges_df = find_all_edges(upcoming_df, fighters_df, elo_ratings)

    print(f"\nData source: {source}")
    print(f"Total lines analyzed: {len(edges_df)}\n")

    flagged = edges_df[edges_df["edge_pct"].abs() >= args.min_edge]

    if flagged.empty:
        print(f"Nothing cleared the {args.min_edge}% edge threshold right now. Market looks efficiently priced.")
    else:
        print(f"🚩 {len(flagged)} line(s) worth a look (|edge| >= {args.min_edge}%):\n")
        print(flagged.to_string(index=False))

    print("\n--- Full board, ranked ---")
    print(edges_df.to_string(index=False))

    print(
        "\nReminder: this flags disagreement with the market, not certainty. "
        "Sanity-check anything flagged against recent form, injuries, and "
        "camp news before deciding it's actually worth anything."
    )


if __name__ == "__main__":
    main()
