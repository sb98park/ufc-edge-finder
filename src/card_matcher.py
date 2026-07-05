"""
Matches computed edges (which only know fighter names) back to the real
upcoming fight card (data/fight_cards.csv) so the site can group everything
by event -> fight, instead of one flat table.
"""

import pandas as pd


def load_fight_cards(path: str = "data/fight_cards.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _split_total_rounds_fighter_field(fighter_field: str) -> set[str]:
    return {name.strip() for name in fighter_field.split(" vs ")}


def group_edges_by_card(edges_df: pd.DataFrame, cards_df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """
    Returns (events, unmatched_edges):
      events: list of {event_name, event_date, fights: [{fighter_a, fighter_b,
               weight_class, card_position, edges: [...]}]}
      unmatched_edges: edges whose fighters aren't on data/fight_cards.csv
               (still useful, just can't be grouped into a known card)
    """
    fights = []
    for _, row in cards_df.iterrows():
        fights.append({
            "event_name": row["event_name"],
            "event_date": row["event_date"],
            "card_position": row["card_position"],
            "weight_class": row["weight_class"],
            "fighter_a": row["fighter_a"],
            "fighter_b": row["fighter_b"],
            "fighters": {row["fighter_a"], row["fighter_b"]},
            "edges": [],
        })

    unmatched_rows = []

    for _, edge in edges_df.iterrows():
        edge_dict = edge.to_dict()
        fighter_field = edge_dict["fighter"]

        if " vs " in fighter_field:
            names = _split_total_rounds_fighter_field(fighter_field)
        else:
            names = {fighter_field}

        matched = False
        for fight in fights:
            if names & fight["fighters"]:
                fight["edges"].append(edge_dict)
                matched = True
                break

        if not matched:
            unmatched_rows.append(edge_dict)

    # group fights into events, preserving card order
    events_map: dict[tuple, dict] = {}
    for fight in fights:
        key = (fight["event_name"], fight["event_date"])
        if key not in events_map:
            events_map[key] = {"event_name": fight["event_name"], "event_date": fight["event_date"], "fights": []}
        # sort each fight's edges by |edge_pct| descending so the juiciest line shows first
        fight["edges"].sort(key=lambda e: abs(e.get("edge_pct", 0)), reverse=True)
        events_map[key]["fights"].append(fight)

    events = list(events_map.values())
    unmatched_df = pd.DataFrame(unmatched_rows) if unmatched_rows else pd.DataFrame()
    return events, unmatched_df


def top_standout_props(edges_df: pd.DataFrame, n: int = 5, min_edge: float = 5.0) -> list[dict]:
    """The headline 'worth a look' props, sorted by biggest absolute edge."""
    if edges_df.empty:
        return []
    standout = edges_df[edges_df["edge_pct"].abs() >= min_edge].copy()
    standout["abs_edge"] = standout["edge_pct"].abs()
    standout = standout.sort_values("abs_edge", ascending=False).head(n)
    return standout.drop(columns="abs_edge").to_dict("records")
