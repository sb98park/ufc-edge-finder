"""
Matches computed edges (which only know fighter names) back to the real
upcoming fight card (data/fight_cards.csv) so the site can group everything
by event -> fight, instead of one flat table.
"""

import re
import unicodedata

import pandas as pd

from src.rationale import explain_edge
from src.model_preview import build_fight_preview, build_full_market_projection
from src.odds_utils import implied_prob_to_american, format_american_odds


def _normalize_name(name: str) -> str:
    """
    Strips accents and standardizes punctuation so minor spelling differences
    between sources (e.g. Polymarket listing 'Benoît Saint Denis' while our
    data has 'Benoit Saint-Denis') don't cause a real fight to silently miss
    its match and get dumped into 'unmatched' instead.
    """
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", normalized.lower()).strip()


def load_fight_cards(path: str = "data/fight_cards.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _split_total_rounds_fighter_field(fighter_field: str) -> set[str]:
    return {name.strip() for name in fighter_field.split(" vs ")}


def group_edges_by_card(
    edges_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    fighters_df: pd.DataFrame | None = None,
    effective_ratings: dict[str, float] | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Returns (events, unmatched_edges):
      events: list of {event_name, event_date, fights: [{fighter_a, fighter_b,
               weight_class, card_position, edges: [...], preview: {...}}]}
      unmatched_edges: edges whose fighters aren't on data/fight_cards.csv
               (still useful, just can't be grouped into a known card)
    """
    fights = []
    for _, row in cards_df.iterrows():
        preview = None
        if fighters_df is not None and effective_ratings is not None:
            preview = build_fight_preview(
                row["fighter_a"], row["fighter_b"], fighters_df, effective_ratings
            )
        fights.append({
            "event_name": row["event_name"],
            "event_date": row["event_date"],
            "card_position": row["card_position"],
            "weight_class": row["weight_class"],
            "fighter_a": row["fighter_a"],
            "fighter_b": row["fighter_b"],
            "fighters": {row["fighter_a"], row["fighter_b"]},
            "fighters_normalized": {_normalize_name(row["fighter_a"]), _normalize_name(row["fighter_b"])},
            "preview": preview,
            "edges": [],
        })

    unmatched_rows = []

    for _, edge in edges_df.iterrows():
        edge_dict = edge.to_dict()
        if fighters_df is not None:
            edge_dict["rationale"] = explain_edge(edge_dict, fighters_df)
        fighter_field = edge_dict["fighter"]

        if " vs " in fighter_field:
            row_pair = _split_total_rounds_fighter_field(fighter_field)
        elif edge_dict.get("opponent"):
            # Moneyline/Method rows: require BOTH the fighter AND their listed
            # opponent to match a tracked fight's exact pair. Matching on the
            # fighter's name alone is what let a stale/unrelated row (e.g. a
            # leftover "vs a different opponent" line) get folded into the
            # wrong fight just because one name happened to overlap.
            row_pair = {edge_dict["fighter"], edge_dict["opponent"]}
        else:
            row_pair = {fighter_field}

        matched = False
        row_pair_normalized = {_normalize_name(n) for n in row_pair}
        for fight in fights:
            if row_pair_normalized == fight["fighters_normalized"]:
                fight["edges"].append(edge_dict)
                matched = True
                break

        if not matched:
            unmatched_rows.append(edge_dict)

    # group fights into events, preserving card order
    events_map: dict[tuple, dict] = {}
    for fight in fights:
        # sort each fight's edges by |edge_pct| descending so the juiciest line shows first
        fight["edges"].sort(key=lambda e: abs(e.get("edge_pct", 0)), reverse=True)

        # fill in model-only projections for any method/rounds markets the
        # live book didn't happen to cover for this fight, so there's always
        # something to look at beyond moneyline
        if fighters_df is not None and effective_ratings is not None:
            live_markets = {e["market"] for e in fight["edges"]}
            projection = build_full_market_projection(
                fight["fighter_a"], fight["fighter_b"], fighters_df, effective_ratings
            )
            model_only = []
            if projection:
                for row in projection["method_rows"] + projection["rounds_rows"] + projection["distance_rows"]:
                    if row["market"] not in live_markets:
                        model_only.append(row)
            fight["model_only_rows"] = model_only

        key = (fight["event_name"], fight["event_date"])
        if key not in events_map:
            events_map[key] = {"event_name": fight["event_name"], "event_date": fight["event_date"], "fights": []}
        events_map[key]["fights"].append(fight)

    events = list(events_map.values())
    unmatched_df = pd.DataFrame(unmatched_rows) if unmatched_rows else pd.DataFrame()
    return events, unmatched_df


def top_standout_props(
    edges_df: pd.DataFrame, fighters_df: pd.DataFrame | None = None, n: int = 5, min_edge: float = 5.0
) -> list[dict]:
    """
    The headline 'worth a look' props. Only positive edges qualify -- a
    negative edge just means the OTHER side of that same line is the value
    play, which will already show up as its own positive-edge entry, so
    showing both is redundant and confusing (looks like two different
    findings when it's really one).
    """
    if edges_df.empty:
        return []
    standout = edges_df[edges_df["edge_pct"] >= min_edge].copy()
    standout = standout.sort_values("edge_pct", ascending=False).head(n)
    records = standout.to_dict("records")
    for r in records:
        try:
            r["model_fair_odds"] = format_american_odds(implied_prob_to_american(r["model_prob"]))
        except (ValueError, ZeroDivisionError):
            r["model_fair_odds"] = "N/A"
        if fighters_df is not None:
            r["rationale"] = explain_edge(r, fighters_df)
    return records
