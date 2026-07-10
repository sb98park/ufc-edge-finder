"""
Matches computed edges (which only know fighter names) back to the real
upcoming fight card (data/fight_cards.csv) so the site can group everything
by event -> fight, instead of one flat table.
"""

import re
import unicodedata

import pandas as pd

from src.rationale import explain_edge, explain_favorite_pick
from src.model_preview import build_fight_preview, build_full_market_projection
from src.odds_utils import implied_prob_to_american, format_american_odds


# A consistent accent color per division, purely for faster visual scanning
# down a long card -- not tied to any model logic.
# Groups card_position into the actual broadcast segments for divider
# purposes -- Main Event and Co-Main Event are individually the most
# important fights, but they're still PART of the "Main Card" broadcast
# segment, not their own separate segments.
SEGMENT_LABELS = {
    "Main Event": "MAIN CARD",
    "Co-Main Event": "MAIN CARD",
    "Main Card": "MAIN CARD",
    "Prelims": "PRELIMINARY CARD",
    "Early Prelims": "EARLY PRELIMS",
}

WEIGHT_CLASS_COLORS = {
    "Strawweight": "#e88fc7",
    "Flyweight": "#5ec9d6",
    "Bantamweight": "#f2a65a",
    "Featherweight": "#b18af2",
    "Lightweight": "#6db3f2",
    "Welterweight": "#6ddc9a",
    "Middleweight": "#f26d6d",
    "Light Heavyweight": "#e8955e",
    "Heavyweight": "#d64545",
}


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


def group_unmatched_by_fight(unmatched_df: pd.DataFrame) -> list[dict]:
    """
    Groups live odds for fights NOT on any tracked card by fighter pair, so
    they show as a genuine preview instead of a flat, hard-to-scan list.
    """
    if unmatched_df.empty:
        return []

    fights: dict[frozenset, dict] = {}
    for _, row in unmatched_df.iterrows():
        row_dict = row.to_dict()
        fighter_field = row_dict.get("fighter", "")
        opponent = row_dict.get("opponent")

        if " vs " in str(fighter_field):
            names = [n.strip() for n in fighter_field.split(" vs ")]
            if len(names) != 2:
                continue
            fighter_a, fighter_b = names
        elif opponent:
            fighter_a, fighter_b = fighter_field, opponent
        else:
            continue

        key = frozenset({fighter_a, fighter_b})
        if key not in fights:
            fights[key] = {"fighter_a": fighter_a, "fighter_b": fighter_b, "edges": []}
        fights[key]["edges"].append(row_dict)

    result = list(fights.values())
    for fight in result:
        fight["edges"].sort(key=lambda e: abs(e.get("edge_pct", 0)), reverse=True)

    # Filter out orphaned single-market noise (e.g. just a stray Under/Over
    # with no moneyline and nothing else) -- keep only fights that have
    # either a real moneyline or multiple market types, since a single
    # isolated rounds line with no other context isn't a useful preview.
    def _is_substantial(fight: dict) -> bool:
        markets = {e.get("market") for e in fight["edges"]}
        has_moneyline = "Moneyline" in markets
        return has_moneyline or len(markets) >= 2

    result = [f for f in result if _is_substantial(f)]
    result.sort(key=lambda f: len(f["edges"]), reverse=True)
    return result


def assign_canonical_fight_ids(upcoming_df: pd.DataFrame, cards_df: pd.DataFrame) -> pd.DataFrame:
    """
    Different odds sources assign their own internal fight IDs -- Polymarket
    might call the McGregor/Holloway fight 'e1' while DraftKings calls it
    '1'. If a moneyline comes from one source and a rounds prop for the
    SAME real fight comes from another, they'd end up with different
    fight_id values and the parlay builder would treat them as two
    unrelated fights instead of bundling them as a same-fight combo.

    This reassigns fight_id based on the normalized fighter pair matched
    against the tracked card, so every row for the same real fight shares
    one consistent ID no matter which source it came from.
    """
    if upcoming_df.empty:
        return upcoming_df

    card_pairs = {}
    for i, row in cards_df.iterrows():
        key = frozenset({_normalize_name(row["fighter_a"]), _normalize_name(row["fighter_b"])})
        card_pairs[key] = f"card_{i}"

    def canonical_id(row):
        fighter_a, fighter_b = row.get("fighter_a"), row.get("fighter_b")
        if not fighter_a or not fighter_b:
            return row.get("fight_id")
        key = frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)})
        return card_pairs.get(key, f"untracked_{'_'.join(sorted(key))}")

    df = upcoming_df.copy()
    df["fight_id"] = df.apply(canonical_id, axis=1)
    return df


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
        is_five_round = str(row.get("card_position", "")).strip() == "Main Event"
        if fighters_df is not None and effective_ratings is not None:
            preview = build_fight_preview(
                row["fighter_a"], row["fighter_b"], fighters_df, effective_ratings, is_five_round=is_five_round
            )
        fights.append({
            "event_name": row["event_name"],
            "event_date": row["event_date"],
            "event_start_time_et": row.get("event_start_time_et", "19:00"),
            "card_position": row["card_position"],
            "segment_label": SEGMENT_LABELS.get(row["card_position"], row["card_position"]),
            "weight_class": row["weight_class"],
            "weight_class_color": WEIGHT_CLASS_COLORS.get(row["weight_class"], "#8a8f9a"),
            "is_womens_division": bool(row.get("is_womens_division", False)),
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
            is_five_round = str(fight.get("card_position", "")).strip() == "Main Event"
            projection = build_full_market_projection(
                fight["fighter_a"], fight["fighter_b"], fighters_df, effective_ratings, is_five_round=is_five_round
            )
            model_only = []
            if projection:
                for row in projection["method_rows"] + projection["rounds_rows"] + projection["distance_rows"]:
                    if row["market"] not in live_markets:
                        model_only.append(row)
            fight["model_only_rows"] = model_only

        key = (fight["event_name"], fight["event_date"])
        if key not in events_map:
            events_map[key] = {
                "event_name": fight["event_name"], "event_date": fight["event_date"],
                "event_start_time_et": fight.get("event_start_time_et", "19:00"), "fights": [],
            }
        events_map[key]["fights"].append(fight)

    events = list(events_map.values())
    unmatched_df = pd.DataFrame(unmatched_rows) if unmatched_rows else pd.DataFrame()
    return events, unmatched_df


LOW_SAMPLE_THRESHOLD = 6  # career fights below this = flagged as limited data


def _sample_size_flag(fighter_field: str, fighters_df: pd.DataFrame | None) -> dict | None:
    """
    Returns {"fighter": name, "fights": n} for whichever named fighter has
    the thinnest record, if any of them are below LOW_SAMPLE_THRESHOLD --
    None if everyone involved has a reasonable sample. fighter_field may be
    a single name or a "A vs B" fight-level string (GoesTheDistance-style
    rows), so this checks all names present, not just the first.
    """
    if fighters_df is None or not fighter_field:
        return None
    names = [n.strip() for n in fighter_field.split(" vs ")]
    thinnest = None
    for name in names:
        row = fighters_df[fighters_df["name"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        total = int(r.get("wins", 0) or 0) + int(r.get("losses", 0) or 0)
        if total < LOW_SAMPLE_THRESHOLD and (thinnest is None or total < thinnest["fights"]):
            thinnest = {"fighter": name, "fights": total}
    return thinnest


def top_favorite_picks(
    edges_df: pd.DataFrame, fighters_df: pd.DataFrame | None = None, n: int = 5,
    min_odds: float = -220, max_odds: float = 160, min_edge: float = 3.0, min_model_prob: float = 0.55,
) -> list[dict]:
    """
    Straight, single-leg picks meant to actually be bet with real size --
    the opposite instinct from the parlay tiers. A -4000 "safe" favorite
    isn't a real pick (no real payout for the risk), and a +900 longshot
    isn't something to put 5-10 units on even if the model likes it, so
    both ends get filtered out by the odds range. Within that range, only
    picks the model has genuine conviction on qualify (min_edge), then
    sorted by model probability -- the highest-probability picks are what
    you'd actually want to size up on, not just the biggest edge number.
    Capped to one per fight so this doesn't turn into five props on the
    same two fighters.

    min_model_prob is a genuine correctness guard, not just a style
    choice: edge_pct alone measures model-vs-market disagreement, which
    says nothing about whether the model actually favors this side. A
    pick at 49.6% model probability can still clear a healthy edge
    threshold (the market may have it even lower) while the model itself
    is calling it a slight underdog -- which has no business being
    labeled a "favorite pick." Confirmed live: this was a real bug, not
    hypothetical.
    """
    if edges_df.empty:
        return []
    candidates = edges_df[
        (edges_df["edge_pct"] >= min_edge)
        & (edges_df["odds_american"] >= min_odds)
        & (edges_df["odds_american"] <= max_odds)
        & (edges_df["model_prob"] >= min_model_prob)
    ].copy()
    if candidates.empty:
        return []

    candidates = candidates.sort_values("model_prob", ascending=False)
    seen_fights = set()
    picks = []
    for _, row in candidates.iterrows():
        fight_id = row.get("fight_id")
        if fight_id in seen_fights:
            continue
        seen_fights.add(fight_id)
        picks.append(row.to_dict())
        if len(picks) >= n:
            break

    for r in picks:
        # Fight-level rows (GoesTheDistance, "Fight Outcome") never set an
        # "opponent" field, since their "fighter" is already the full
        # matchup string. When mixed into a DataFrame with rows that DO
        # have one, pandas fills the gap with NaN -- which is truthy in
        # Python, so a template check like {% if p.opponent %} doesn't
        # actually filter it out, it just prints the literal word "nan".
        if pd.isna(r.get("opponent")):
            r["opponent"] = None
        r["low_sample"] = _sample_size_flag(r["fighter"], fighters_df)
        try:
            r["model_fair_odds"] = format_american_odds(implied_prob_to_american(r["model_prob"]))
        except (ValueError, ZeroDivisionError):
            r["model_fair_odds"] = "N/A"
        if fighters_df is not None:
            if r["market"] == "Moneyline":
                r["rationale"] = explain_favorite_pick(r, fighters_df)
            else:
                r["rationale"] = explain_edge(r, fighters_df)
    return picks


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
        if pd.isna(r.get("opponent")):
            r["opponent"] = None
        r["low_sample"] = _sample_size_flag(r["fighter"], fighters_df)
        try:
            r["model_fair_odds"] = format_american_odds(implied_prob_to_american(r["model_prob"]))
        except (ValueError, ZeroDivisionError):
            r["model_fair_odds"] = "N/A"
        if fighters_df is not None:
            r["rationale"] = explain_edge(r, fighters_df)
    return records
