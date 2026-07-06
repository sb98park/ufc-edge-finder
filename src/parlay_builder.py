"""
Builds multi-leg parlays out of the individual live-odds edges already
computed for the tracked card. Two flavors:

  - "Bankroll Builder": modest 2-3 leg combos landing around +100 to +300,
    built from legs the model actually likes (win probability > 50%).
  - "Lotto Parlays": longer-shot 3-5 leg combos at +1000 or higher, built
    from whichever legs give the best combined hit probability for that
    payout tier -- still long shots, just the least-long of the long shots.

Real-money sportsbooks won't let you parlay two outcomes from the same
fight (they're not independent), so that's enforced here too. Parlay math
otherwise assumes each leg is independent, which is a simplification --
fights on the same card aren't perfectly independent in reality (e.g. a
judging-heavy night), but there's no clean way to quantify that
correlation from public data, so it's called out here rather than faked.
"""

import itertools

import pandas as pd

from src.odds_utils import american_to_decimal, decimal_to_american, format_american_odds


def _leg_label(row: dict) -> str:
    """Human-readable description of exactly what this leg is."""
    market = row["market"]
    if market == "Moneyline":
        return f"{row['fighter']} ML ({format_american_odds(row['odds_american'])})"
    elif market.startswith("Method"):
        method = market.replace("Method: ", "")
        return f"{row['fighter']} by {method} ({format_american_odds(row['odds_american'])})"
    elif "Total Rounds" in market or market in ("TotalRounds",):
        return f"{row['fighter']} {market} ({format_american_odds(row['odds_american'])})"
    return f"{row['fighter']} — {market} ({format_american_odds(row['odds_american'])})"


def _build_candidate_legs(tracked_edges: list[dict]) -> list[dict]:
    """Only real, book-priced legs (not model-only projections) go into a parlay slip."""
    legs = []
    for row in tracked_edges:
        if row.get("odds_american") is None or row.get("model_prob") is None:
            continue
        legs.append({
            "fight_id": row["fight_id"],
            "label": _leg_label(row),
            "model_prob": row["model_prob"],
            "odds_american": row["odds_american"],
            "decimal_odds": american_to_decimal(row["odds_american"]),
            "market": row["market"],
            "fighter": row["fighter"],
        })
    return legs


def _combine(legs: tuple[dict, ...]) -> dict:
    combined_decimal = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_decimal *= leg["decimal_odds"]
        combined_prob *= leg["model_prob"]
    combined_american = decimal_to_american(combined_decimal)
    return {
        "legs": [l["label"] for l in legs],
        "leg_details": list(legs),
        "combined_american": round(combined_american),
        "combined_american_display": format_american_odds(combined_american),
        "combined_prob": round(combined_prob, 4),
    }


def _find_parlays(
    legs: list[dict],
    leg_counts: tuple[int, ...],
    min_american: float,
    max_american: float | None,
    min_leg_prob: float,
    max_results: int,
) -> list[dict]:
    eligible = [l for l in legs if l["model_prob"] >= min_leg_prob]
    results = []

    for count in leg_counts:
        if len(eligible) < count:
            continue
        for combo in itertools.combinations(eligible, count):
            # can't parlay two outcomes from the same fight -- not independent
            fight_ids = [l["fight_id"] for l in combo]
            if len(set(fight_ids)) != len(fight_ids):
                continue

            parlay = _combine(combo)
            if parlay["combined_american"] < min_american:
                continue
            if max_american is not None and parlay["combined_american"] > max_american:
                continue
            results.append(parlay)

    # Best = highest combined hit probability within the target payout tier
    results.sort(key=lambda p: p["combined_prob"], reverse=True)
    return results[:max_results]


def build_bankroll_builder_parlays(tracked_edges: list[dict], max_results: int = 3) -> list[dict]:
    """2-3 leg combos landing roughly +100 to +300, from legs the model favors (>50%)."""
    legs = _build_candidate_legs(tracked_edges)
    return _find_parlays(
        legs, leg_counts=(2, 3), min_american=100, max_american=320,
        min_leg_prob=0.50, max_results=max_results,
    )


def build_lotto_parlays(tracked_edges: list[dict], max_results: int = 3) -> list[dict]:
    """3-5 leg combos at +1000 or higher, ranked by best combined hit probability among the long shots."""
    legs = _build_candidate_legs(tracked_edges)
    return _find_parlays(
        legs, leg_counts=(3, 4, 5), min_american=1000, max_american=None,
        min_leg_prob=0.15, max_results=max_results,
    )
