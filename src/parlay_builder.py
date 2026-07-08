"""
Builds multi-leg parlays out of the individual live-odds edges already
computed for the tracked card. Two flavors:

  - "Bankroll Builder": modest 2-3 leg combos landing around +100 to +300,
    built from legs the model actually likes (win probability > 50%).
  - "Lotto Parlays": longer-shot 3-5 leg combos at +1000 or higher, built
    from whichever legs give the best combined hit probability for that
    payout tier -- still long shots, just the least-long of the long shots.

Legs from the SAME fight CAN be combined -- e.g. "Fighter A Moneyline" +
"Under 2.5 Rounds" -- as long as they're not actually contradictory or
redundant. Two rules keep this sane:
  1. At most ONE "who wins / how" leg per fight (Moneyline or Method --
     never both, since Method already implies the Moneyline pick and
     combining them is either redundant, same fighter, or contradictory,
     different fighters).
  2. At most ONE "fight length" leg per fight (Total Rounds or Goes The
     Distance), with contradictions excluded (e.g. "wins by Decision"
     can't coexist with "Under 2.5 rounds" or "Ends In Finish").

Cross-fight legs are still assumed independent, which is a real
simplification -- fights on the same card aren't perfectly independent
in reality, but there's no clean way to quantify that from public data.
"""

import itertools

import pandas as pd

from src.odds_utils import american_to_decimal, decimal_to_american, format_american_odds, implied_prob_to_american

WINNER_FAMILY = {"Moneyline"}  # "Method: X" markets are matched by prefix below
LENGTH_FAMILY_PREFIXES = ("Total Rounds", "Fight Outcome")


def _leg_label(row: dict, odds_display: str) -> str:
    """Human-readable description of exactly what this leg is."""
    market = row["market"]
    if market == "Moneyline":
        return f"{row['fighter']} ML ({odds_display})"
    elif market.startswith("Method"):
        method = market.replace("Method: ", "")
        return f"{row['fighter']} by {method} ({odds_display})"
    elif market.startswith("Total Rounds"):
        line_desc = market.replace("Total Rounds ", "")
        return f"{row['fighter']} {line_desc} rounds ({odds_display})"
    elif market.startswith("Fight Outcome"):
        outcome = market.replace("Fight Outcome: ", "")
        return f"{row['fighter']} — {outcome} ({odds_display})"
    return f"{row['fighter']} — {market} ({odds_display})"


def _leg_family(market: str) -> str:
    if market == "Moneyline" or market.startswith("Method"):
        return "winner"
    if any(market.startswith(p) for p in LENGTH_FAMILY_PREFIXES):
        return "length"
    return "other"


def _is_contradiction(leg_a: dict, leg_b: dict) -> bool:
    """
    Blocks both real contradictions AND redundant pairs. A redundant pair
    isn't impossible to happen together -- it's the SAME claim stated twice
    (e.g. "wins by KO/TKO" already means "ends in finish"), so combining
    them double-counts one signal as if it were two independent risks.
    """
    markets = {leg_a["market"], leg_b["market"]}
    is_decision = any(m.startswith("Method: DEC") for m in markets)
    is_finish_method = any(m.startswith("Method: KO") or m.startswith("Method: SUB") for m in markets)
    is_under = any("Under" in m for m in markets)
    is_ends_in_finish = any("Ends In Finish" in m for m in markets)
    is_goes_distance = any("Goes The Distance" in m for m in markets)

    if is_decision and (is_under or is_ends_in_finish):
        return True  # winning by decision means it went the full distance
    if is_finish_method and is_goes_distance:
        return True  # a finish contradicts "goes the distance"
    if is_finish_method and is_ends_in_finish:
        return True  # redundant: "wins by KO/TKO or SUB" already IS "ends in finish"
    if is_decision and is_goes_distance:
        return True  # redundant: "wins by decision" already IS "goes the distance"
    return False


def _build_candidate_pieces(tracked_edges: list[dict], model_only_by_fight: dict | None = None) -> list[dict]:
    """
    Builds the atomic units that can be cross-fight-combined: either a
    single leg, or a valid same-fight (winner + length) pairing. Each piece
    is tagged with its fight_id so the outer combination step still
    enforces "no two pieces from the same fight."

    Includes model-only projected legs (no live book price) when a fight
    has no live props at all -- e.g. a fighter with a standout stat like
    Terrance McKinney's first-round finish rate can still be a real parlay
    idea even without a live book line for it. These are clearly labeled
    "(model)" in the leg text rather than presented as a real bettable price.
    """
    real_legs = [
        row for row in tracked_edges
        if row.get("odds_american") is not None and row.get("model_prob") is not None
    ]

    by_fight: dict = {}
    for row in real_legs:
        by_fight.setdefault(row["fight_id"], []).append(row)

    pieces = []
    for fight_id, legs in by_fight.items():
        winner_legs = [l for l in legs if _leg_family(l["market"]) == "winner"]
        length_legs = [l for l in legs if _leg_family(l["market"]) == "length"]

        # single-leg pieces (either family alone)
        for leg in winner_legs + length_legs:
            pieces.append({
                "fight_id": fight_id,
                "label": _leg_label(leg, format_american_odds(leg["odds_american"])),
                "model_prob": leg["model_prob"],
                "decimal_odds": american_to_decimal(leg["odds_american"]),
                "is_model": False,
            })

        # combined winner+length pieces, skipping real contradictions
        for w in winner_legs:
            for l in length_legs:
                if _is_contradiction(w, l):
                    continue
                pieces.append({
                    "fight_id": fight_id,
                    "label": f"{_leg_label(w, format_american_odds(w['odds_american']))} + {_leg_label(l, format_american_odds(l['odds_american']))}",
                    "model_prob": w["model_prob"] * l["model_prob"],
                    "decimal_odds": american_to_decimal(w["odds_american"]) * american_to_decimal(l["odds_american"]),
                    "is_model": False,
                })

    # Model-only projected pieces -- only added for fights that had NO real
    # legs at all, so a fight with live data isn't diluted with unpriced
    # guesses when real prices already exist for it. Capped to the top 2
    # per fight (not all ~9 possible projections) -- with 11 fights on a
    # card, including every projection exploded the combinatorial search
    # space enough to hang the process (confirmed: caused an OOM kill).
    if model_only_by_fight:
        for fight_id, rows in model_only_by_fight.items():
            if fight_id in by_fight:
                continue
            top_rows = sorted(rows, key=lambda r: r["model_prob"], reverse=True)[:2]
            for row in top_rows:
                try:
                    proj_odds = implied_prob_to_american(row["model_prob"])
                except (ValueError, ZeroDivisionError):
                    continue
                pieces.append({
                    "fight_id": fight_id,
                    "label": _leg_label(row, f"{format_american_odds(proj_odds)}"),
                    "model_prob": row["model_prob"],
                    "decimal_odds": american_to_decimal(proj_odds),
                    "is_model": True,
                })

    return pieces


def _combine(pieces: tuple[dict, ...]) -> dict:
    combined_decimal = 1.0
    combined_prob = 1.0
    legs = []
    fight_ids = []
    for piece in pieces:
        combined_decimal *= piece["decimal_odds"]
        combined_prob *= piece["model_prob"]
        legs.append({"label": piece["label"], "is_model": piece.get("is_model", False)})
        fight_ids.append(piece["fight_id"])
    combined_american = decimal_to_american(combined_decimal)
    return {
        "legs": legs,
        "has_model_legs": any(l["is_model"] for l in legs),
        "fight_ids": fight_ids,
        "combined_american": round(combined_american),
        "combined_american_display": format_american_odds(combined_american),
        "combined_prob": round(combined_prob, 4),
    }


def _find_parlays(
    pieces: list[dict],
    leg_counts: tuple[int, ...],
    min_american: float,
    max_american: float | None,
    min_leg_prob: float,
    max_results: int,
    label: str = "parlay",
) -> list[dict]:
    eligible = [p for p in pieces if p["model_prob"] >= min_leg_prob]

    # Hard safety cap: combinations of size 5 from a pool of even ~100
    # pieces is tens of millions of combos -- confirmed this can hang the
    # process. Capping the pool (keeping the most-likely pieces first)
    # keeps the search fast regardless of how large the input ever gets.
    MAX_POOL_SIZE = 30
    if len(eligible) > MAX_POOL_SIZE:
        eligible = sorted(eligible, key=lambda p: p["model_prob"], reverse=True)[:MAX_POOL_SIZE]

    results = []
    best_miss = None  # track the closest we got, even if nothing qualified

    for count in leg_counts:
        if len(eligible) < count:
            continue
        for combo in itertools.combinations(eligible, count):
            fight_ids = [p["fight_id"] for p in combo]
            if len(set(fight_ids)) != len(fight_ids):
                continue  # no two pieces from the same fight

            parlay = _combine(combo)

            if best_miss is None or abs(parlay["combined_american"] - min_american) < abs(best_miss["combined_american"] - min_american):
                best_miss = parlay

            if parlay["combined_american"] < min_american:
                continue
            if max_american is not None and parlay["combined_american"] > max_american:
                continue
            results.append(parlay)

    if not results:
        distinct_fights = len({p["fight_id"] for p in eligible})
        print(f"[{label}] no combos found: {len(eligible)} eligible pieces across {distinct_fights} distinct fights "
              f"(need >= {min(leg_counts)} distinct fights). "
              f"Closest miss: {best_miss['combined_american_display'] if best_miss else 'none tried'} "
              f"(target: {min_american:+.0f}{'+' if max_american is None else f' to {max_american:+.0f}'})")
        return []

    results.sort(key=lambda p: p["combined_prob"], reverse=True)
    return _select_diverse(results, max_results)


def _select_diverse(results: list[dict], max_results: int) -> list[dict]:
    """
    Picking the top N by raw probability tends to produce near-duplicates --
    if one leg has an unusually high individual probability, almost every
    top-ranked combo ends up including it. This tries progressively looser
    overlap tolerances (0 shared legs first, then 1, then 2...) rather than
    jumping straight from "zero overlap" to "no constraint at all" -- lotto
    in particular has a much smaller pool of genuinely long-shot-priced legs
    than bankroll does, so a hard 0-or-unlimited jump was reusing the same
    1-2 standout legs across every slot far more than necessary.
    """
    max_possible_overlap = max((len(r["fight_ids"]) for r in results), default=1)
    for max_shared in range(0, max_possible_overlap + 1):
        selected = []
        for parlay in results:
            fight_id_set = set(parlay["fight_ids"])
            too_similar = any(
                len(fight_id_set & set(chosen["fight_ids"])) > max_shared
                for chosen in selected
            )
            if not too_similar:
                selected.append(parlay)
            if len(selected) >= max_results:
                return selected
        if len(selected) >= max_results:
            return selected

    return selected


def build_bankroll_builder_parlays(tracked_edges: list[dict], model_only_by_fight: dict | None = None, max_results: int = 3) -> list[dict]:
    """2-3 piece combos landing roughly +100 to +300, from legs the model favors (>50%)."""
    pieces = _build_candidate_pieces(tracked_edges, model_only_by_fight)
    return _find_parlays(
        pieces, leg_counts=(2, 3), min_american=100, max_american=320,
        min_leg_prob=0.50, max_results=max_results, label="bankroll",
    )


def build_lotto_parlays(tracked_edges: list[dict], model_only_by_fight: dict | None = None, max_results: int = 3) -> list[dict]:
    """+1000 or higher combos, 2-5 pieces -- leg count doesn't matter, only the payout does."""
    pieces = _build_candidate_pieces(tracked_edges, model_only_by_fight)
    return _find_parlays(
        pieces, leg_counts=(2, 3, 4, 5), min_american=1000, max_american=None,
        min_leg_prob=0.15, max_results=max_results, label="lotto",
    )


def build_moonshot_parlays(tracked_edges: list[dict], model_only_by_fight: dict | None = None, max_results: int = 3) -> list[dict]:
    """
    +5000 or higher, any leg count from 2 up to 8. This has essentially no
    business hitting -- it's the "why not" tier, built purely for fun. Even
    the longest of long shots still gets ranked by the model's best combined
    probability among everything that clears the bar, so it's the "best
    worst bet" rather than a totally random pile of legs.
    """
    pieces = _build_candidate_pieces(tracked_edges, model_only_by_fight)
    return _find_parlays(
        pieces, leg_counts=(2, 3, 4, 5, 6, 7, 8), min_american=5000, max_american=None,
        min_leg_prob=0.05, max_results=max_results, label="moonshot",
    )
