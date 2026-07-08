"""
Model track record: logs the model's prediction for each tracked-card fight
on every run (keeping the latest prediction per fight, since the model can
shift as new data comes in before fight night -- same way a sportsbook line
moves right up until the bell), then compares against actual results once
they're recorded.

Also tracks Closing Line Value (CLV): the moneyline price on the model's
favorite the FIRST time it was logged (pick_odds) vs. the last known price
before the fight (closing_odds). If the market moved TOWARD the model's
side by closing (shortened), the model beat the closing line -- a real,
outcome-independent signal that the model saw value before the market
caught up, not just "got lucky" on a coinflip result.

There's no live results API for this -- results have to be added manually
to data/fight_results.csv after a card happens (event_name, fighter_a,
fighter_b, winner, method). Until then, the track record section stays
honestly empty rather than faking a number.
"""

import csv
import os

from src.odds_utils import american_to_decimal

PREDICTIONS_LOG_PATH = "data/predictions_log.csv"
FIELDNAMES = [
    "event_name", "fighter_a", "fighter_b", "favorite", "favorite_prob",
    "confidence_label", "likely_method", "pick_odds", "closing_odds", "last_updated",
]


def _favorite_moneyline_odds(fight: dict, favorite: str) -> float | None:
    for edge in fight.get("edges", []):
        if edge.get("market") == "Moneyline" and edge.get("fighter") == favorite:
            return edge.get("odds_american")
    return None


def log_predictions(events: list[dict], generated_at: str) -> None:
    """Keeps the LATEST prediction per (event, fighter_a, fighter_b), overwriting older entries for the same fight."""
    existing = {}
    if os.path.exists(PREDICTIONS_LOG_PATH):
        with open(PREDICTIONS_LOG_PATH, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["event_name"], row["fighter_a"], row["fighter_b"])
                existing[key] = row

    for event in events:
        for fight in event.get("fights", []):
            preview = fight.get("preview")
            if not preview:
                continue
            key = (fight["event_name"], fight["fighter_a"], fight["fighter_b"])
            current_odds = _favorite_moneyline_odds(fight, preview["favorite"])
            prior = existing.get(key)

            # pick_odds is set ONCE -- the first time we see a live price for
            # this fight's favorite -- and never overwritten after that, so
            # it genuinely represents "the price when the model first had a
            # read on this fight," not a moving target.
            pick_odds = prior.get("pick_odds") if prior and prior.get("pick_odds") not in (None, "") else None
            if pick_odds is None and current_odds is not None:
                pick_odds = current_odds

            # closing_odds updates every run a live price is available,
            # so whatever it holds when the fight actually happens is
            # naturally the last real price seen -- the closing line.
            closing_odds = current_odds if current_odds is not None else (prior.get("closing_odds") if prior else None)

            existing[key] = {
                "event_name": fight["event_name"],
                "fighter_a": fight["fighter_a"],
                "fighter_b": fight["fighter_b"],
                "favorite": preview["favorite"],
                "favorite_prob": preview["favorite_prob"],
                "confidence_label": preview["confidence_label"],
                "likely_method": preview["likely_method"],
                "pick_odds": pick_odds if pick_odds is not None else "",
                "closing_odds": closing_odds if closing_odds is not None else "",
                "last_updated": generated_at,
            }

    os.makedirs(os.path.dirname(PREDICTIONS_LOG_PATH), exist_ok=True)
    with open(PREDICTIONS_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in existing.values():
            writer.writerow(row)


def _pair_key(fighter_a: str, fighter_b: str) -> frozenset:
    return frozenset({fighter_a.strip().lower(), fighter_b.strip().lower()})


def _clv_result(pick_odds, closing_odds) -> dict | None:
    """
    Beating the closing line means the pick-time price was BETTER value than
    the closing price for the same side -- i.e. the market moved TOWARD the
    model's favorite by fight night (odds shortened), meaning the model saw
    something the market hadn't fully priced in yet. This is independent of
    whether the bet actually won: a fighter can lose straight-up while the
    model still correctly anticipated real market movement, which is the
    whole point of CLV as a model-quality metric distinct from raw record.
    """
    if not pick_odds or not closing_odds:
        return None
    try:
        pick_prob = 1 / american_to_decimal(float(pick_odds))
        closing_prob = 1 / american_to_decimal(float(closing_odds))
    except (ValueError, ZeroDivisionError):
        return None
    beat_clv = closing_prob > pick_prob
    return {
        "pick_odds": float(pick_odds), "closing_odds": float(closing_odds),
        "pick_prob": round(pick_prob, 4), "closing_prob": round(closing_prob, 4),
        "beat_clv": beat_clv,
        "clv_pct": round((closing_prob - pick_prob) * 100, 1),
    }


def compute_track_record(results_csv_path: str = "data/fight_results.csv") -> dict | None:
    """
    Joins logged predictions against recorded results. Returns None if there
    are no recorded results yet (honest empty state, not a fabricated stat).
    """
    if not os.path.exists(results_csv_path) or not os.path.exists(PREDICTIONS_LOG_PATH):
        return None

    with open(results_csv_path, newline="") as f:
        results = list(csv.DictReader(f))
    if not results:
        return None

    with open(PREDICTIONS_LOG_PATH, newline="") as f:
        predictions = list(csv.DictReader(f))

    pred_by_key = {_pair_key(p["fighter_a"], p["fighter_b"]): p for p in predictions}

    matched = []
    for result in results:
        if not result.get("winner"):
            continue
        key = _pair_key(result["fighter_a"], result["fighter_b"])
        pred = pred_by_key.get(key)
        if not pred:
            continue
        correct = pred["favorite"].strip().lower() == result["winner"].strip().lower()
        clv = _clv_result(pred.get("pick_odds"), pred.get("closing_odds"))
        matched.append({
            "event_name": result["event_name"],
            "fighter_a": result["fighter_a"],
            "fighter_b": result["fighter_b"],
            "predicted_favorite": pred["favorite"],
            "confidence_label": pred["confidence_label"],
            "actual_winner": result["winner"],
            "correct": correct,
            "clv": clv,
        })

    if not matched:
        return None

    total = len(matched)
    correct_count = sum(1 for m in matched if m["correct"])

    by_confidence = {}
    for label in ("High Confidence", "Medium Confidence", "Low Confidence"):
        subset = [m for m in matched if m["confidence_label"] == label]
        if subset:
            by_confidence[label] = {
                "total": len(subset),
                "correct": sum(1 for m in subset if m["correct"]),
                "accuracy_pct": round(sum(1 for m in subset if m["correct"]) / len(subset) * 100, 1),
            }

    clv_eligible = [m for m in matched if m["clv"] is not None]
    clv_stats = None
    if clv_eligible:
        clv_beats = sum(1 for m in clv_eligible if m["clv"]["beat_clv"])
        clv_stats = {
            "total": len(clv_eligible),
            "beat": clv_beats,
            "beat_pct": round(clv_beats / len(clv_eligible) * 100, 1),
            "avg_clv_pct": round(sum(m["clv"]["clv_pct"] for m in clv_eligible) / len(clv_eligible), 1),
        }

    return {
        "total": total,
        "correct": correct_count,
        "accuracy_pct": round(correct_count / total * 100, 1),
        "by_confidence": by_confidence,
        "clv_stats": clv_stats,
        "results": matched,
    }
