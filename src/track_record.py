"""
Model track record: logs the model's prediction for each tracked-card fight
on every run (keeping the latest prediction per fight, since the model can
shift as new data comes in before fight night -- same way a sportsbook line
moves right up until the bell), then compares against actual results once
they're recorded.

There's no live results API for this -- results have to be added manually
to data/fight_results.csv after a card happens (event_name, fighter_a,
fighter_b, winner, method). Until then, the track record section stays
honestly empty rather than faking a number.
"""

import csv
import os

PREDICTIONS_LOG_PATH = "data/predictions_log.csv"
FIELDNAMES = ["event_name", "fighter_a", "fighter_b", "favorite", "favorite_prob", "confidence_label", "likely_method", "last_updated"]


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
            existing[key] = {
                "event_name": fight["event_name"],
                "fighter_a": fight["fighter_a"],
                "fighter_b": fight["fighter_b"],
                "favorite": preview["favorite"],
                "favorite_prob": preview["favorite_prob"],
                "confidence_label": preview["confidence_label"],
                "likely_method": preview["likely_method"],
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
        matched.append({
            "event_name": result["event_name"],
            "fighter_a": result["fighter_a"],
            "fighter_b": result["fighter_b"],
            "predicted_favorite": pred["favorite"],
            "confidence_label": pred["confidence_label"],
            "actual_winner": result["winner"],
            "correct": correct,
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

    return {
        "total": total,
        "correct": correct_count,
        "accuracy_pct": round(correct_count / total * 100, 1),
        "by_confidence": by_confidence,
        "results": matched,
    }
