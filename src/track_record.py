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

Results are fetched automatically (see results_fetcher.py, ESPN-first);
this file only reads whatever's already in data/fight_results.csv, so if
that's empty the track record section stays honestly empty rather than
faking a number.
"""

import csv
import datetime as dt
import json
import os

from src.odds_utils import american_to_decimal
from src.card_matcher import _normalize_name

PREDICTIONS_LOG_PATH = "data/predictions_log.csv"
FIELDNAMES = [
    "event_name", "fighter_a", "fighter_b", "favorite", "favorite_prob",
    "confidence_label", "likely_method", "pick_odds", "closing_odds",
    "favorite_prob_history", "last_updated", "is_lock_of_week",
]
MOMENTUM_HISTORY_CAP = 10
MOMENTUM_THRESHOLD = 0.03  # 3 percentage points -- below this, treat as noise/stable
LOCK_OF_WEEK_MAX = 3  # cap, not a target -- a card with only one real standout gets one lock, not three padded-out picks


def _loose_name(name: str) -> tuple:
    parts = _normalize_name(name).split()
    return (parts[0], parts[-1]) if parts else (_normalize_name(name),)


def _favorite_moneyline_odds(fight: dict, favorite: str) -> float | None:
    """
    Confirmed real gap in production (July 2026): 11 of 12 fights on one
    card never got pick_odds/closing_odds logged despite Polymarket
    genuinely having moneyline markets for all of them (confirmed by the
    user, not assumed) -- exact-string matching here is a strong
    suspect, since the exact same class of mismatch (a live source's
    fighter-name spelling not exactly matching this project's own
    canonical name -- middle names, accents, hyphenation) has already
    been confirmed multiple times elsewhere in this codebase for
    different data sources. Tries progressively looser matching only
    as needed: exact string first (cheapest, zero false-positive risk),
    then accent/punctuation-normalized (reuses the same normalization
    already proven for Polymarket name variance in line_movement.py),
    then a narrow first+last-word match for a present/missing middle
    name specifically -- not full fuzzy matching, to avoid conflating
    two different real people who happen to share a first or last name.

    Logs the real, raw fighter names actually present in this fight's
    Moneyline edges when even the loose match fails, so if this
    hypothesis turns out wrong, the next run's logs show the real
    mismatch instead of this staying an unexplained silent gap again.
    """
    ml_edges = [e for e in fight.get("edges", []) if e.get("market") == "Moneyline"]

    for edge in ml_edges:
        if edge.get("fighter") == favorite:
            return edge.get("odds_american")

    norm_favorite = _normalize_name(favorite)
    for edge in ml_edges:
        if _normalize_name(str(edge.get("fighter", ""))) == norm_favorite:
            return edge.get("odds_american")

    loose_favorite = _loose_name(favorite)
    for edge in ml_edges:
        if _loose_name(str(edge.get("fighter", ""))) == loose_favorite:
            return edge.get("odds_american")

    if ml_edges:
        print(f"[track_record] no Moneyline odds match for favorite {favorite!r} in "
              f"{fight.get('fighter_a')!r} vs {fight.get('fighter_b')!r} even after "
              f"exact/normalized/loose matching -- raw fighter names on this fight's "
              f"Moneyline edges: {[e.get('fighter') for e in ml_edges]!r}")
    return None


def log_predictions(events: list[dict], generated_at: str, decided_keys: set | None = None) -> None:
    """
    Keeps the LATEST prediction per (event, fighter_a, fighter_b),
    overwriting older entries for the same fight -- EXCEPT for fights in
    decided_keys, which are skipped entirely once they have a confirmed
    result.

    This matters for genuine track-record integrity, not just tidiness:
    without this, a fight's logged "prediction" keeps getting silently
    overwritten by a fresh predict_matchup() call on every regeneration
    for as long as the card stays in "This Weekend" (through the day
    after the event) -- meaning ongoing model tuning could retroactively
    change what a fight's prediction "was," after the outcome is already
    known. That's not a real prediction anymore, it's hindsight wearing
    a prediction's clothes. Confirmed this was live: a real fight's
    logged pick changed after the card concluded, purely from routine
    site regenerations picking up unrelated model refinements made
    afterward.
    """
    decided_keys = decided_keys or set()
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
            fighter_key = frozenset({fight["fighter_a"].strip().lower(), fight["fighter_b"].strip().lower()})
            if fighter_key in decided_keys:
                continue  # locked in -- don't let post-result model changes rewrite history
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

            # Rolling favorite_prob history, for the momentum indicator --
            # if the model's favorite FLIPS between runs, start a fresh
            # history rather than comparing probabilities across two
            # different fighters, which wouldn't mean anything.
            prior_favorite = prior.get("favorite") if prior else None
            try:
                prior_history = json.loads(prior.get("favorite_prob_history") or "[]") if prior else []
            except (json.JSONDecodeError, TypeError):
                prior_history = []

            if prior_favorite != preview["favorite"]:
                new_history = [{"prob": preview["favorite_prob"], "date": generated_at}]
            else:
                new_history = (prior_history + [{"prob": preview["favorite_prob"], "date": generated_at}])[-MOMENTUM_HISTORY_CAP:]

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
                "favorite_prob_history": json.dumps(new_history),
                "last_updated": generated_at,
                "is_lock_of_week": (prior.get("is_lock_of_week", "") if prior else ""),
            }

    _assign_locks_of_week(existing, events, decided_keys)

    os.makedirs(os.path.dirname(PREDICTIONS_LOG_PATH), exist_ok=True)
    with open(PREDICTIONS_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in existing.values():
            writer.writerow(row)


def _assign_locks_of_week(existing: dict, events: list[dict], decided_keys: set) -> None:
    """
    Lock of the Week = the top (up to LOCK_OF_WEEK_MAX) High Confidence
    picks for a given event, ranked by exact probability -- not just
    tier membership, since "High Confidence" itself spans a wide 75-100%
    range and a 76% pick isn't really a "lock" next to a 94% one on the
    same card.

    Only recomputed for fights NOT in decided_keys, for the same reason
    predictions themselves get frozen once a result exists: without that
    guard, the lock designation could silently shift after the fact
    (e.g. a late model tweak nudges one pick's probability past another's)
    which would rewrite a claim that was supposed to be made in advance,
    not in hindsight. Decided fights simply keep whatever lock status
    they already had going into the card.
    """
    by_event: dict[str, list[str]] = {}
    for event in events:
        event_name = event["event_name"]
        for fight in event.get("fights", []):
            key = (event_name, fight["fighter_a"], fight["fighter_b"])
            fighter_key = frozenset({fight["fighter_a"].strip().lower(), fight["fighter_b"].strip().lower()})
            if fighter_key in decided_keys or key not in existing:
                continue
            by_event.setdefault(event_name, []).append(key)

    for event_name, keys in by_event.items():
        high_conf_keys = [k for k in keys if existing[k]["confidence_label"] == "High Confidence"]
        high_conf_keys.sort(key=lambda k: float(existing[k]["favorite_prob"]), reverse=True)
        lock_keys = set(high_conf_keys[:LOCK_OF_WEEK_MAX])
        for k in keys:
            existing[k]["is_lock_of_week"] = "true" if k in lock_keys else "false"


def compute_momentum(favorite_prob_history_json: str) -> dict | None:
    """
    Compares the oldest vs newest retained probability for the model's
    current favorite. Returns None if there's not enough history yet, or
    if the model's read has genuinely been stable (below the noise
    threshold) -- this should stay quiet most of the time, since the
    model's inputs don't change often; when it DOES show a real move,
    that's usually because something concrete changed (an injury/missed-
    weight update, a data correction), which is worth surfacing.
    """
    try:
        history = json.loads(favorite_prob_history_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return None
    if len(history) < 2:
        return None
    oldest, newest = history[0]["prob"], history[-1]["prob"]
    delta = newest - oldest
    if abs(delta) < MOMENTUM_THRESHOLD:
        return None
    return {"direction": "up" if delta > 0 else "down", "delta_pct": round(delta * 100, 1)}


def load_momentum_by_key() -> dict:
    """{(fighter_a, fighter_b): momentum_dict_or_None} for every logged fight."""
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return {}
    result = {}
    with open(PREDICTIONS_LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            key = _pair_key(row["fighter_a"], row["fighter_b"])
            result[key] = compute_momentum(row.get("favorite_prob_history", ""))
    return result


MIN_RESULTS_FOR_CALIBRATION = 8  # below this, buckets are too noisy to be meaningful
CALIBRATION_BINS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]


def _compute_calibration(matched: list[dict]) -> dict | None:
    """
    Buckets predictions by predicted probability and compares the average
    PREDICTED probability in each bucket against the ACTUAL fraction that
    came in correct -- a real calibration check, not just an accuracy
    number. A model that says "70% confident" should win about 70% of
    those picks over time; this is what actually tests that, rather than
    just reporting a single blended accuracy figure that could hide
    systematic over- or under-confidence.

    Returns a "not ready" marker below a minimum sample size -- a
    calibration curve from 3 results is noise dressed up as insight, not a
    real signal yet.
    """
    eligible = [m for m in matched if m.get("favorite_prob") is not None]
    if len(eligible) < MIN_RESULTS_FOR_CALIBRATION:
        return {"ready": False, "total": len(eligible), "needed": MIN_RESULTS_FOR_CALIBRATION}

    points = []
    for lo, hi in CALIBRATION_BINS:
        bucket = [m for m in eligible if lo <= m["favorite_prob"] < hi]
        if not bucket:
            continue
        predicted_avg = sum(m["favorite_prob"] for m in bucket) / len(bucket)
        actual_rate = sum(1 for m in bucket if m["correct"]) / len(bucket)
        points.append({
            "predicted": round(predicted_avg, 3),
            "actual": round(actual_rate, 3),
            "n": len(bucket),
        })

    total_n = sum(p["n"] for p in points)
    weighted_gap = sum((p["actual"] - p["predicted"]) * p["n"] for p in points) / total_n if total_n else 0
    if abs(weighted_gap) < 0.05:
        summary = "Across every confidence level, our picks won almost exactly as often as we said they would — the model isn't over- or under-selling itself."
    elif weighted_gap > 0:
        summary = f"On average, our picks have actually won about {round(weighted_gap*100)} points MORE often than the confidence we stated — if anything, we've been modest, not overselling."
    else:
        summary = f"On average, our picks have won about {round(abs(weighted_gap)*100)} points LESS often than the confidence we stated — a real sign of overconfidence worth watching."

    return {"ready": True, "total": len(eligible), "points": points, "summary": summary}


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


UNITS_BY_CONFIDENCE = {
    "High Confidence": 5.0,
    "Medium Confidence": 3.0,
    "Low Confidence": 1.0,
}


def _units_result(confidence_label, pick_odds, correct: bool) -> float | None:
    """
    Units won/lost on this pick, sized by confidence tier (5/3/1 for
    High/Medium/Low) and priced using the REAL market odds at pick time
    (pick_odds, from Polymarket) -- deliberately never the model's own
    probability, which would just be grading the model against itself.
    A win returns unit_size * (decimal_odds - 1) (profit only, stake not
    included); a loss is the full unit_size. Returns None when pick_odds
    isn't available -- excluded from the aggregate rather than guessed,
    same honesty standard as CLV and the market-baseline stat.
    """
    unit_size = UNITS_BY_CONFIDENCE.get(confidence_label)
    if unit_size is None:
        return None
    try:
        decimal_odds = american_to_decimal(float(pick_odds))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return round(unit_size * (decimal_odds - 1), 2) if correct else round(-unit_size, 2)


def _method_matches(predicted_method, actual_method) -> bool | None:
    """
    Compares the model's predicted method against the real outcome's
    method, normalized to a broad category -- the model predicts
    "Decision" without guessing unanimous/split/majority, while
    fight_results.csv logs the specific variant ("Decision - Unanimous"),
    so a straight string comparison would call every correct decision
    prediction a miss. Normalizes both sides to the same small set of
    buckets (KO/TKO, Submission, Decision, DQ) before comparing.
    Returns None if either side is missing/unparseable.
    """
    if not predicted_method or not actual_method:
        return None
    def _bucket(m: str) -> str:
        m = str(m).strip().upper()
        if m.startswith("DECISION") or m in ("DEC", "S-DEC", "U-DEC", "M-DEC"):
            return "DECISION"
        if "KO" in m or "TKO" in m:
            return "KO/TKO"
        if "SUB" in m:
            return "SUBMISSION"
        if "DQ" in m:
            return "DQ"
        return m
    return _bucket(predicted_method) == _bucket(actual_method)


def _favorite_won(pick_odds, correct: bool) -> bool | None:
    """
    Derives whether the MARKET's favorite won this fight, independent of
    whether the model's pick agreed with the market. Negative pick_odds
    means the model picked the market favorite; positive means it picked
    the underdog. Combined with whether that pick was correct, this
    covers all four cases without needing the other side's odds stored
    anywhere -- exactly one fighter is the favorite and exactly one wins,
    so the sign + correctness fully determines the answer:
      favorite picked & won      -> favorite won
      favorite picked & lost     -> favorite lost (underdog won)
      underdog picked & won      -> favorite lost (this pick WAS the upset)
      underdog picked & lost     -> favorite won
    Returns None when pick_odds is missing/zero/unparseable (can't tell).
    """
    try:
        odds = float(pick_odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    picked_favorite = odds < 0
    return correct if picked_favorite else not correct


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
        # Method correctness only means something when the winner pick was
        # ALSO right -- "predicted the wrong fighter, but nailed the
        # method" isn't a real signal worth scoring, so this is only
        # computed (non-None) for already-correct picks.
        method_correct = _method_matches(pred.get("likely_method"), result.get("method")) if correct else None
        units_result = _units_result(pred["confidence_label"], pred.get("pick_odds"), correct)
        matched.append({
            "event_name": result["event_name"],
            "fighter_a": result["fighter_a"],
            "fighter_b": result["fighter_b"],
            "predicted_favorite": pred["favorite"],
            "favorite_prob": float(pred["favorite_prob"]) if pred.get("favorite_prob") not in (None, "") else None,
            "confidence_label": pred["confidence_label"],
            "predicted_method": pred.get("likely_method"),
            "actual_method": result.get("method"),
            "method_correct": method_correct,
            "actual_winner": result["winner"],
            "correct": correct,
            "clv": clv,
            "favorite_won": _favorite_won(pred.get("pick_odds"), correct),
            "units_result": units_result,
            "unit_size": UNITS_BY_CONFIDENCE.get(pred["confidence_label"]),
            "is_lock_of_week": pred.get("is_lock_of_week") is True or str(pred.get("is_lock_of_week")).strip().lower() == "true",
            "date_added": result.get("date_added", ""),
            "card_position": result.get("card_position"),
        })

    # Most recent first -- what someone checking in on the site actually
    # cares about, not raw file-insertion order (which isn't guaranteed
    # to be chronological, especially once the automated fetcher and
    # manual entries are both writing to the same file). Unparseable/
    # missing dates sort last rather than crashing or landing at random.
    def _sort_key(m):
        try:
            return dt.datetime.strptime(m["date_added"], "%Y-%m-%d")
        except (ValueError, TypeError):
            return dt.datetime.min
    matched.sort(key=_sort_key, reverse=True)

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

    calibration = _compute_calibration(matched)

    accuracy_pct = round(correct_count / total * 100, 1)
    sparkline = _log_and_load_accuracy_sparkline(correct_count, total, accuracy_pct)

    results_by_event = _group_results_by_event(matched)

    # Lock of the Week: all-time record on the model's own top-conviction
    # picks specifically -- a genuinely different (and harder to hide
    # behind) claim than the blended accuracy number, since these are
    # picked out IN ADVANCE as the picks the model would most stand
    # behind, not selected with the benefit of hindsight.
    lock_picks = [m for m in matched if m.get("is_lock_of_week")]
    lock_record = None
    if lock_picks:
        lock_correct = sum(1 for m in lock_picks if m["correct"])
        lock_record = {
            "correct": lock_correct,
            "total": len(lock_picks),
            "accuracy_pct": round(lock_correct / len(lock_picks) * 100, 1),
        }

    # Units/ROI tracking: sized by confidence tier, priced with the real
    # market odds at pick time -- never the model's own probability,
    # which would just be grading the model against itself instead of
    # against what was actually available to bet. Only counts picks with
    # real odds on record, same partial-coverage honesty as CLV and the
    # market baseline above.
    units_eligible = [m for m in matched if m["units_result"] is not None]
    units_stats = None
    if units_eligible:
        total_units = round(sum(m["units_result"] for m in units_eligible), 2)
        total_staked = sum(UNITS_BY_CONFIDENCE.get(m["confidence_label"], 0) for m in units_eligible)
        by_tier = {}
        for tier in ("High Confidence", "Medium Confidence", "Low Confidence"):
            tier_picks = [m for m in units_eligible if m["confidence_label"] == tier]
            if tier_picks:
                by_tier[tier] = {
                    "units": round(sum(m["units_result"] for m in tier_picks), 2),
                    "count": len(tier_picks),
                    "unit_size": UNITS_BY_CONFIDENCE[tier],
                }
        # Running total needs chronological order (oldest first) for the
        # sparkline to read left-to-right correctly -- matched is sorted
        # most-recent-first for the list display, so reverse it here.
        # Starts at an explicit 0 baseline (the model's actual starting
        # point before any tracked results existed), not just the first
        # pick's own result -- otherwise the very first data point would
        # misleadingly look like where the series "started."
        running = [0.0]
        cumulative = 0.0
        for m in reversed(units_eligible):
            cumulative += m["units_result"]
            running.append(round(cumulative, 2))
        units_stats = {
            "total_units": total_units,
            "total_staked": total_staked,
            "roi_pct": round(total_units / total_staked * 100, 1) if total_staked else None,
            "eligible_count": len(units_eligible),
            "event_count": len({m["event_name"] for m in units_eligible}),
            "by_tier": by_tier,
            "running_total": running,
        }

    # Event Summary: an at-a-glance digest per event -- built once per
    # event group and reused both for the latest event (top-level,
    # unchanged shape for template compatibility) and for every past
    # event too, so "how did this card go" isn't something only the
    # most recent event gets to show.
    for group in results_by_event:
        group["summary"] = _build_event_summary(group)
    latest_event_summary = results_by_event[0]["summary"] if results_by_event else None

    # Model vs. market baseline: is the model's accuracy actually beating
    # the "just pick every favorite" strategy, or is it riding a card full
    # of obvious favorites winning? Only computed over the subset with
    # usable odds -- a partial-coverage stat honestly labeled beats a
    # complete-looking one that's silently wrong for missing rows.
    fav_known = [m for m in matched if m["favorite_won"] is not None]
    market_baseline = None
    if fav_known:
        fav_wins = sum(1 for m in fav_known if m["favorite_won"])
        market_baseline = {
            "total": len(fav_known),
            "favorite_win_pct": round(fav_wins / len(fav_known) * 100, 1),
            "model_accuracy_pct": round(sum(1 for m in fav_known if m["correct"]) / len(fav_known) * 100, 1),
        }

    return {
        "total": total,
        "correct": correct_count,
        "accuracy_pct": accuracy_pct,
        "by_confidence": by_confidence,
        "clv_stats": clv_stats,
        "calibration": calibration,
        "results": matched,
        "results_by_event": results_by_event,
        "market_baseline": market_baseline,
        "units_stats": units_stats,
        "latest_event_summary": latest_event_summary,
        "lock_record": lock_record,
        "accuracy_sparkline": sparkline,
    }


def _group_results_by_event(matched: list[dict]) -> list[dict]:
    """
    Groups already-sorted (most-recent-first) results under their event
    name. Groups explicitly by event_name rather than relying on the
    date sort putting same-event entries adjacent to each other --
    entries logged at different times of the same night could plausibly
    carry slightly different date_added values, which would silently
    break a groupby that assumes adjacency. Event groups themselves are
    ordered most-recent-first; fights WITHIN each group are ordered by
    real billing rank (Main Event first, working down to Early Prelims)
    rather than insertion order, which had no real meaning.
    """
    groups: dict[str, list[dict]] = {}
    for m in matched:
        groups.setdefault(m["event_name"], []).append(m)

    def _latest_date(entries: list[dict]) -> str:
        return max((e["date_added"] for e in entries), default="")

    billing_rank = {"Main Event": 0, "Co-Main Event": 1, "Main Card": 2, "Prelims": 3, "Early Prelims": 4}

    def _sort_within_event(entries: list[dict]) -> list[dict]:
        # Missing card_position (e.g. an older result logged before this
        # field existed) falls back to keeping its original relative
        # position rather than being scattered to an arbitrary spot --
        # stable sort with a rank that doesn't discriminate among unknowns.
        return sorted(entries, key=lambda e: billing_rank.get(e.get("card_position"), 99))

    ordered_event_names = sorted(groups.keys(), key=lambda name: _latest_date(groups[name]), reverse=True)
    return [{"event_name": name, "results": _sort_within_event(groups[name])} for name in ordered_event_names]


def _build_event_summary(group: dict) -> dict:
    """
    At-a-glance digest for one tracked event: record, accuracy, units,
    perfect-prop count, and a conditional brag headline. Built per-event
    (not just for the single most recent one) so every past event, once
    expanded, shows the same summary it would have shown when it WAS the
    latest event -- otherwise a real, earned "Perfect on every Medium &
    High Confidence pick" headline would silently vanish the moment a
    newer event took over as "latest," which is exactly what happened
    before this was generalized.
    """
    results = group["results"]
    correct = sum(1 for m in results if m["correct"])
    units_eligible = [m for m in results if m["units_result"] is not None]

    # The overall record includes Low Confidence picks, which are
    # near-coinflips by design and DILUTE how the model actually did
    # on the calls it was actually confident about -- surfaced
    # separately since "perfect on every real conviction pick" is a
    # genuinely different (and more meaningful) claim than the blended
    # record, not just a more flattering way to say the same thing.
    high_medium = [m for m in results if m["confidence_label"] in ("High Confidence", "Medium Confidence")]
    high_medium_correct = sum(1 for m in high_medium if m["correct"])

    # Tiered and conditional on purpose -- a headline this site can't
    # back up with the actual numbers is worse than no headline at
    # all, so this only fires when the data genuinely earns it, and
    # says less (or nothing) when it doesn't.
    brag_headline = None
    if len(high_medium) >= 2 and high_medium_correct == len(high_medium):
        brag_headline = {"text": f"Perfect on every Medium & High Confidence pick ({high_medium_correct}/{len(high_medium)})", "tier": "gold"}
    elif results and correct / len(results) >= 0.75:
        brag_headline = {"text": f"Strong card — {correct}/{len(results)} correct", "tier": "green"}

    return {
        "event_name": group["event_name"],
        "correct": correct,
        "incorrect": len(results) - correct,
        "total": len(results),
        "accuracy_pct": round(correct / len(results) * 100, 1) if results else 0,
        "perfect_prop_count": sum(1 for m in results if m["correct"] and m["method_correct"]),
        "units": round(sum(m["units_result"] for m in units_eligible), 2) if units_eligible else None,
        "units_eligible": len(units_eligible),
        "high_medium_correct": high_medium_correct,
        "high_medium_total": len(high_medium),
        "brag_headline": brag_headline,
        "lock_picks": [m for m in results if m.get("is_lock_of_week")],
    }


ACCURACY_HISTORY_PATH = "data/accuracy_history.csv"


def _log_and_load_accuracy_sparkline(correct: int, total: int, accuracy_pct: float) -> list[float] | None:
    """
    Appends today's accuracy snapshot to a small running history file, then
    returns the accuracy_pct series for a sparkline -- genuinely forward-
    tracking only, same honesty standard as the rest of Track Record. A
    snapshot is only appended if it differs from the last logged one, so
    routine reruns with no new results don't pad the file with duplicate
    points. Returns None until there are at least 2 distinct points, since
    a single dot isn't a trend.
    """
    today = dt.date.today().isoformat()
    rows = []
    if os.path.exists(ACCURACY_HISTORY_PATH):
        with open(ACCURACY_HISTORY_PATH, newline="") as f:
            rows = list(csv.DictReader(f))

    if not rows or int(rows[-1]["correct"]) != correct or int(rows[-1]["total"]) != total:
        rows.append({"date": today, "correct": str(correct), "total": str(total), "accuracy_pct": str(accuracy_pct)})
        with open(ACCURACY_HISTORY_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "correct", "total", "accuracy_pct"])
            writer.writeheader()
            writer.writerows(rows)

    if len(rows) < 2:
        return None
    return [float(r["accuracy_pct"]) for r in rows]
