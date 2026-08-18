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
import math
import os

from src.odds_utils import american_to_decimal, american_to_implied_prob, remove_vig_two_way
from src.card_matcher import _normalize_name
from src.rationale import explain_settled as _explain_settled, falsifier_fired as _falsifier_fired

PREDICTIONS_LOG_PATH = "data/predictions_log.csv"
FIELDNAMES = [
    "event_name", "fighter_a", "fighter_b", "favorite", "favorite_prob",
    "confidence_label", "likely_method", "pick_odds", "closing_odds", "opponent_odds",
    # THE DE-VIGGED PROBABILITY AT EACH OF THOSE TWO MOMENTS, and the pair CLV
    # is actually graded on. See _clv_result for why the American prices above
    # cannot do that job any more.
    "pick_fair_prob", "closing_fair_prob",
    "favorite_prob_history", "last_updated", "is_lock_of_week", "voided",
    # THE RISK THE BLURB NAMED, captured pre-fight. Every pick's commentary
    # ends by naming the strongest thing arguing against it; after the fight
    # the honest question is whether that is what beat it. It cannot be
    # recovered later -- regenerating the blurb would run it against ratings
    # that already absorbed the result, the exact contamination the frozen
    # pick restore above exists to prevent. Blank on every row logged before
    # this column existed, which is correct rather than backfillable.
    "pick_falsifier",
]
# A two-way price this lopsided is a market in the act of resolving, not a
# line anyone could have bet. 0.97 is deliberately looser than the 0.9995
# these actually print, so a genuine -3000 blowout favourite still counts
# while nothing that resolved can sneak through.
_SETTLED_PROB = 0.97


def _is_settled_price(american) -> bool:
    """True when an American price implies a probability past _SETTLED_PROB."""
    try:
        p = american_to_implied_prob(float(american))
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return p is None or p >= _SETTLED_PROB or p <= (1.0 - _SETTLED_PROB)


# Below this many settled picks a band record is a story about a handful of
# fights. Ten is the same floor _compute_calibration uses before it will draw
# a bin, kept identical so the two cannot disagree about what counts.
MIN_BAND_RECORD = 10


def _settled_note(won, method, falsifier_fired, band_record):
    """explain_settled with no original text -- just the appended line."""
    return _explain_settled("", won, method, falsifier_fired, band_record).strip()


MOMENTUM_HISTORY_CAP = 10
MOMENTUM_THRESHOLD = 0.03  # 3 percentage points -- below this, treat as noise/stable
LOCK_OF_WEEK_MAX = 3  # cap, not a target -- a card with only one real standout gets one lock, not three padded-out picks
# Absolute floor a pick must clear to be called a lock, on top of the cap.
# Without it the label was purely RELATIVE to whatever else was on the card,
# which produced this in real logged data: Alden Coria at 76.9% was a lock
# (his card had only two High Confidence picks) while Magomed Tuchalov at
# 86.3% was NOT (his card had five, and he placed fourth). A pick 9.4 points
# worse wore the label purely because of which card it landed on. A floor
# makes "lock" mean something absolute; the cap then keeps a stacked card
# from crowning six of them.
# 0.82 chosen from the user's real predictions_log across 5 cards: it keeps
# 7 locks, drops the two weakest (76.9%, 79.1%), and leaves 2 of 5 cards
# with none -- which is the point, not a cost. Provisional on a small
# sample (13 High Confidence picks); revisit once more locks have resolved.
LOCK_OF_WEEK_MIN_PROB = 0.82


def _loose_name(name: str) -> tuple:
    parts = _normalize_name(name).split()
    return (parts[0], parts[-1]) if parts else (_normalize_name(name),)


def _favorite_moneyline_edge(fight: dict, favorite: str) -> dict | None:
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
            return edge

    norm_favorite = _normalize_name(favorite)
    for edge in ml_edges:
        if _normalize_name(str(edge.get("fighter", ""))) == norm_favorite:
            return edge

    loose_favorite = _loose_name(favorite)
    for edge in ml_edges:
        if _loose_name(str(edge.get("fighter", ""))) == loose_favorite:
            return edge

    if ml_edges:
        print(f"[track_record] no Moneyline odds match for favorite {favorite!r} in "
              f"{fight.get('fighter_a')!r} vs {fight.get('fighter_b')!r} even after "
              f"exact/normalized/loose matching -- raw fighter names on this fight's "
              f"Moneyline edges: {[e.get('fighter') for e in ml_edges]!r}")
    return None


def _favorite_moneyline_odds(fight: dict, favorite: str) -> float | None:
    """The matched edge's American price, or None."""
    edge = _favorite_moneyline_edge(fight, favorite)
    return edge.get("odds_american") if edge else None


def _fair_prob_of(edge: dict | None) -> float | None:
    """
    The de-vigged probability behind a moneyline edge.

    This is the quantity CLV is graded on, because unlike the American price
    it does not change when the SOURCE changes -- see _clv_result.
    """
    if not edge:
        return None
    p = edge.get("book_fair_prob")
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    return p if 0.0 < p < 1.0 else None


# THE MOMENT REAL SPORTSBOOK PRICES FIRST REACHED THIS PIPELINE.
#
# Before this, TheRundown was configured but never actually invoked -- the
# date list it needed was built from a field Polymarket does not have -- so
# every price ever written to predictions_log came from Polymarket, which is
# peer-to-peer and carries no margin. For those rows the quoted price IS the
# fair line, which is what makes the migration below sound rather than a
# guess. After this instant a price may carry vig, and de-vigging it by
# assumption would be exactly the fabrication this all exists to avoid.
_BOOK_PRICES_LIVE_FROM = "2026-08-18T19:39:17+00:00"


def _backfill_legacy_fair_probs(row: dict) -> dict:
    """
    Fill pick_fair_prob / closing_fair_prob on rows written before those
    columns existed. Runs on every load, and is a no-op once a row has them.

    WHY THIS LIVES IN THE PIPELINE RATHER THAN A ONE-OFF SCRIPT. It WAS a
    one-off script, and the data commit it produced was silently dropped by a
    rebase against a concurrent CI auto-refresh -- so the 77 recovered rows
    vanished and the next build rewrote the file without the columns at all.
    A migration that has to be remembered is a migration that gets clobbered;
    this one repairs itself on every run no matter what else touches the file.

    Only rows predating _BOOK_PRICES_LIVE_FROM are touched, because only for
    those is the stored price known to be vig-free. A later row missing its
    fair value is left blank and simply goes ungraded, which is the honest
    outcome -- CLV skips what it cannot compare.
    """
    if not row:
        return row
    last = str(row.get("last_updated") or "")
    for src, dst in (("pick_odds", "pick_fair_prob"),
                     ("closing_odds", "closing_fair_prob")):
        if str(row.get(dst) or "").strip():
            continue
        # An empty last_updated is treated as legacy: nothing written since
        # the columns landed can lack it.
        if last and _iso_after(last, _BOOK_PRICES_LIVE_FROM):
            continue
        price = row.get(src)
        if price in (None, ""):
            continue
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if _is_settled_price(price):
            continue        # the result wearing a closing line -- see _clv_result
        p = american_to_implied_prob(price)
        if 0.0 < p < 1.0:
            row[dst] = round(p, 4)
    return row


def _parse_log_stamp(value: str):
    """
    last_updated as a tz-aware datetime, or None.

    THE LOG DOES NOT STORE ISO. generate_site writes "%Y-%m-%d %I:%M %p ET"
    in America/New_York -- "2026-08-18 02:19 PM ET" -- which fromisoformat
    cannot read. Parsing only ISO meant every row on file failed to parse,
    and since an unparseable stamp is treated as legacy, EVERY future row
    would have qualified for a migration meant only for the historical ones.
    """
    from datetime import datetime, timezone
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        pass
    if text.endswith(" ET"):
        try:
            from zoneinfo import ZoneInfo
            naive = datetime.strptime(text[:-3].strip(), "%Y-%m-%d %I:%M %p")
            return naive.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _iso_after(a: str, b: str) -> bool:
    """True when timestamp a is strictly later than b."""
    pa, pb = _parse_log_stamp(a), _parse_log_stamp(b)
    if pa is None or pb is None:
        # An unparseable stamp is treated as legacy, matching a blank one --
        # the conservative direction, since these rows predate the columns.
        return False
    return pa > pb


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
                existing[key] = _backfill_legacy_fair_probs(row)

    for event in events:
        for fight in event.get("fights", []):
            preview = fight.get("preview")
            if not preview:
                continue
            key = (fight["event_name"], fight["fighter_a"], fight["fighter_b"])
            fighter_key = frozenset({fight["fighter_a"].strip().lower(), fight["fighter_b"].strip().lower()})
            if fighter_key in decided_keys:
                continue  # locked in -- don't let post-result model changes rewrite history
            _fav_edge = _favorite_moneyline_edge(fight, preview["favorite"])
            current_odds = _fav_edge.get("odds_american") if _fav_edge else None
            current_fair = _fair_prob_of(_fav_edge)
            opponent_name = fight["fighter_b"] if preview["favorite"] == fight["fighter_a"] else fight["fighter_a"]
            current_opponent_odds = _favorite_moneyline_odds(fight, opponent_name)
            prior = existing.get(key)

            # pick_odds is set ONCE -- the first time we see a live price for
            # this fight's favorite -- and never overwritten after that, so
            # it genuinely represents "the price when the model first had a
            # read on this fight," not a moving target.
            pick_odds = prior.get("pick_odds") if prior and prior.get("pick_odds") not in (None, "") else None
            capturing_pick_odds_now = pick_odds is None and current_odds is not None
            if capturing_pick_odds_now:
                pick_odds = current_odds

            # Set once, on the same run and from the same edge as pick_odds,
            # so the pair genuinely describes one moment.
            pick_fair_prob = prior.get("pick_fair_prob") if prior and prior.get("pick_fair_prob") not in (None, "") else None
            if pick_fair_prob is None and capturing_pick_odds_now:
                pick_fair_prob = current_fair

            # opponent_odds is captured ON THE RUN PICK_ODDS IS FIRST CAPTURED,
            # or not at all -- which is what the "same moment" claim above
            # actually requires and what two independent set-once blocks did
            # not deliver. They were independent: each side filled in on the
            # first run its OWN price resolved, and _favorite_moneyline_odds
            # needs three fallback tiers precisely because name-matching against
            # Polymarket fails often enough that one side resolving days after
            # the other is the expected case, not an edge case. The result is in
            # the shipped log: Du Plessis/Usman holds -195 against -240, an
            # implied sum of 1.367, which no simultaneous two-way market can
            # produce (the other 90 pairs sit at a median of 1.019 and top out
            # at 1.103). That row then reads as an underdog pick and drags the
            # published underdog-vs-market baseline.
            #
            # Deliberately does NOT gate pick_odds on the opponent's price
            # being available: pick_odds alone still drives units, CLV and
            # favorite_won, and _is_underdog documents its own fallback for a
            # row where only pick_odds was ever captured. So the cost of an
            # unresolvable opponent price is now a missing second price rather
            # than a fabricated pairing.
            opponent_odds = prior.get("opponent_odds") if prior and prior.get("opponent_odds") not in (None, "") else None
            if capturing_pick_odds_now and opponent_odds is None:
                opponent_odds = current_opponent_odds

            # closing_odds updates every run a live price is available,
            # so whatever it holds when the fight actually happens is
            # naturally the last real price seen -- the closing line.
            #
            # EXCEPT WHEN THE MARKET HAS ALREADY SETTLED. A Polymarket
            # contract does not delist the moment the horn sounds; it prints
            # 0.9995 / 0.0005 while it resolves. Those ticks arrive before
            # this fight lands in decided_keys, so they were being written
            # straight into closing_odds -- 14 of 77 graded rows hold a
            # |price| >= 5000.
            #
            # That is not a closing line, it is the RESULT wearing one. And
            # CLV is the single number on this site claiming to show edge
            # INDEPENDENTLY of results, so letting the outcome in through
            # this door makes it circular: a won pick is guaranteed to
            # "beat the close" at 0.9995. Nearly a quarter of the sample
            # was grading the model against its own scoreboard.
            #
            # Anything past _SETTLED_PROB is refused and the last genuine
            # pre-settlement price is kept instead.
            closing_odds = prior.get("closing_odds") if prior else None
            closing_fair_prob = prior.get("closing_fair_prob") if prior else None
            if current_odds is not None and not _is_settled_price(current_odds):
                closing_odds = current_odds
                # Advances together with closing_odds and under the same
                # settled-price guard, so the fair pair can never be one run
                # out of step with the price pair.
                if current_fair is not None:
                    closing_fair_prob = current_fair

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
                "opponent_odds": opponent_odds if opponent_odds is not None else "",
                "pick_fair_prob": pick_fair_prob if pick_fair_prob is not None else "",
                "closing_fair_prob": closing_fair_prob if closing_fair_prob is not None else "",
                "favorite_prob_history": json.dumps(new_history),
                "last_updated": generated_at,
                "is_lock_of_week": (prior.get("is_lock_of_week", "") if prior else ""),
                "voided": (prior.get("voided", "") if prior else ""),
                # SET ONCE, like pick_odds. The named risk belongs to the call
                # as it was made; a later build re-deriving it from a nudged
                # model would quietly rewrite what we said we were worried
                # about.
                "pick_falsifier": ((prior.get("pick_falsifier") if prior and prior.get("pick_falsifier") else None)
                                   or preview.get("pick_falsifier") or ""),
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
        high_conf_keys = [
            k for k in keys
            if existing[k]["confidence_label"] == "High Confidence"
            and float(existing[k]["favorite_prob"]) >= LOCK_OF_WEEK_MIN_PROB
        ]
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


def load_logged_predictions_by_key() -> dict:
    """
    {(fighter_a, fighter_b): {favorite, favorite_prob, confidence_label,
    likely_method}} -- the prediction AS IT WAS LOGGED, for restoring onto a
    fight whose result is already known.

    WHY THIS EXISTS. The fight card rebuilds its preview from scratch on
    every run, and once a fight is decided the inputs to that preview have
    already been contaminated by its own outcome: fighter_backfill rewrites
    both fighters' W/L records from ESPN, and merge_results_into_history
    feeds the bout into the ratings. Re-running the model against that data
    is not a prediction, it is a lookup -- and it reliably "picks" whoever
    actually won.

    Observed live during UFC 330: the card had Mansur Abdul-Malik at 51%,
    he was submitted, and within two builds the same card was displaying
    Dustin Stoltzfus as the pick at 67%. The track record stayed correct
    throughout, because it reads this log -- so the card was contradicting
    the accuracy figure printed elsewhere on the same page, always in the
    model's favour.

    This log is written before a fight resolves and frozen afterwards (see
    decided_keys in log_predictions), which makes it the only honest source
    for what was actually called.
    """
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return {}
    out = {}
    with open(PREDICTIONS_LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("favorite"):
                continue
            try:
                prob = float(row.get("favorite_prob") or 0) or None
            except (TypeError, ValueError):
                prob = None
            out[_pair_key(row["fighter_a"], row["fighter_b"])] = {
                "favorite": row["favorite"],
                "favorite_prob": prob,
                "confidence_label": row.get("confidence_label") or None,
                "pick_falsifier": row.get("pick_falsifier") or None,
                "likely_method": row.get("likely_method") or None,
            }
    return out


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
        # Operator-facing drift warning (log only, not site copy -- the
        # user-facing summary below already covers the aggregate story).
        # Fires per-bin when the actual win rate lands more than 15
        # points from the average stated confidence, with n>=10 so a
        # couple of unlucky results in a thin bucket doesn't cry wolf.
        # 15pp at n=10 is still within plausible binomial noise, so this
        # is a "look at this" nudge, not a statistical verdict -- but
        # it's exactly the early-drift signal that previously had no way
        # to surface anywhere except manually eyeballing the chart.
        if len(bucket) >= 10 and abs(actual_rate - predicted_avg) > 0.15:
            direction = "OVERconfident" if actual_rate < predicted_avg else "UNDERconfident"
            print(f"[track_record] CALIBRATION DRIFT: {lo:.0%}-{min(hi,1.0):.0%} bin is {direction} "
                  f"-- predicted avg {predicted_avg:.1%} but actual win rate {actual_rate:.1%} "
                  f"over n={len(bucket)} picks. Worth a look if this persists across refreshes.")

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


def _clv_result(pick_odds, closing_odds,
                pick_fair_prob=None, closing_fair_prob=None) -> dict | None:
    """
    Beating the closing line means the market moved TOWARD the model's pick by
    fight night -- the model saw something the market had not fully priced.
    Independent of whether the bet won: a fighter can lose straight-up while
    the model still correctly anticipated real market movement, which is the
    whole point of CLV as a model-quality metric distinct from raw record.

    GRADED ON THE DE-VIGGED PROBABILITY, NOT THE AMERICAN PRICE, and that is a
    correctness requirement rather than a refinement.

    `odds_american` is no longer one feed's number. It is the best BETTABLE
    price where a sportsbook quoted and the vig-free fair line where none did,
    and books post lines progressively through the week -- so a pick captured
    before its book arrived and closed after it flips basis mid-flight. The
    price then appears to shorten by exactly the vig. Measured: a bet whose
    price never moved at all scored beat_clv=True with +1.7 to +2.4% CLV on
    every one of three test prices, always in the flattering direction. That
    would have inflated the one number on this site claiming to show edge
    INDEPENDENTLY of results.

    De-vigging removes precisely the quantity that changes when the source
    does, so fair-at-pick against fair-at-close compares like with like no
    matter which feed quoted at either end. It is also the better statistical
    object: CLV should measure the market's view of the true probability
    moving, not the bookmaker's margin changing.

    Falls back to the American pair for rows logged before those columns
    existed, and says which basis it used so a mixed sample can be split.
    """
    def _f(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if 0.0 < v < 1.0 else None

    pf, cf = _f(pick_fair_prob), _f(closing_fair_prob)
    if pf is not None and cf is not None:
        pick_prob, closing_prob, basis = pf, cf, "fair"
    else:
        # LEGACY ROWS ONLY. Same guards as before: no price, no grade; and a
        # settled market at 0.9995 is the RESULT wearing a closing line, which
        # would score beat_clv=True on every winning pick by construction.
        if not pick_odds or not closing_odds:
            return None
        if _is_settled_price(closing_odds):
            return None
        try:
            pick_prob = 1 / american_to_decimal(float(pick_odds))
            closing_prob = 1 / american_to_decimal(float(closing_odds))
        except (ValueError, ZeroDivisionError):
            return None
        # NaN SURVIVES EVERY GUARD ABOVE. `not float("nan")` is False, so a
        # missing price passes the emptiness check; NaN compares False against
        # the settled threshold; and the arithmetic raises nothing, it just
        # propagates. The result was 24 rows scoring beat_clv=False with a NaN
        # clv_pct, which then poisoned the mean of the whole cohort. Same
        # IEEE 754 trap the single-sided de-vig documents.
        if not (math.isfinite(pick_prob) and math.isfinite(closing_prob)):
            return None
        basis = "price"

    out = {
        "pick_prob": round(pick_prob, 4), "closing_prob": round(closing_prob, 4),
        "beat_clv": closing_prob > pick_prob,
        "clv_pct": round((closing_prob - pick_prob) * 100, 1),
        # Which pair this was graded on. A sample mixing the two is not one
        # sample, and without this nobody downstream can tell.
        "basis": basis,
    }
    try:
        out["pick_odds"] = float(pick_odds)
        out["closing_odds"] = float(closing_odds)
    except (TypeError, ValueError):
        pass
    return out


UNITS_BY_CONFIDENCE = {
    "High Confidence": 5.0,
    "Medium Confidence": 3.0,
    "Low Confidence": 1.0,
}
# A Lock of the Week is a stronger conviction call than a regular High
# Confidence pick (it's the single best pick on the card, not just one
# of however many clear favorites) -- staked heavier to reflect that,
# and routed to its OWN tier below rather than double-counted inside
# High Confidence too.
LOCK_OF_WEEK_UNITS = 10.0


def _units_result(unit_size: float | None, pick_odds, correct: bool) -> float | None:
    """
    Units won/lost on this pick, sized by the resolved unit_size the
    caller passes in (5/3/1 for High/Medium/Low Confidence, or
    LOCK_OF_WEEK_UNITS for a Lock of the Week -- resolving which one
    happens at the call site, since that's where is_lock_of_week is
    known) and priced using the REAL market odds at pick time
    (pick_odds, from Polymarket) -- deliberately never the model's own
    probability, which would just be grading the model against itself.
    A win returns unit_size * (decimal_odds - 1) (profit only, stake not
    included); a loss is the full unit_size. Returns None when pick_odds
    isn't available -- excluded from the aggregate rather than guessed,
    same honesty standard as CLV and the market-baseline stat.
    """
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


# A real two-way moneyline's two implied probabilities sum to a little over
# 1.00 -- that sum IS the book's margin. Across the 90 coherent pairs already
# in predictions_log.csv the median is 1.019 and the maximum 1.103. So a pair
# summing far outside that band is not a wide market, it is two prices that
# were never quoted at the same time: the log predates the same-run capture
# above and holds at least one such row (-195 against -240, sum 1.367).
#
# Bounds set well clear of anything real rather than snug against the observed
# range, so this only ever refuses the impossible. An incoherent pair is
# reported as "opponent price unknown", NOT as a reason to discard the row:
# _is_underdog then falls back to the sign check it already documents, and
# _market_expectation skips the row the same way it skips a missing price.
# Same principle as _clv_result refusing a settled closing price -- decline to
# use a number that cannot mean what it claims, rather than rewrite history.
_COHERENT_PAIR_MIN, _COHERENT_PAIR_MAX = 0.90, 1.25


def _coherent_opponent_odds(pick_odds, opponent_odds):
    """opponent_odds if it can be a simultaneous quote against pick_odds, else None."""
    if pick_odds is None or opponent_odds is None:
        return None
    try:
        total = american_to_implied_prob(float(pick_odds)) + american_to_implied_prob(float(opponent_odds))
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return opponent_odds if _COHERENT_PAIR_MIN <= total <= _COHERENT_PAIR_MAX else None


def _is_underdog(pick_odds, opponent_odds) -> bool | None:
    """
    Relative definition, per explicit user instruction: a pick is the
    underdog if their own price implies a LOWER win probability than
    their opponent's price does -- not simply "positive American odds."
    This distinction is real, not theoretical: confirmed directly in
    this project's own data that King Green was priced -105 against
    Terrance McKinney's -115 -- both negative, but Green was still the
    relative underdog, priced worse than his opponent even though
    neither number was positive. A simple ">0" check would have missed
    this entirely.

    Falls back to the simple sign check when opponent_odds genuinely
    isn't known (e.g. an older tracked fight where only pick_odds was
    ever captured) -- a reasonable approximation, since exactly one side
    of a real moneyline is negative in the large majority of cases; it
    just can't catch the rarer both-negative case without knowing the
    other side.

    Returns None when pick_odds itself is missing -- "unknown" and
    "confirmed not an underdog" are different claims, and a caller
    filtering for correct underdog picks should treat them differently
    (exclude, not count as False).
    """
    if pick_odds is None:
        return None
    # An incoherent pair is treated exactly like a missing one -- see
    # _coherent_opponent_odds. The relative comparison below is only better
    # than the sign check when both prices come from the same moment.
    opponent_odds = _coherent_opponent_odds(pick_odds, opponent_odds)
    if opponent_odds is None:
        return pick_odds > 0
    try:
        pick_prob = 1 / american_to_decimal(float(pick_odds))
        opp_prob = 1 / american_to_decimal(float(opponent_odds))
    except (ValueError, ZeroDivisionError, TypeError):
        return pick_odds > 0
    return pick_prob < opp_prob


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

    # VOIDED predictions (cancelled fights, marked via
    # scripts/mark_fight_cancelled.py) are removed before matching ever
    # happens: a void counts as if the prediction was never made -- it
    # must not touch accuracy, confidence tiers, locks, CLV, or units in
    # EITHER direction. Filtering here (rather than relying on "no result
    # ever appears for a cancelled fight") also protects against a result
    # for the same pairing surfacing later -- e.g. the fight getting
    # rescheduled to a future card, where the OLD pick shouldn't silently
    # count against the NEW fight.
    predictions = [p for p in predictions if str(p.get("voided", "")).strip().lower() != "true"]

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
        clv = _clv_result(pred.get("pick_odds"), pred.get("closing_odds"),
                          pred.get("pick_fair_prob"), pred.get("closing_fair_prob"))
        # Method correctness only means something when the winner pick was
        # ALSO right -- "predicted the wrong fighter, but nailed the
        # method" isn't a real signal worth scoring, so this is only
        # computed (non-None) for already-correct picks.
        method_correct = _method_matches(pred.get("likely_method"), result.get("method")) if correct else None
        is_lock = pred.get("is_lock_of_week") is True or str(pred.get("is_lock_of_week")).strip().lower() == "true"
        resolved_unit_size = LOCK_OF_WEEK_UNITS if is_lock else UNITS_BY_CONFIDENCE.get(pred["confidence_label"])
        units_result = _units_result(resolved_unit_size, pred.get("pick_odds"), correct)
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
            "pick_falsifier": pred.get("pick_falsifier") or None,
            "pick_odds": float(pred["pick_odds"]) if pred.get("pick_odds") not in (None, "") else None,
            "opponent_odds": float(pred["opponent_odds"]) if pred.get("opponent_odds") not in (None, "") else None,
            "units_result": units_result,
            "unit_size": resolved_unit_size,
            "is_lock_of_week": is_lock,
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
    # BAND RECORD, for the morning-after line. Grouped on the same displayed
    # integer percent the closer bands on, so a pick's blurb and its settled
    # note cannot disagree about which band it is in.
    _bands = {}
    for m in matched:
        fp = m.get("favorite_prob")
        if fp is None:
            continue
        pct = round(float(fp) * 100)
        b = ("coinflip" if pct < 55 else "lean" if pct < 65 else
             "solid" if pct < 75 else "strong")
        w, n = _bands.get(b, (0, 0))
        _bands[b] = (w + (1 if m["correct"] else 0), n + 1)

    # ONE FLAT LINE PER SETTLED PICK. See rationale.explain_settled for why the
    # tone is identical whether it won or lost, and why the band record
    # publishes whatever it is.
    #
    # falsifier_fired is None throughout: the counterargument each blurb names
    # is not stored on the prediction row, so whether it is what happened
    # cannot be known after the fact without regenerating the blurb from
    # post-fight ratings -- which is precisely the contamination
    # generate_site.py's frozen-pick restore exists to prevent. It needs a
    # column, captured pre-fight. Until then the note carries the result and
    # the band and says nothing it cannot support.
    for m in matched:
        fp = m.get("favorite_prob")
        if fp is None:
            m["settled_note"] = None
            continue
        pct = round(float(fp) * 100)
        b = ("coinflip" if pct < 55 else "lean" if pct < 65 else
             "solid" if pct < 75 else "strong")
        # ONLY THE PART THE ROW DOES NOT ALREADY SHOW. explain_settled's full
        # line opens with "Won by unanimous decision", and this row prints the
        # winner and the method two lines above -- restating them is the
        # number-stuffing the review warned about. What is genuinely new is
        # how the model's picks AT THIS CONFIDENCE have actually done, which
        # appears nowhere else on the page.
        rec = _bands.get(b)
        if rec and rec[1] >= MIN_BAND_RECORD:
            w, n = rec
            m["band_note"] = f"Picks at this confidence are {w}-{n - w} this season"
        else:
            m["band_note"] = None

        # WHETHER THE NAMED RISK IS WHAT HAPPENED. None for every pick logged
        # before pick_falsifier existed, for wins, and for risks a result
        # cannot adjudicate -- see rationale.falsifier_fired. None prints
        # nothing, which is the point: this is only worth saying when it is
        # actually known.
        fired = _falsifier_fired(m.get("pick_falsifier"), bool(m["correct"]), m.get("actual_method"))
        m["falsifier_note"] = ("The risk we flagged is what happened" if fired is True
                               else "Not for the reason we flagged" if fired is False
                               else None)

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
        # THE SAMPLE IS MIXED, and pooling it silently would hide that.
        # Rows logged before the fair-probability columns existed are graded
        # on the American price pair, which is the basis that inflated CLV
        # whenever a pick changed source mid-week. Those rows are still
        # counted -- throwing away months of history would be worse -- but the
        # split is published so the headline can be read with the right amount
        # of trust, and so the price-basis share visibly shrinks as fair-basis
        # rows accumulate.
        by_basis = {}
        for m in clv_eligible:
            b = m["clv"].get("basis", "price")
            by_basis[b] = by_basis.get(b, 0) + 1
        fair_only = [m for m in clv_eligible if m["clv"].get("basis") == "fair"]
        # CARDS, NOT PICKS, is the honest denominator for how much this is
        # worth reading. Thirteen picks from one night share a market regime,
        # a slate of opponents and one set of late-money conditions; they are
        # nothing like thirteen independent observations. The published
        # percentage is over picks, but the card count sits beside it so the
        # sample cannot look larger than it is.
        clv_cards = len({m.get("event_name") for m in clv_eligible if m.get("event_name")})
        clv_stats = {
            "cards": clv_cards,
            "total": len(clv_eligible),
            "beat": clv_beats,
            "beat_pct": round(clv_beats / len(clv_eligible) * 100, 1),
            "avg_clv_pct": round(sum(m["clv"]["clv_pct"] for m in clv_eligible) / len(clv_eligible), 1),
            "by_basis": by_basis,
            # The uncontaminated subset, reported separately because it is the
            # only part of this that is trustworthy without qualification.
            "fair_total": len(fair_only),
            "fair_beat_pct": (round(sum(1 for m in fair_only if m["clv"]["beat_clv"])
                                    / len(fair_only) * 100, 1) if fair_only else None),
            "fair_avg_clv_pct": (round(sum(m["clv"]["clv_pct"] for m in fair_only)
                                       / len(fair_only), 1) if fair_only else None),
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
        # "Perfect calls": both the winner AND the method called correctly
        # in advance -- the strongest claim a pick can make, counted here
        # so the template can headline it without recomputing.
        perfect_calls = sum(1 for m in lock_picks if m["correct"] and m.get("method_correct") is True)
        lock_record = {
            "correct": lock_correct,
            "total": len(lock_picks),
            "accuracy_pct": round(lock_correct / len(lock_picks) * 100, 1),
            "perfect_calls": perfect_calls,
            # Full history for the scrollable Track Record card -- matched
            # is already sorted most-recent-first, so this inherits that.
            "picks": lock_picks,
        }

    # Correct underdog calls: arguably a more impressive claim than raw
    # accuracy, since it's specifically "the model saw something the
    # market itself didn't." Sorting by raw pick_odds descending puts
    # the biggest underdog first -- American odds order continuously
    # from biggest favorite (most negative) to biggest underdog (most
    # positive) straight through the -100/+100 boundary, so this holds
    # correctly for both a genuine positive-odds underdog and the
    # both-sides-negative case (e.g. -105 vs. -115) alike.
    underdog_hits = sorted(
        [m for m in matched if m["correct"] and m.get("pick_odds") is not None and _is_underdog(m["pick_odds"], m.get("opponent_odds"))],
        key=lambda m: m["pick_odds"], reverse=True,
    )
    underdog_record = None
    if underdog_hits:
        underdog_record = {
            "count": len(underdog_hits),
            "biggest": underdog_hits[0],
            "hits": underdog_hits,
        }

    # Favorite vs. underdog accuracy: a genuinely different question than
    # underdog_record above (which only ever looks at correct underdog
    # picks, for the highlight reel) -- this is the real hit rate in
    # EACH group, correct and incorrect alike, since "how often do we
    # nail underdogs specifically" needs the denominator of every
    # underdog pick attempted, not just the wins.
    market_position_odds_known = [m for m in matched if m.get("pick_odds") is not None]
    favorite_picks = [m for m in market_position_odds_known if not _is_underdog(m["pick_odds"], m.get("opponent_odds"))]
    underdog_picks = [m for m in market_position_odds_known if _is_underdog(m["pick_odds"], m.get("opponent_odds"))]
    # THE HIT RATE ALONE IS NOT INTERPRETABLE, and publishing it without a
    # baseline actively misleads. Underdog picks are SUPPOSED to lose most of
    # the time -- that is what being an underdog means -- so "43%" reads as
    # failure against an instinctive 50% when the honest comparison is the
    # market's own number for those specific fighters, around 35%. Judged
    # that way the same 43% is the model beating the price it was offered.
    #
    # This is not hypothetical. Analysing UFC 330 I compared underdog picks
    # to 50%, concluded the model's disagreements "carry negative
    # information", and was wrong: measured against the market's implied
    # probability the picks came in AHEAD in both groups. If that mistake is
    # available with the whole dataset open, a reader looking at a bare
    # percentage has no chance.
    #
    # Both sides' prices are needed and de-vigged, since raw implied
    # probabilities sum to ~1.05 and the margin would otherwise show up as
    # phantom model edge.
    def _market_expectation(picks):
        """Mean de-vigged market probability on the fighter we picked."""
        vals = []
        for m in picks:
            po = m.get("pick_odds")
            # Same coherence gate as _is_underdog: de-vigging a pair that was
            # never simultaneous produces a "market expectation" the market
            # never held, and this figure is published as the baseline the
            # model's accuracy is compared against.
            oo = _coherent_opponent_odds(po, m.get("opponent_odds"))
            if po is None or oo is None:
                continue
            try:
                fair_pick, _ = remove_vig_two_way(
                    american_to_implied_prob(po), american_to_implied_prob(oo)
                )
            except (ValueError, TypeError, ZeroDivisionError):
                continue
            vals.append(fair_pick)
        return round(sum(vals) / len(vals) * 100, 1) if vals else None

    market_position_accuracy = None
    if favorite_picks or underdog_picks:
        fav_correct = sum(1 for m in favorite_picks if m["correct"])
        dog_correct = sum(1 for m in underdog_picks if m["correct"])

        def _group(picks, correct_n):
            acc = round(correct_n / len(picks) * 100, 1) if picks else None
            exp = _market_expectation(picks)
            return {
                "correct": correct_n,
                "total": len(picks),
                "accuracy_pct": acc,
                # What the market gave these same picks. None when prices are
                # missing -- shown as absent rather than guessed, same
                # partial-coverage honesty as CLV.
                "market_expected_pct": exp,
                # Signed gap. Positive means the picks beat their price.
                "vs_market_pct": (round(acc - exp, 1)
                                  if acc is not None and exp is not None else None),
            }

        market_position_accuracy = {
            "favorites": _group(favorite_picks, fav_correct),
            "underdogs": _group(underdog_picks, dog_correct),
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
        total_staked = sum(m["unit_size"] for m in units_eligible if m["unit_size"] is not None)
        by_tier = {}
        lock_picks_units = [m for m in units_eligible if m["is_lock_of_week"]]
        if lock_picks_units:
            by_tier["Lock of the Week"] = {
                "units": round(sum(m["units_result"] for m in lock_picks_units), 2),
                "count": len(lock_picks_units),
                "unit_size": LOCK_OF_WEEK_UNITS,
            }
        for tier in ("High Confidence", "Medium Confidence", "Low Confidence"):
            # A lock is ALWAYS High Confidence by definition, but gets its
            # own tier above -- excluded here so it's counted once, at its
            # real 10-unit weight, not also folded into the 5-unit tier.
            tier_picks = [m for m in units_eligible if m["confidence_label"] == tier and not m["is_lock_of_week"]]
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
        # Per-point metadata for the interactive chart. The cumulative figure
        # tells you where you STOOD; the pick's own result tells you what
        # happened AT that point, which is the actual reason to scrub -- a
        # downward step becomes "this fight cost 10U" rather than an
        # unexplained dip. Index 0 is the synthetic origin, so it gets a null
        # entry to stay aligned with `running`.
        running_points = [None]
        cumulative = 0.0
        for m in reversed(units_eligible):
            cumulative += m["units_result"]
            running.append(round(cumulative, 2))
            running_points.append({
                "fight": f'{m["predicted_favorite"]} vs {m["fighter_b"] if m["predicted_favorite"] == m["fighter_a"] else m["fighter_a"]}',
                "pick": m["predicted_favorite"],
                "units": round(m["units_result"], 2),
                "won": bool(m.get("correct")),
                "tier": m.get("confidence_label") or "",
            })
        units_stats = {
            "total_units": total_units,
            "total_staked": total_staked,
            "roi_pct": round(total_units / total_staked * 100, 1) if total_staked else None,
            "eligible_count": len(units_eligible),
            "event_count": len({m["event_name"] for m in units_eligible}),
            "by_tier": by_tier,
            "running_total": running,
            "running_points": running_points,
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
        "underdog_record": underdog_record,
        "market_position_accuracy": market_position_accuracy,
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

    def _card_display_order() -> dict:
        """
        {pair_key: position} from the card files, which are already stored
        main-event-first -- so ascending position IS "last fight of the night
        at the top", with no inference required.
        """
        order = {}
        for path in ("data/fight_cards.csv", "data/future_cards.csv"):
            if not os.path.exists(path):
                continue
            try:
                with open(path, newline="") as f:
                    for row in csv.DictReader(f):
                        key = _pair_key(row.get("fighter_a", ""), row.get("fighter_b", ""))
                        if key not in order:
                            order[key] = len(order)
            except (OSError, csv.Error):
                continue
        return order

    card_order = _card_display_order()

    def _sort_within_event(entries: list[dict]) -> list[dict]:
        # LATEST FIGHT FIRST. Billing rank alone only orders the GROUPS --
        # five fights all sharing "Main Card" (rank 2) came out in whatever
        # order they happened to be stored, so within a segment the list was
        # effectively arbitrary rather than chronological.
        # Two stable passes: date_added descending first (results are
        # recorded as each fight finishes, so it tracks fight order within a
        # card), then billing rank. Python's sort is stable, so the second
        # pass preserves the date order inside each equal rank.
        # Missing card_position still falls back to rank 99 rather than
        # being scattered to an arbitrary spot.
        # PREFER THE CARD FILE'S OWN ORDER when it covers every fight here.
        # The date_added fallback below can't order fights within a segment:
        # date_added is a DATE, not a timestamp, so every fight from one night
        # ties and the "stable sort preserves date order" reasoning has no
        # order to preserve. What survives is fight_results.csv's insertion
        # order, which is the order fights FINISHED -- earliest first, the
        # exact reverse of what this function is supposed to produce.
        # Older cards happened to look right only because their results were
        # written in card order rather than as they concluded.
        # Requires ALL entries to be found, so a partially-covered event
        # falls back wholesale rather than interleaving two different
        # orderings, which would be worse than either alone.
        keys = [_pair_key(e.get("fighter_a", ""), e.get("fighter_b", "")) for e in entries]
        if card_order and all(k in card_order for k in keys):
            return sorted(entries, key=lambda e: card_order[_pair_key(e.get("fighter_a", ""), e.get("fighter_b", ""))])

        by_recency = sorted(entries, key=lambda e: str(e.get("date_added") or ""), reverse=True)
        return sorted(by_recency, key=lambda e: billing_rank.get(e.get("card_position"), 99))

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
