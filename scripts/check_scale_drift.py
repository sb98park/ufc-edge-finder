"""
Does the model's probability scale still mean what the thresholds assume?

WHY THIS EXISTS. LOCK_OF_WEEK_MIN_PROB was set to 0.82 in July on 5 cards and
13 High Confidence picks, and flagged provisional. Nine cards later locks had
stopped appearing entirely -- and the reason was not that the model got worse
or the cards got weaker. THE SCALE MOVED UNDER THE CONSTANT. Measured over the
nine cards with logged prices:

    model p90    Spearman rho -0.67, p = 0.050
    market p90   Spearman rho +0.00, p = 1.000

Nobody noticed for six weeks. It was found because the owner said "I don't see
lock of the weeks in the future cards" -- the same discovery route as every
other thing this repo has an alarm for now.

Every absolute probability threshold in this project -- the lock floor, the
0.75/0.60 tier bars, MIN_RECORD gates -- assumes the number 0.78 means the
same thing in November that it meant in July. Nothing checked that.

THE COMPARISON IS AGAINST THE MARKET, AND THAT IS THE WHOLE DESIGN. A run of
genuinely competitive cards compresses the model's top end for an honest
reason, and a monitor watching the model alone cannot tell that apart from
drift -- it would cry wolf every time the UFC books three flat Fight Nights in
a row. The market's own p90 is the control: when both fall together the cards
got tighter, when the model falls and the market does not, the scale moved.

WHY IT IS NOT A GATE. A moving scale is information, not corruption, and
CLAUDE.md is explicit that a gate must never fail for a number that
legitimately moved -- these gates freeze the whole site on stale data when
they trip, on fight night included. This prints, writes its own block in
source_health.json, and exits 0 ALWAYS.

WHAT IT DELIBERATELY DOES NOT DO: recommend a new floor as though the number
were fitted. The sample is a handful of cards; the honest output is "the scale
has moved this far, the floor now selects this much, here is what would
restore its original selectivity" -- three measurements and an arithmetic
restatement, not a fit. Re-deriving the constant stays a decision someone
makes and records, as section 1 requires.

Run: python3 scripts/check_scale_drift.py
Read-only except for its own block in data/source_health.json.
"""

import datetime as dt
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.track_record import LOCK_OF_WEEK_MIN_PROB  # noqa: E402

HEALTH = "data/source_health.json"

# The era the live floor was calibrated against, and the era being judged.
# Both are counts of CARDS, not fights: the card is the unit that varies, and
# ~14 picks inside one card are not 14 independent draws on "how lopsided is
# this weekend".
BASELINE_CARDS = 6
TRAILING_CARDS = 6

# HOW FAR THE MODEL-MINUS-MARKET GAP MUST MOVE before this says anything.
# 0.04 at the p90 is roughly half the drift that actually went unnoticed
# (~0.07), so it fires well before a repeat while staying clear of the
# card-to-card noise, which runs +-0.02 on a quiet stretch.
DRIFT_PP = 0.04

# ...AND the live floor must ALSO be selecting outside half-to-double its
# original rate. BOTH conditions, because either alone has a benign
# explanation: the gap can move on two odd cards, and selectivity legitimately
# collapses on a genuinely weak stretch. Only together do they mean the
# threshold has stopped measuring what it was set to measure.
SELECTIVITY_BAND = (0.5, 2.0)


def _implied(odds) -> float | None:
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o != o or o == 0:
        return None
    return (-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)


def _event_order() -> dict:
    """event_name -> a sortable date. Results first, then scheduled cards."""
    order = {}
    try:
        res = pd.read_csv("data/fight_results.csv")
        for e, g in res.groupby("event_name"):
            order[str(e)] = str(g["date_added"].min())[:10]
    except (OSError, pd.errors.EmptyDataError, KeyError):
        pass
    for path in ("data/fight_cards.csv", "data/future_cards.csv"):
        try:
            c = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if "event_name" not in c.columns or "event_date" not in c.columns:
            continue
        for e, g in c.groupby("event_name"):
            order.setdefault(str(e), str(g["event_date"].min())[:10])
    return order


def _per_card(log: pd.DataFrame, order: dict) -> list[dict]:
    """
    One row per card: the model's p90 and the market's, side by side.

    ONLY CARDS WE CAN PLACE IN TIME. An event with no results and no row on a
    tracked card is a stale orphan -- a card renamed mid-week leaves its old
    name behind in the log, with real-looking probabilities frozen at whatever
    the model said before the rename. Ordering by a date it does not have
    would drop it at one end of the series and skew whichever window it landed
    in. Confirmed present: 'Ankalaev vs. Rountree Jr.' is 16 rows duplicating
    13 of 'Ankalaev vs. Guskov'.
    """
    rows = []
    for event, g in log.groupby("event_name"):
        when = order.get(str(event))
        if not when:
            continue
        model, market = [], []
        for _, r in g.iterrows():
            try:
                p = float(r["favorite_prob"])
            except (TypeError, ValueError, KeyError):
                continue
            if p != p:
                continue
            model.append(p)
            a, b = _implied(r.get("pick_odds")), _implied(r.get("opponent_odds"))
            if a is None or b is None or (a + b) <= 0:
                continue
            # De-vig, then take the FAVOURITE side whichever corner it is --
            # the question is how lopsided the card looked, not who we picked.
            market.append(max(a / (a + b), b / (a + b)))
        if len(model) < 5 or len(market) < 5:
            continue
        ms, ks = pd.Series(model), pd.Series(market)
        rows.append({
            "event": str(event), "date": when, "fights": len(model),
            "priced": len(market),
            "model_p90": round(float(ms.quantile(0.90)), 4),
            "market_p90": round(float(ks.quantile(0.90)), 4),
            "gap_p90": round(float(ms.quantile(0.90) - ks.quantile(0.90)), 4),
            "clearing_floor": int((ms >= LOCK_OF_WEEK_MIN_PROB).sum()),
            "selectivity": round(float((ms >= LOCK_OF_WEEK_MIN_PROB).mean()), 4),
        })
    return sorted(rows, key=lambda r: (r["date"], r["event"]))


def analyse(cards: list[dict], log: pd.DataFrame) -> dict:
    """
    The whole judgement, as a pure function of already-loaded data.

    Separated from main() so the tests can drive it with synthetic cards
    instead of whatever data/ happens to hold today. A previous alarm's test
    asserted on the live source_health.json and froze CI for two hours when
    the file legitimately changed; this one must never be able to do that.
    """
    base, tail = cards[:BASELINE_CARDS], cards[-TRAILING_CARDS:]
    base_gap = sum(c["gap_p90"] for c in base) / len(base)
    tail_gap = sum(c["gap_p90"] for c in tail) / len(tail)
    gap_move = tail_gap - base_gap

    # Selectivity is measured on PICKS, not averaged over cards: a 16-fight
    # card and a 10-fight card are not equal evidence about how often the
    # floor lets something through.
    base_sel = sum(c["clearing_floor"] for c in base) / sum(c["fights"] for c in base)
    tail_sel = sum(c["clearing_floor"] for c in tail) / sum(c["fights"] for c in tail)

    # What floor would restore the baseline rate on the trailing window? Pure
    # arithmetic on the empirical quantile -- an "equivalent", not a fit.
    tail_names = {c["event"] for c in tail}
    tail_probs = []
    for _, r in log[log["event_name"].astype(str).isin(tail_names)].iterrows():
        try:
            p = float(r["favorite_prob"])
        except (TypeError, ValueError):
            continue
        if p == p:
            tail_probs.append(p)
    equiv = (round(float(pd.Series(tail_probs).quantile(1 - base_sel)), 3)
             if tail_probs and 0 < base_sel < 1 else None)

    drifted = abs(gap_move) > DRIFT_PP
    lo, hi = SELECTIVITY_BAND
    off_band = base_sel > 0 and not (lo * base_sel <= tail_sel <= hi * base_sel)
    # BOTH, or it is not drift. See SELECTIVITY_BAND.
    verdict = "drifted" if (drifted and off_band) else ("watch" if (drifted or off_band) else "ok")
    return {"base": base, "tail": tail, "base_gap": base_gap, "tail_gap": tail_gap,
            "gap_move": gap_move, "base_sel": base_sel, "tail_sel": tail_sel,
            "equivalent_floor": equiv, "drifted": drifted, "off_band": off_band,
            "verdict": verdict}


def main() -> int:
    try:
        log = pd.read_csv("data/predictions_log.csv")
    except (OSError, pd.errors.EmptyDataError):
        print("[drift] no predictions_log -- nothing to measure")
        return 0
    for col in ("event_name", "favorite_prob"):
        if col not in log.columns:
            print(f"[drift] predictions_log has no {col!r} column -- skipping")
            return 0

    cards = _per_card(log, _event_order())
    if len(cards) < BASELINE_CARDS + 2:
        print(f"[drift] {len(cards)} placeable card(s); need "
              f"{BASELINE_CARDS + 2} before a baseline means anything")
        return 0

    stats = analyse(cards, log)

    base, tail = stats["base"], stats["tail"]
    base_gap, tail_gap, gap_move = stats["base_gap"], stats["tail_gap"], stats["gap_move"]
    base_sel, tail_sel = stats["base_sel"], stats["tail_sel"]
    equiv, verdict = stats["equivalent_floor"], stats["verdict"]

    print(f"{'card':44s} {'n':>3} {'model p90':>10} {'mkt p90':>8} {'gap':>7} {'>=floor':>8}")
    for c in cards:
        mark = "  <" if c in tail else ""
        print(f"{c['event'][:44]:44s} {c['fights']:3d} {c['model_p90']:10.3f} "
              f"{c['market_p90']:8.3f} {c['gap_p90']:+7.3f} {c['clearing_floor']:8d}{mark}")

    print(f"\n  floor in force              {LOCK_OF_WEEK_MIN_PROB:.2f}")
    print(f"  model-minus-market p90      baseline {base_gap:+.3f} "
          f"(first {len(base)} cards)  ->  trailing {tail_gap:+.3f} "
          f"(last {len(tail)})   move {gap_move:+.3f}")
    print(f"  the floor's selectivity     baseline {base_sel:.1%}  ->  trailing {tail_sel:.1%}")
    if equiv is not None:
        print(f"  floor restoring that rate   {equiv:.3f} on the trailing window")

    drifted, off_band = stats["drifted"], stats["off_band"]
    if verdict == "drifted":
        print(f"\n  DRIFTED. The model's top end moved {gap_move:+.3f} against a market "
              f"that is the control here, AND the floor now selects {tail_sel:.1%} "
              f"against {base_sel:.1%} when it was set.")
        print(f"  {LOCK_OF_WEEK_MIN_PROB:.2f} no longer picks out what it was chosen to pick out. "
              f"Re-deriving it is a decision to make and record, not something to "
              f"read off this line -- {len(tail)} cards is a small sample.")
    elif drifted:
        print(f"\n  WATCH. The scale moved {gap_move:+.3f} but the floor still selects "
              f"{tail_sel:.1%} against {base_sel:.1%}, inside the band. Not actionable yet.")
    elif off_band:
        print(f"\n  WATCH. The floor selects {tail_sel:.1%} against {base_sel:.1%}, but the "
              f"model tracks the market to within {gap_move:+.3f} -- so the CARDS are "
              f"flatter, not the scale. Nothing to change.")
    else:
        print(f"\n  OK. Scale tracks the market to within {gap_move:+.3f} and the floor "
              f"selects {tail_sel:.1%} against {base_sel:.1%}.")

    payload = {}
    try:
        with open(HEALTH, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        pass
    if not isinstance(payload, dict):
        payload = {}
    payload["scale_drift"] = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict,
        "floor_in_force": LOCK_OF_WEEK_MIN_PROB,
        "baseline_cards": len(base), "trailing_cards": len(tail),
        "baseline_gap_p90": round(base_gap, 4), "trailing_gap_p90": round(tail_gap, 4),
        "gap_move": round(gap_move, 4),
        "baseline_selectivity": round(base_sel, 4),
        "trailing_selectivity": round(tail_sel, 4),
        "equivalent_floor": equiv,
        "cards": cards,
    }
    try:
        os.makedirs("data", exist_ok=True)
        with open(HEALTH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError as exc:
        print(f"[drift] not written ({exc}) -- continuing")
    return 0          # an alarm, never a brake


if __name__ == "__main__":
    sys.exit(main())
