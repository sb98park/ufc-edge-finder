"""
Recompute one fight's logged prediction after a DATA correction.

WHEN THIS IS LEGITIMATE, and when it isn't. A track record is only worth
anything if entries aren't quietly adjusted after the fact, so this is not a
general "change a pick" tool. It exists for one case: a prediction was
generated from demonstrably wrong INPUT data, the error was identified
before the fight, and the fix simply hadn't shipped yet.

Real case: Borislav Nikolić was stored at 2-1 because the backfill read the
scoreboard's promotion-scoped record instead of his 16-2 career mark. His
opponent was a 79% High Confidence pick built on a phantom debutant.

WHAT IT DOES NOT DO: hide the change. The correction is appended to
favorite_prob_history with a note, so the old and new probabilities both
remain visible. A record that shows "this was corrected, here's why" is more
trustworthy than one that silently reads as if the error never happened --
and the whole reason to bother is trustworthiness.

WHOLE-CARD MODE (--event) exists for a second, narrower case: the logged
predictions for an entire card drifted away from what was actually published
BEFORE the fights, so the log no longer matches what anyone saw. That is the
same class of problem as a bad input -- the log is wrong about the past --
and it is NOT a licence to re-run a card after seeing results and keep the
better numbers. The audit note goes into every row it touches, so a card
"corrected" for the wrong reason is visible as such forever.

Usage:
    python3 scripts/recompute_prediction.py "Vologdin"          # dry run, one fight
    python3 scripts/recompute_prediction.py "Vologdin" --apply
    python3 scripts/recompute_prediction.py --event "Gamrot"    # dry run, whole card
    python3 scripts/recompute_prediction.py --event "Gamrot" --apply
    ... optionally --note "why this correction was made"
"""

import datetime as dt
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.elo import EloRatingSystem  # noqa: E402
from src.power_rating import build_effective_ratings  # noqa: E402
from src.model_preview import build_fight_preview  # noqa: E402

LOG = "data/predictions_log.csv"


def main():
    argv = sys.argv[1:]
    apply = "--apply" in argv
    event_mode = "--event" in argv
    note = ""
    if "--note" in argv:
        ni = argv.index("--note")
        if ni + 1 < len(argv):
            note = argv[ni + 1]
            argv = argv[:ni] + argv[ni + 2:]
    args = [a for a in argv if a not in ("--apply", "--event")]
    if not args:
        print(__doc__)
        sys.exit(1)
    needle = args[0].lower()

    log = pd.read_csv(LOG)

    # Accent-folded search. A plain substring match can't find "Uroš Medić"
    # from "medic" -- ć is not c -- so every accented fighter needed their
    # opponent's name used instead. That's the fifth place diacritics have
    # broken a lookup in this codebase; fold both sides and it stops.
    import unicodedata as _ud

    def _fold(t):
        return "".join(ch for ch in _ud.normalize("NFKD", str(t).lower())
                       if not _ud.combining(ch))

    needle_folded = _fold(needle)
    if event_mode:
        # Match the EVENT, not a fighter, so one run covers a whole card.
        mask = log["event_name"].map(lambda v: needle_folded in _fold(v))
        if not mask.any():
            print(f"No logged predictions for an event matching {args[0]!r}. Events in the log:")
            for name in log["event_name"].dropna().unique():
                print(f"  - {name}")
            sys.exit(1)
        print(f"EVENT MODE: {int(mask.sum())} logged prediction(s) across "
              f"{log.loc[mask, 'event_name'].nunique()} event(s) matching {args[0]!r}.\n")
    else:
        mask = (log["fighter_a"].map(lambda v: needle_folded in _fold(v)) |
                log["fighter_b"].map(lambda v: needle_folded in _fold(v)))
        if not mask.any():
            print(f"No logged prediction matching {args[0]!r}.")
            sys.exit(1)

    fighters = pd.read_csv("data/fighters.csv")
    history = pd.read_csv("data/fight_history.csv")
    try:
        weight_hist = pd.read_csv("data/fighter_weight_class_history.csv")
    except FileNotFoundError:
        weight_hist = None

    elo = EloRatingSystem()
    elo.build_from_history(history)
    eff = build_effective_ratings(fighters, elo.ratings, history)
    now = dt.datetime.now().isoformat(timespec="seconds")

    for i in log.index[mask]:
        row = log.loc[i]
        a, b = str(row["fighter_a"]), str(row["fighter_b"])
        # build_fight_preview, not predict_matchup: the log stores favorite /
        # favorite_prob / confidence_label, which are the preview's shape.
        # predict_matchup returns raw prob_a / prob_b and no label.
        preview = build_fight_preview(a, b, fighters, eff,
                                      weight_class_history_df=weight_hist)
        if not preview:
            print(f"  {a} vs {b}: no preview produced -- check both fighters exist in fighters.csv")
            continue

        old_fav, old_prob = str(row["favorite"]), float(row["favorite_prob"])
        old_label = str(row["confidence_label"])
        # The CSV stores this as a string, so "False" is TRUTHY -- a plain
        # str() check fired the lock warning on a 64.6% pick that was never a
        # lock. Test the value, not its presence.
        _lock_raw = str(row.get("is_lock_of_week", "")).strip().lower()
        old_lock = _lock_raw in ("true", "yes", "1")
        new_fav, new_prob = preview["favorite"], preview["favorite_prob"]
        new_label = preview["confidence_label"]

        print(f"\n{a} vs {b}")
        print(f"   favorite   {old_fav:22} -> {new_fav}")
        print(f"   probability{old_prob:>10.1%}            -> {new_prob:.1%}")
        print(f"   confidence {old_label:22} -> {new_label}")
        if old_fav != new_fav:
            print("   NOTE: the corrected data flips which fighter the model favours.")
        if old_lock and new_prob < 0.82:
            print("   NOTE: was a Lock of the Week; the new probability is below the 82% floor.")

        if not apply:
            continue

        try:
            hist = json.loads(row.get("favorite_prob_history") or "[]")
        except (json.JSONDecodeError, TypeError):
            hist = []
        # The correction is APPENDED, never overwritten -- the original
        # probability stays in the history so the change is auditable.
        hist.append({
            "prob": new_prob, "date": now,
            "note": note or f"corrected from {old_prob:.3f} after opponent record fix",
        })
        log.at[i, "favorite"] = new_fav
        log.at[i, "favorite_prob"] = new_prob
        log.at[i, "confidence_label"] = new_label
        log.at[i, "likely_method"] = preview.get("likely_method", row.get("likely_method"))
        log.at[i, "favorite_prob_history"] = json.dumps(hist)
        log.at[i, "last_updated"] = now
        if old_lock and new_prob < 0.82:
            # Match the column's existing dtype. pandas reads this column as
            # bool when every value is True/False, and writing "" into a bool
            # column raises rather than coercing -- which crashed mid-run,
            # after the console had already printed the change.
            if pd.api.types.is_bool_dtype(log["is_lock_of_week"]):
                log.at[i, "is_lock_of_week"] = False
            else:
                log.at[i, "is_lock_of_week"] = ""

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return
    log.to_csv(LOG, index=False)
    print("\nWritten. Commit data/predictions_log.csv.")


if __name__ == "__main__":
    main()
