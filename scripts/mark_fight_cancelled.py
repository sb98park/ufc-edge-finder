"""
Mark a fight as CANCELLED (e.g. a late pull-out before the card).

Does exactly two things:
1. fight_cards.csv: sets cancelled=True on the matching row. The fight
   STAYS on the displayed card with a big cancellation banner (see
   templates/site.html), and resync_tracked_card_order treats cancelled
   rows as pinned -- they're expected to be missing from ESPN's fresh
   data, so the orphan-drop logic never removes them.
2. predictions_log.csv: sets voided=True on the matching prediction
   row(s). track_record.py skips voided rows entirely at match time --
   the pick counts as a VOID: it never enters accuracy, confidence-tier,
   lock, or units math in either direction, exactly as if the prediction
   had never been made. (The Locks display still shows the lock itself,
   flagged cancelled -- it genuinely WAS the lock of the week; only the
   STATS treat it as nonexistent.)

Usage:
  python3 scripts/mark_fight_cancelled.py "Islam Dulatov" "Wellington Turman"           # dry run
  python3 scripts/mark_fight_cancelled.py "Islam Dulatov" "Wellington Turman" --apply   # write
"""

import sys

import pandas as pd

FIGHT_CARDS = "data/fight_cards.csv"
PREDICTIONS_LOG = "data/predictions_log.csv"


def _norm(s) -> str:
    return str(s).strip().lower()


def main():
    args = [a for a in sys.argv[1:] if a != "--apply"]
    apply = "--apply" in sys.argv
    if len(args) != 2:
        print(__doc__)
        sys.exit(1)
    target = {_norm(args[0]), _norm(args[1])}

    cards = pd.read_csv(FIGHT_CARDS)
    if "cancelled" not in cards.columns:
        cards["cancelled"] = False
    # An all-empty flag column reads back from CSV as float64 (all-NaN),
    # and pandas 2.x raises assigning bool True into float64 -- coerce.
    cards["cancelled"] = cards["cancelled"].astype("object")
    card_mask = cards.apply(lambda r: {_norm(r["fighter_a"]), _norm(r["fighter_b"])} == target, axis=1)
    print(f"fight_cards.csv rows matching: {int(card_mask.sum())}")
    for _, r in cards[card_mask].iterrows():
        print(f"  -> {r['fighter_a']} vs {r['fighter_b']} ({r['event_name']}, {r['card_position']})")

    preds = pd.read_csv(PREDICTIONS_LOG)
    if "voided" not in preds.columns:
        preds["voided"] = False
    preds["voided"] = preds["voided"].astype("object")
    pred_mask = preds.apply(lambda r: {_norm(r["fighter_a"]), _norm(r["fighter_b"])} == target, axis=1)
    print(f"predictions_log.csv rows matching: {int(pred_mask.sum())}")
    for _, r in preds[pred_mask].iterrows():
        lock = " [LOCK OF THE WEEK]" if str(r.get("is_lock_of_week")).strip().lower() == "true" else ""
        print(f"  -> pick: {r['favorite']} ({r.get('confidence_label')}){lock}")

    if not card_mask.any() and not pred_mask.any():
        print("\nNothing matched -- check the exact fighter name spellings against the CSVs.")
        sys.exit(1)

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to mark cancelled+voided.")
        return

    cards.loc[card_mask, "cancelled"] = True
    cards.to_csv(FIGHT_CARDS, index=False)
    preds.loc[pred_mask, "voided"] = True
    preds.to_csv(PREDICTIONS_LOG, index=False)
    print(f"\nDone -- {int(card_mask.sum())} fight(s) marked cancelled, "
          f"{int(pred_mask.sum())} prediction(s) voided. Now run generate_site.py and push.")


if __name__ == "__main__":
    main()
