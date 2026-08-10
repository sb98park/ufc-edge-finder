"""
Add a fight to the card by hand, for bouts ESPN hasn't published yet.

WHY THIS EXISTS. card_discovery adds new fights automatically, but only ones
ESPN's feed actually lists, and ESPN can lag a genuine late booking by days.
Both of UFC Vegas 120's late additions were reported by every MMA outlet and
carried by Tapology and Wikipedia while ESPN still showed the old lineup.
Waiting for the feed means the card is wrong for the fight that is most
interesting to a reader: the one that just changed.

Rows written here get manually_added=True, which PINS them in
resync_tracked_card_order. That flag is load-bearing, not bookkeeping: a
hand-entered fight is by definition one ESPN doesn't list, so without the pin
it would look orphaned on the next resync, climb the orphan streak, and be
auto-marked CANCELLED within about fifteen minutes -- taking its prediction
with it. The flag says a human has better information than the feed right
now, so the feed's silence is not evidence against the fight.

When ESPN does catch up, the fresh row matches this one on fighter names and
the normal path takes over; the pin simply stops mattering.

AFTER RUNNING THIS, run generate_site.py. Its fighter backfill will try to
pull roster rows for anyone new. If a fighter can't be found (a true debutant
ESPN has no athlete page for), predict_matchup returns None and the whole
preview is stripped, leaving only the moneyline chart -- an honest degradation
rather than an error, but worth knowing before it surprises you.

Usage (dry run first, always):
  python3 scripts/add_fight_manually.py \
      --event "UFC Fight Night: Gamrot vs. Salkilld" \
      --fighter-a "Miles Johns" --fighter-b "Gianni Vazquez" \
      --weight-class "Featherweight" --position "Prelims" \
      --replacement-for "Jessie Rosas" --short-notice "Gianni Vazquez"

  ... then re-run with --apply to write.

  --replacement-for  marks the bout with the Replacement badge and records
                     who withdrew (shown as the badge's tooltip).
  --short-notice     sets short_notice=1 in fighters.csv for that fighter.
                     Name ONLY the fighter who actually stepped in late.
                     matchup_model applies the penalty as a DIFFERENCE
                     between the two corners, so flagging both fighters
                     cancels out to exactly zero adjustment.

CREATING ROSTER ROWS (--create-roster-row plus --a-record / --b-record):

  A hand-added fight gets NO roster row from the normal backfill, and the
  reason is structural rather than a bug. ensure_roster_rows() resolves ESPN
  athlete ids by fetching the SCOREBOARD FOR THE EVENT DATE and reading that
  event's competitors -- so the lookup only ever contains fighters ESPN
  already lists on the card. A fight added by hand is by definition one ESPN
  doesn't list, so its fighters can never be resolved that way. Note this is
  NOT the same as "ESPN has no page for them": the route goes through the
  card, not through athlete search, so even a fighter with a full ESPN
  profile stays unresolvable until the feed catches up.
  Without a row, predict_matchup returns None and the whole preview is
  stripped -- no confidence, no tale of the tape, no waterfall, just the
  moneyline chart.

  A MINIMAL row is enough to restore the preview, and is honest rather than
  a fudge: fighters in this situation are almost always debutants or
  near-debutants with no UFC history, so Elo correctly falls back to the flat
  1500 prior and the recency-weighted stats fall back to career values that
  don't exist yet. That is exactly the documented path for debut fights,
  which are measurably the model's BEST cohort -- not a degraded one.

  Records are as reported publicly; pass them explicitly rather than letting
  this script guess:
      --create-roster-row --a-record "15-5" --b-record "14-5-1" \
      --a-country "United States" --b-country "Mexico"

  Optional physicals, omitted rather than invented if you don't have them:
      --a-height 68 --a-reach 70 --b-height 66 --b-reach 68
"""

import argparse
import sys
import unicodedata

import pandas as pd

# BOTH card files. fight_cards.csv is only the CURRENT card; anything further
# out lives in future_cards.csv, and late additions are at least as likely
# there -- the UFC.com cross-check found one on a card five weeks away. The
# original version read only the first file and reported the event as
# nonexistent, listing the one event it could see.
FIGHT_CARDS = "data/fight_cards.csv"
FUTURE_CARDS = "data/future_cards.csv"
CARD_FILES = [FIGHT_CARDS, FUTURE_CARDS]
FIGHTERS = "data/fighters.csv"


def _fold(v) -> str:
    folded = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(folded.lower().split())


def _parse_record(rec: str):
    """
    "14-5-1" or "15-5" -> (wins, losses, draws). Returns None on anything
    else rather than guessing -- a silently mis-parsed record would feed the
    model a wrong career line, which is worse than having no row at all.
    """
    parts = [p.strip() for p in str(rec).replace("\u2013", "-").split("-")]
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        return None
    w, l = int(parts[0]), int(parts[1])
    d = int(parts[2]) if len(parts) == 3 else 0
    return w, l, d


def _create_roster_rows(pending: list[dict], fighters_path: str = FIGHTERS) -> int:
    """
    Write minimal fighters.csv rows for fighters the ESPN backfill can't reach.

    Only ever fills what was explicitly supplied. A partial row is fine --
    build_fight_preview handles missing physicals -- whereas a fabricated
    complete one would look authoritative while being invented.
    """
    fighters = pd.read_csv(fighters_path)
    existing = set(fighters["name"].map(_fold))
    new_rows = [r for r in pending if _fold(r["name"]) not in existing]
    if not new_rows:
        print("  Both fighters already have roster rows -- nothing to create.")
        return 0
    for col in {k for r in new_rows for k in r}:
        if col not in fighters.columns:
            fighters[col] = None
    fighters = pd.concat([fighters, pd.DataFrame(new_rows)], ignore_index=True)
    fighters.to_csv(fighters_path, index=False)
    for r in new_rows:
        print(f"  Created roster row: {r['name']} "
              f"({r.get('wins','?')}-{r.get('losses','?')}, {r.get('weight_class','?')})")
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--event", required=True, help="event_name exactly as it appears in fight_cards.csv")
    ap.add_argument("--fighter-a", required=True)
    ap.add_argument("--fighter-b", required=True)
    ap.add_argument("--weight-class", required=True)
    ap.add_argument("--position", required=True, help='e.g. "Prelims", "Main Card", "Early Prelims"')
    ap.add_argument("--replacement-for", default="", help="name of the fighter who withdrew")
    ap.add_argument("--short-notice", default="", help="fighter who took the bout late -- ONE name")
    ap.add_argument("--womens", action="store_true", help="women's division bout")
    ap.add_argument("--create-roster-row", action="store_true",
                    help="write minimal fighters.csv rows for whichever corner lacks one")
    for side in ("a", "b"):
        ap.add_argument(f"--{side}-record", default="", help=f'fighter {side.upper()} record, e.g. "14-5-1"')
        ap.add_argument(f"--{side}-country", default="")
        ap.add_argument(f"--{side}-height", type=float, default=None, help="inches")
        ap.add_argument(f"--{side}-reach", type=float, default=None, help="inches")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # Find which file holds the event, and write back to that same one.
    target_path, cards, event_rows = None, None, None
    for path in CARD_FILES:
        try:
            df = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        rows = df[df["event_name"].astype(str).str.strip() == args.event.strip()]
        if not rows.empty:
            target_path, cards, event_rows = path, df, rows
            break
    if event_rows is None:
        print(f"No rows found for event {args.event!r}. Events available:")
        for path in CARD_FILES:
            try:
                df = pd.read_csv(path)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                continue
            for name in df["event_name"].dropna().unique():
                print(f"  - {name}   [{path}]")
        sys.exit(1)
    print(f"(event found in {target_path})")

    pair = {_fold(args.fighter_a), _fold(args.fighter_b)}
    existing = cards.apply(lambda r: {_fold(r["fighter_a"]), _fold(r["fighter_b"])} == pair, axis=1)
    already_on_card = bool(existing.any())
    if already_on_card:
        print(f"That bout is already on the card ({int(existing.sum())} row(s)).")
        for _, r in cards[existing].iterrows():
            print(f"  -> {r['fighter_a']} vs {r['fighter_b']} ({r['event_name']}, {r['card_position']})")
        # Still worth continuing IF the run is here to create roster rows: the
        # fight and the roster row are separate failures, and the common case
        # is discovering the missing row only after the fight was added. Adding
        # the bout twice is what must not happen -- that part is skipped below.
        if not args.create_roster_row:
            print("Nothing to do. Pass --create-roster-row if you're here to add missing roster rows.")
            sys.exit(1)
        print("Continuing anyway to handle roster rows -- the bout itself will NOT be added again.\n")

    # Inherit every event-level field from a sibling row rather than asking for
    # them: date, start times and any other per-event column must match the
    # rest of the card exactly or this fight sorts and renders out of step.
    template = event_rows.iloc[-1].to_dict()
    new_row = {c: template.get(c) for c in cards.columns}
    new_row.update({
        "fighter_a": args.fighter_a.strip(),
        "fighter_b": args.fighter_b.strip(),
        "weight_class": args.weight_class.strip(),
        "card_position": args.position.strip(),
        "manually_added": True,
        "cancelled": False,
    })
    if "is_womens_division" in cards.columns:
        new_row["is_womens_division"] = bool(args.womens)
    if args.replacement_for:
        new_row["replacement"] = True
        new_row["replaced_fighter"] = args.replacement_for.strip()
    # Anything carried over from the template that is per-FIGHT rather than
    # per-EVENT would otherwise be silently inherited -- a copied result or
    # orphan counter would be actively wrong on a brand-new bout.
    for col in ("result_label", "result_round_time", "winner", "_orphan_streak", "is_lock_of_week"):
        if col in new_row:
            new_row[col] = None

    # Assembled before the dry-run report so --create-roster-row is fully
    # previewable, and so a malformed record fails BEFORE anything is written.
    pending_roster = []
    if args.create_roster_row:
        roster_names = set(pd.read_csv(FIGHTERS)["name"].map(_fold))
        for side, name in (("a", args.fighter_a), ("b", args.fighter_b)):
            if _fold(name) in roster_names:
                continue
            rec = getattr(args, f"{side}_record")
            if not rec:
                print(f"ERROR: {name} has no roster row and no --{side}-record was given. "
                      f"Without a record there is nothing worth writing -- pass the record "
                      f"or drop --create-roster-row.")
                sys.exit(1)
            parsed = _parse_record(rec)
            if not parsed:
                print(f"ERROR: could not parse --{side}-record {rec!r}. Expected \"W-L\" or \"W-L-D\".")
                sys.exit(1)
            w, l, d = parsed
            row = {"name": name.strip(), "weight_class": args.weight_class.strip(),
                   "wins": w, "losses": l}
            if d:
                row["draws"] = d
            if getattr(args, f"{side}_country"):
                row["country"] = getattr(args, f"{side}_country").strip()
            if getattr(args, f"{side}_height") is not None:
                row["height_in"] = getattr(args, f"{side}_height")
            if getattr(args, f"{side}_reach") is not None:
                row["reach_in"] = getattr(args, f"{side}_reach")
            pending_roster.append(row)

    if not already_on_card:
        print(f"Would add to {args.event!r} in {target_path}:")
        for k in ("fighter_a", "fighter_b", "weight_class", "card_position",
                  "replacement", "replaced_fighter", "manually_added"):
            if k in new_row and new_row.get(k) not in (None, ""):
                print(f"    {k}: {new_row[k]}")
    if args.short_notice:
        print(f"    short_notice=1 in {FIGHTERS} for: {args.short_notice}")
    for row in pending_roster:
        print(f"    NEW roster row: {row}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to write.")
        return

    if not already_on_card:
        for col in ("manually_added", "replacement", "replaced_fighter"):
            if col not in cards.columns:
                cards[col] = None
            # All-empty flag columns read back from CSV as float64 (all-NaN),
            # and pandas 2.x raises assigning a bool or str into float64.
            cards[col] = cards[col].astype("object")

        # Appended, not inserted: resync_tracked_card_order reorders the card
        # against ESPN on the next run anyway, and until then last is the safe
        # place -- it can't displace a fight whose position IS confirmed.
        cards = pd.concat([cards, pd.DataFrame([new_row])], ignore_index=True)
        cards.to_csv(target_path, index=False)
        print(f"\nAdded to {target_path}.")

    # BEFORE the short-notice step, deliberately. The flag is set by matching
    # a name in fighters.csv, so if the incoming fighter's row is being
    # created in this same run it has to exist first -- otherwise the flag
    # silently finds nothing, which is exactly the dead end the previous
    # version hit on Gianni Vazquez.
    if pending_roster:
        _create_roster_rows(pending_roster)

    if args.short_notice:
        fighters = pd.read_csv(FIGHTERS)
        if "short_notice" not in fighters.columns:
            fighters["short_notice"] = 0
        mask = fighters["name"].map(_fold) == _fold(args.short_notice)
        if not mask.any():
            print(f"  NOTE: {args.short_notice!r} still has no row in {FIGHTERS}, so short_notice "
                  f"was not set. Re-running this command won't help -- it exits early once the "
                  f"bout is on the card. Either re-run WITH --create-roster-row and a record for "
                  f"that fighter, or set the column directly.")
        else:
            fighters.loc[mask, "short_notice"] = 1
            fighters.to_csv(FIGHTERS, index=False)
            print(f"  Set short_notice=1 for {args.short_notice}.")

    print("\nNow run generate_site.py, then scripts/lint_site.py, then push.")


if __name__ == "__main__":
    main()
