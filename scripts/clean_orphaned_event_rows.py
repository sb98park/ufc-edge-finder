"""
Rows stranded in predictions_log by a card that changed its name.

WHAT HAPPENS. ESPN renames a card when its headliner changes: "UFC Fight
Night: Ankalaev vs. Rountree Jr." became "... vs. Guskov" on fight day.
log_predictions keys on (event_name, fighter_a, fighter_b), so after a rename
every bout misses its own prior row and gets written afresh under the new
name. The rows under the OLD name are then unreachable forever -- they never
match a result, so they never grade, and nothing ever visits them again.

track_record.log_predictions already migrates a row across a rename, by
looking the pair up when the exact key misses. THAT FIX CANNOT REACH THIS
ONE, because it sits behind

    if fighter_key in decided_keys:
        continue

and by the time this rename was noticed the card had already graded. Every
bout on it short-circuits before the migration is reached. Same shape as
card_discovery.deduplicate_tracked_fights: repairing the matching logic does
nothing about what an earlier, buggier run already wrote to disk.

TWO DIFFERENT REPAIRS, and conflating them would lose real information.

  DUPLICATES -- the same pair, graded under the card's real name. The
  prediction survives in full under the correct event; the stranded copy is
  a pure artefact of the rename and carries nothing the kept row does not.
  These are REMOVED.

  CANCELLED BOOKINGS -- fights that were on the card before the reshuffle and
  never happened at all. These are real calls the model made and they are NOT
  deleted; they are marked voided and re-pointed to the card's real name,
  which is what `voided` is for and what scripts/mark_fight_cancelled.py does
  by hand. Voiding also defuses a live hazard that track_record's own comment
  names: if Ankalaev vs. Rountree Jr. is ever rebooked, an unvoided July
  prediction would silently grade against a fight held months later.

WHY IT ASKS RATHER THAN ACTS. Section 1 says the published record is the
owner's to correct, explicitly, recorded in a commit message -- so this
REPORTS by default and exits 0, and only writes under --apply. It is not
wired into the build as a mutation and must not be: a pass that quietly
deletes published rows is precisely what section 1 forbids.

Usage:
  python3 scripts/clean_orphaned_event_rows.py            # report, exit 0
  python3 scripts/clean_orphaned_event_rows.py --apply     # write
"""

import sys

import pandas as pd

LOG = "data/predictions_log.csv"
RESULTS = "data/fight_results.csv"
CARDS = ("data/fight_cards.csv", "data/future_cards.csv")


def _pair(a, b):
    return frozenset({str(a).strip().lower(), str(b).strip().lower()})


def _load():
    log = pd.read_csv(LOG)
    try:
        res = pd.read_csv(RESULTS)
    except (OSError, pd.errors.EmptyDataError):
        res = pd.DataFrame(columns=["event_name", "fighter_a", "fighter_b"])
    live_events = set()
    for p in CARDS:
        try:
            c = pd.read_csv(p)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if "event_name" in c.columns:
            live_events |= {str(e) for e in c["event_name"].dropna()}
    return log, res, live_events


def analyse(log, res, live_events):
    """Classify every row. Pure, so the tests can drive it with fixtures."""
    # An event is REAL if it produced results or is on a tracked card.
    # An EMPTY results frame has no columns at all, so this cannot index it
    # blind -- a repo with no graded card yet would raise KeyError here and
    # take the whole check down rather than reporting nothing.
    res_events = (res["event_name"].dropna() if "event_name" in getattr(res, "columns", [])
                  else [])
    real_events = {str(e) for e in res_events} | set(live_events)

    # A DUPLICATE IS A PAIR THAT ALREADY HAS A ROW UNDER A REAL EVENT NAME --
    # not merely one that graded. Islam Dulatov vs Wellington Turman was
    # cancelled off this very card, so it never graded, but it HAS a row under
    # the real name carrying voided=true. Keying on "graded" classified it as
    # a cancelled booking and would have voided-and-re-pointed it into a
    # second Guskov row: a duplicate created by the duplicate remover.
    kept_under = {}
    for _, r in log.iterrows():
        if str(r["event_name"]) in real_events:
            kept_under.setdefault(_pair(r["fighter_a"], r["fighter_b"]), str(r["event_name"]))

    drop, void = [], []
    if "event_name" not in getattr(log, "columns", []):
        return [], []
    for i, r in log.iterrows():
        ev = str(r["event_name"])
        if ev in real_events:
            continue                      # the event exists; nothing stranded
        pk = _pair(r["fighter_a"], r["fighter_b"])
        if pk in kept_under:
            # The identical pair already has a row under a REAL event name, so
            # this copy carries nothing that one does not. The `ev in
            # real_events` guard above guarantees we never drop that keeper.
            drop.append({"idx": i, "event": ev, "a": r["fighter_a"], "b": r["fighter_b"],
                         "kept_under": kept_under[pk]})
        else:
            already = str(r.get("voided", "")).strip().lower() == "true"
            void.append({"idx": i, "event": ev, "a": r["fighter_a"], "b": r["fighter_b"],
                         "favorite": r.get("favorite"), "prob": r.get("favorite_prob"),
                         "already_voided": already})
    return drop, void


def main() -> int:
    apply = "--apply" in sys.argv
    try:
        log, res, live_events = _load()
    except (OSError, pd.errors.EmptyDataError) as e:
        print(f"[orphans] cannot read inputs ({e}) -- nothing to do")
        return 0

    drop, void = analyse(log, res, live_events)
    if not drop and not void:
        print("[orphans] no rows stranded under an event that does not exist")
        return 0

    orphan_events = sorted({d["event"] for d in drop} | {v["event"] for v in void})
    print(f"[orphans] {len(orphan_events)} event name(s) in predictions_log that produced no "
          f"result and sit on no tracked card:")
    for e in orphan_events:
        print(f"    {e!r}")

    print(f"\n  REMOVE -- {len(drop)} duplicate row(s); the same pair already has a row under the real name:")
    for d in drop:
        print(f"    {d['a']} vs {d['b']}   kept under {d['kept_under']!r}")
    print(f"\n  VOID + RE-POINT -- {len(void)} booking(s) that never happened:")
    for v in void:
        tag = "  (already voided)" if v["already_voided"] else ""
        print(f"    {v['a']} vs {v['b']}   picked {v['favorite']} {v['prob']}{tag}")

    if not apply:
        print("\n  DRY RUN. Re-run with --apply to write. This edits the published record, "
              "so it is the owner's call and belongs in a commit message.")
        return 0

    # Re-point the voided rows onto the card's real name, which is what the
    # rename produced -- same physical event, new headline. log_predictions
    # already re-points on rename, so this is the established operation, not
    # a new liberty with the key.
    targets = {d["kept_under"] for d in drop}
    if len(targets) != 1:
        print(f"\n  REFUSING TO APPLY: the duplicates point at {len(targets)} different real "
              f"events ({sorted(targets)}), so there is no single name to re-point the "
              f"cancelled bookings onto. Resolve by hand.")
        return 0
    real_name = targets.pop()

    for v in void:
        log.at[v["idx"], "voided"] = "true"
        log.at[v["idx"], "event_name"] = real_name
    log = log.drop(index=[d["idx"] for d in drop]).reset_index(drop=True)
    log.to_csv(LOG, index=False)
    print(f"\n  APPLIED: dropped {len(drop)} duplicate row(s); voided and re-pointed "
          f"{len(void)} cancelled booking(s) onto {real_name!r}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
