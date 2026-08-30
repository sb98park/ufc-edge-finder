"""
Replay a real card's confirmation sequence through the live code, before a card
runs rather than during one.

WHY THIS EXISTS. Twenty-three commits landed between 2026-08-29 and 2026-08-30,
at least seven of which only change behaviour while a card is running:
results_pending flagging, bout_order as chronology, the card_is_over handover,
name-order folding, the LIVE NOW suppression, the espnLiveKey guard and the
per-card plays record. Their first real test was going to be a Saturday with
money live -- which is exactly how the last card lost live mode for hours.

THE SEQUENCE IS REAL, not invented. It is the order results actually landed on
2026-08-29, recovered from this repo's own commit history of fight_results.csv:

    07:12   +1   Liu Ce vs Levi Rodrigues Jr.        (alone, out of order)
    07:26   +8   the prelim block
    08:01   +2   Sumudaerji, Kai Asakura
    08:28   +1   Denise Gomes
    08:58   +1   the main event

That first batch is the pathology. Liu Ce sits ninth in the night but confirmed
first, because ESPN spells him "Ce Liu" and the exact-string match missed him,
so his result had to be entered by hand. One out-of-order confirmation used to
delete the eight unconfirmed fights before him.

IT IMPORTS THE REAL FUNCTIONS. generate_site.card_is_over and
schedule.apply_live_corrections are called directly rather than re-implemented;
a rehearsal that re-implements the logic drifts from it silently.

Writes nothing: SCHEDULE_STATE_PATH is redirected to a temp dir. Exits non-zero
on any failed assertion, so it can become a gate later.

Run: python3 scripts/rehearse_fight_night.py
"""

import datetime as dt
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.schedule as sch
from generate_site import card_is_over

EVENT = "UFC Fight Night: Nurmagomedov vs. Song"
EVENT_DATE = "2026-08-29"
START_ET = "07:00"

# (label, fighters confirmed by this point) -- cumulative, in the real order.
REAL_SEQUENCE = [
    ("07:12  one result, out of order", [("Liu Ce", "Levi Rodrigues Jr.")]),
    ("07:26  the prelim block", [
        ("Liu Ce", "Levi Rodrigues Jr."), ("Andre Lima", "Namsrai Batbayar"),
        ("Bilal Hasan", "Nilson Rojas"), ("Cam Nelson", "Ding Meng"),
        ("Francesco Nuzzi", "Xiao Long"), ("Hector Santiago", "Lawrence Lui"),
        ("Jingnan Xiong", "Julia Polastri"), ("Rei Tsuruya", "Kevin Borjas"),
        ("Sean Woodson", "Jack Jenkins")]),
    ("08:01  main card opens", None),      # + Sumudaerji, Kai Asakura
    ("08:28  co-main", None),              # + Denise Gomes
    ("08:58  main event", None),           # + Umar
]
_LATER = [("Sumudaerji", "Alex Perez"), ("Kai Asakura", "Aoriqileng"),
          ("Denise Gomes", "Yan Xiaonan"), ("Song Yadong", "Umar Nurmagomedov")]

FAILURES = []


def check(label, cond, detail=""):
    tag = "ok  " if cond else "FAIL"
    print(f"    [{tag}] {label}{('  -> ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


def key(a, b):
    return frozenset({a.strip().lower(), b.strip().lower()})


CARD_FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tests", "fixtures", "card_2026_08_29.csv")


def load_card():
    """
    The 2026-08-29 card, from a FROZEN FIXTURE rather than data/fight_cards.csv.

    That file holds only the current card, so this rehearsal would have read an
    empty frame the moment the next card was promoted -- every loop body would
    have iterated nothing and every check would have passed vacuously. This repo
    has already shipped exactly that failure once: test_plays_ledger asserted
    "every play is written: got 0" against an empty ledger and stayed green for
    as long as the bettable-venue rule had been live.

    A rehearsal has to keep its card.
    """
    rows = pd.read_csv(CARD_FIXTURE).to_dict("records")
    if not rows:
        raise SystemExit("rehearsal fixture is empty -- refusing to pass vacuously")
    return rows


def schedule_for(confirmed_pairs, now_hhmm):
    """Fresh state every call -- apply_live_corrections persists an anchor."""
    sch.SCHEDULE_STATE_PATH = os.path.join(tempfile.mkdtemp(), "state.json")
    sched = sch.build_fight_schedule(load_card(), EVENT_DATE, START_ET)
    finished = {key(a, b) for a, b in confirmed_pairs}
    now = dt.datetime.fromisoformat(f"{EVENT_DATE}T{now_hhmm}:00-04:00")
    kept, _ = sch.apply_live_corrections(sched, finished, now=now)
    return sched, kept


def phase_1_real_sequence():
    print("\nPHASE 1 -- the real 2026-08-29 sequence, through today's code")
    card = load_card()
    live_total = len([c for c in card
                      if str(c.get("cancelled", "")).lower() != "true"])
    cumulative = []
    times = ["07:20", "07:35", "08:05", "08:30", "09:00"]
    batches = [REAL_SEQUENCE[0][1], REAL_SEQUENCE[1][1],
               REAL_SEQUENCE[1][1] + _LATER[:2],
               REAL_SEQUENCE[1][1] + _LATER[:3],
               REAL_SEQUENCE[1][1] + _LATER]
    for (label, _), pairs, t in zip(REAL_SEQUENCE, batches, times):
        cumulative = pairs
        full, kept = schedule_for(cumulative, t)
        print(f"\n  {label}   ({len(cumulative)} confirmed)")

        # 1. NOTHING UNCONFIRMED MAY VANISH. The original bug deleted eight.
        confirmed_keys = {key(a, b) for a, b in cumulative}
        unconfirmed = [f for f in full
                       if key(f["fighter_a"], f["fighter_b"]) not in confirmed_keys]
        kept_keys = {key(f["fighter_a"], f["fighter_b"]) for f in kept}
        missing = [f for f in unconfirmed
                   if key(f["fighter_a"], f["fighter_b"]) not in kept_keys]
        check("no unconfirmed fight dropped from the schedule",
              not missing,
              f"{len(missing)} dropped: {[f['fighter_a'] for f in missing][:3]}" if missing else "")

        # 2. THE BROWSER MUST STILL BE POLLING. insideCardWindow reads the ends.
        check("schedule still spans the card (poll window intact)",
              len(kept) >= len(unconfirmed), f"{len(kept)} entries")

        # 3. THE SECTIONS MUST NOT HAND OVER MID-CARD.
        over = card_is_over(len(cumulative), live_total, 0, True)
        expected = len(cumulative) >= live_total
        check("card_is_over matches whether the card is actually finished",
              over == expected, f"{len(cumulative)}/{live_total} -> {over}")

        # 4. LIVE/NEXT MUST BE A REAL, UNCONFIRMED FIGHT.
        eligible = [f for f in kept if not f.get("results_pending")]
        if eligible:
            nxt = eligible[0]
            check("next-up is not an already-confirmed fight",
                  key(nxt["fighter_a"], nxt["fighter_b"]) not in confirmed_keys,
                  f"{nxt['fighter_a']} vs {nxt['fighter_b']}")


def phase_2_clean_sequence():
    print("\n\nPHASE 2 -- a card whose results land in order (the Sept 5 shape)")
    card = load_card()
    live = [c for c in card if str(c.get("cancelled", "")).lower() != "true"]
    total = len(live)
    ordered = sorted(live, key=lambda c: int(float(c.get("bout_order") or 0)))
    cumulative = []
    for i, c in enumerate(ordered, start=1):
        cumulative.append((c["fighter_a"], c["fighter_b"]))
        _, kept = schedule_for(cumulative, "09:00")
        over = card_is_over(len(cumulative), total, 0, True)
        if i < total:
            if over:
                check(f"after {i}/{total} the card is NOT over", False, "handed over early")
        else:
            check(f"after {i}/{total} the card IS over", over)
    stuck = [f for f in kept if f.get("results_pending")]
    check("a fully-confirmed card leaves nothing flagged results-pending", not stuck)


def phase_3_pathologies():
    print("\n\nPHASE 3 -- the cases that must not strand or misfire")
    card = load_card()
    total = len([c for c in card if str(c.get("cancelled", "")).lower() != "true"])
    check("one result never confirms: holds on Sunday",
          card_is_over(total - 1, total, 1, True) is False)
    check("one result never confirms: releases on Monday",
          card_is_over(total - 1, total, 2, True) is True)
    check("a future card is never 'over'",
          card_is_over(0, total, 0, False) is False)
    # THIS ASSERTION WAS WRONG THE FIRST TIME, and correcting it rather than
    # the code is the point. total == 0 with the card already happened means a
    # card every fight of which was cancelled -- there is nothing left to show,
    # so handing over IS correct. What must hold is that it does not hand over
    # on fight day itself, and that a future card never does.
    check("a card of all-cancelled fights hands over via the backstop",
          card_is_over(0, 0, 5, True) is True)
    check("...but not on fight day itself",
          card_is_over(0, 0, 0, True) is False)

    # bout_order must survive the rows being reordered
    rows = load_card()
    base = [(f["fighter_a"], f["fighter_b"])
            for f in sch.build_fight_schedule(rows, EVENT_DATE, START_ET)]
    shuffled = list(reversed(rows))
    after = [(f["fighter_a"], f["fighter_b"])
             for f in sch.build_fight_schedule(shuffled, EVENT_DATE, START_ET)]
    check("fight order survives the card file being reordered", base == after)


def main():
    print(f"Rehearsing {EVENT} against the current code")
    print(f"card: {len(load_card())} rows")
    phase_1_real_sequence()
    phase_2_clean_sequence()
    phase_3_pathologies()
    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed. The card-night paths behave as intended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
