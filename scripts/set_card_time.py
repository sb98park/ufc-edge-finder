"""
Set a card's start time by hand.

WHY THIS EXISTS. event_start_time_et is normally derived from ESPN's own
competition times, but falls back to 19:00 (US primetime) when ESPN hasn't
published them yet. Everything downstream is computed FROM that value --
src/schedule.py lays out prelims and main card as offsets from it -- so a
wrong start time quietly mis-states the countdown, the "Main Card at X"
banner line, and every per-fight estimate.

That default is wrong for every INTERNATIONAL card. A European or Middle
Eastern event runs early US time: prelims 10:00 ET, main card 13:00 ET.
Left at the default it renders as a 21:00 main card -- eight hours out.

card_discovery.py now refreshes the stored time whenever ESPN offers a real
one, so this self-corrects going forward. But if ESPN still has no times for
an already-stored card, nothing can correct it automatically, and that's
what this script is for.

NOTE the convention: event_start_time_et is the PRELIMS start, not the main
card. src/schedule.py derives the main card from it. So for a card with
10:00 prelims and 13:00 main card, pass 10:00.

Usage:
    python3 scripts/set_card_time.py "Medic" 10:00              # prelims only, dry run
    python3 scripts/set_card_time.py "Medic" 10:00 13:00        # prelims + main card
    python3 scripts/set_card_time.py "Medic" 10:00 13:00 --apply

Pass the MAIN CARD time as a second argument whenever you know it. Without it
the main card is only ESTIMATED as a fixed offset from prelims, which assumes
a constant-length prelim block -- and that doesn't hold (a US card runs
19:00 -> 21:00, a short international one 10:00 -> 13:00).
"""

import re
import sys

import pandas as pd

FILES = ["data/fight_cards.csv", "data/future_cards.csv"]


def main():
    args = [a for a in sys.argv[1:] if a != "--apply"]
    apply = "--apply" in sys.argv
    if len(args) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    needle, new_time = args[0].lower(), args[1]
    main_time = args[2] if len(args) == 3 else None

    if not re.fullmatch(r"\d{1,2}:\d{2}", new_time):
        print(f"'{new_time}' isn't HH:MM (24-hour, Eastern). Example: 10:00")
        sys.exit(1)
    hh, mm = (int(x) for x in new_time.split(":"))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        print(f"'{new_time}' isn't a real time.")
        sys.exit(1)
    new_time = f"{hh:02d}:{mm:02d}"
    if main_time is not None:
        if not re.fullmatch(r"\d{1,2}:\d{2}", main_time):
            print(f"'{main_time}' isn't HH:MM (24-hour, Eastern).")
            sys.exit(1)
        mh, mmn = (int(x) for x in main_time.split(":"))
        if not (0 <= mh <= 23 and 0 <= mmn <= 59):
            print(f"'{main_time}' isn't a real time.")
            sys.exit(1)
        main_time = f"{mh:02d}:{mmn:02d}"

    touched = False
    for path in FILES:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if df.empty or "event_name" not in df.columns:
            continue
        mask = df["event_name"].astype(str).str.lower().str.contains(needle, regex=False)
        if not mask.any():
            continue
        touched = True
        for name in df.loc[mask, "event_name"].unique():
            old = df.loc[df["event_name"] == name, "event_start_time_et"].dropna().unique()
            print(f"{path}: {name}")
            print(f"   prelims start {list(old) or ['(unset)']} -> {new_time}")
            # Main card is derived downstream; show it so the change is checkable
            if main_time:
                print(f"   main card {main_time} ET (published, exact)")
            else:
                print(f"   main card ESTIMATED at {(hh + 2) % 24:02d}:{mm:02d} ET "
                      f"— pass it as a 3rd argument to set it exactly")
        if apply:
            df.loc[mask, "event_start_time_et"] = new_time
            if main_time:
                if "event_main_card_time_et" not in df.columns:
                    df["event_main_card_time_et"] = ""
                df.loc[mask, "event_main_card_time_et"] = main_time
            df.to_csv(path, index=False)

    if not touched:
        print(f"No event matching {args[0]!r} found in fight_cards.csv or future_cards.csv.")
        sys.exit(1)
    if apply:
        print("\nWritten. Run generate_site.py, then commit the data file(s) that changed.")
    else:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
