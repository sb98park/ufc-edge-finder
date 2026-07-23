"""
Resets combat_edge_checked=False for a specific list of fighters, so the
NEXT run of generate_site.py gives them a genuine fresh Combat Edge
attempt. This matters specifically because the eligibility check in
fighter_backfill.py (`already_checked["combat_edge"]`) only looks at
whether a check ever happened -- not which network it happened from. The
39 fighters in Section 5's STUCK list got that flag set to True back when
every attempt came from GitHub Actions' blocked cloud IPs. Since then,
we've confirmed (via the relay test) that the block really is IP-based --
which means it should NOT apply to a run from your own home network,
including a plain local `python3 generate_site.py` run, something you
already do on every push. The flag is just stale, not the underlying
situation.

This is a genuinely different, much lower-effort path than standing up a
self-hosted GitHub Actions runner (which would still be needed if you
want this to happen automatically on the 5-minute schedule -- but method-
of-victory data barely changes fighter to fight, so a single successful
local run is likely enough to permanently fill these in without needing
that at all).

Run locally (matters that it's YOUR network, not the scheduled job):
    python3 scripts/reset_combat_edge_checked.py
Then immediately run generate_site.py as usual -- that's the run that'll
actually attempt Combat Edge again for these fighters.

Read-only preview by default -- pass --apply to actually write the reset.
"""
import sys

import pandas as pd

FIGHTERS_PATH = "data/fighters.csv"

# The 39 STUCK fighters from the July 2026 audit (section 5) -- both
# combat_edge_checked and wikipedia_checked were True with method data
# still missing. Only resetting combat_edge_checked: wikipedia_checked
# should stay as-is, since Wikipedia's block (or lack of a page) is a
# separate, unrelated situation this fix doesn't touch.
STUCK_FIGHTERS = [
    "Mateusz Rębecki", "Jovan Leka", "Max Gimenis", "Hailey Cowan", "Nina Milošević",
    "Dennis Buzukja", "Bogdan Grad", "Josias Musasa", "Mark Vologdin", "Oban Elliott",
    "Michael Oliveira", "Ludovit Klein", "Gilbert Urbina", "Vlasto Čepo", "Duško Todorović",
    "Robert Valentin", "Aleksandar Rakic", "Jan Blachowicz", "Daniel Rodriguez", "Uroš Medić",
    "Steven Asplund", "Guilherme Pat", "Diego Ferreira", "Louie Sutherland", "José Montanha",
    "Bruno Lopes", "Diyar Nurgozhay", "Alexia Thainara", "Tresean Gore", "Mansur Abdul-Malik",
    "Kauê Fernandes", "Ce Liu", "Rei Tsuruya", "Kevin Borjas", "Alex Perez", "Sumudaerji",
    "Trevor Peek", "Kurtis Campbell", "Mario Pinto",
]


def _find_row_index(fighters: pd.DataFrame, name: str):
    exact = fighters.index[fighters["name"] == name]
    if len(exact) > 0:
        return exact[0]
    def key(s):
        parts = str(s).split()
        return (parts[0].lower(), parts[-1].lower()) if parts else ("", "")
    target = key(name)
    for idx, csv_name in fighters["name"].items():
        if key(csv_name) == target:
            return idx
    return None


def main():
    apply = "--apply" in sys.argv
    fighters = pd.read_csv(FIGHTERS_PATH)

    # Some of the STUCK list have since been manually backfilled with
    # real data this session (they no longer have a method-data gap at
    # all) -- skip those rather than needlessly re-flip their flag.
    method_cols = ["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]

    to_reset, already_filled, not_found = [], [], []
    for name in STUCK_FIGHTERS:
        idx = _find_row_index(fighters, name)
        if idx is None:
            not_found.append(name)
            continue
        if not fighters.loc[idx, method_cols].isna().any():
            already_filled.append(name)
            continue
        to_reset.append((idx, name))

    print(f"Would reset combat_edge_checked=False for {len(to_reset)} fighter(s):")
    for _, name in to_reset:
        print(f"  - {name}")
    if already_filled:
        print(f"\nSkipped {len(already_filled)} already filled by manual backfill this session: {already_filled}")
    if not_found:
        print(f"\nNot found in {FIGHTERS_PATH}: {not_found}")

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply to actually reset these.")
        return

    for idx, _ in to_reset:
        fighters.loc[idx, "combat_edge_checked"] = False
    fighters.to_csv(FIGHTERS_PATH, index=False)
    print(f"\nDone -- {len(to_reset)} fighter(s) reset. Now run generate_site.py locally "
          f"(not via GitHub Actions) so the retry happens from your own network.")


if __name__ == "__main__":
    main()
