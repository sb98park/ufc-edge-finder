"""
Corrects 11 fighters flagged by scripts/audit_fighter_data.py's consistency
check (July 2026): their win_pct/loss counts didn't sum to their own
ko/sub/dec method breakdown. This is NOT a fill-empty-cells script like
manual_backfill_method_breakdown.py -- every value here deliberately
OVERWRITES an existing, wrong value. That's a more consequential kind of
edit (wins/losses feed Elo's fallback stats-prior, power ratings, and
finish-rate math elsewhere), so every correction below is logged with its
OLD value alongside the NEW one, and every number is sourced against at
least two independent corroborating sources (Wikipedia's own infobox,
cross-checked against Sherdog/Tapology/ESPN's fight-by-fight history)
before being trusted enough to overwrite real data.

ROOT CAUSE, confirmed and fixed separately in src/results_fetcher.py:
sync_fighter_records() incremented wins/losses unconoditionally but only
incremented the matching ko_wins/sub_wins/dec_wins column when the fight's
method string mapped to a known prefix -- a Disqualification result (no
entry in _METHOD_TO_PREFIX) would silently increment the total while
leaving the method breakdown one short. Confirmed responsible for 2 of the
11 corrections here (Stoltzfus, Song Yadong both had exactly this DQ gap).
That fix stops this specific cause from recurring going forward; this
script repairs the damage already sitting in the data from before the fix
existed. For the other fighters below (Gamrot, Luque, Blanchfield -- large
gaps -- and Yanez, Gibson, Ponzinibbio, Fortune, Magny -- smaller ones),
no matching code-level cause was found despite checking for the two most
likely candidates (duplicate-processing across scheduled runs, and
draw-mishandling) -- both came back clean on inspection. These may be
older data-entry inconsistencies predating the sync mechanism, not an
active bug. Said so plainly rather than claim a root cause not actually
confirmed.

Run once against the real, live data/fighters.csv:
    python3 scripts/correct_fighter_data_errors.py
"""
import pandas as pd

FIGHTERS_PATH = "data/fighters.csv"

# Fighters whose METHOD BREAKDOWN was wrong (wins/losses fields were
# already correct). Format: (ko_wins, sub_wins, dec_wins, ko_losses,
# sub_losses, dec_losses). Each verified via Wikipedia's own infobox,
# summing exactly to the fighter's real wins/losses.
METHOD_BREAKDOWN_CORRECTIONS = {
    # Large, genuinely corrupted gaps (method sum exceeded real wins by
    # 8-14) -- not simple staleness, something was actually wrong.
    "Mateusz Gamrot":    (8, 6, 12, 0, 1, 3),    # 26-4
    "Vicente Luque":     (11, 10, 3, 2, 3, 7),   # 24-12
    "Erin Blanchfield":  (2, 5, 7, 0, 0, 2),     # 14-2
    # Off-by-one gaps
    "Tyrell Fortune":    (11, 1, 6, 1, 1, 1),    # 18-3
    "Magomed Tuchalov":  (5, 1, 0, 0, 0, 0),     # 6-0 -- his last-fight win
                                                   # over Caio Machado (KO,
                                                   # backfilled earlier this
                                                   # session) was never added
                                                   # to ko_wins; this is that
                                                   # 4->5 correction.
    # Disqualification-result gaps -- root cause confirmed and fixed in
    # src/results_fetcher.py; DQ folded into the "dec" bucket for schema
    # consistency (ends by ruling, not a finish, same as a decision).
    "Dustin Stoltzfus":  (3, 6, 7, 2, 2, 4),     # 16-8 (dec_wins includes the 1 DQ win)
    "Song Yadong":       (9, 4, 10, 2, 0, 7),    # 23-9 (dec_losses includes the 1 DQ loss)
    # Caught via a real audit run + the new Combat Edge cross-check (July
    # 2026): Robert Valentin's Combat Edge win-sentence (11) disagreed with
    # our recorded wins (12) - researched his full career fight-by-fight
    # history (18 fights across Tapology/ESPN/Combat Edge, all agreeing on
    # 11-6-0 plus 1 no-contest that doesn't count either way) and confirmed
    # OUR number was the stale/wrong one, not Combat Edge's. See
    # RECORD_CORRECTIONS below for the matching wins-field fix.
    "Robert Valentin":   (3, 7, 1, 3, 1, 2),     # 11-6 (corrected from stale 12-6)
    "Tresean Gore":      (2, 4, 1, 2, 0, 2),     # 7-4, reconstructed from a real ESPN fight-by-fight table
}

# Fighters whose WINS or LOSSES field itself was wrong (their method
# breakdown was already correct and is left untouched). Format: (correct
# wins, correct losses). Verified against Wikipedia + at least one of
# Sherdog/Tapology/ESPN's own fight-by-fight history.
RECORD_CORRECTIONS = {
    # 19-6 recorded, real is 17-6-1 (includes a March 2026 majority draw
    # vs. Ricky Simon that doesn't add to either side) -- both wins AND
    # losses were off.
    "Adrian Yanez":          (17, 6),
    "Cody Gibson":           (21, 12),
    "Santiago Ponzinibbio":  (30, 9),
    "Neil Magny":            (31, 14),
    "Robert Valentin":       (11, 6),   # 12->11 wins, real record is 11-6-0 (+1 NC)
}


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
    fighters = pd.read_csv(FIGHTERS_PATH)
    changed = 0

    print("=== Method breakdown corrections ===")
    for name, (ko_w, sub_w, dec_w, ko_l, sub_l, dec_l) in METHOD_BREAKDOWN_CORRECTIONS.items():
        idx = _find_row_index(fighters, name)
        if idx is None:
            print(f"  '{name}' not found -- skipping")
            continue
        old = tuple(fighters.loc[idx, c] for c in ["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"])
        new = (ko_w, sub_w, dec_w, ko_l, sub_l, dec_l)
        for col, val in zip(["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"], new):
            fighters.loc[idx, col] = val
        print(f"  {name}: {old} -> {new}")
        changed += 1

    print()
    print("=== Wins/losses field corrections ===")
    for name, (wins, losses) in RECORD_CORRECTIONS.items():
        idx = _find_row_index(fighters, name)
        if idx is None:
            print(f"  '{name}' not found -- skipping")
            continue
        old = (fighters.loc[idx, "wins"], fighters.loc[idx, "losses"])
        fighters.loc[idx, "wins"] = wins
        fighters.loc[idx, "losses"] = losses
        print(f"  {name}: wins/losses {old} -> ({wins}, {losses})")
        changed += 1

    fighters.to_csv(FIGHTERS_PATH, index=False)
    print()
    print(f"Done -- {changed}/{len(METHOD_BREAKDOWN_CORRECTIONS) + len(RECORD_CORRECTIONS)} fighters corrected.")
    print("Re-run scripts/audit_fighter_data.py afterward to confirm all 11 are now clean.")
    print("Still need generate_site.py + commit + git pull --no-rebase + push for this to go live.")


if __name__ == "__main__":
    main()
