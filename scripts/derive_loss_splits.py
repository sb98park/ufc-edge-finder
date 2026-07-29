"""
Fill missing KO/SUB/DEC loss splits by COUNTING them from fight history.

WHY. ko_losses / sub_losses / dec_losses come from scraping "Loss by
knockout" language off Combat Edge, and that loss-side text is simply absent
for many lower-profile fighters -- which is why prelim cards show "3-—"
where main-card fighters show "3-1". It's a source gap, not a parse bug, so
no amount of scraper work fixes it.

But the answer is already in data we hold: every fight in the results file
carries a method and a winner, so a fighter's losses by method can just be
COUNTED. That's strictly better than scraping it -- counted splits can't
disagree with the record shown beside them.

CONSERVATIVE BY DESIGN: only fills values that are MISSING. A scraped split
is left alone, because it covers a fighter's whole career including bouts
outside the UFC, while counting only sees what's in the results file. Where
counting would UNDERSTATE a career, the scraped number is the better one.

Usage:
    python3 scripts/derive_loss_splits.py            # dry run
    python3 scripts/derive_loss_splits.py --apply
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.card_matcher import _normalize_name  # noqa: E402

FIGHTERS = "data/fighters.csv"
RESULTS = next((p for p in ("data/ufc_fight_results.csv",
                            "/mnt/user-data/uploads/ufc_fight_results.csv")
                if os.path.exists(p)), None)


def bucket(method):
    m = str(method).upper()
    if "KO" in m or "TKO" in m:
        return "ko_losses"
    if "SUB" in m:
        return "sub_losses"
    if "DEC" in m:
        return "dec_losses"
    return None          # DQ, overturned, no contest -- not a method loss


def main():
    apply = "--apply" in sys.argv
    if not RESULTS:
        print("Need ufc_fight_results.csv in data/.")
        sys.exit(1)

    res = pd.read_csv(RESULTS)
    res.columns = [c.strip() for c in res.columns]

    # BOUT is "A vs. B" and OUTCOME is W/L relative to the FIRST name, so the
    # loser is whichever side OUTCOME doesn't favour.
    counts = {}
    for r in res.to_dict("records"):
        bout, outcome = str(r.get("BOUT", "")), str(r.get("OUTCOME", "")).strip()
        if " vs. " not in bout:
            continue
        a, b = [x.strip() for x in bout.split(" vs. ", 1)]
        if outcome.startswith("W"):
            loser = b
        elif outcome.startswith("L"):
            loser = a
        else:
            continue                      # draw / NC -- nobody takes a loss
        col = bucket(r.get("METHOD"))
        if not col:
            continue
        key = _normalize_name(loser)
        counts.setdefault(key, {"ko_losses": 0, "sub_losses": 0, "dec_losses": 0})
        counts[key][col] += 1

    fighters = pd.read_csv(FIGHTERS)
    for c in ("ko_losses", "sub_losses", "dec_losses"):
        if c not in fighters.columns:
            fighters[c] = pd.NA

    filled = {c: 0 for c in ("ko_losses", "sub_losses", "dec_losses")}
    rows_touched, examples = 0, []
    for i, row in fighters.iterrows():
        key = _normalize_name(str(row["name"]))
        got = counts.get(key)
        if not got:
            continue
        touched = False
        for c in filled:
            if pd.isna(row.get(c)):
                fighters.at[i, c] = got[c]
                filled[c] += 1
                touched = True
        if touched:
            rows_touched += 1
            if len(examples) < 6:
                examples.append(f"{row['name']}: KO {got['ko_losses']} / "
                                f"SUB {got['sub_losses']} / DEC {got['dec_losses']}")

    print(f"fighters in results file with countable losses: {len(counts)}")
    print(f"fighters.csv rows updated: {rows_touched}")
    for c, n in filled.items():
        print(f"   {c:12} filled {n}")
    if examples:
        print("\nexamples:")
        for e in examples:
            print("  ", e)
    if not rows_touched:
        print("\nNothing missing -- every fighter already has loss splits.")

    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return
    fighters.to_csv(FIGHTERS, index=False)
    print("\nWritten. Commit data/fighters.csv.")


if __name__ == "__main__":
    main()
