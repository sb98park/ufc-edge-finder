"""
Set the display order of a card by hand.

Needed because the fights ESPN cannot order for us are exactly the ones
that need ordering: a cancelled bout ESPN has dropped, and a hand-added
bout ESPN has not published. resync_tracked_card_order re-inserts pinned
rows next to the row they previously followed, so an order set here is
preserved across refreshes rather than being pushed back to the end.

THE FILE RUNS MAIN EVENT FIRST, so prelims are stored in REVERSE
chronological order -- the card opener is the LAST row, not the first.
Edit DESIRED below to match the file order you want, then run it.
Rows not named are kept and appended rather than dropped.

Usage:  python3 scripts/reorder_card.py
"""

import unicodedata
import pandas as pd

DESIRED = [
    ("Mateusz Gamrot", "Quillan Salkilld"),
    ("Diego Ferreira", "Billy Quarantillo"),
    ("Darren Elkins", "Yadier del Valle"),
    ("Amanda Lemos", "Alexia Thainara"),
    ("Ty Miller", "Billy Ray Goff"),
    ("Steven Asplund", "Guilherme Pat"),
    ("Diyar Nurgozhay", "Bruno Lopes"),
    ("Louie Sutherland", "José Montanha"),
    ("Manoel Sousa", "Richie Miranda"),
    ("Miles Johns", "Gianni Vazquez"),
    ("Miles Johns", "Jessie Rosas"),
    ("Juliana Miller", "Ravena Oliveira"),
    ("Gigi Canuto", "Carol Foro"),
]

def fold(v):
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())

d = pd.read_csv("data/fight_cards.csv")
keys = d.apply(lambda r: frozenset({fold(r["fighter_a"]), fold(r["fighter_b"])}), axis=1)
order, seen = [], set()
for a, b in DESIRED:
    want = frozenset({fold(a), fold(b)})
    hit = [i for i in d.index if keys[i] == want and i not in seen]
    if not hit:
        print(f"  NOT FOUND, skipping: {a} vs {b}")
        continue
    order.append(hit[0]); seen.add(hit[0])
# Anything not named keeps its relative position at the end rather than
# being dropped -- a silent deletion here would remove a real fight.
leftover = [i for i in d.index if i not in seen]
if leftover:
    print(f"  {len(leftover)} row(s) not named in DESIRED, appended in existing order")
d.loc[order + leftover].to_csv("data/fight_cards.csv", index=False)
print("\nNew order:")
for _, r in pd.read_csv("data/fight_cards.csv").iterrows():
    print(f"  {r['card_position']:<15} {r['fighter_a']} vs {r['fighter_b']}")
