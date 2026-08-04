"""
Add fights to fight_history.csv by hand, when ESPN can't be reached.

WHEN TO USE THIS. fight_history.csv has three sources and each can stall:
etl_fight_history.py depends on a raw dataset that lags weeks;
merge_results_into_history.py only covers cards the site itself watched; and
backfill_history_from_espn.py needs ESPN, which throttles an IP to 403 after
a burst of calls.

That leaves real gaps with real consequences -- Quillan Salkilld's May 2026 KO
of Beneil Dariush was in none of them, so the model read a 4-fight streak
against a true 5: below the fight-fact threshold, and one step short in the
streak bonus.

UFC FIGHTS ONLY. fight_history.csv is the graph Elo is built on, and it is
UFC-only by design (see etl_fight_history.py). Adding regional or DWCS bouts
would put fighters with no shared opponents into the same rating pool and
quietly distort every rating derived from it. A fighter's pre-UFC record
already reaches the model through fighters.csv career columns.

Edit ADDITIONS below, then run. Same safety as the other merges: append only,
duplicates skipped on an unordered accent-folded name pair plus date.

Usage:
    python3 scripts/add_history_manually.py            # dry run
    python3 scripts/add_history_manually.py --apply
"""

import sys
import unicodedata

import pandas as pd

HISTORY = "data/fight_history.csv"

# date, winner, loser, method
# The winner goes first; the script writes fighter_a/fighter_b accordingly.
ADDITIONS = [
    # Populated from the gap scan: every card fighter whose most recent
    # fight was missing from history. Draws are excluded (the file is
    # decisive fights only) and two entries were dropped as same-fight
    # date drift -- Quarantillo and Magny already appear under the
    # previous day's date, so adding them would duplicate, not fill.
    # Mirrored pairs (the same bout reported from both corners) are left
    # in; the unordered dedupe key collapses them.
    ("2026-03-28", "Alexa Grasso", "Maycee Barber", ""),
    ("2026-03-28", "Alexia Thainara", "Bruna Brasil", ""),
    ("2026-03-14", "Gillian Robertson", "Amanda Lemos", ""),
    ("2026-05-30", "Cody Haddon", "Aoriqileng", ""),
    ("2026-03-28", "Navajo Stirling", "Bruno Lopes", ""),
    ("2026-05-23", "Ce Liu", "Igor Barabanov", ""),
    ("2026-04-11", "Josh Hokit", "Curtis Blaydes", ""),
    ("2026-04-25", "Eric McConico", "Rodolfo Vieira", ""),
    ("2026-04-11", "Mateusz Gamrot", "Esteban Ribovics", ""),
    ("2026-03-14", "Gillian Robertson", "Amanda Lemos", ""),
    ("2026-04-04", "Thomas Petersen", "Guilherme Pat", ""),
    ("2019-08-23", "Hugo Cunha", "Henrique da Silva Lopes", ""),
    ("2026-04-18", "JJ Aldrich", "Jamey-Lyn Horth", ""),
    ("2026-06-07", "Jessie Rosas", "Erick Ruano", ""),
    ("2026-06-06", "Iwo Baraniewski", "Junior Tafa", ""),
    ("2026-05-30", "Kai Asakura", "Cameron Smotherman", ""),
    ("2025-09-06", "Kauê Fernandes", "Harry Hardwick", ""),
    ("2026-04-11", "Vicente Luque", "Kelvin Gastelum", ""),
    ("2026-06-20", "Kevin Borjas", "Andre Lima", ""),
    ("2026-03-21", "Danny Silva", "Kurtis Campbell", ""),
    ("2026-05-02", "Louie Sutherland", "Tai Tuivasa", ""),
    ("2026-05-30", "Luis Felipe Dias", "Yi Sak Lee", ""),
    ("2026-03-14", "Manoel Sousa", "Bolaji Oki", ""),
    ("2026-03-28", "Yousri Belgaroui", "Mansur Abdul-Malik", ""),
    ("2026-03-21", "Mario Pinto", "Felipe Franco", ""),
    ("2026-04-11", "Mateusz Gamrot", "Esteban Ribovics", ""),
    ("2024-08-28", "Marco Tulio", "Matthieu Letho Duclos", ""),
    ("2026-03-21", "Michael Page", "Sam Patterson", ""),
    ("2025-12-14", "Melquizael Costa", "Morgan Charrière", ""),
    ("2026-03-14", "Myktybek Orolbai", "Chris Curtis", ""),
    ("2026-03-21", "Nathaniel Wood", "Losene Keita", ""),
    ("2026-06-27", "Nursulton Ruziboev", "Andrey Pulyaev", ""),
    ("2026-05-02", "Quillan Salkilld", "Beneil Dariush", ""),
    ("2026-03-21", "Shanelle Dyer", "Ravena Oliveira", ""),
    ("2026-05-30", "Rei Tsuruya", "Luis Gurule", ""),
    ("2026-05-22", "Richie Miranda", "Robert Varricchio", ""),
    ("2026-04-25", "Ryan Spann", "Marcus Buchecha", ""),
    ("2026-05-16", "Salahdine Parnasse", "Kenneth Cross", ""),
    ("2026-05-30", "Song Yadong", "Deiveson Figueiredo", ""),
    ("2026-03-14", "Vitor Petrino", "Steven Asplund", ""),
    ("2026-05-02", "Steve Erceg", "Tim Elliott", ""),
    ("2026-04-04", "Tresean Gore", "Azamat Bekoev", ""),
    ("2026-04-11", "Vicente Luque", "Kelvin Gastelum", ""),
    ("2026-05-09", "Alexander Volkov", "Waldo Cortes Acosta", ""),
    ("2026-03-28", "Yousri Belgaroui", "Mansur Abdul-Malik", ""),
    ("2026-06-06", "Édgar Cháirez", "Bruno Silva", ""),
]


def fold(n):
    s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def key(a, b, d):
    return (frozenset({fold(a), fold(b)}), str(d)[:10])


def main():
    apply = "--apply" in sys.argv
    hist = pd.read_csv(HISTORY)
    have = {key(r["fighter_a"], r["fighter_b"], r.get("date")) for _, r in hist.iterrows()}

    # Two different reasons to skip, reported separately -- lumping them
    # under "already present" made a mirrored pair look like it was in
    # history, which is confusing when the same fight also appears in the
    # NEW list from the other corner.
    in_history = {key(r["fighter_a"], r["fighter_b"], r.get("date"))
                  for _, r in hist.iterrows()}
    new_rows, dupes, mirrored = [], [], []
    for date, winner, loser, method in ADDITIONS:
        k = key(winner, loser, date)
        if k in in_history:
            dupes.append(f"{date} {winner} vs {loser}")
            continue
        if k in have:
            mirrored.append(f"{date} {winner} vs {loser}")
            continue
        have.add(k)
        new_rows.append({"date": date, "fighter_a": winner, "fighter_b": loser,
                         "winner": winner, "method": method})

    print(f"history rows : {len(hist)}")
    if dupes:
        print(f"already in history ({len(dupes)}):")
        for d in dupes:
            print(f"   {d}")
    if mirrored:
        print(f"same fight from the other corner, collapsed ({len(mirrored)}):")
        for d in mirrored:
            print(f"   {d}")
    print(f"NEW to add   : {len(new_rows)}")
    for r in new_rows:
        print(f"   {r['date']}  {r['fighter_a']} def. {r['fighter_b']}  [{r['method']}]")

    if not new_rows:
        print("\nNothing to add.")
        return
    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return

    out = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True).sort_values("date")
    out.to_csv(HISTORY, index=False)
    print(f"\nWritten: {len(hist)} -> {len(out)} rows.")
    print("Re-run generate_site.py -- ratings, streaks and facts all read this.")


if __name__ == "__main__":
    main()
