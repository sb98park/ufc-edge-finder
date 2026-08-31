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

TWO LISTS, AND THE DIFFERENCE MATTERS. ADDITIONS is UFC bouts. NON_UFC is
regional ones, which carry a promotion and are excluded from the Elo graph by
elo.build_from_history.

That split replaces the old blanket "UFC fights only" rule, and the reason for
the old rule was real: adding Michael Aljarouj's 15 regional bouts with no
promotion set moved him +194.5 Elo and dragged 269 other fighters with him
through three shared opponents, 33 of them on the current roster. Elo scores a
win against the opponent's own rating, and a regional opponent with no other
results in the graph sits at the 1500 default -- so beating a French regional
flyweight scored exactly like beating an average UFC fighter.

What changed is that Elo is no longer the only consumer. A regional bout is
worthless as evidence of RELATIVE strength and decisive as evidence of
ACTIVITY: a man who fought in April 2025 is not carrying five years of ring
rust, whoever he fought. So non-UFC rows count for history coverage, layoff
and last_fight_date, and are invisible to Elo. That is the same distinction
etl_fight_history.py already draws when it excludes draws and overturned
results from the graph while still counting them toward last_fight_date.

Edit ADDITIONS below, then run. Same safety as the other merges: append only,
duplicates skipped on an unordered accent-folded name pair plus date.

Usage:
    python3 scripts/add_history_manually.py            # dry run
    python3 scripts/add_history_manually.py --apply
"""

import sys
import unicodedata

import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_matcher import fight_key   # noqa: E402

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
    # "Ce Liu" here, but the spine stores him as "Liu Ce" (CLAUDE.md s4).
    # fold() lowercases and strips accents; it does not reorder name parts,
    # so the dedupe key missed it and a re-run would have written a THIRD
    # copy of a fight already present twice. Corrected to the spine spelling.
    ("2026-05-23", "Liu Ce", "Igor Barabanov", ""),
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

# date, fighter_a, fighter_b, winner (None for a no contest), method, promotion.
# A no contest is stored with an EMPTY winner, which is now a first-class
# state everywhere downstream -- Elo skips it, track_record treats it as a
# push, and _is_settled counts it as a settled result so convergence stops
# chasing it.
NON_UFC = [
    # Michael Aljarouj, read off Tapology by the owner 2026-08-31, ahead of
    # his UFC debut on 2026-09-05. ESPN's athlete eventlog has exactly two
    # events for him -- the 2021 Brave CF loss and the upcoming card -- so
    # none of this is reachable from any source the build can call.
    # Transcription checked against Tapology's own running record column:
    # 0-0 through 13-3 with two no contests reconciles exactly.
    # Cancelled bouts (6 of them) are omitted; they never happened.
    ("2017-04-08", "Michael Aljarouj", "Charly Hoang", "Michael Aljarouj", "SUB", "100% FIGHT"),
    ("2017-05-20", "Michael Aljarouj", "Ulrich Agbessi", "Michael Aljarouj", "SUB", "100% FIGHT"),
    ("2017-07-01", "Mathieu Morciano", "Michael Aljarouj", "Mathieu Morciano", "DEC", "HFC"),
    ("2017-10-28", "Michael Aljarouj", "Ulrich Agbessi", "Michael Aljarouj", "KO/TKO", "100% FIGHT"),
    # Three opponent names are truncated by Tapology's own column width
    # ("Franck Lebouyon...", "Alexandre Guille...", "Sylvain Sommerei...").
    # Left as shown rather than guessed at: these rows exist to prove Aljarouj
    # was active, they are excluded from Elo, and none of the three appears
    # anywhere else in the spine, so the exact spelling changes nothing.
    ("2017-10-28", "Michael Aljarouj", "Franck Lebouyon", "Michael Aljarouj", "KO/TKO", "100% FIGHT"),
    ("2018-01-18", "Mickael Kanguichev", "Michael Aljarouj", "Mickael Kanguichev", "KO/TKO", "100% FIGHT"),
    ("2018-04-14", "Michael Aljarouj", "Lucas Tenório", "Michael Aljarouj", "KO/TKO", "100% FIGHT"),
    ("2019-03-09", "Michael Aljarouj", "Ronny Gomez", None, "NC", "100% FIGHT"),
    ("2019-06-29", "Michael Aljarouj", "Alexandre Guille", "Michael Aljarouj", "SUB", "100% FIGHT"),
    ("2019-12-21", "Michael Aljarouj", "Mickael Kanguichev", "Michael Aljarouj", "DEC", "100% FIGHT"),
    ("2022-10-08", "Michael Aljarouj", "Sylvain Sommerei", "Michael Aljarouj", "KO/TKO", "MMAGP"),
    ("2023-03-04", "Michael Aljarouj", "Ezzoubair Bouarsa", "Michael Aljarouj", "DEC", "MMAGP"),
    ("2023-10-21", "Michael Aljarouj", "Nuno Costa", "Michael Aljarouj", "KO/TKO", "MMAGP"),
    # 2024-01-27 promotion shown only as a logo; not identified. Recorded as
    # unknown-but-not-UFC, which is all the Elo filter needs to know.
    ("2024-01-27", "Michael Aljarouj", "Liridon Ramani", "Michael Aljarouj", "DEC", "Regional (unidentified)"),
    ("2024-06-08", "Michael Aljarouj", "Leno Rodrigo", "Michael Aljarouj", "KO/TKO", "Hexagone MMA"),
    ("2024-11-23", "Michael Aljarouj", "Pedro Nobre", "Michael Aljarouj", "KO/TKO", "Hexagone MMA"),
    ("2025-04-12", "Michael Aljarouj", "Romero dos Reis", None, "NC", "Hexagone MMA"),
]


# Bout identity comes from src/card_matcher.fight_key -- ONE definition for
# the whole spine (CLAUDE.md s4: do not add a thirteenth name fold). The local
# fold here stripped accents but not punctuation, so it would have written a
# second copy of any bout whose other spelling used a hyphen or apostrophe.
def key(a, b, d) -> tuple:
    return fight_key(a, b, d)


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
                         "winner": winner, "method": method, "promotion": ""})

    for date, fa, fb, winner, method, promotion in NON_UFC:
        k = key(fa, fb, date)
        if k in in_history:
            dupes.append(f"{date} {fa} vs {fb}")
            continue
        if k in have:
            mirrored.append(f"{date} {fa} vs {fb}")
            continue
        have.add(k)
        # An empty winner IS the no contest. Do not substitute a placeholder:
        # elo skips the row on exactly this condition, and a sentinel string
        # would be read as a phantom fighter.
        new_rows.append({"date": date, "fighter_a": fa, "fighter_b": fb,
                         "winner": winner or "", "method": method,
                         "promotion": promotion})

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
        verb = "def." if r["winner"] else "vs"
        promo = f"  ({r['promotion']})" if r.get("promotion") else ""
        print(f"   {r['date']}  {r['fighter_a']} {verb} {r['fighter_b']}  [{r['method']}]{promo}")

    if not new_rows:
        print("\nNothing to add.")
        return
    if not apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return

    if "promotion" not in hist.columns:
        # Blank on every pre-existing row, which the Elo filter reads as UFC --
        # so introducing the column cannot move a single existing rating.
        hist["promotion"] = ""
    # kind="stable": pandas defaults to quicksort, which is NOT stable, so a
    # plain sort_values("date") silently reshuffles rows WITHIN a date. Elo
    # replays row by row, so that alone moved 271 fighters (Don Frye +23.8)
    # with zero data change. Measured 2026-08-31.
    out = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True).sort_values("date", kind="stable")
    out["promotion"] = out["promotion"].fillna("")
    out.to_csv(HISTORY, index=False)
    print(f"\nWritten: {len(hist)} -> {len(out)} rows.")
    print("Re-run generate_site.py -- ratings, streaks and facts all read this.")


if __name__ == "__main__":
    main()
