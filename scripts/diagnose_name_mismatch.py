"""
Diagnostic (July 2026): finds fighters who appear on the fight card but
whose name doesn't EXACTLY match their row in fighters.csv.

The comparison table (and several model-preview lookups) find a fighter's
stats by exact-string match of the card name against fighters.csv's
'name' column. If the two differ by even one character -- a trailing
space, an accent, "Jr." vs "Jr", a curly vs straight apostrophe -- the
lookup silently fails and the table renders dashes for KO/Sub/Dec even
though the real data is sitting in fighters.csv under the other spelling.

This script only PRINTS findings. It changes nothing. Run:
    python3 scripts/diagnose_name_mismatch.py
"""
import pandas as pd
import unicodedata


def _norm(s: str) -> str:
    # accent-strip + collapse whitespace + lowercase, for loose comparison
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.split()).lower()


def _show_diff(card_name: str, csv_name: str):
    print(f"    card       : {card_name!r}")
    print(f"    fighters.csv: {csv_name!r}")
    if len(card_name) != len(csv_name):
        print(f"    (length differs: card={len(card_name)} vs csv={len(csv_name)})")
    for i, (c1, c2) in enumerate(zip(card_name, csv_name)):
        if c1 != c2:
            print(f"    first difference at position {i}: {c1!r} (U+{ord(c1):04X}) vs {c2!r} (U+{ord(c2):04X})")
            break


def main():
    fighters = pd.read_csv("data/fighters.csv")
    cards = pd.read_csv("data/fight_cards.csv")

    fighter_names = list(fighters["name"])
    exact = set(fighter_names)
    norm_index = {}
    for n in fighter_names:
        norm_index.setdefault(_norm(n), []).append(n)

    card_names = sorted(set(cards["fighter_a"]) | set(cards["fighter_b"]))

    hard_mismatch = []   # no match even loosely -- genuinely not in fighters.csv
    soft_mismatch = []   # loose match exists but exact match fails -- THE BUG

    for cn in card_names:
        if cn in exact:
            continue
        candidates = norm_index.get(_norm(cn), [])
        if candidates:
            soft_mismatch.append((cn, candidates[0]))
        else:
            hard_mismatch.append(cn)

    print("=" * 70)
    if soft_mismatch:
        print(f"NAME-MISMATCH BUG FOUND: {len(soft_mismatch)} fighter(s) are on the card")
        print("with a DIFFERENT spelling than their fighters.csv row. The exact-match")
        print("lookup fails, so their KO/Sub/Dec render as dashes despite real data")
        print("existing under the other spelling:")
        print()
        for cn, csvn in soft_mismatch:
            print(f"  {cn!r}:")
            _show_diff(cn, csvn)
            print()
    else:
        print("No soft name mismatches found -- every card name that exists in")
        print("fighters.csv matches exactly. The dashes are NOT caused by a name")
        print("mismatch; the cause is something else.")
    print("=" * 70)

    if hard_mismatch:
        print()
        print(f"Separately, {len(hard_mismatch)} card fighter(s) have NO row in fighters.csv")
        print("at all (not even a loose match) -- these genuinely aren't backfilled yet:")
        for cn in hard_mismatch:
            print(f"  {cn!r}")


if __name__ == "__main__":
    main()
