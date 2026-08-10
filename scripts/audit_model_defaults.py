"""
Which model terms are firing on real data, and which on defaults.

Every term below reads a fighters.csv column through a hardcoded fallback.
When BOTH corners are missing the column the fallbacks cancel and the term is
merely inert. When only ONE corner is missing, the model compares a real
number against an invented one and produces an edge that reflects who has a
data file rather than anything about the fight -- and it always favours the
fighter with more history.

This reports both failure modes per column, for the current card.
"""
import pandas as pd

# column -> (term it feeds, hardcoded default in matchup_model)
TERMS = {
    "strike_accuracy_pct": ("striking (GATED)", 45),
    "td_accuracy_pct":     ("wrestling fallback (GATED)", 20),
    "td_defense_pct":      ("wrestling fallback (GATED)", 65),
    "td_per_15":           ("wrestling, primary path", None),
    "slpm":                ("striking volume", None),
    "sapm":                ("striking volume", None),
    "height_in":           ("height", 70),
    "sub_wins":            ("submission threat", 0),
    "ko_losses":           ("durability", 0),
    "sub_losses":          ("durability", 0),
    "age":                 ("age cliff", None),
    "missed_weight_count": ("missed weight", None),
    "last_fight_date":     ("layoff / quick return", None),
}

def blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "nan")

fighters = pd.read_csv("data/fighters.csv").set_index("name")
cards = pd.read_csv("data/fight_cards.csv")

print(f"{'column':<22} {'term':<30} {'both blank':>11} {'ONE blank':>10}")
print("-" * 78)
for col, (term, default) in TERMS.items():
    if col not in fighters.columns:
        print(f"{col:<22} {term:<30} {'COLUMN MISSING':>22}")
        continue
    both = one = 0
    for _, r in cards.iterrows():
        try:
            a = fighters.loc[r["fighter_a"], col]
            b = fighters.loc[r["fighter_b"], col]
        except KeyError:
            continue
        ba, bb = blank(a), blank(b)
        if ba and bb:
            both += 1
        elif ba or bb:
            one += 1
    # GATED columns are safe when asymmetric -- matchup_model now requires
    # both corners to have real data or the term contributes nothing, so a
    # one-blank fight there produces zero, not a phantom edge. Only ungated
    # columns still manufacture one.
    gated = "GATED" in term
    flag = ("  <-- asymmetric, but GATED (contributes 0, safe)" if one and gated
            else "  <-- ASYMMETRIC: phantom edge" if one else "")
    print(f"{col:<22} {term:<30} {both:>11} {one:>10}{flag}")

print("\n'both blank' = term contributes nothing (harmless).")
print("'ONE blank'  = real number vs invented default -> phantom edge.")
