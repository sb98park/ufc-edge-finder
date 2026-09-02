"""
One fighter, one identity, whatever a source calls him.

Jose Delgado was split across every file at once: two rows in fighters.csv,
4 bouts in fight_history under one spelling and 14 under the other (two
separate nodes in the Elo graph), all 22 rows of his per-bout stats under only
one, and the card pointing at the spelling with neither. His scouting drawer
rendered empty and his rating was built from 14 of 18 bouts.

Folding cannot fix this. Accents, punctuation and token order all fold; a
MIDDLE NAME does not, and matching on first+last alone would merge two
different people.
"""
import sys

import pandas as pd

sys.path.insert(0, ".")
from src.fighter_history import build_fighter_history, fold_name    # noqa: E402
from src.names import (NAME_ALIASES, _normalize_name,               # noqa: E402
                       canonical_name, fight_key)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


check("an aliased name folds to its canonical form",
      _normalize_name("Jose Miguel Delgado") == _normalize_name("Jose Delgado"))
check("canonical_name returns the stored spelling",
      canonical_name("Jose Miguel Delgado") == "Jose Delgado")
check("canonical_name is idempotent",
      canonical_name(canonical_name("Jose Miguel Delgado")) == "Jose Delgado")
check("an unaliased name is returned unchanged",
      canonical_name("Someone Entirely Else") == "Someone Entirely Else")
check("None and blank survive", canonical_name(None) is None and canonical_name("") == "")
check("an alias matches regardless of accents or punctuation",
      canonical_name("José-Miguel  Delgado") == "Jose Delgado")
check("fight_key collapses the two spellings onto one bout",
      fight_key("Jean Silva", "Jose Miguel Delgado", "2026-09-12")
      == fight_key("Jean Silva", "Jose Delgado", "2026-09-12"))
check("two genuinely different fighters are NOT merged",
      canonical_name("Rolando Delgado") == "Rolando Delgado"
      and _normalize_name("Rolando Delgado") != _normalize_name("Jose Delgado"))
check("every alias key is already folded",
      all(k == " ".join(k.split()) and k == k.lower() for k in NAME_ALIASES))

# The data itself must hold one identity, or Elo still sees two nodes.
h = pd.read_csv("data/fight_history.csv")
f = pd.read_csv("data/fighters.csv")
for variant in NAME_ALIASES.values():
    pass
stale = {v for v in NAME_ALIASES}
def _folded_col(df, col):
    return {_normalize_name(x) for x in df[col].dropna().astype(str)}

raw_names = set(h["fighter_a"].dropna().astype(str)) | set(h["fighter_b"].dropna().astype(str))
check("no spine row still carries a non-canonical spelling",
      not any(canonical_name(n) != n for n in raw_names))
check("no roster row carries a non-canonical spelling",
      not any(canonical_name(str(n)) != str(n) for n in f["name"].dropna()))
check("the roster holds one row per fighter",
      len({_normalize_name(str(n)) for n in f["name"].dropna()}) == len(f))

for c in ("data/fight_cards.csv", "data/future_cards.csv"):
    try:
        cards = pd.read_csv(c)
    except (OSError, pd.errors.EmptyDataError):
        continue
    bad = [n for col in ("fighter_a", "fighter_b")
           for n in cards[col].dropna().astype(str) if canonical_name(n) != n]
    check(f"{c} carries only canonical spellings", not bad)

# And the drawer must find him under the spelling a card would use.
hist = build_fighter_history(["Jose Miguel Delgado"])
check("the scouting drawer resolves an aliased fighter",
      len(hist.get(fold_name("Jose Miguel Delgado")) or []) > 0)

# ===== THE RAW-KEYED LOOKUPS, WHICH THE FOLD DOES NOT REACH =====
# The alias made every fold-based lookup resolve, and that was mistaken for
# the whole job. predict_matchup matches the roster by EXACT string equality
# and reads elo by EXACT dict key, and both miss silently -- an unknown name
# is an empty frame and a defaulted 1500 rating, never an error.
#
# So canonicalising the spine and the roster, without canonicalising what the
# ODDS FEED calls him, published the Noche main event at 77.4% off a defaulted
# rating where the real one says 48.7%. Synthetic fixtures on purpose: this
# must keep testing the mechanism after the live card moves on.
from src.matchup_model import predict_matchup

if NAME_ALIASES:
    alias_raw, alias_canon = next(iter(NAME_ALIASES.items()))
    other = "Zzz Testfighter"
    roster = pd.DataFrame([
        {"name": alias_canon, "weight_class": "Featherweight", "wins": 12, "losses": 2,
         "height_inches": 68.0, "reach_inches": 70.0, "slpm": 4.0, "sapm": 3.0,
         "strike_accuracy_pct": 50.0, "td_accuracy_pct": 40.0, "td_defense_pct": 60.0},
        {"name": other, "weight_class": "Featherweight", "wins": 10, "losses": 3,
         "height_inches": 69.0, "reach_inches": 71.0, "slpm": 4.2, "sapm": 3.1,
         "strike_accuracy_pct": 51.0, "td_accuracy_pct": 41.0, "td_defense_pct": 61.0},
    ])
    ratings = {alias_canon: 1680.0, other: 1713.0}   # keyed canonically, as elo builds them

    canon_side = predict_matchup(other, alias_canon, roster, ratings)
    alias_side = predict_matchup(other, alias_raw, roster, ratings)
    check("an aliased name still resolves to a matchup", alias_side is not None)
    check("the aliased spelling predicts identically to the canonical one",
          canon_side is not None and alias_side is not None
          and abs(canon_side["prob_a"] - alias_side["prob_a"]) < 1e-12)
    check("the returned matchup echoes the canonical spelling",
          alias_side is not None and alias_side["fighter_b"] == alias_canon)
    check("the aliased name does not fall back to the default rating",
          alias_side is not None and abs(alias_side["prob_a"] - 0.5) > 1e-6)

print(f"test_name_aliases: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
