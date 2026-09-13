"""
Every path that writes a fighter name must canonicalise it first.

The alias table is only as good as the writers that consult it. Four paths
appended names straight from ESPN, and over one weekend that produced:

  fight_history   "Jean Silva vs Jose Miguel Delgado" -- froze the refresh
                  for 75 minutes on a fight night, because
                  tests/test_name_aliases is a hard gate before any data
                  mutation and the spine is append-only
  fight_results   the Noche main event logged a SECOND time as "Sean King
                  III", which card_discovery read as a replacement: it
                  cancelled the original bout and voided its prediction, a
                  correct pick removed from the published record
  fighters        a second roster row for "Sean King III" with its own 8-0
                  record, invisible to the duplicate check because that
                  compares unresolved spellings
  fight_history   seven spine bouts under the same split identity

One root cause, four doors. This test is a structural guard: it asserts the
canonicalisation is still present in each writer, so adding a fifth door
without one fails here rather than on a Saturday.

Reads source text, not behaviour, because behaviour needs ESPN.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.names import canonical_name  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# path -> what it writes, for the failure message
WRITERS = {
    "src/results_fetcher.py": "fight_results.csv",
    "src/fighter_backfill.py": "fighters.csv",
    "scripts/backfill_history_from_espn.py": "fight_history.csv",
    "scripts/merge_results_into_history.py": "fight_history.csv",
}

for rel, writes in WRITERS.items():
    src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
    check(f"{rel} imports canonical_name (writes {writes})",
          "canonical_name" in src and re.search(r"import .*canonical_name|canonical_name\s*\(", src))
    check(f"{rel} actually calls it", bool(re.search(r"canonical_name\s*\(", src)))

# --- the alias itself still resolves the cases that caused this -----------
check("Jose Miguel Delgado folds", canonical_name("Jose Miguel Delgado") == "Jose Delgado")
check("canonical_name is idempotent",
      canonical_name(canonical_name("Jose Miguel Delgado")) == "Jose Delgado")
check("an unaliased name is returned unchanged",
      canonical_name("Jessie Rosas") == "Jessie Rosas")
check("None and blank survive",
      canonical_name(None) is None and canonical_name("") == "")

# --- the data that ships must already be clean ---------------------------
# The spine is append-only, so a bad row there is permanent until edited by
# hand. This is the same invariant test_name_aliases enforces; repeated here
# against the writers' own output files so a regression names the writer.
import csv  # noqa: E402

for rel in ("data/fight_history.csv", "data/fight_results.csv",
            "data/fighters.csv", "data/fight_cards.csv"):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        continue
    bad = []
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for c in ("name", "fighter_a", "fighter_b", "winner"):
                v = r.get(c)
                if v and canonical_name(v) != v:
                    bad.append(f"{v!r}->{canonical_name(v)!r}")
    check(f"{rel} carries only canonical spellings" + (f" (found {bad[:3]})" if bad else ""),
          not bad)

print(f"{'PASS' if not fail else 'FAIL'}: test_ingest_canonicalises -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
