"""
The method probabilities must be frozen with the label they explain.

likely_method alone cannot answer the only question that matters about the
method model -- whether its PROBABILITIES are calibrated -- because it is
the largest cell, not the model's opinion. That gap produced two confident,
wrong reports in one week: "the model over-predicts decisions" (it says
~50% against a 46.9% base rate) and "it never predicts submissions" (16.5%
against 19.8%). Both were the argmax; neither could be checked, because the
cells were never stored.

p_ko / p_sub / p_dec close that. They are SET ONCE, like pick_odds and
pick_falsifier, because they are the claim as it was published -- and they
cannot be backfilled: regenerating them later runs the model against
ratings that already absorbed the result.

Synthetic fixtures only; nothing here reads data/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.track_record import FIELDNAMES, _frozen_method_probs  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


GRID = {"method_distribution": {"ko": 0.34, "sub": 0.16, "decision": 0.50}}

check("the three columns are in the schema",
      all(c in FIELDNAMES for c in ("p_ko", "p_sub", "p_dec")))

# --- first publication ----------------------------------------------------
f = _frozen_method_probs(None, GRID)
check("a fresh row records the grid", (f["p_ko"], f["p_sub"], f["p_dec"]) == (0.34, 0.16, 0.5))
check("the three sum to 1", abs(sum(float(f[k]) for k in f) - 1.0) < 1e-9)

# --- SET ONCE -------------------------------------------------------------
# The whole point: a later build with a moved model must not rewrite the
# claim, exactly as pick_odds is not rewritten when the price moves.
prior = {"p_ko": "0.10", "p_sub": "0.20", "p_dec": "0.70"}
later = _frozen_method_probs(prior, {"method_distribution": {"ko": 0.9, "sub": 0.05, "decision": 0.05}})
check("a logged grid survives a moved model",
      (later["p_ko"], later["p_sub"], later["p_dec"]) == ("0.10", "0.20", "0.70"))

# --- MISSING IS BLANK, NEVER ZERO ----------------------------------------
# method_model returns None on a missing feature rather than substituting a
# default. A fabricated 0.0 would read as "the model thinks this cannot
# happen" instead of "we never asked", which is the distinction these
# columns exist to preserve.
for label, preview in (("no preview", {}),
                       ("no distribution", {"method_distribution": None}),
                       ("distribution is not a dict", {"method_distribution": "nope"})):
    b = _frozen_method_probs(None, preview)
    check(f"{label} -> blanks", (b["p_ko"], b["p_sub"], b["p_dec"]) == ("", "", ""))
    check(f"{label} -> not zeros", not any(v == 0 or v == 0.0 for v in b.values()))

partial = _frozen_method_probs(None, {"method_distribution": {"ko": 0.4, "decision": 0.5}})
check("a partial grid blanks only the missing cell",
      partial["p_ko"] == 0.4 and partial["p_dec"] == 0.5 and partial["p_sub"] == "")

# A row logged before the columns existed has blanks, and must be filled on
# the next publication rather than staying blank forever -- the freeze is on
# a RECORDED value, not on the absence of one.
check("a pre-existing row with blanks still accepts a first grid",
      _frozen_method_probs({"p_ko": "", "p_sub": "", "p_dec": ""}, GRID)["p_dec"] == 0.5)

# --- idempotence ----------------------------------------------------------
once = _frozen_method_probs(None, GRID)
twice = _frozen_method_probs({k: str(v) for k, v in once.items()}, GRID)
check("re-running on its own output changes nothing",
      [float(v) for v in twice.values()] == [float(v) for v in once.values()])

print(f"{'PASS' if not fail else 'FAIL'}: test_frozen_method_probs -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
