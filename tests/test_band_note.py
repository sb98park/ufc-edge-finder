"""
"Picks at this confidence are W-L" must describe the confidence on the row.

It did not. The band was a separate partition of the probability --
coinflip/lean/solid/strong cut at 55/65/75 -- and it rounded before comparing.
Umar Nurmagomedov at 0.747 was labelled Medium (0.747 < 0.75, read raw) and
banded "strong" (round(74.7) = 75), so a Medium pick that LOST was counted
against the High cohort and carried its note: "Picks at this confidence are
20-2" printed under a row labelled Medium. Without him it is 20-1.
"""
import re
import sys

sys.path.insert(0, ".")
from src.track_record import (MIN_BAND_RECORD, _band_of,          # noqa: E402
                             compute_track_record)

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


# The band IS the displayed tier, not a re-derivation from the probability.
check("the band is read off the row's label",
      _band_of({"confidence_label": "Medium Confidence", "favorite_prob": 0.747})
      == "Medium Confidence")
check("a 0.747 pick is NOT banded with the 0.75+ cohort",
      _band_of({"confidence_label": "Medium Confidence", "favorite_prob": 0.747})
      != "High Confidence")
check("a missing label yields no band", _band_of({"favorite_prob": 0.9}) is None)
check("a blank label yields no band",
      _band_of({"confidence_label": "  ", "favorite_prob": 0.9}) is None)
# A tier carries hysteresis and card-level monotonicity, so it cannot be
# re-derived from the probability -- two picks at the same number may sit in
# different tiers, and the band must follow the label either way.
check("two picks at one probability follow their own labels",
      _band_of({"confidence_label": "High Confidence", "favorite_prob": 0.744})
      != _band_of({"confidence_label": "Medium Confidence", "favorite_prob": 0.744}))

r = compute_track_record()
if not r:
    print("test_band_note: no record to check")
    sys.exit(0)

rows = [m for m in r["results"] if m.get("band_note")]
check("some rows carry a band note", len(rows) > 0)

# Every note must state the record of the cohort the row is actually in.
by_tier = {}
for m in r["results"]:
    lbl = m.get("confidence_label")
    if lbl:
        w, n = by_tier.get(lbl, (0, 0))
        by_tier[lbl] = (w + (1 if m["correct"] else 0), n + 1)

mismatched = 0
for m in rows:
    w, n = by_tier[m["confidence_label"]]
    if f"are {w}-{n - w} this season" not in m["band_note"]:
        mismatched += 1
check("every band note matches its own tier's record", mismatched == 0)

# The cohorts must partition the record, or some pick is counted twice or not
# at all.
tot_w = sum(w for w, _ in by_tier.values())
tot_n = sum(n for _, n in by_tier.values())
check(f"the tiers partition the record ({tot_w}/{tot_n} vs {r['correct']}/{r['total']})",
      (tot_w, tot_n) == (r["correct"], r["total"]))

# A thin cohort must stay silent rather than publish a record of three fights.
for lbl, (w, n) in by_tier.items():
    has_note = any(m.get("band_note") for m in r["results"]
                   if m.get("confidence_label") == lbl)
    if n < MIN_BAND_RECORD:
        check(f"{lbl} is too thin ({n}) to publish a record", not has_note)

# The specific regression.
umar = [m for m in r["results"] if "Umar" in str(m.get("predicted_favorite"))]
if umar:
    u = umar[0]
    check("Umar is banded Medium, where his label puts him",
          u["confidence_label"] == "Medium Confidence"
          and _band_of(u) == "Medium Confidence")
    hw, hn = by_tier.get("High Confidence", (0, 0))
    check(f"the High cohort no longer carries his loss ({hw}-{hn - hw})",
          u["correct"] is False and hn > 0 and (hw, hn - hw) != (20, 2))

print(f"test_band_note: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
