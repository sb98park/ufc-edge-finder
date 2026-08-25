"""
The paywall's closed set, and the failure mode it exists to prevent.

WHAT WENT WRONG. favorite_picks was never added to MEMBER_ONLY_CONTEXT, so the
free payload shipped the model's picks, their probabilities and their edges for
as long as that section existed. scripts/check_free_build.py -- the hard gate
whose entire job is catching exactly this -- passed green on every one of those
builds, because it asserts that values the redaction REMOVED are absent, and a
key that was never redacted removes nothing and asserts nothing.

So the gate could catch a redaction that had broken and was structurally blind
to one that was never written. These tests pin the fix: every render-context
key must be classified, and an unclassified one fails loudly.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import tiering  # noqa: E402

FAILURES = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:60s} got {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


print("\nthe two lists are a partition, not two overlapping opinions")
overlap = set(tiering.MEMBER_ONLY_CONTEXT) & set(tiering.FREE_CONTEXT)
check("no key is both free and member-only", sorted(overlap), [])
check("no key is listed twice as member-only",
      len(tiering.MEMBER_ONLY_CONTEXT), len(set(tiering.MEMBER_ONLY_CONTEXT)))
check("no key is listed twice as free",
      len(tiering.FREE_CONTEXT), len(set(tiering.FREE_CONTEXT)))

print("\nAN UNCLASSIFIED KEY FAILS THE BUILD")
# The whole point. A new context key carrying next Saturday's picks must not be
# able to reach a free payload just because nobody remembered this file.
base = {k: [] for k in tiering.MEMBER_ONLY_CONTEXT}
base.update({k: [] for k in tiering.FREE_CONTEXT})
try:
    tiering.redact_context(dict(base))
    check("a fully classified context redacts cleanly", True, True)
except RuntimeError as exc:
    check(f"a fully classified context redacts cleanly ({exc})", False, True)

leaky = dict(base, tomorrows_picks=[{"fighter": "A", "model_prob": 0.91}])
try:
    tiering.redact_context(leaky)
    check("an unknown key raises", False, True)
except RuntimeError as exc:
    check("an unknown key raises", True, True)
    check("  ...and names it", "tomorrows_picks" in str(exc), True)
    check("  ...and says what to do about it",
          "MEMBER_ONLY_CONTEXT" in str(exc) and "FREE_CONTEXT" in str(exc), True)

print("\nthe second leak: one member-only limb on an otherwise free key")
# whats_new_snapshot's `movements` is market data and belongs to everyone; its
# `standout` list is built from standout_props BEFORE redaction and was
# shipping the five clearest reads with the edge attached.
ctx = dict(base, whats_new_snapshot={
    "standout": [{"key": "Rei Tsuruya|Moneyline", "label": "Rei Tsuruya Moneyline",
                  "edge_pct": -6.05}],
    "movements": [{"key": "X|Moneyline", "label": "X Moneyline", "pct_change": 41.6}],
})
out, removed, assertable = tiering.redact_context(ctx)
check("the model limb is emptied", out["whats_new_snapshot"]["standout"], [])
check("  ...and the market limb survives",
      len(out["whats_new_snapshot"]["movements"]), 1)
check("  ...and the removed values reach the leak manifest",
      any("Rei Tsuruya" in v for v in removed), True)

print("\nand the keys that were actually leaking are now member-only")
for key in ("favorite_picks", "standout_props", "lock_picks", "plays_rows"):
    check(f"{key} is behind the wall", key in tiering.MEMBER_ONLY_CONTEXT, True)

print("\n" + ("-" * 72))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print(f"the closed set holds -- {len(tiering.MEMBER_ONLY_CONTEXT)} member-only, "
      f"{len(tiering.FREE_CONTEXT)} free, 0 unclassified")
