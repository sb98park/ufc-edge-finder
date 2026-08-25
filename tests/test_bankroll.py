"""
The bankroll, and the two things it must never do.

A unit has always meant "1% of bankroll" while the bankroll was a number
nobody tracked -- so every figure on the site was a FLAT-STAKE result. Same
picks, same unit sizes, replayed over the graded record: +55.9% flat against
+70.1% compounded. The accounting was costing more than most rule changes
would gain.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bankroll import (  # noqa: E402
    load, apply_settled, summarise, STARTING_MULTIPLE,
)
from src.plays import UNIT_AS_BANKROLL_PCT  # noqa: E402

FAILURES = []


def check(label, got, want, tol=1e-6):
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:58s} got {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def row(pid, result, units, price, at):
    return {"play_id": pid, "result": result, "units": units,
            "odds_american": price, "graded_at": at}


fresh = load("/nonexistent-so-this-is-a-fresh-start")
check("a fresh bankroll is exactly its starting multiple",
      fresh["multiple"], STARTING_MULTIPLE)

print("\none unit is one per cent, of what there is NOW")
s = apply_settled(fresh, [row("a", "lost", 10.0, -300, "1")])
check("a 10U loss costs exactly 10%", s["multiple"], 0.90)
s2 = apply_settled(s, [row("b", "lost", 10.0, -300, "2")])
# 0.90 * 0.90, not 1.00 - 0.20. The second loss is smaller in cash because the
# bankroll it is sized against is smaller. That is the whole mechanism.
check("  ...so a second 10U loss costs less than the first",
      s2["multiple"], 0.81)
check("  ...which flat staking would have called 0.80",
      round(1 - 2 * 10 * UNIT_AS_BANKROLL_PCT, 2), 0.80)

print("\nSETTLING TWICE IS THE FAILURE THAT WOULD NOT LOOK LIKE ONE")
# This runs on every build. A bankroll that compounded the same win twice
# would be fiction inside a week, and would look plausible the whole time.
again = apply_settled(s2, [row("a", "lost", 10.0, -300, "1"),
                           row("b", "lost", 10.0, -300, "2")])
check("replaying settled plays moves nothing", again["multiple"], s2["multiple"])
check("  ...and does not double-count them", len(again["settled"]), 2)

print("\nungraded and void plays do not move it")
s3 = apply_settled(fresh, [row("c", "", 5.0, -190, ""),
                           row("d", "void", 5.0, -190, "9")])
check("neither an ungraded nor a voided play settles",
      s3["multiple"], STARTING_MULTIPLE)
check("  ...and neither is recorded as settled", len(s3["settled"]), 0)

print("\nwhat the sequence does and does not change")
# The module docstring claimed order changed the multiple. It does not: the
# multiple is a product of per-bet factors and multiplication commutes. What
# the order changes is the PATH -- the peak, and the drawdown from it.
win, loss = row("w", "won", 10.0, 100, "1"), row("l", "lost", 10.0, 100, "2")
a = apply_settled(load("/nonexistent"), [win, loss])
b = apply_settled(load("/nonexistent"), [dict(loss, graded_at="1"), dict(win, graded_at="2")])
check("win-then-loss and loss-then-win land on the same multiple",
      a["multiple"], b["multiple"])
check("  ...and it is below evens, which is the cost of variance",
      a["multiple"] < 1.0, True)
# The peak is where the two paths genuinely part: winning first sets a high
# water mark of 1.10, losing first never rises above the 1.0 it started at.
# (Drawdown happens to match here -- both are 10% -- because a fall from 1.10
# to 0.99 and a fall from 1.00 to 0.90 are the same proportion. The peak is
# the honest discriminator, and it is also the number the next drawdown will
# be measured against.)
check("winning first sets a higher water mark", a["peak"] > b["peak"], True)
check("  ...at exactly the post-win multiple", a["peak"], 1.1)
check("  ...while losing first never rises above its start", b["peak"], 1.0)

print("\nthe drawdown is measured from the peak, and never forgets")
seq = [row("p1", "won", 10.0, 100, "1"), row("p2", "lost", 10.0, 100, "2"),
       row("p3", "won", 10.0, 100, "3")]
s4 = apply_settled(load("/nonexistent"), seq)
check("recovered above 1.0", s4["multiple"] > 1.0, True)
check("  ...and the drawdown it took to get there is still on the record",
      s4["max_drawdown_pct"] > 0, True)

print("\nwhat the page prints")
out = summarise(s4)
check("growth is against the start, not the peak",
      out["growth_pct"], round((s4["multiple"] - 1) * 100, 1))
check("and the settled count is the number of graded plays", out["settled"], 3)

print("\n" + ("-" * 68))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print("the bankroll holds")
