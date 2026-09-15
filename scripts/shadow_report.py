#!/usr/bin/env python3
"""
What the paper record says so far, and whether it is enough to act on.

card_plays.DISCRETIONARY_PLAYS is "a pause, not a verdict -- turn it back on
when the discretionary moneylines have enough of a record to argue with."
This prints that record and answers the second half of the sentence, so the
decision is a reading rather than a feeling.

THE BAR IS DELIBERATELY EXPLICIT. A tier is reported as decidable only when a
bootstrap interval over its own settled rows excludes zero. That is a low bar
for confidence and a high one for patience: at roughly two discretionary
plays a card it takes months, which is the honest cost of not betting on
noise. The counterfactual that motivated the ledger sat at n=17 with a 95%
interval of [-41%, +102%] on the one tier that looked positive.

Nothing here is staked. See src/shadow_ledger.

    python3 scripts/shadow_report.py
"""

import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shadow_ledger import load, summarise  # noqa: E402

MIN_SETTLED = 30          # below this a bootstrap interval is theatre


def bootstrap_roi(units, staked, n=20000, seed=17):
    if not units or staked <= 0:
        return None
    rng = random.Random(seed)
    out = [sum(units[rng.randrange(len(units))] for _ in units) / staked * 100
           for _ in range(n)]
    out.sort()
    return out[int(0.025 * n)], out[int(0.975 * n)]


def main():
    rows = load()
    if not rows:
        print("No paper rows yet. The ledger starts filling on the next card "
              "that offers a discretionary candidate.")
        return 0

    settled = [r for r in rows if r.get("result") in ("won", "lost")]
    s = summarise(rows)
    print(f"PAPER RECORD -- not staked, not in the published numbers\n")
    print(f"  rows on file   {len(rows)}")
    print(f"  settled        {len(settled)}")
    print(f"  notional       {s['units']:+.2f}U on {s['staked']:.1f}U staked"
          + (f"  (ROI {s['roi_pct']:+.1f}%)" if s.get("roi_pct") is not None else ""))
    print(f"  events         {len({r['event_name'] for r in settled})}")

    if not settled:
        print("\nNothing has settled yet -- no read is available.")
        return 0

    groups = collections.defaultdict(list)
    for r in settled:
        groups[r.get("tier") or "Prop"].append(r)

    print(f"\n{'tier':<20} {'n':>4} {'staked':>8} {'units':>9} {'ROI':>8}  {'95% CI':>20}  verdict")
    for tier, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        units = [float(r["units_result"]) for r in g]
        staked = sum(float(r["units"]) for r in g)
        total = sum(units)
        roi = total / staked * 100 if staked else 0.0
        ci = bootstrap_roi(units, staked)
        if len(g) < MIN_SETTLED:
            verdict = f"too few (need {MIN_SETTLED})"
            ci_txt = "--"
        else:
            lo, hi = ci
            ci_txt = f"[{lo:+.0f}%, {hi:+.0f}%]"
            verdict = ("WORTH STAKING" if lo > 0 else
                       "clearly negative" if hi < 0 else "still undecidable")
        print(f"{tier:<20} {len(g):>4} {staked:>7.1f}U {total:>+8.2f}U {roi:>+7.1f}%  "
              f"{ci_txt:>20}  {verdict}")

    ready = [t for t, g in groups.items() if len(g) >= MIN_SETTLED]
    print()
    if not ready:
        need = MIN_SETTLED - max(len(g) for g in groups.values())
        print(f"No tier has {MIN_SETTLED} settled paper plays yet -- "
              f"the nearest needs {need} more. DISCRETIONARY_PLAYS stays off.")
    else:
        print(f"Tiers with enough settled rows to read: {', '.join(sorted(ready))}.")
        print("A 'WORTH STAKING' verdict is the argument for flipping "
              "DISCRETIONARY_PLAYS, not the act of flipping it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
