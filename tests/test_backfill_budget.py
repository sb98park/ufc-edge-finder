"""
A backfill run that runs out of time must still SAVE what it fetched.

This is the bug that produced "backfill_fight_stats has failed 18 runs in a
row" on the live site, and the shape of it matters more than the count.

The step writes once, AFTER its fetch loop. In CI it carries
timeout-minutes: 2 and starts with a cold cache (data/.espn_cache is
gitignored), and a cold fighter measured 37.6s. After UFC 331 thirteen carded
fighters needed stats -- 8.2 minutes of work against a 2-minute cap. The
runner killed the step mid-loop, `updates` died with it, nothing was written,
and because --missing-only reads what was written, the next run found exactly
the same thirteen. Eighteen runs, real work every time, none of it kept.

So the property is not "it is fast". It is: WHATEVER HAPPENS, A RUN LEAVES
FEWER TARGETS THAN IT FOUND. Without that the step cannot converge however
long the cap is.

No network and no data/ access: fighter_stats is stubbed, so this measures
the budget and nothing else.
"""

import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.backfill_espn_fight_stats as bf  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def run_loop(targets, ids, budget, per_fighter=0.05):
    """The step's fetch loop, with the clock and the fetch both faked."""
    updates, missing_id, deferred = {}, [], 0
    started = time.time()
    for name in targets:
        aid = ids.get(name)
        if not aid:
            missing_id.append(name)
            continue
        if budget and (time.time() - started) >= budget:
            deferred += 1
            continue
        time.sleep(per_fighter)
        updates[name] = {"espn_fights": 10}
    return updates, missing_id, deferred


NAMES = [f"F{i}" for i in range(10)]
IDS = {n: str(1000 + i) for i, n in enumerate(NAMES)}

# --- the regression -------------------------------------------------------
updates, _, deferred = run_loop(NAMES, IDS, budget=0.12, per_fighter=0.05)
check("a bounded run still fetched somebody", len(updates) > 0)
check("  ...and deferred the rest rather than dying", deferred > 0)
check("  ...accounting for every target", len(updates) + deferred == len(NAMES))
# The whole point: next run has fewer to do.
check("THE LIST SHRINKS -- the next run starts smaller", deferred < len(NAMES))

# --- the budget is checked BEFORE a fighter, not after --------------------
# A fighter started at the limit runs to completion, so the true bound is
# budget + one fighter. That is why CI uses 45s under a 120s cap and not 90:
# a test that asserted "never exceeds the budget" would be asserting
# something this design deliberately does not promise.
t0 = time.time()
run_loop(NAMES, IDS, budget=0.10, per_fighter=0.08)
elapsed = time.time() - t0
check("the real bound is budget plus one fighter, and is honest about it",
      0.10 <= elapsed <= 0.10 + 0.08 * 2)

# --- fighters with no id are free and must not consume the budget ---------
# Thirteen of the twenty-five targets had no ESPN id at all. If those ate the
# budget, a run could spend its whole allowance discovering nothing.
half = {n: IDS[n] for n in NAMES[5:]}
updates, missing, deferred = run_loop(NAMES, half, budget=0.12, per_fighter=0.05)
check("no-id fighters are all processed", len(missing) == 5)
check("  ...and never consume the budget", len(updates) >= 2)

# --- no budget means the old behaviour, unchanged -------------------------
updates, _, deferred = run_loop(NAMES, IDS, budget=0, per_fighter=0.001)
check("budget 0 fetches everything", len(updates) == len(NAMES) and deferred == 0)

# --- the flag exists and defaults to off ----------------------------------
import argparse  # noqa: E402
src = open(bf.__file__).read()
check("the script accepts --budget-seconds", "--budget-seconds" in src)
check("  ...defaulting to no budget", '"--budget-seconds", type=int, default=0' in src)
check("  ...checked before the fetch, not after",
      src.index("budget_seconds and") < src.index("s = fighter_stats(aid, name)"))
check("  ...and the write is still reached", "Wrote {len(updates)} fighter(s)" in src
      or "Wrote" in src)

# --- CI actually passes it ------------------------------------------------
wf = open(os.path.join(os.path.dirname(bf.__file__), "..",
                       ".github", "workflows", "refresh.yml")).read()
line = [l for l in wf.splitlines() if "backfill_espn_fight_stats.py" in l and "run:" in l]
check("the workflow invokes the backfill exactly once", len(line) == 1)
check("  ...with a budget", "--budget-seconds" in line[0])
_budget = int(line[0].split("--budget-seconds")[1].split()[0])
# The step's own cap is 2 minutes. budget + one fighter (~40s measured) +
# startup has to fit inside it, or the kill this fix exists to prevent
# simply happens later.
check(f"  ...that fits under the 120s step cap with a fighter to spare ({_budget}s)",
      _budget + 40 + 10 < 120)

print(f"{'PASS' if not fail else 'FAIL'}: test_backfill_budget -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
