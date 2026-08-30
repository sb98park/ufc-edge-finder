---
name: ship-check
description: >
  Run the full verification suite before committing in this repo — build, the three
  hard CI gates, the seven test files, the linter, and the published-record
  invariant. Use before every commit, and any time you need to know whether the tree
  is releasable.
---

# Ship check

CI runs these in this order and a failure in any of them stops the site publishing.
Run all of it; the step most often skipped is the last, and it is the one that
protects the published record.

## Capture the record BEFORE you start

```bash
python3 -c "
from src.track_record import compute_track_record
r=compute_track_record(); us=r['units_stats']
print(us['total_units'], us['by_tier']['Lock of the Week']['count'], f\"{r['correct']}/{r['total']}\")"
```

Write the three numbers down. A code change must not move them.

## The suite

```bash
# 1. build -- must exit 0 and write the page
python3 generate_site.py | tail -2

# 2. the three hard gates (they run AFTER the build, BEFORE the commit step in CI,
#    so a failure freezes the site on stale data)
for g in check_free_build check_stake_schedule check_plays_ledger; do
  printf "%-24s" "$g"; python3 scripts/$g.py >/dev/null 2>&1 && echo PASS || echo FAIL; done

# 3. the seven test files -- run as PLAIN SCRIPTS, which is how CI runs them.
#    `pytest` collects nothing here.
for t in tests/test_*.py; do python3 "$t" >/dev/null 2>&1 || echo "FAILED: $t"; done

# 4. the linter (~2 min). Expect: 0 failures, 3 known market-label warnings.
python3 scripts/lint_site.py | tail -3
```

## Then re-check the record

Re-run the capture command. **The three numbers must be identical.** If they moved and
you did not deliberately intend it, stop and find out why before committing — a code
change that shifts the published record is the failure this repo most cares about.

## Notes

- `data/` will be dirty after the build. That is expected churn; `git checkout -- data/`
  before committing source. See the `data-commit` skill if the data change is the point.
- The linter is slow. Start it in the background and do other verification while it runs,
  but do not commit before reading its result.
- A gate failing for a number that legitimately moved — rather than a structural problem —
  is a bug in the gate, not a reason to bypass it. `check_stake_schedule` froze the site
  this way once already.
