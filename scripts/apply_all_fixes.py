"""
Combined one-time fix (July 2026): runs BOTH the manual method-breakdown
backfill AND the stale-fight removal in a single step, so there's only
one command to run rather than two separate ones to remember.

Run this once against the real, live data files:
    python scripts/apply_all_fixes.py

IMPORTANT: after this finishes, you still need to:
    1. python generate_site.py   (regenerates docs/index.html)
    2. commit and push           (so the live site actually updates)
This script only updates the underlying data/ CSV files - it does not
regenerate the site or push anything itself.
"""
import manual_backfill_method_breakdown
import remove_stale_fight

if __name__ == "__main__":
    print("=== Step 1: manual method-breakdown backfill ===")
    manual_backfill_method_breakdown.main()
    print()
    print("=== Step 2: removing stale Nurmagomedov vs Martinez fight ===")
    remove_stale_fight.main()
    print()
    print("=== Done with both steps. Next, run: ===")
    print("    python generate_site.py")
    print("    (then commit and push)")
