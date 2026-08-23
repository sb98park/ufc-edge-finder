#!/usr/bin/env python3
"""
The published track record must never move because of a stake change.

Every figure on the site -- the headline units, the ROI, the curve, the tier
breakdown -- is priced off STAKE_SCHEDULE. If a future edit to that schedule
were applied retroactively, every one of those numbers would silently change,
and the record members have been reading would stop describing anything that
was ever actually published. That is the single worst thing this codebase
could do quietly, so it is checked rather than trusted.

Run it in CI. Exit 1 means a stake change has reached back into history.

    python3 scripts/check_stake_schedule.py
"""
import sys

sys.path.insert(0, ".")

from src import track_record as tr  # noqa: E402


# The record as published. Update these ONLY when the underlying results
# change (a new card grades, a fight is voided) -- never to accommodate a
# stake edit. If a stake edit moves them, the edit is the bug.
#
# Captured 2026-08-23, after 7 cards, immediately before Medium's stake was
# cut from 3U to 2U effective 2026-08-25.
FROZEN = {
    "eligible_count": 88,
    "total_units": 63.44,
    "roi_pct": 22.4,
    "by_tier": {
        "Lock of the Week":  {"units": 46.96, "count": 9,  "graded_at": [10.0]},
        "High Confidence":   {"units": 11.80, "count": 11, "graded_at": [5.0]},
        "Medium Confidence": {"units": -7.01, "count": 35, "graded_at": [3.0]},
        "Low Confidence":    {"units": 11.69, "count": 33, "graded_at": [1.0]},
    },
}


def main() -> int:
    record = tr.compute_track_record()
    if not record or not record.get("units_stats"):
        print("FAIL  no track record to check")
        return 1

    stats = record["units_stats"]
    failures = []

    # 1. The headline figures are exactly what was published.
    for field in ("eligible_count", "total_units", "roi_pct"):
        want, got = FROZEN[field], stats.get(field)
        status = "ok" if got == want else "CHANGED"
        print(f"  {field:16} published {want!r:>10}  now {got!r:>10}  {status}")
        if got != want:
            failures.append(f"{field}: published {want!r}, now {got!r}")

    # 2. Each tier's units and the stakes its history was GRADED at.
    #    unit_size (the forward stake) is deliberately not checked -- moving
    #    it is the whole point of a cutover. unit_sizes is the history.
    print()
    for tier, want in FROZEN["by_tier"].items():
        got = stats.get("by_tier", {}).get(tier)
        if not got:
            failures.append(f"{tier}: tier disappeared from the breakdown")
            print(f"  {tier:20} MISSING")
            continue
        graded_at = got.get("unit_sizes", [got.get("unit_size")])
        ok = (got["units"] == want["units"]
              and got["count"] == want["count"]
              and graded_at == want["graded_at"])
        print(f"  {tier:20} {got['units']:+8.2f}U over {got['count']:3}p "
              f"graded at {graded_at}  now staking {got.get('unit_size')}U  "
              f"{'ok' if ok else 'CHANGED'}")
        if got["units"] != want["units"]:
            failures.append(f"{tier} units: published {want['units']}, now {got['units']}")
        if got["count"] != want["count"]:
            failures.append(f"{tier} count: published {want['count']}, now {got['count']}")
        if graded_at != want["graded_at"]:
            failures.append(f"{tier} was graded at {want['graded_at']}, now {graded_at}")

    # 3. The schedule itself must stay newest-first, with exactly one
    #    open-ended fallback and that fallback last. A schedule out of order
    #    silently grades everything at whichever entry happens to match first.
    print()
    dates = [d for d, _ in tr.STAKE_SCHEDULE]
    if dates != sorted(dates, reverse=True):
        failures.append(f"STAKE_SCHEDULE is not newest-first: {dates}")
    if dates.count("") != 1 or dates[-1] != "":
        failures.append("STAKE_SCHEDULE needs exactly one open-ended entry, and it must be last")
    print(f"  STAKE_SCHEDULE   {dates}  "
          f"{'ok' if not any('SCHEDULE' in f for f in failures) else 'MALFORMED'}")

    print()
    if failures:
        print("FAIL  a stake change has reached back into the published record:")
        for f in failures:
            print(f"      - {f}")
        print("\n      If results genuinely changed (new card graded, fight voided),")
        print("      update FROZEN in this file. If a stake edit did this, date it")
        print("      in STAKE_SCHEDULE instead of changing history.")
        return 1

    print("PASS  the published record is untouched by the current stake ladder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
