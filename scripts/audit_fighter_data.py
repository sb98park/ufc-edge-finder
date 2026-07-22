"""
Full-roster data audit -- checks data/fighters.csv for completeness and
internal-consistency issues across EVERY tracked fighter, not just this
weekend's card. This runs against your real data (this only works run
locally; the sandbox's own copy is stale and not representative).

Checks:
  1. Missing critical display fields (reach, height, stance, method
     breakdown, last-fight data) -- same fields that show as "—" on the
     site when absent.
  2. Internal consistency: does ko_wins + sub_wins + dec_wins actually
     equal the fighter's recorded wins? Same for losses. A mismatch here
     means either the record itself or the method breakdown is wrong --
     worth knowing which fighters to double-check.
  3. Physically implausible values: height/reach outside a sane human
     range, negative win/loss counts, method counts exceeding total wins.
  4. Fighters where reach_in == height_in exactly (the still-open,
     unconfirmed question from earlier this session -- lists every case,
     not just the ones already spotted, so you can see if it's a couple
     fighters or a systemic pattern).

Run: python3 scripts/audit_fighter_data.py
Read-only -- makes no changes to any file.
"""
import pandas as pd

FIGHTERS_PATH = "data/fighters.csv"

CRITICAL_FIELDS = [
    "reach_in", "height_in", "stance", "weight_class",
    "ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses",
    "last_fight_date", "last_fight_result", "last_fight_opponent", "last_fight_method",
]

PLAUSIBLE_HEIGHT_IN = (55, 90)
PLAUSIBLE_REACH_IN = (55, 95)


def main():
    df = pd.read_csv(FIGHTERS_PATH)
    print(f"Auditing {len(df)} fighters in {FIGHTERS_PATH}\n")

    # 1. Missing critical fields
    print("=" * 70)
    print("1. MISSING CRITICAL FIELDS")
    print("=" * 70)
    missing_any = 0
    for _, r in df.iterrows():
        gaps = [c for c in CRITICAL_FIELDS if c not in df.columns or pd.isna(r.get(c))]
        if gaps:
            missing_any += 1
            print(f"  {r['name']}: missing {', '.join(gaps)}")
    if not missing_any:
        print("  none -- every tracked fighter has all critical fields filled")
    else:
        print(f"\n  {missing_any}/{len(df)} fighters have at least one gap")

    # 2. Win/loss vs. method-breakdown consistency
    print()
    print("=" * 70)
    print("2. METHOD BREAKDOWN VS. RECORD CONSISTENCY")
    print("=" * 70)
    mismatches = 0
    for _, r in df.iterrows():
        if pd.isna(r.get("ko_wins")) or pd.isna(r.get("sub_wins")) or pd.isna(r.get("dec_wins")):
            continue  # already caught by check 1, don't double-report
        method_wins = (r.get("ko_wins", 0) or 0) + (r.get("sub_wins", 0) or 0) + (r.get("dec_wins", 0) or 0)
        if pd.notna(r.get("wins")) and int(method_wins) != int(r["wins"]):
            mismatches += 1
            print(f"  {r['name']}: wins={int(r['wins'])} but ko+sub+dec wins sum to {int(method_wins)}")
        if not (pd.isna(r.get("ko_losses")) or pd.isna(r.get("sub_losses")) or pd.isna(r.get("dec_losses"))):
            method_losses = (r.get("ko_losses", 0) or 0) + (r.get("sub_losses", 0) or 0) + (r.get("dec_losses", 0) or 0)
            if pd.notna(r.get("losses")) and int(method_losses) != int(r["losses"]):
                mismatches += 1
                print(f"  {r['name']}: losses={int(r['losses'])} but ko+sub+dec losses sum to {int(method_losses)}")
    if not mismatches:
        print("  none -- every fighter's method breakdown sums correctly to their record")

    # 3. Implausible values
    print()
    print("=" * 70)
    print("3. PHYSICALLY IMPLAUSIBLE OR NEGATIVE VALUES")
    print("=" * 70)
    implausible = 0
    for _, r in df.iterrows():
        if pd.notna(r.get("height_in")) and not (PLAUSIBLE_HEIGHT_IN[0] <= r["height_in"] <= PLAUSIBLE_HEIGHT_IN[1]):
            implausible += 1
            print(f"  {r['name']}: height_in={r['height_in']} outside plausible range {PLAUSIBLE_HEIGHT_IN}")
        if pd.notna(r.get("reach_in")) and not (PLAUSIBLE_REACH_IN[0] <= r["reach_in"] <= PLAUSIBLE_REACH_IN[1]):
            implausible += 1
            print(f"  {r['name']}: reach_in={r['reach_in']} outside plausible range {PLAUSIBLE_REACH_IN}")
        for col in ("wins", "losses", "ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"):
            if pd.notna(r.get(col)) and r[col] < 0:
                implausible += 1
                print(f"  {r['name']}: {col}={r[col]} is negative")
    if not implausible:
        print("  none -- no implausible values found")

    # 4. reach_in == height_in (the open, unconfirmed question from earlier this session)
    print()
    print("=" * 70)
    print("4. REACH == HEIGHT EXACTLY (open question -- real ESPN quirk or a bug?)")
    print("=" * 70)
    equal_count = 0
    for _, r in df.iterrows():
        if pd.notna(r.get("reach_in")) and pd.notna(r.get("height_in")) and r["reach_in"] == r["height_in"]:
            equal_count += 1
            print(f"  {r['name']}: reach_in == height_in == {r['height_in']}")
    if not equal_count:
        print("  none in this dataset")
    else:
        print(f"\n  {equal_count}/{len(df)} fighters affected", end="")
        pct = equal_count / len(df) * 100
        if pct >= 15:
            print(f" ({pct:.0f}% -- high enough that this looks systemic, not isolated coincidences)")
        else:
            print(f" ({pct:.0f}%)")

    print()
    print("=" * 70)
    print("5. METHOD-DATA GAPS: STUCK (needs manual research) VS QUEUED (will resolve on its own)")
    print("=" * 70)
    method_cols = ["ko_wins", "sub_wins", "dec_wins", "ko_losses", "sub_losses", "dec_losses"]
    has_checked_cols = "combat_edge_checked" in df.columns and "wikipedia_checked" in df.columns
    if not has_checked_cols:
        print("  combat_edge_checked/wikipedia_checked columns not found in this file -- skipping.")
        print("  (Expected if this is an older snapshot predating those columns.)")
    else:
        missing_method = df[df[method_cols].isna().any(axis=1)]
        stuck, queued = [], []
        for _, r in missing_method.iterrows():
            ce_checked = bool(r.get("combat_edge_checked", False))
            wiki_checked = bool(r.get("wikipedia_checked", False))
            if ce_checked and wiki_checked:
                stuck.append(r["name"])
            else:
                queued.append(r["name"])
        print(f"  STUCK (both sources exhausted, needs manual research): {len(stuck)}")
        for n in stuck:
            print(f"    - {n}")
        print(f"\n  QUEUED (at least one source not yet tried, will resolve via future scheduled runs): {len(queued)}")
        for n in queued:
            print(f"    - {n}")
        print(f"\n  Only the {len(stuck)} STUCK fighters need manual research/backfill scripts.")
        print(f"  The other {len(queued)} should fill in on their own given more scheduled runs --")
        print(f"  re-run this audit after a few refresh cycles to see that number shrink.")

    print()
    print("=" * 70)
    print("Audit complete. This is read-only -- nothing was changed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
