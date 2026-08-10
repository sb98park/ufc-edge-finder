"""
Audit what the radar chart is ACTUALLY plotting, per fighter on the card.

WHY THIS IS NEEDED. compute_radar_metrics() coerces every missing input to
zero:

    striking_acc = float(row.get("strike_accuracy_pct") or 0)

A fighter with no striking data is therefore drawn at 0 on that axis -- which
does not read as "unknown", it reads as "the worst striker on the card". The
chart cannot distinguish absent data from a genuinely terrible number, and a
collapsed polygon is a strong visual claim to make about a fighter nobody has
data on. radar_chart.py's own docstring asserts the six axes are "all fully
populated for the whole roster"; this script tests that claim rather than
trusting it, because it stopped being true the moment a hand-added debutant
got a minimal roster row.

Also reports SCHEMA columns the radar ignores, since several inputs a betting
read would want are already sitting in fighters.csv unused.

Usage:  python3 scripts/audit_radar_coverage.py
        python3 scripts/audit_radar_coverage.py --all   # whole roster, not just the card
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Column -> the axis it feeds. Mirrors compute_radar_metrics exactly; if that
# function changes, this must change with it or the audit quietly goes stale.
AXIS_INPUTS = {
    "Knockdown Rate":        ["espn_fights", "knockdowns_per_fight"],
    "Submission Threat":     ["wins", "sub_wins"],
    "Striking Pace":         ["espn_fights", "sig_strikes_att_per_fight"],
    "Damage Resistance":     ["espn_fights", "sig_strikes_absorbed_per_fight"],
    "Submission Resistance": ["losses", "sub_losses"],
    "Distance Rate":         ["wins", "losses", "dec_wins", "dec_losses"],
}

# In fighters.csv, never plotted.
UNUSED = ["slpm", "sapm", "slpm_r", "sapm_r", "td_per_15", "td_per_15_r",
          "first_round_finish_pct", "age", "reach_in", "missed_weight_count",
          "strike_accuracy_pct", "td_accuracy_pct", "td_defense_pct",
          "knockdowns_absorbed_per_fight", "td_att_per_fight"]


def _blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "nan")


def main():
    everyone = "--all" in sys.argv
    fighters = pd.read_csv("data/fighters.csv")

    if everyone:
        names = set(fighters["name"])
        scope = "whole roster"
    else:
        cards = pd.read_csv("data/fight_cards.csv")
        names = set(cards["fighter_a"]) | set(cards["fighter_b"])
        scope = "fighters on data/fight_cards.csv"

    on = fighters[fighters["name"].isin(names)]
    print(f"Radar coverage for {len(on)} of {len(names)} {scope}")
    if len(on) < len(names):
        missing_rows = sorted(names - set(fighters["name"]))
        print(f"  NO ROSTER ROW AT ALL ({len(missing_rows)}): {missing_rows}")
        print("  -> these get no preview at all, so no radar either.")
    print()

    # Per-axis coverage across the group.
    print("AXIS COVERAGE (how many fighters have real data for each axis)")
    for axis, cols in AXIS_INPUTS.items():
        have = on.apply(lambda r: all(not _blank(r.get(c)) for c in cols), axis=1).sum()
        print(f"  {axis:<22} {have}/{len(on)}")
    print()

    # The dangerous cases: plotted as 0, indistinguishable from "bad".
    # Missing inputs now render as a polygon BREAK, not a zero, so this is a
    # completeness report rather than a bug hunt.
    print("FIGHTERS WITH BLANK AXES (rendered as a gap, not as zero)")
    any_bad = False
    for _, r in on.iterrows():
        blanks = [axis for axis, cols in AXIS_INPUTS.items()
                  if any(_blank(r.get(c)) for c in cols)]
        if blanks:
            any_bad = True
            print(f"  {r['name']:<24} -> {len(blanks)}/6 blank: {', '.join(blanks)}")
    if not any_bad:
        print("  none -- every fighter on the card has all six axes.")
    print()

    print("IN fighters.csv BUT NEVER PLOTTED")
    for c in UNUSED:
        if c in fighters.columns:
            have = (~on[c].map(_blank)).sum()
            print(f"  {c:<24} present for {have}/{len(on)} on this card")


if __name__ == "__main__":
    main()
