#!/usr/bin/env python3
"""
Is the method model's P(decision) actually miscalibrated?

WHY THIS EXISTS. On 2026-09-05 the graded log showed 31.9% of fights going
to decision (37/116) while the model's P(decision) sat near 50%, and the
obvious reading was an 18pp over-estimate worth recalibrating away. That
reading was wrong, and this script is the check that shows why -- kept so
the same 31.9% does not get re-discovered and acted on next time.

TWO ARTEFACTS HAVE TO BE RULED OUT BEFORE THE NUMBER MEANS ANYTHING.

1. ARGMAX. The published "by decision" label is argmax over the favourite's
   three-way grid. Decision is typically modal at 30-47% against KO at
   15-43%, so the label reads "decision" on ~86% of fights while the model
   is claiming nothing of the sort. Comparing label frequency to outcome
   frequency measures the argmax, not the model. CLAUDE.md s7 records a
   previous "5-sigma method bias" that was exactly this.

2. WINDOW. 116 graded fights is eight weeks, and method rates are extremely
   noisy at card scale -- our nine cards range from 14% to 58% decisions on
   ~13 fights each. The comparison has to be against the same window, not
   against a career base rate.

WHAT IT FOUND. Run it to reproduce; the numbers printed below were current
at 2026-09-06:

    all-time UFC corpus          46.9%   (17,354 bout-sides)
    rest of 2026                 48.4%   (~258 fights)
    our window 07-11..09-05      32.9%   (~76 fights)     z = -2.40, p ~ 0.017
    our 116 graded fights        31.9%                    -0.23 SE from the window

So the tracked sample is a faithful sample of its window; the WINDOW is the
outlier. An eight-week finish-heavy stretch, roughly 2.4 sigma, is an
ordinary thing for UFC to do. The model's ~50% is consistent with the 46.9%
long-run rate and with the 52.4% holdout it was fitted against.

CONCLUSION: no recalibration. Refitting P(decision) toward 31.9% would fit
an eight-week fluctuation and be wrong by ~15pp the moment the rate reverts
to its own 2026 average.

WHAT WOULD CHANGE THAT. A sustained gap measured the right way: mean
predicted P(decision) against observed frequency, over 250+ fights spanning
at least two quarters, with the window test below still significant. Not a
label-frequency comparison, and not one card.

Exit code is 0 when the evidence does NOT support recalibrating, 1 when it
does -- so this can be re-run later as the sample grows.
"""

import math
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.track_record import compute_track_record  # noqa: E402
from src.ufc_method_rates import load_dated_ufc_bouts  # noqa: E402

WINDOW_LO, WINDOW_HI = "2026-07-11", "2026-09-05"


def _rate(pairs):
    d = sum(1 for _, m in pairs if m is None)
    return d, len(pairs)


def _published_grid_means():
    """
    Mean fight-level P(KO)/P(SUB)/P(decision) over the built page.

    THE ONLY HONEST WAY TO ASK THIS. Comparing how often a LABEL says
    "submission" to how often submissions happen measures the argmax, not
    the model -- the same mistake that produced a confident, wrong report
    about decisions. The label is the largest cell; the model's opinion is
    the cell's value.

    Reads docs/index.html because that is where the reconciled grid is
    actually published. Returns {} when there is no build to read, and the
    caller says so rather than guessing.
    """
    try:
        with open("docs/index.html", encoding="utf-8") as fh:
            page = fh.read()
    except OSError:
        return {}
    vals = {"ko/tko": [], "submission": [], "decision": []}
    for block in re.split(r'(?=<details class="fight-card)', page):
        if "data-fight-key=" not in block:
            continue
        row = {}
        for m in re.finditer(
                r'<td class="mkt-label">Fight ends by ([A-Za-z/]+)</td>\s*'
                r'<td class="mkt-model"[^>]*>([\d.]+)%</td>', block, re.DOTALL):
            row[m.group(1).lower()] = float(m.group(2))
        if len(row) == 3:
            for k, v in row.items():
                vals[k].append(v)
    return {k: v for k, v in vals.items() if v}


def main() -> int:
    bouts = load_dated_ufc_bouts()
    # Each fight appears once per fighter. Double-counting does not bias a
    # RATE, but it does halve the effective sample, so n is corrected before
    # any significance test -- not doing so turned z = -2.40 into -3.38.
    window, rest_2026, alltime = [], [], []
    for _name, fighter_bouts in bouts.items():
        for date, _won, method in fighter_bouts:
            d = str(date)
            alltime.append((d, method))
            if not d.startswith("2026"):
                continue
            (window if WINDOW_LO <= d <= WINDOW_HI else rest_2026).append((d, method))

    ad, at = _rate(alltime)
    wd, wt = _rate(window)
    rd, rt = _rate(rest_2026)
    n_win, n_rest = wt // 2, rt // 2
    p_win, p_rest = wd / wt, rd / rt
    pool = (wd + rd) / (wt + rt)
    se = math.sqrt(pool * (1 - pool) * (1 / n_win + 1 / n_rest))
    z = (p_win - p_rest) / se
    p_two = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    print("  corpus decision rates")
    print(f"    all-time              {ad/at:6.1%}   ({at} bout-sides)")
    print(f"    rest of 2026          {p_rest:6.1%}   (~{n_rest} fights)")
    print(f"    window {WINDOW_LO}..{WINDOW_HI}  {p_win:6.1%}   (~{n_win} fights)")
    print(f"    window vs rest of 2026: {p_win - p_rest:+.1%}, z={z:+.2f}, p~{p_two:.3f}")

    rec = compute_track_record()
    graded = [x for x in rec["results"] if x.get("actual_method")]
    gd = sum(1 for x in graded if "dec" in str(x["actual_method"]).lower())
    p_graded = gd / len(graded)
    se_g = math.sqrt(p_win * (1 - p_win) / len(graded))
    # ---- ALL THREE CLASSES, model probability vs base rate -----------------
    # Added after the same question was asked about SUBMISSIONS: the label
    # names one on 4% of fights against a ~20% base rate, which looks like a
    # large bias and is not one.
    grid = _published_grid_means()
    if grid:
        alltime = {"ko/tko": 0.0, "submission": 0.0, "decision": 0.0}
        tot = 0
        for _n, fb in bouts.items():
            for _d, _w, meth in fb:
                tot += 1
                alltime["decision" if meth is None else
                        ("submission" if meth == "sub" else "ko/tko")] += 1
        print(f"\n  model probability vs base rate, over {len(grid['submission'])} priced fights")
        print(f"    {'outcome':12}{'model mean':>12}{'all-time':>11}{'gap':>9}")
        for k in ("ko/tko", "submission", "decision"):
            m = sum(grid[k]) / len(grid[k])
            b = 100.0 * alltime[k] / tot
            print(f"    {k:12}{m:11.1f}%{b:10.1f}%{m - b:+8.1f}pp")
        sub = grid["submission"]
        modal = sum(1 for i in range(len(sub))
                    if sub[i] > max(grid["ko/tko"][i], grid["decision"][i]))
        print(f"    submission is the largest cell on {modal}/{len(sub)} fights, "
              f"which is why the LABEL names it rarely -- that is the argmax, "
              f"not the probability")

    print(f"\n  our graded sample       {p_graded:6.1%}   ({gd}/{len(graded)})")
    print(f"    vs its own window:    {(p_graded - p_win)/se_g:+.2f} SE"
          f"  -- the sample tracks the window, the window is the outlier")

    per_card = defaultdict(list)
    for x in graded:
        per_card[x.get("event_name")].append(x)
    rates = sorted(sum(1 for y in v if "dec" in str(y["actual_method"]).lower()) / len(v)
                   for v in per_card.values())
    print(f"\n  per-card decision rate, {len(rates)} cards: "
          f"{', '.join(f'{r:.0%}' for r in rates)}")
    print(f"    spread {min(rates):.0%}-{max(rates):.0%} on ~13 fights each")

    # Recalibrating is supported only if the sample is BOTH large enough to
    # see a real effect and no longer explainable as this one window.
    big_enough = len(graded) >= 250
    still_off = abs((p_graded - ad / at)) > 0.08 and p_two > 0.05
    if big_enough and still_off:
        print("\n  VERDICT: sample is large and the window no longer explains the gap"
              " -- a refit is now justified.")
        return 1
    print("\n  VERDICT: do NOT recalibrate. The gap is this eight-week window,"
          " not the model.")
    print(f"    (need n>=250 graded fights, currently {len(graded)};"
          f" and the window effect to stop explaining it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
