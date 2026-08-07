"""
Measure WHEN finishes actually happen, by scheduled length.

WHAT THIS CHECKS. src/method_model.py distributes finishes across rounds with
a hand-written shape:

    3-round   [0.40, 0.33, 0.27]
    5-round   [0.30, 0.24, 0.19, 0.15, 0.12]

Those drive every Under/Over number on the site -- P(Under X.5) is
P(finish) times the cumulative share before that mark. They were written to be
front-loaded, which is the real pattern, and never fitted. Every other part of
the method stack has holdout numbers behind it; this doesn't.

The shares only need to be RIGHT IN PROPORTION -- P(finish) comes from the
validated fight-level model, so an error here misallocates finishes between
lines without changing the total. That makes it lower stakes than the model
itself, and worth an hour rather than a week.

Reports the observed distribution, the current assumption, and the resulting
error on each betting line.

Run: python3 research_finish_timing.py
"""

import pandas as pd

import research_survival_model as R
from src.method_model import _ROUND_FINISH_SHARE, finish_share_before

RESULTS = "data/ufc_fight_results.csv"


def main():
    try:
        res = pd.read_csv(RESULTS)
    except FileNotFoundError:
        print(f"Needs {RESULTS}.")
        return
    res.columns = [c.strip() for c in res.columns]

    rnd_col = next((c for c in ("ROUND", "Round", "round") if c in res.columns), None)
    fmt_col = next((c for c in ("TIME FORMAT", "TimeFormat", "time_format") if c in res.columns), None)
    mth_col = next((c for c in ("METHOD", "Method", "method") if c in res.columns), None)
    if not rnd_col or not mth_col:
        print(f"Columns not found. Available: {list(res.columns)[:12]}")
        return

    counts = {3: {}, 5: {}}
    for r in res.to_dict("records"):
        ev = R.classify(str(r.get(mth_col, "")))
        if ev not in (R.KO, R.SUB):          # finishes only
            continue
        raw = str(r.get(fmt_col, "") or "")
        sched = 5 if raw.strip().startswith("5") else 3
        try:
            rnd = int(float(r[rnd_col]))
        except (TypeError, ValueError):
            continue
        if not 1 <= rnd <= sched:
            continue
        counts[sched][rnd] = counts[sched].get(rnd, 0) + 1

    for sched in (3, 5):
        c = counts[sched]
        n = sum(c.values())
        if n < 100:
            print(f"\n{sched}-round: only {n} finishes, too few to fit")
            continue
        observed = [c.get(i + 1, 0) / n for i in range(sched)]
        current = _ROUND_FINISH_SHARE[sched]

        print(f"\n{'='*58}")
        print(f"{sched}-ROUND FIGHTS -- {n} finishes")
        print(f"{'='*58}")
        print(f"  {'round':7}{'observed':>11}{'assumed':>10}{'gap':>9}")
        for i in range(sched):
            print(f"  R{i+1:<6}{observed[i]:10.1%}{current[i]:10.1%}{observed[i]-current[i]:+9.1%}")

        print(f"\n  effect on each line (share of finishes BEFORE the mark):")
        print(f"  {'line':8}{'observed':>11}{'assumed':>10}{'gap':>9}")
        worst = 0.0
        for k in range(sched):
            line = k + 0.5
            obs = sum(observed[:k]) + (observed[k] * 0.5 if k < len(observed) else 0)
            asm = finish_share_before(line, sched)
            worst = max(worst, abs(obs - asm))
            flag = "  <-- OFF" if abs(obs - asm) > 0.05 else ""
            print(f"  Under {line:<3}{obs:10.1%}{asm:10.1%}{obs-asm:+9.1%}{flag}")

        print()
        if worst <= 0.03:
            print(f"  Assumption holds -- worst line off by {worst:.1%}. No change needed.")
        else:
            print(f"  Worst line off by {worst:.1%}. Replace the shares with:")
            print(f"    {sched}: {[round(v, 3) for v in observed]},")
            print(f"  A gap here misallocates finishes BETWEEN lines; the total")
            print(f"  P(finish) is unaffected, so only Under/Over pricing moves.")


if __name__ == "__main__":
    main()
