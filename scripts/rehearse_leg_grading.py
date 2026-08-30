"""
Prove the two settlement implementations still agree.

WHY THIS EXISTS. Slip settlement is written TWICE: once in JavaScript in
templates/site.html (gradeCondition / gradeLeg / slipState, which run live in the
browser during a card) and once in Python in src/parlay_grader.py (which settles
the ledger afterwards). parlay_grader's own docstring says the second was written
from the first. Two implementations of the same rules drift, and the drift is
silent -- the reader sees one answer during the card and a different one in the
record.

AND THEY DISAGREE ABOUT THE CLOCK BY DESIGN. ESPN's scoreboard counts DOWN inside
a round, so the JS computes elapsed as (300 - remaining). fight_results.csv stores
ELAPSED. Both are five characters of mm:ss and neither announces which it is. Feed
one convention to the other and every Over/Under rounds leg silently inverts --
the docstring calls this out as the one thing that would flip results rather than
fail loudly, and rounds legs have been the entire content of pinned slips.

So each case below is defined ONCE, semantically (round, seconds elapsed), and
rendered into both representations. If anyone ever "fixes" one side's convention
to match the other, this fails.

THE JS IS SLICED OUT OF THE TEMPLATE, not copied here. A copy would be a third
implementation and would drift from both.

Run: python3 scripts/rehearse_leg_grading.py     (requires node)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import parlay_grader as pg

TEMPLATE = "templates/site.html"
FAILURES = []


def slice_js() -> str:
    """Lift the grading functions out of the shipped template."""
    src = open(TEMPLATE, encoding="utf-8").read()
    wanted = []
    for name in ("canonicalKey", "gradeCondition", "gradeLeg", "slipState"):
        m = re.search(r"\n(\s*)function " + name + r"\s*\(", src)
        if not m:
            raise SystemExit(f"could not find function {name} in {TEMPLATE}")
        start = m.start() + 1
        i = src.index("{", m.end() - 1)
        depth, j = 0, i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        wanted.append(src[start:j + 1])
    return "const T = true, F = false, U = null;\n" + "\n".join(wanted)


def run_js(cases) -> list:
    js = slice_js() + """
const cases = JSON.parse(process.argv[2]);
const out = cases.map(c => {
  const fight = c.fight;
  const verdicts = c.conditions.map(cond => gradeCondition(cond, fight));
  return verdicts.map(v => v === true ? "T" : v === false ? "F" : v === 'void' ? "void" : "U");
});
console.log(JSON.stringify(out));
"""
    d = tempfile.mkdtemp()
    p = os.path.join(d, "grade.js")
    open(p, "w").write(js)
    try:
        r = subprocess.run(["node", p, json.dumps(cases)],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise SystemExit(f"node failed: {r.stderr[:400]}")
        return json.loads(r.stdout)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def py_verdicts(fight_py, conditions):
    out = []
    for c in conditions:
        v = pg.grade_condition(c, fight_py)
        out.append("T" if v is True else "F" if v is False else "void" if v == pg.VOID else "U")
    return out


# Each case: the fight stated ONCE, then rendered into both conventions.
#   end_round / elapsed_s -> python end_time (elapsed), js end_clock (remaining)
CASES = [
    ("KO r2 at 3:12 elapsed, Over 1.5", 2, 192, "ko/tko", False,
     [{"kind": "rounds", "op": "over", "line": 1.5}]),
    ("KO r2 at 3:12 elapsed, Under 1.5", 2, 192, "ko/tko", False,
     [{"kind": "rounds", "op": "under", "line": 1.5}]),
    ("KO r2 at 0:30 elapsed, Over 1.5 (just past)", 2, 30, "ko/tko", False,
     [{"kind": "rounds", "op": "over", "line": 1.5}]),
    ("KO r1 at 4:30 elapsed, Under 1.5", 1, 270, "ko/tko", False,
     [{"kind": "rounds", "op": "under", "line": 1.5}]),
    ("KO r2 at 4:30 elapsed, Over 2.5 (just under)", 2, 270, "ko/tko", False,
     [{"kind": "rounds", "op": "over", "line": 2.5}]),
    # CASES CHOSEN TO BE SENSITIVE TO THE CLOCK CONVENTION. A rounds leg only
    # inverts when the line sits INSIDE the finishing round -- line == (round-1)
    # + 0.5 -- and the elapsed time falls on the opposite side of that round's
    # midpoint from the remaining time. Without these the suite is nearly blind
    # to the very mistake it exists to catch: measured on the first draft, only
    # 1 of 3 rounds cases flipped when the countdown was fed to the ledger
    # grader, so it would have passed two times in three.
    ("KO r2 at 1:00 elapsed, Over 1.5  [sensitive]", 2, 60, "ko/tko", False,
     [{"kind": "rounds", "op": "over", "line": 1.5}]),
    ("KO r3 at 3:20 elapsed, Over 2.5  [sensitive]", 3, 200, "ko/tko", False,
     [{"kind": "rounds", "op": "over", "line": 2.5}]),
    ("KO r3 at 1:40 elapsed, Under 2.5 [sensitive]", 3, 100, "ko/tko", False,
     [{"kind": "rounds", "op": "under", "line": 2.5}]),
    ("KO r4 at 3:20 elapsed, Over 3.5  [sensitive]", 4, 200, "ko/tko", False,
     [{"kind": "rounds", "op": "over", "line": 3.5}]),
    ("KO r5 at 1:40 elapsed, Under 4.5 [sensitive]", 5, 100, "ko/tko", False,
     [{"kind": "rounds", "op": "under", "line": 4.5}]),
    ("decision, Over 2.5", 3, 300, "decision", True,
     [{"kind": "rounds", "op": "over", "line": 2.5}]),
    ("decision, Under 2.5", 3, 300, "decision", True,
     [{"kind": "rounds", "op": "under", "line": 2.5}]),
    ("sub r1 2:00, winner + method", 1, 120, "submission", False,
     [{"kind": "winner", "fighter": "Rei Tsuruya"},
      {"kind": "method", "any_of": ["submission"]}]),
    ("sub r1 2:00, wrong winner", 1, 120, "submission", False,
     [{"kind": "winner", "fighter": "Kevin Borjas"}]),
    ("decision, distance=true", 3, 300, "decision", True,
     [{"kind": "distance", "value": True}]),
]


def main():
    if not shutil.which("node"):
        print("node not installed -- skipping (this check needs it)")
        return 0
    print("Differential: browser JS vs src/parlay_grader.py, same fights\n")

    js_cases, py_fights = [], []
    for label, rnd, elapsed, method, went_distance, conds in CASES:
        remaining = 300 - elapsed
        js_cases.append({
            "fight": {
                "winner": "Rei Tsuruya", "no_winner": False, "cancelled": False,
                "method_slug": method, "went_distance": went_distance,
                "end_round": rnd,
                # REMAINING -- what ESPN's scoreboard shows
                "end_clock": f"{remaining // 60}:{remaining % 60:02d}",
            },
            "conditions": conds,
        })
        py_fights.append({
            "winner": "Rei Tsuruya", "no_winner": False, "cancelled": False,
            "method_slug": method, "went_distance": went_distance,
            "end_round": rnd,
            # ELAPSED -- what fight_results.csv stores
            "end_time": f"{elapsed // 60}:{elapsed % 60:02d}",
        })

    js_out = run_js(js_cases)
    for (label, rnd, elapsed, *_rest), jsv, pyf, case in zip(
            CASES, js_out, py_fights, js_cases):
        pyv = py_verdicts(pyf, case["conditions"])
        agree = jsv == pyv
        tag = "ok  " if agree else "FAIL"
        print(f"  [{tag}] {label}")
        print(f"         js(remaining {case['fight']['end_clock']}) = {jsv}   "
              f"py(elapsed {pyf['end_time']}) = {pyv}")
        if not agree:
            FAILURES.append(label)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} DISAGREEMENT(S) between the live grader and the ledger grader:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("Both graders agree on every case. The clock conventions still line up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
