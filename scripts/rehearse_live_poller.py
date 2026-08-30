"""
Run the browser's live-scoreboard handler against a real ESPN payload.

applyLiveScoreboard is what turns an ESPN poll into what a reader sees during a
card: which row shows a result, which fight the banner calls live, and whether
the server's ground-truth live key survives. It had no test, and its first
exercise of this session's changes -- the LIVE NOW suppression and the
espnLiveKey guard -- was going to be a Saturday with money on the board.

IT NEEDS NO DOM. Every DOM touch is delegated to paintResult, paintMethod,
update and gradeAllSlips, so the handler itself can run under node against
stubs, and what it MEANS -- which fight matched, what label was built, what
liveKey came out -- is fully observable. The five delegates are stubbed and
their calls recorded.

THE PAYLOAD IS REAL: tests/fixtures/espn_scoreboard_2026_08_29.json, fetched
from site.api.espn.com for the card that ran, trimmed to the fields this handler
actually reads so it stays legible and cannot drift on fields nothing uses. The
mid-card and draw variants are DERIVED from it by rewriting status, which is
stated at each use -- the real card finished with all thirteen bouts post.

The JS is sliced out of templates/site.html rather than copied.

Run: python3 scripts/rehearse_live_poller.py     (requires node)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "templates", "site.html")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "espn_scoreboard_2026_08_29.json")
CARD = os.path.join(ROOT, "tests", "fixtures", "card_2026_08_29.csv")

FAILURES = []


def check(label, cond, detail=""):
    print(f"    [{'ok  ' if cond else 'FAIL'}] {label}{('  -> ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(label)


def slice_fn(src, name):
    m = re.search(r"\n\s*function " + name + r"\s*\(", src)
    if not m:
        raise SystemExit(f"could not find {name} in the template")
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
    return src[m.start() + 1:j + 1]


def build_js():
    src = open(TEMPLATE, encoding="utf-8").read()
    return "\n".join(slice_fn(src, n) for n in ("canonicalKey", "applyLiveScoreboard"))


HARNESS = r"""
const input = JSON.parse(process.argv[2]);
let schedule = input.schedule;
let espnLiveKey = input.espnLiveKey;      // the server-baked ground truth
const calls = { paintResult: [], paintMethod: [], update: 0, gradeAllSlips: 0 };

// The five delegates. applyLiveScoreboard touches no DOM itself, so stubbing
// these is enough to observe everything it decides.
function paintResult(f) { calls.paintResult.push(f.fighter_a + '|' + f.fighter_b); }
function paintMethod(f, m) { calls.paintMethod.push(f.fighter_a + '|' + m); }
function update() { calls.update++; }
function gradeAllSlips() { calls.gradeAllSlips++; }
function fetchLiveMethods() { return Promise.resolve(null); }   // resolves later; not under test

applyLiveScoreboard(input.data);

console.log(JSON.stringify({
  espnLiveKey: espnLiveKey,
  calls: calls,
  schedule: schedule.map(f => ({
    k: f.fighter_a + ' vs ' + f.fighter_b,
    winner: f.winner || null,
    result_label: f.result_label || null,
    no_winner: !!f.no_winner,
  })),
}));
"""


def run(data, schedule, espn_live_key=None):
    js = build_js() + HARNESS
    d = tempfile.mkdtemp()
    p = os.path.join(d, "poll.js")
    open(p, "w").write(js)
    try:
        payload = json.dumps({"data": data, "schedule": schedule, "espnLiveKey": espn_live_key})
        r = subprocess.run(["node", p, payload], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise SystemExit(f"node failed: {r.stderr[:500]}")
        return json.loads(r.stdout)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def schedule_from_card():
    import csv
    out = []
    for row in csv.DictReader(open(CARD, encoding="utf-8")):
        if str(row.get("cancelled", "")).strip().lower() == "true":
            continue
        out.append({"fighter_a": row["fighter_a"], "fighter_b": row["fighter_b"],
                    "estimated_start_iso": "2026-08-29T07:00:00-04:00",
                    "estimated_end_iso": "2026-08-29T07:15:00-04:00"})
    return out


def main():
    if not shutil.which("node"):
        print("node not installed -- skipping (this check needs it)")
        return 0
    real = json.load(open(FIXTURE))
    sched = schedule_from_card()
    print(f"Live poller against the real 2026-08-29 payload "
          f"({len(real['events'][0]['competitions'])} competitions, {len(sched)} carded fights)\n")

    print("  A. the real payload, card complete")
    out = run(real, sched, espn_live_key="song yadong|umar nurmagomedov")
    painted = len(out["calls"]["paintResult"])
    check("every carded fight got a result painted", painted == len(sched), f"{painted}/{len(sched)}")
    check("every fight carries a winner", all(f["winner"] for f in out["schedule"]))
    labelled = [f for f in out["schedule"] if f["result_label"]]
    check("result labels built for all", len(labelled) == len(sched))
    check("nothing is flagged no_winner on a decided card",
          not any(f["no_winner"] for f in out["schedule"]))
    check("slips were re-graded once", out["calls"]["gradeAllSlips"] == 1)
    check("nothing in progress -> live key cleared, since this payload IS our card",
          out["espnLiveKey"] is None, repr(out["espnLiveKey"]))

    print("\n  B. mid-card: one bout rewritten to state 'in' (derived from the real payload)")
    mid = json.loads(json.dumps(real))
    tgt = mid["events"][0]["competitions"][9]
    tgt["status"]["type"] = {"state": "in", "completed": False, "description": "In Progress"}
    for c in tgt["competitors"]:
        c["winner"] = False
    names = [c["athlete"]["fullName"] for c in tgt["competitors"]]
    out = run(mid, schedule_from_card(), espn_live_key=None)
    want = "|".join(sorted(n.lower() for n in names))
    check("the in-progress bout becomes the live key", out["espnLiveKey"] == want,
          f"{out['espnLiveKey']!r} for {names}")
    check("the live bout is NOT painted as finished",
          f"{names[0]}|{names[1]}" not in out["calls"]["paintResult"] and
          f"{names[1]}|{names[0]}" not in out["calls"]["paintResult"])

    print("\n  C. a payload about a different day (the guard added this session)")
    out = run({"events": []}, schedule_from_card(),
              espn_live_key="song yadong|umar nurmagomedov")
    check("a payload that does not cover our card must NOT wipe the live key",
          out["espnLiveKey"] == "song yadong|umar nurmagomedov", repr(out["espnLiveKey"]))

    print("\n  D. a draw: completed with no winner flagged (derived)")
    draw = json.loads(json.dumps(real))
    d0 = draw["events"][0]["competitions"][0]
    for c in d0["competitors"]:
        c["winner"] = False
    dn = [c["athlete"]["fullName"] for c in d0["competitors"]]
    out = run(draw, schedule_from_card(), espn_live_key=None)
    row = next((f for f in out["schedule"] if dn[0] in f["k"]), None)
    check("the drawn bout is flagged no_winner", bool(row and row["no_winner"]),
          repr(row["k"]) if row else "row not found")
    check("and labelled a draw, not a loss",
          bool(row and row["result_label"] and "Draw" in row["result_label"]),
          repr(row["result_label"]) if row else "")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("The live poller behaves correctly on a real payload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
