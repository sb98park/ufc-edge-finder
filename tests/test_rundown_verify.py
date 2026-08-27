"""
The Rundown harness, and the four failures it exists to tell apart.

scripts/verify_rundown.py is the only thing standing between a silent feed and
a build that looks healthy while carrying no book prices, so it gets tested
the same way the ledger does -- against a fixture, never against the network.
The fixture carries one healthy fight priced at the real DraftKings open from
2026-08-26 (-625 / +455), one fight priced BELOW an implied sum of 1.0, and one
quoted by an affiliate the client does not know.
"""

import sys, os, re, json, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

SCRIPT = os.path.join(os.path.dirname(HERE), "scripts", "verify_rundown.py")
FIXTURE = os.path.join(HERE, "fixtures", "rundown_sample.json")

FAILURES = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:58s} got {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


def flat(text):
    """Whitespace-insensitive, so a column-width tweak is not a test failure."""
    return " ".join(text.split())


def run(*args, env=None):
    e = dict(os.environ)
    e.pop("RUNDOWN_API_KEY", None)      # never let a real key reach a test
    e.update(env or {})
    p = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True,
                       text=True, cwd=os.path.dirname(HERE), env=e)
    return p.returncode, p.stdout + p.stderr


def healthy_fixture():
    """The sample with the sub-1.0 fight removed."""
    d = json.load(open(FIXTURE))
    d["events"] = [e for e in d["events"] if "Alpha Fighter" not in json.dumps(e)]
    path = os.path.join(tempfile.mkdtemp(), "ok.json")
    json.dump(d, open(path, "w"))
    return path


print("\na missing key is a fact about the shell, not a failure")
code, out = run()
check("exits clean", code, 0)
check("says the key is unset", "RUNDOWN_API_KEY is not set" in out, True)
check("does not present it as a bug", "not a bug" in out, True)

print("\nthe planted faults are all found")
code, out = run("--fixture", FIXTURE)
check("exits 1", code, 1)
check("catches the sub-1.0 implied sum", "below 1.0 is an arbitrage" in out, True)
check("names the book and the fight",
      "DraftKings prices Alpha Fighter|Beta Fighter" in out, True)
check("reports the unknown affiliate", "DROPPED" in out and "'77': 2" in out, True)
check("does not fail the build over it", out.count("FAIL  1 problem") == 1, True)

print("\nthe vig it measures matches the one computed by hand")
# -625 / +455 is the real DraftKings open, and 4.22% is what de-vigging it by
# hand off the screenshot gave. If this moves, one of the two was wrong.
check("DraftKings 4.22%", "+4.22% vig" in out, True)
check("both books measured on the same fight",
      "FanDuel 1.0423" in flat(out), True)

print("\ncoverage is measured per market, not assumed")
check("totals present on one fight of three",
      "TotalRounds 1/3 fight(s)" in flat(out), True)
check("and from a single book", "0 with more than one book" in out, True)

print("\na clean feed passes")
ok_path = healthy_fixture()
code, out = run("--fixture", ok_path)
check("exits 0", code, 0)
check("says so", "PASS" in out, True)

print("\nthe quota projection replays the real schedule")
code, out = run("--fixture", ok_path, "--date", "2026-08-29")
check("costs the payload rather than a remembered number",
      "point(s) per pull, measured from this payload" in flat(out), True)
check("names the allowance and the margin", "allowance 17,000 of 20,000" in flat(out), True)
check("walks a week, ending on the card", "<- fight day" in out, True)
# Parsed rather than string-matched: the point is that the spacing column
# decreases as the card approaches, and that survives a format change.
_gaps = [int(m) for m in re.findall(r"^\s+2026-\d\d-\d\d\s+\d+\s+[\d,]+\s+[\d.]+%\s+(\d+)m",
                                    out, re.M)]
check("the schedule has a row per day", len(_gaps) >= 5, True)
check("and tightens monotonically toward the card",
      _gaps == sorted(_gaps, reverse=True), True)
check("  ...ending at the fight-day floor", _gaps[-1] <= 20 if _gaps else False, True)


print("\nthe cadence ramps toward the card")
import datetime as dt  # noqa: E402
from src.rundown_source import (  # noqa: E402
    plan_pull, DAILY_POINT_CAP, BUDGET_SAFETY, FIGHT_DAY_FLOOR_SECONDS,
)

CARD = "2026-08-29"


def at(days_before, hour=12, points=0, cost=104):
    when = dt.datetime.combine(
        dt.date(2026, 8, 29) - dt.timedelta(days=days_before),
        dt.time(hour, 0), tzinfo=dt.timezone.utc)
    return plan_pull([CARD], {"points": points, "last_cost": cost}, when), when


check("a week out is six-hourly", at(7)[0]["interval"], 6 * 3600)
check("three days out is hourly", at(3)[0]["interval"], 3600)
check("the day before is twenty minutes", at(1)[0]["interval"], 20 * 60)
check("fight day is tighter than the day before",
      at(0)[0]["interval"] < at(1)[0]["interval"], True)
check("  ...and never below the floor",
      at(0)[0]["interval"] >= FIGHT_DAY_FLOOR_SECONDS, True)

print("\nfight day paces itself off what is left")
_early, _ = at(0, hour=6)
_late, _ = at(0, hour=22)
check("a nearly spent budget slows down",
      at(0, points=16_000)[0]["interval"] > at(0, points=0)[0]["interval"], True)
check("a dearer card slows down too",
      at(0, cost=400)[0]["interval"] > at(0, cost=104)[0]["interval"], True)

print("\nthe hard stop is what makes it a guarantee")
_allow = int(DAILY_POINT_CAP * BUDGET_SAFETY)
check("affordable with budget left", at(0, points=0)[0]["affordable"], True)
check("refused once the allowance is gone",
      at(0, points=_allow)[0]["affordable"], False)
check("  ...and the allowance sits under the real cap", _allow < DAILY_POINT_CAP, True)


def simulate(cost):
    """Walk a week minute by minute and return the worst day's spend."""
    worst, budget, last, day = 0, None, None, None
    t = dt.datetime(2026, 8, 22, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 8, 31, tzinfo=dt.timezone.utc)
    while t < end:
        d = t.strftime("%Y-%m-%d")
        if d != day:
            day, budget, last = d, {"points": 0, "last_cost": cost}, None
        p = plan_pull([CARD], budget, t)
        if p["affordable"] and (last is None or (t - last).total_seconds() >= p["interval"]):
            budget["points"] += cost
            last = t
            worst = max(worst, budget["points"])
        t += dt.timedelta(minutes=1)
    return worst


print("\nA WHOLE WEEK STAYS UNDER THE CAP, whatever a pull turns out to cost")
# The repo has carried two different figures for the cost of a pull -- ~104
# with main_line totals and ~208 without -- and neither has been measured
# against the live feed. 400 is included because the guarantee has to hold
# when the estimate is wrong, not only when it is right.
for _cost in (104, 208, 400, 900):
    _worst = simulate(_cost)
    check(f"worst day at {_cost} pts/pull stays under {DAILY_POINT_CAP:,}",
          _worst <= DAILY_POINT_CAP, True)

print("\n" + ("-" * 70))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print("the harness tells the four failures apart")
