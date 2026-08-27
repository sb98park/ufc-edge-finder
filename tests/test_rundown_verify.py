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

print("\nthe quota projection is arithmetic, not a remembered number")
code, out = run("--fixture", ok_path, "--cadence-minutes", "1")
check("scales with the cadence", "1440 pull(s)/day" in out, True)
code, out = run("--fixture", ok_path, "--cadence-minutes", "0.5")
check("fails when the cap would be blown", code, 1)
check("says which knob to turn", "MIN_SECONDS_BETWEEN_PULLS" in out, True)

print("\n" + ("-" * 70))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print("the harness tells the four failures apart")
