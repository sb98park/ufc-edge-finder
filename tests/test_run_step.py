"""
The wrapper that watches the non-blocking steps must never become a gate.

It replaces `|| true` on seven steps in refresh.yml. If it can exit non-zero,
a dead ESPN feed freezes a site that real money is staked against -- which is
the precise outcome `|| true` was there to prevent, reintroduced by the thing
meant to improve on it. So "always exits 0" is the load-bearing property and
most of these tests are that.

Runs the real script as a subprocess with cwd set to a temp directory. That is
the whole isolation mechanism and it is worth stating plainly: src.source_health
writes to the RELATIVE path "data/source_health.json", so a different cwd is a
different file. Nothing here touches the live one -- an earlier alarm's test
asserted on that file and froze CI for two hours.
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "scripts", "run_step.py")

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


def run(_health, *args):
    """Run run_step.py from the temp dir, so it writes the temp health file."""
    return subprocess.run([sys.executable, RUNNER, *args],
                          capture_output=True, text=True, cwd=tmp_dir)


with tempfile.TemporaryDirectory() as tmp_dir:
    os.makedirs(os.path.join(tmp_dir, "data"), exist_ok=True)
    health = os.path.join(tmp_dir, "data", "source_health.json")

    def steps():
        try:
            with open(health) as fh:
                return json.load(fh).get("steps", {})
        except (OSError, ValueError):
            return {}

    # 1. A COMMAND THAT DIES MUST STILL LEAVE THE WRAPPER AT 0.
    r = run(health, "boom", "--", sys.executable, "-c", "raise NameError('fold')")
    check("a crashing step does not fail the wrapper", r.returncode == 0)
    check("...and is recorded as failed", steps().get("boom", {}).get("ok") is False)
    check("...with the traceback kept", "NameError" in (steps()["boom"].get("error") or ""))
    check("...and a streak of 1", steps()["boom"].get("consecutive_failures") == 1)

    # 2. A COMMAND THAT CANNOT EVEN BE LAUNCHED. The wrapper must record it
    #    rather than raising out of itself -- this is the import-error shape.
    r = run(health, "missing", "--", "definitely-not-a-real-binary-xyz")
    check("an unlaunchable command does not fail the wrapper", r.returncode == 0)
    check("...and is recorded", steps().get("missing", {}).get("ok") is False)

    # 3. Streaks accumulate, which is what separates broken from flaky.
    for _ in range(2):
        run(health, "boom", "--", sys.executable, "-c", "raise SystemExit(3)")
    check("consecutive failures accumulate", steps()["boom"]["consecutive_failures"] == 3)
    check("the exit code is kept", steps()["boom"]["exit_code"] == 3)

    # 4. A success clears the streak and stamps last_ok.
    run(health, "boom", "--", sys.executable, "-c", "print('fine')")
    check("a success clears the streak", steps()["boom"]["consecutive_failures"] == 0)
    check("...stamps last_ok", bool(steps()["boom"]["last_ok"]))
    check("...and drops the stale traceback", steps()["boom"]["error"] is None)

    # 5. ONE STEP MUST NOT ERASE ANOTHER. The health file had exactly this bug
    #    once: a writer built a fresh dict and dropped every key it did not own.
    check("an unrelated step survives", steps().get("missing", {}).get("ok") is False)

    # 6. THE RUNNER-KILL SHAPE, which no post-hoc recorder can see for itself:
    #    an attempt that started and never finished.
    blob = json.load(open(health))
    blob["steps"]["killed"] = {"ok": True, "in_flight": True,
                               "started_at": "2026-01-01T00:00:00+00:00",
                               "consecutive_failures": 0, "last_ok": "2026-01-01T00:00:00+00:00"}
    json.dump(blob, open(health, "w"))
    r = run(health, "killed", "--", sys.executable, "-c", "print('recovered')")
    check("a killed previous attempt is noticed", steps()["killed"].get("killed_previous_run") is True)
    check("...and said so out loud", "never finished" in r.stdout)
    check("...and the block is no longer in flight", steps()["killed"].get("in_flight") is False)

    # 7. Malformed usage cannot break a build either.
    check("no arguments exits 0", run(health).returncode == 0)
    check("a name with no command exits 0", run(health, "lonely").returncode == 0)
    check("--report exits 0", run(health, "--report").returncode == 0)

print(f"test_run_step: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
