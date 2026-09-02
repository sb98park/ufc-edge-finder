"""
Run a CI step that is allowed to fail, and record whether it did.

WHY. Seven steps in refresh.yml run as `... || true`, which is correct -- a
backfill that cannot reach ESPN must not freeze a site that real money is
staked against. But `|| true` discards the exit code, and nothing else looked
at it, so a step that stopped working stayed broken silently.

That is not hypothetical. scripts/backfill_history_from_espn.py raised
NameError on EVERY run for two days after a refactor deleted a helper it still
called. CI swallowed it every five minutes. It surfaced only because a fighter
on an imminent card had no scouting data and the owner asked why.

A STEP CANNOT REPORT ON ITSELF. The failure mode is the script dying before
its own reporting line -- an import error, a NameError at the top of main(), a
kill. So the recorder has to sit OUTSIDE the process, which is what this is:
it runs the command as a subprocess and writes the outcome whatever happens.

It ALWAYS EXITS 0 and replaces the `|| true` rather than sitting beside it,
so a step's failure still cannot freeze the build. What changes is that the
failure is now written to data/source_health.json, which is committed, so it
survives the run that produced it -- the same reasoning src/source_health.py
already records for feeds.

last_ok and consecutive_failures come from the PREVIOUS block, so "broken
since Tuesday" is distinguishable from "flaked once", which is the distinction
that decides whether anyone needs to act.

Output is captured and re-printed rather than streamed, so a long step prints
nothing until it finishes. Accepted: the alternative is a pump loop, and this
file must be too simple to be the thing that breaks.

Usage:
  python3 scripts/run_step.py <name> -- <command...>
  python3 scripts/run_step.py --report
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.source_health import record, PATH  # noqa: E402

ERROR_TAIL_CHARS = 800
# Failing this many runs in a row stops being a flake. At a 5-minute cadence
# it is ~15 minutes of a step being dead, which is short enough to catch a
# real break and long enough to ride out one ESPN 403.
FAILURE_STREAK_ALARM = 3


def _steps() -> dict:
    try:
        with open(PATH, encoding="utf-8") as fh:
            blob = json.load(fh)
        s = blob.get("steps") if isinstance(blob, dict) else None
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def report() -> int:
    steps = _steps()
    if not steps:
        print("[steps] nothing recorded yet")
        return 0
    print(f"{'step':34s} {'state':>9} {'streak':>7}  last ok")
    broken = []
    for name in sorted(steps):
        s = steps[name] or {}
        okay = bool(s.get("ok"))
        streak = int(s.get("consecutive_failures") or 0)
        state = ("RUNNING" if s.get("in_flight")
                 else ("ok" if okay else f"FAILED({s.get('exit_code')})"))
        print(f"{name[:34]:34s} {state:>9} {streak:>7}  {s.get('last_ok') or 'never'}")
        if streak >= FAILURE_STREAK_ALARM:
            broken.append((name, streak, s))
    if broken:
        print()
        for name, streak, s in broken:
            print(f"  {name} has failed {streak} runs in a row; last success "
                  f"{s.get('last_ok') or 'never'}.")
            tail = (s.get("error") or "").strip().splitlines()
            if tail:
                print(f"      {tail[-1][:160]}")
    return 0                                    # a report, never a brake


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] == "--report":
        return report()
    if "--" not in argv:
        print("usage: run_step.py <name> -- <command...>")
        return 0                                # never break the build on usage
    name = argv[0]
    cmd = argv[argv.index("--") + 1:]
    if not cmd:
        print(f"[steps] {name}: no command given")
        return 0

    prior = _steps().get(name) or {}

    # A RUNNER KILL CANNOT BE CAUGHT BY EITHER OF US. Two of these steps carry
    # timeout-minutes, and when the runner kills a step at the limit it kills
    # this wrapper too -- there is no exit code to record and no chance to
    # record one. refresh.yml already documents that class: on 2026-08-16
    # every build from 03:22 failed because an optional step got slow enough
    # to hit its limit, and the site sat nine hours stale.
    #
    # So the fact that an attempt STARTED is written before the attempt runs.
    # If this block is still in flight when the next run reads it, the
    # previous attempt died without finishing, and that is the one shape of
    # failure a post-hoc recorder can never see for itself.
    killed_last_time = bool(prior.get("in_flight"))
    if killed_last_time:
        prior = dict(prior)
        prior["consecutive_failures"] = int(prior.get("consecutive_failures") or 0) + 1
        print(f"[steps] {name}: previous attempt at {prior.get('started_at')} never finished "
              f"-- killed at a timeout, counting it as a failure")

    started = time.time()
    _mark = _steps()
    _mark[name] = {**prior, "in_flight": True,
                   "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    record("steps", _mark)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:                    # noqa: BLE001
        # A command that cannot even be launched is exactly the case that must
        # still be recorded rather than raising out of the wrapper.
        code, out, err = 127, "", f"{type(exc).__name__}: {exc}"
    took = round(time.time() - started, 1)

    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)

    ok = code == 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    steps = _steps()
    steps[name] = {
        "ok": ok,
        "in_flight": False,
        "exit_code": code,
        "ran_at": now,
        "duration_s": took,
        "last_ok": now if ok else prior.get("last_ok"),
        "consecutive_failures": 0 if ok else int(prior.get("consecutive_failures") or 0) + 1,
        "killed_previous_run": killed_last_time,
        # Only on failure: a passing step's stderr is usually progress chatter,
        # and keeping it would churn the committed file every single run.
        "error": None if ok else (err or out or "")[-ERROR_TAIL_CHARS:],
    }
    record("steps", steps)

    if ok:
        print(f"[steps] {name}: ok in {took}s")
    else:
        streak = steps[name]["consecutive_failures"]
        print(f"[steps] {name}: FAILED exit {code} after {took}s "
              f"({streak} run(s) in a row; last ok {prior.get('last_ok') or 'never'})")
    return 0            # ALWAYS. This replaces `|| true`, it does not add a gate.


if __name__ == "__main__":
    sys.exit(main())
