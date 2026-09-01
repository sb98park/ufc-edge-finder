"""
data/source_health.json must accumulate, and the parlay builder must say why
it produced nothing.

Four reporters write to this file. The first version of the live_props writer
built a fresh dict and wrote it, deleting every key it did not own -- a health
file that erases half its own health reads as "nothing to report". The merge
lived inline in that one writer, so every new reporter had to rediscover it.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, ".")
import src.source_health as sh                                   # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


tmp = tempfile.mkdtemp()
_real = sh.PATH
sh.PATH = os.path.join(tmp, "health.json")
try:
    check("writes into a file that does not exist yet", sh.record("alpha", {"n": 1}))
    check("a second reporter does not clobber the first", sh.record("beta", [1, 2]))
    d = json.load(open(sh.PATH))
    check("both blocks survive", d.get("alpha") == {"n": 1} and d.get("beta") == [1, 2])
    check("a timestamp is stamped", bool(d.get("at")))

    sh.record("alpha", {"n": 2})
    d = json.load(open(sh.PATH))
    check("a reporter can update its OWN block", d["alpha"] == {"n": 2})
    check("and updating one leaves the other alone", d["beta"] == [1, 2])

    # A corrupt file must not lose the write, and must not raise.
    open(sh.PATH, "w").write("{not json")
    check("a corrupt file is recovered from, not raised on", sh.record("gamma", 3))
    check("the recovered file holds the new block", json.load(open(sh.PATH))["gamma"] == 3)

    # An unwritable path must return False rather than take the build down.
    sh.PATH = os.path.join(tmp, "nope", "\0bad", "h.json")
    check("an unwritable path returns False instead of raising", sh.record("d", 1) is False)
finally:
    sh.PATH = _real
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------- the parlay builder's reason
import src.parlay_builder as pb                                  # noqa: E402

tmp = tempfile.mkdtemp()
sh.PATH = os.path.join(tmp, "health.json")
try:
    sh.record("unrelated", {"keep": "me"})
    pb._record_reason("bankroll", "no bettable venue", {"reference_only_legs": 21})
    pb._record_reason("lotto", "no combination met the target", {"eligible_pieces": 4})
    d = json.load(open(sh.PATH))
    check("the parlay block exists", isinstance(d.get("parlay"), dict))
    check("each tier keeps its own reason",
          set(d["parlay"]) == {"bankroll", "lotto"})
    check("the reason is carried", d["parlay"]["bankroll"]["reason"] == "no bettable venue")
    check("the detail is carried", d["parlay"]["bankroll"]["reference_only_legs"] == 21)
    check("one tier does not overwrite the other",
          d["parlay"]["lotto"]["eligible_pieces"] == 4)
    check("an unrelated reporter's block is untouched", d["unrelated"] == {"keep": "me"})

    # A later success must replace the tier's stale failure reason.
    pb._record_reason("bankroll", "built", {"slips": 2})
    d = json.load(open(sh.PATH))
    check("a success overwrites the tier's stale reason",
          d["parlay"]["bankroll"]["reason"] == "built"
          and "reference_only_legs" not in d["parlay"]["bankroll"])
    check("and the other tier still says what it said",
          d["parlay"]["lotto"]["reason"] == "no combination met the target")
finally:
    sh.PATH = _real
    shutil.rmtree(tmp, ignore_errors=True)

# STRUCTURAL ONLY, NOT CONTENT. This asserted that the shipped
# data/source_health.json carries a `parlay` block -- and it does after a
# build, but data/ is reverted before committing so CI never sees one, and the
# test suite is a HARD GATE. It failed every run and froze the site on stale
# data, which is precisely what CLAUDE.md s2 says a gate must never do: fail
# for something that legitimately varies rather than for a structural fault.
#
# The behaviour is already covered above against synthetic fixtures, which is
# where behaviour belongs. All this checks is that whatever is committed is
# readable and shaped like a health file -- true whether or not a build has
# run since.
_shipped = json.load(open("data/source_health.json"))
check("the shipped health file is a json object", isinstance(_shipped, dict))
check("every block in it is keyed by a string",
      all(isinstance(k, str) for k in _shipped))
if "parlay" in _shipped:
    check("a shipped parlay block is well formed",
          isinstance(_shipped["parlay"], dict)
          and all(isinstance(v, dict) and "reason" in v
                  for v in _shipped["parlay"].values()))

print(f"test_source_health: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
