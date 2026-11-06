"""
A redacted probability is a leak only when it is a NUMBER, not digits inside
a bigger one.

check_free_build is the paywall's last line: it refuses to deploy when a
model value survives into the free payload. It matched with `value in
haystack`, and the file already carries one scar from that -- "letter-spacing:
-0.2px matched a redacted value and reported a leak that was a CSS
declaration" -- patched by stripping style and script blocks.

The same collision then happened where stripping cannot reach. On 2026-11-01
the probability 0.6396 was found inside

    data-target-offset="40.6396425204"

a progress-ring geometry attribute. 87.96459 is 2*pi*14, so that offset
encodes 53.8% and has nothing to do with 63.96%; the digits merely collide.
The gate failed twenty-nine consecutive runs -- about two and a half hours of
no deploys -- until the probability moved on its own.

FAILING CLOSED IS STILL THE RULE. Only the LEADING side is relaxed, because
only a preceding digit or dot can make a match part of a longer number. A
trailing digit is left alone on purpose: 0.6396425204 standing on its own is
the same quantity at more precision, and that is still paid content.

Synthetic strings only; nothing here reads docs/ or data/.
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spec = importlib.util.spec_from_file_location(
    "check_free_build",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "check_free_build.py"))
_cfb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfb)
leak_at = _cfb._leak_at

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


# --- the regression --------------------------------------------------------
RING = ('<circle stroke-dasharray="87.96459420000001" '
        'data-target-offset="40.6396425204" class="ring-progress"/>')
check("the ring-geometry collision is not a leak", leak_at("0.6396", RING) is None)
check("  ...and the gate would have deployed", leak_at("0.6396", RING) is None)

# --- real leaks must still be caught ---------------------------------------
for label, hay in (
        ("in text", "<span>0.6396</span>"),
        ("in an attribute", '<div data-p="0.6396"></div>'),
        ("at the very start", "0.6396 trailing text"),
        ("after a bracket", "(0.6396)"),
        ("after a comma", "a,0.6396,b"),
):
    check(f"a real leak {label} is caught", leak_at("0.6396", hay) is not None)

# A longer, more precise number that STARTS with the redacted value is the
# same quantity with more digits -- still paid content, still a leak.
check("more precision is still a leak",
      leak_at("0.6396", '<div data-p="0.6396425204"></div>') is not None)

# --- embedded digits are not leaks -----------------------------------------
for label, hay in (
        ("preceded by a digit", 'x="10.6396"'),
        ("preceded by a dot", 'x="1.0.6396"'),
        ("mid-number", 'offset="140.63962"'),
):
    check(f"{label} is not a leak", leak_at("0.6396", hay) is None)

# --- it must scan PAST a false positive to a real one ----------------------
# The first occurrence being embedded must not hide a genuine leak later in
# the same document. This is why the helper loops instead of testing find()
# once.
both = 'data-target-offset="40.6396425204" ... <span>0.6396</span>'
pos = leak_at("0.6396", both)
check("a later real leak is still found past an embedded one", pos is not None)
check("  ...and it points at the real one, not the ring",
      pos is not None and both[pos - 6:pos] == "<span>")

# --- absent means absent ---------------------------------------------------
check("a value that is not there returns None",
      leak_at("0.9999", '<div data-p="0.6396"></div>') is None)
check("an empty haystack is not a leak", leak_at("0.6396", "") is None)

print(f"{'PASS' if not fail else 'FAIL'}: test_free_build_leak_match -- {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
