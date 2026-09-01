"""
Every waterfall factor description must fit its box.

The .wf-why box is 152px wide and clamped to two lines. Measured in a real
browser at 375px: a 55-character string wraps to two full lines and is not
clipped; the 103-character adjustment-cap sentence that shipped before was
clipped, and it truncated at "so no..." -- precisely where it was about to
explain why the cap exists.

CHARACTERS ARE A PROXY, and a deliberately conservative one. The longest
description that renders on ONE line is 36 characters, so two lines is
roughly 72; the limit here is 64, which leaves room for a wide-glyph string
without needing a browser in CI.
"""
import sys

sys.path.insert(0, ".")

ok = fail = 0
LIMIT = 64


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


import re                                                        # noqa: E402
import pathlib                                                   # noqa: E402

src = pathlib.Path("src/matchup_model.py").read_text(encoding="utf-8")

# The FACTORS table: ("key", "Label", "plain-language explanation")
block = src[src.index("    FACTORS = ["):]
block = block[:block.index("\n    ]")]
descriptions = re.findall(r'\(\s*"[^"]+",\s*"[^"]+",\s*"([^"]*)"', block)
check("the FACTORS table was parsed", len(descriptions) >= 5)
for d in descriptions:
    check(f"{d[:40]!r} fits ({len(d)} chars)", len(d) <= LIMIT)

# The base row and the cap row are built by hand rather than from FACTORS.
for literal in re.findall(r'step\(\s*"[^"]+",\s*\n?\s*"([^"]+)"', src):
    check(f"{literal[:40]!r} fits ({len(literal)} chars)", len(literal) <= LIMIT)

# The cap row is an f-string across two source lines; reconstruct it.
from src.matchup_model import ADJUSTMENT_TOTAL_CAP                # noqa: E402
cap = f"capped at {ADJUSTMENT_TOTAL_CAP:.0f} points so factors cannot outweigh the gap"
check(f"the adjustment-cap copy fits ({len(cap)} chars)", len(cap) <= LIMIT)
check("the cap copy is still present in the source",
      "cannot outweigh the gap" in src)
check("the sentence that truncated is gone",
      "so no pile-up" not in src)

print(f"test_waterfall_copy: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
