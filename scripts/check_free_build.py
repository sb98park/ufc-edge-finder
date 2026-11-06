#!/usr/bin/env python3
"""
Fails the build if the FREE payload contains any of the model layer.

This is the one check in the repo whose failure is not recoverable by
shipping a fix afterwards. Once a free payload containing next Saturday's
picks has been served, those picks are public: it is cached, it is scraped,
and no later deploy takes it back. Everything else here can be fixed forward.
That is why this runs as a hard gate rather than a warning, and why it asserts
on VALUES rather than on the absence of a few CSS class names -- a class name
can be renamed by an unrelated commit and the check would quietly pass
forever while leaking.

WHAT IT ASSERTS

  1. Every scalar the redaction removed from the upcoming card is absent
     from the free payload. generate_site.py --tier free writes the manifest
     of exactly those values, so the check tracks the redaction automatically
     rather than duplicating a list that would drift out of step with it.

  2. The structural markers of the model layer do not appear inside the
     upcoming-card section. Belt and braces on top of (1): catches a whole
     block rendering through some path the manifest never saw.

  3. The manifest is not suspiciously empty. A redaction that silently
     stopped removing anything would otherwise pass every value assertion
     in this file by having nothing to assert.

Usage:
    python3 scripts/check_free_build.py [--free build/free.html]
                                        [--manifest build/redacted-manifest.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Structural giveaways of the model layer. Checked only inside the upcoming
# card, because every one of these legitimately appears on PAST fights, which
# are free.
MODEL_MARKERS = (
    "model-pick-badge",
    "waterfall-block",
    'data-nav="why"',
    "confidence-badge",
    "lock-pick-card",
)

# Values so short or common that finding them in 6MB of HTML proves nothing.
# A model probability of 0.5 renders as "50", which appears in coordinates,
# timestamps and fighter stats. Only distinctive values are worth asserting.
MIN_VALUE_LEN = 6


def _leak_at(value: str, haystack: str):
    """
    Where `value` appears as a NUMBER, not as digits inside a bigger one.

    A bare `value in haystack` is what this used to do, and the file already
    carries one scar from it -- "letter-spacing: -0.2px matched a redacted
    value and reported a leak that was a CSS declaration", fixed by stripping
    style and script. The same collision then happened in rendered HTML, where
    stripping cannot help: on 2026-11-01 the probability 0.6396 was found
    inside data-target-offset="40.6396425204", a progress-ring geometry
    attribute. 87.96459 is 2*pi*14, so that offset encodes 53.8% and has
    nothing to do with 63.96% -- the digits merely collide. The gate blocked
    every deploy for twenty-nine consecutive runs until the probability moved.

    So a match is only real when it is not preceded by a digit or a dot, which
    is exactly what makes it part of a longer number. A genuine leak is
    preceded by a quote, a space, a bracket or a tag -- never by a digit.

    DELIBERATELY STILL FAILS CLOSED on a trailing digit: if the manifest holds
    0.6396 and the page carries 0.6396425204 standing on its own, that is the
    same quantity at more precision and it is still paid content. Only the
    leading side is a false-positive source.
    """
    start = 0
    while True:
        i = haystack.find(value, start)
        if i < 0:
            return None
        before = haystack[i - 1] if i else ""
        if not (before.isdigit() or before == "."):
            return i
        start = i + 1


def _ungraded_cards(html: str) -> list[str]:
    """
    Every fight card that has NOT been graded yet.

    Scoped per CARD, not per section. Scoping by section was wrong and
    produced a confident false failure: the card being sold is partly graded
    on fight night, and a graded fight legitimately shows its pick badge and
    waterfall because that is the free tier's proof. Six such badges inside
    the section read as six leaks when nothing had leaked at all.

    `finished-row` is the class the template puts on a card with a result,
    so it is the same signal the redaction keys on.
    """
    cards = re.split(r'(?=<details class="fight-card)', html)
    return [
        c for c in cards
        if c.startswith('<details class="fight-card') and "finished-row" not in c[:400]
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", default="build/free.html")
    ap.add_argument("--manifest", default="build/redacted-manifest.json")
    args = ap.parse_args()

    for path in (args.free, args.manifest):
        if not os.path.exists(path):
            print(f"FAIL  missing {path} -- run: python3 generate_site.py --tier free")
            return 1

    html = open(args.free).read()
    # Search the RENDERED page only. Stylesheets and scripts are full of
    # incidental numbers -- "letter-spacing: -0.2px" matched a redacted value
    # and reported a leak that was a CSS declaration.
    haystack = re.sub(r"<style\b.*?</style>|<script\b.*?</script>", "",
                      html, flags=re.S | re.I)
    manifest = json.load(open(args.manifest))
    values = manifest.get("values", [])

    failures: list[str] = []

    # (3) the redaction must actually have done something
    if len(values) < 10:
        failures.append(
            f"redaction manifest holds only {len(values)} value(s) -- the redaction "
            f"looks like it stopped working, so the assertions below prove nothing"
        )

    # (1) no redacted value may survive anywhere in the free payload
    checked = leaked = 0
    for value in values:
        s = str(value)
        if len(s) < MIN_VALUE_LEN:
            continue
        checked += 1
        where = _leak_at(s, haystack)
        if where is not None:
            leaked += 1
            if leaked <= 8:
                context = re.sub(r"\s+", " ", haystack[max(0, where - 70):where + 70])
                failures.append(f"redacted value {s!r} present in the free payload: ...{context}...")

    if leaked > 8:
        failures.append(f"...and {leaked - 8} further redacted value(s) present")

    # (2) no model structure inside any UNGRADED fight card
    ungraded = _ungraded_cards(haystack)
    if not ungraded:
        failures.append("found no ungraded fight cards -- the check has nothing to verify, "
                        "which usually means the card markup changed")
    for card in ungraded:
        for marker in MODEL_MARKERS:
            if marker in card:
                name = re.search(r'data-fight-key="([^"]*)"', card)
                who = name.group(1) if name else "unknown fight"
                failures.append(f"model marker {marker!r} rendered on ungraded fight: {who}")

    print(f"free payload    {len(html):,} bytes")
    print(f"ungraded cards  {len(ungraded)} checked for model markup")
    print(f"values checked {checked} of {len(values)} in the manifest "
          f"({len(values) - checked} too short to be distinctive)")

    if failures:
        print(f"\nFAIL  {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        print("\nThe free payload contains paid content. Do not deploy.")
        return 1

    print("\nPASS  no model output found in the free payload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
