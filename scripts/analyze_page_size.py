"""
What is actually making docs/index.html big?

The page reached 2.8 MB after card discovery started working and six events
appeared instead of one. Before optimising anything, this reports where the
bytes are -- and, more importantly, what actually ships.

RAW SIZE IS NOT TRANSFER SIZE. GitHub Pages serves gzip, and this page is
about as compressible as HTML gets: thousands of near-identical table rows,
repeated class names, and SVG path data made of the same few characters.
A 2.8 MB file can leave the server well under 400 KB. Optimising the raw
number without checking the compressed one is optimising the wrong figure.

Usage:
    python3 scripts/analyze_page_size.py
"""

import gzip
import os
import re
import sys

PAGE = "docs/index.html"

# Ordered so the most specific patterns are measured first; each byte is
# attributed once, to the first pattern that claims it.
SECTIONS = [
    ("inline <svg> blocks", r"<svg\b.*?</svg>"),
    ("<style> blocks", r"<style>.*?</style>"),
    ("<script> blocks", r"<script\b.*?</script>"),
    ("<table> markup", r"<table\b.*?</table>"),
    ("HTML comments", r"<!--.*?-->"),
]


def main():
    if not os.path.exists(PAGE):
        print(f"{PAGE} missing -- run generate_site.py first.")
        sys.exit(1)
    raw = open(PAGE, "rb").read()
    text = raw.decode("utf-8", "replace")
    n = len(raw)
    gz = len(gzip.compress(raw, 6))

    print(f"raw        {n/1024:>9,.0f} KB")
    print(f"gzipped    {gz/1024:>9,.0f} KB   ({gz/n*100:.0f}% of raw -- this is what a visitor downloads)")
    print(f"br (est)   {gz*0.85/1024:>9,.0f} KB   (brotli typically ~15% under gzip)\n")

    remaining = text
    print(f"  {'section':<24}{'raw KB':>10}{'share':>8}{'count':>8}")
    print("  " + "-" * 50)
    claimed = 0
    for label, pat in SECTIONS:
        found = re.findall(pat, remaining, re.DOTALL)
        size = sum(len(f.encode("utf-8")) for f in found)
        claimed += size
        remaining = re.sub(pat, "", remaining, flags=re.DOTALL)
        print(f"  {label:<24}{size/1024:>10,.0f}{size/n*100:>7.0f}%{len(found):>8}")
    other = n - claimed
    print(f"  {'everything else':<24}{other/1024:>10,.0f}{other/n*100:>7.0f}%")

    # The repeated-artifact question: how much is one fight worth?
    fights = len(re.findall(r'class="[^"]*fight-card', text))
    if fights:
        print(f"\n  {fights} fight card(s) on the page -> ~{n/fights/1024:.0f} KB each raw, "
              f"~{gz/fights/1024:.0f} KB each gzipped")

    # Biggest single repeated strings are the real optimisation targets: a
    # long string repeated N times collapses to almost nothing under gzip,
    # so it matters far less than its raw share suggests.
    print("\n  LARGEST REPEATED BLOCKS (raw cost x occurrences)")
    svgs = re.findall(r"<svg\b.*?</svg>", text, re.DOTALL)
    seen = {}
    for s in svgs:
        seen[s] = seen.get(s, 0) + 1
    dupes = sorted(((len(v.encode()) * c, len(v.encode()), c) for v, c in seen.items() if c > 1),
                   reverse=True)[:5]
    if dupes:
        for total, each, count in dupes:
            print(f"     {each/1024:>6.1f} KB x {count:<4} = {total/1024:>7.1f} KB "
                  f"-- identical, so a <use> reference or gzip already handles it")
    else:
        print("     none -- no identical SVG is emitted twice")

    print("\n  READ THIS AS: optimise the GZIPPED number. Raw size that comes from")
    print("  repetition is nearly free once compressed; unique content is not.")


if __name__ == "__main__":
    main()
