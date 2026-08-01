"""
Pre-push checks for the bug classes this project actually produces.

Every check below exists because the bug it catches SHIPPED at least once and
was found by a human on a real device. None are hypothetical. The point is
that each cost multiple rounds of "I don't see the change" or "this section
is broken", and every one is detectable in seconds from the rendered output.

Run: python3 scripts/lint_site.py
Exit code 1 if anything fails, so it can gate a push.
"""

import json
import os
import re
import subprocess
import sys

RENDERED = "docs/index.html"
FAILURES, WARNINGS = [], []


def fail(check, detail):
    FAILURES.append((check, detail))


def warn(check, detail):
    WARNINGS.append((check, detail))


def load():
    if not os.path.exists(RENDERED):
        print("docs/index.html missing -- run generate_site.py first.")
        sys.exit(1)
    return open(RENDERED, encoding="utf-8").read()


def check_css_braces(c):
    """
    A stray } silently kills EVERY rule after it. Cost ~8 rounds: the CSS was
    verifiably present in the file and simply never applied, which looked
    exactly like a deploy problem.
    """
    css = re.search(r"<style>(.*?)</style>", c, re.DOTALL)
    if not css:
        return fail("css-braces", "no <style> block found")
    body = re.sub(r"/\*.*?\*/", " ", css.group(1), flags=re.DOTALL)
    opens, closes = body.count("{"), body.count("}")
    if opens != closes:
        return fail("css-braces", f"{opens} open vs {closes} close")
    depth = 0
    for i, ch in enumerate(body):
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            line = body[:i].count("\n") + 1
            return fail("css-braces", f"stray closing brace near line {line} of the style block")


def check_js_syntax(c):
    """A syntax error anywhere kills every later handler in the same block."""
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", c, re.DOTALL)
    if not scripts:
        return fail("js-syntax", "no inline script found")
    biggest = max(scripts, key=len)
    open("/tmp/_lint.js", "w").write(biggest)
    try:
        r = subprocess.run(["node", "--check", "/tmp/_lint.js"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fail("js-syntax", r.stderr.strip().splitlines()[0] if r.stderr else "node --check failed")
        return
    except FileNotFoundError:
        # node isn't installed everywhere. A linter that CRASHES is worse
        # than one that skips a check, so fall back to a structural balance
        # test -- it won't catch every syntax error, but it does catch the
        # unclosed brace or paren that a bad edit actually produces.
        pass

    stripped = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', biggest)
    stripped = re.sub(r"'(?:[^'\\\n]|\\.)*'", "''", stripped)
    stripped = re.sub(r"`(?:[^`\\]|\\.)*`", "``", stripped)
    stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.DOTALL)
    stripped = re.sub(r"//[^\n]*", "", stripped)
    for opener, closer, label in (("{", "}", "braces"), ("(", ")", "parens"), ("[", "]", "brackets")):
        o, cl = stripped.count(opener), stripped.count(closer)
        if o != cl:
            fail("js-syntax", f"unbalanced {label}: {o} open vs {cl} close "
                              f"(node not installed -- structural check only)")
    warn("js-syntax", "node not installed; ran a structural check only. "
                      "`brew install node` enables full syntax validation.")


def _function_body(src, start):
    """Extract a function body by brace matching from its opening paren."""
    i = src.find("{", start)
    if i < 0:
        return ""
    depth, j = 0, i
    while j < len(src):
        depth += (src[j] == "{") - (src[j] == "}")
        if depth == 0:
            return src[i:j + 1]
        j += 1
    return src[i:i + 4000]


def _strip_nested(body):
    """Remove nested function and arrow bodies, leaving only top-level code."""
    out, depth, i = [], 0, 0
    while i < len(body):
        if body.startswith("function", i) or body.startswith("=>", i):
            j = body.find("{", i)
            if j >= 0:
                d, k = 0, j
                while k < len(body):
                    d += (body[k] == "{") - (body[k] == "}")
                    if d == 0:
                        break
                    k += 1
                i = k + 1
                continue
        out.append(body[i])
        i += 1
    return "".join(out)


def check_deferred_inits(c):
    """
    The page body is emitted AFTER the inline script, so any getElementById
    at parse time returns null and an early-return guard silently wires
    NOTHING. Hit four separate times in one feature -- the filter chips were
    dead for days before anyone noticed.
    """
    for m in re.finditer(r"function (init\w+)\s*\(", c):
        name = m.group(1)
        body = _function_body(c, m.end())
        # Only IMMEDIATE queries matter. A lookup inside a nested callback
        # runs when that callback fires, which is normally after load -- a
        # fixed character window flagged both of those and the next
        # function's body too, so the check has to brace-match and strip
        # nested functions before looking.
        top = _strip_nested(body)
        if "getElementById" not in top and "querySelector" not in top:
            continue
        if f"DOMContentLoaded', {name}" not in c and f'DOMContentLoaded", {name}' not in c:
            fail("deferred-init", f"{name}() queries the DOM at init time but is never deferred")


def check_swipe_exclusions(c):
    """
    A horizontally scrollable or draggable strip that isn't excluded from the
    document-level swipe handler navigates the page instead of scrolling
    itself. Caught on the header pill and again on the Record tab's card bar.
    """
    m = re.search(r"closest\('([^']*table-scroll[^']*)'\)\)\s*\{\s*\n\s*tracking = false", c)
    if not m:
        return warn("swipe-exclusions", "could not locate the exclusion list")
    excluded = {s.strip() for s in m.group(1).split(",")}
    css = re.search(r"<style>(.*?)</style>", c, re.DOTALL).group(1)
    for mm in re.finditer(r"([^{}]+)\{[^}]*overflow-x:\s*auto", css):
        for sel in mm.group(1).split(","):
            sel = sel.strip().split()[0] if sel.strip() else ""
            if sel.startswith(".") and sel not in excluded:
                fail("swipe-exclusions", f"{sel} scrolls horizontally but isn't excluded")


def check_unrendered_jinja(c):
    """
    Nested {{ }} inside an existing expression prints as literal text. Shipped
    once as "{{ icon_underdog() }} Underdog" visible on the live site.
    """
    for pat, label in ((r"\{\{[^}]*\{\{", "nested {{"), (r"\{%[^%]*\{%", "nested {%")):
        if re.search(pat, c):
            fail("jinja", f"{label} reached the rendered page")
    # Any {{ }} at all. The original pattern required a BARE variable, so it
    # missed "{{ icon_underdog() }} Underdog" -- which is the exact form that
    # shipped to the live site. A clean render contains zero of these.
    leftover = re.findall(r"\{\{[^{}]{0,90}\}\}", c)
    if leftover:
        fail("jinja", f"{len(leftover)} unrendered expression(s), e.g. {leftover[0][:50]!r}")


def check_exact_name_matching():
    """
    Polymarket writes "Uros Medic" while the roster holds "Uroš Medić". An
    exact == match silently drops every accented fighter -- it removed their
    per-fighter KO lines and duplicated round totals before being found.
    """
    for path in ("src/edge_finder.py", "src/card_matcher.py"):
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8").read()
        for m in re.finditer(r'fighters_df\[fighters_df\["name"\] == ([^\]]+)\]', src):
            line = src[:m.start()].count("\n") + 1
            # Skip the tolerant helper itself -- it performs the exact match
            # first BY DESIGN, then falls back to accent folding.
            defs = [d.start() for d in re.finditer(r"^def \w+", src[:m.start()], re.M)]
            enclosing = src[defs[-1]:m.start()] if defs else ""
            if enclosing.startswith("def _find_fighter"):
                continue
            warn("name-matching", f"{path}:{line} matches fighter names exactly")


def check_market_string_consistency():
    """
    Modules coin their own market labels -- "FightMethod" vs "Fight Method:",
    "GoesTheDistance" vs "Fight Outcome:". Four separate bugs came from
    downstream code string-matching against the wrong one.
    """
    labels = {}
    for path in ("src/edge_finder.py", "src/model_preview.py", "src/polymarket_source.py"):
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8").read()
        for m in re.finditer(r'"market":\s*f?"([A-Za-z][^"{]*)"', src):
            labels.setdefault(re.sub(r"[:\s].*$", "", m.group(1)), set()).add(os.path.basename(path))
    norm = {}
    for label in labels:
        norm.setdefault(label.lower().replace(" ", ""), set()).add(label)
    for key, variants in norm.items():
        if len(variants) > 1:
            warn("market-labels", f"same market spelled {sorted(variants)}")


def check_probability_coherence():
    """
    Fight-level KO + SUB + DEC must sum to 1. They came from three different
    models once and summed to 103.8% on screen.
    """
    try:
        sys.path.insert(0, ".")
        from src.method_model import method_probabilities
        d = method_probabilities(0.1, 0.04, 0.7, 0.35, 0.5, 0.3, 3)
        if d and abs(sum(d.values()) - 1.0) > 1e-6:
            fail("prob-coherence", f"method probabilities sum to {sum(d.values()):.4f}")
    except Exception as e:
        warn("prob-coherence", f"could not evaluate ({e})")


def main():
    c = load()
    check_css_braces(c)
    check_js_syntax(c)
    check_deferred_inits(c)
    check_swipe_exclusions(c)
    check_unrendered_jinja(c)
    check_exact_name_matching()
    check_market_string_consistency()
    check_probability_coherence()

    for label, items in (("FAIL", FAILURES), ("WARN", WARNINGS)):
        for check, detail in items:
            print(f"  {label}  [{check}] {detail}")
    if not FAILURES and not WARNINGS:
        print("  all checks passed")
    print(f"\n{len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
