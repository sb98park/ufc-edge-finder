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

    Checks EVERY <style> block, not just the first. This used to be a
    re.search for one block, which was correct while the page had exactly
    one -- then a second, tiny <style> was added at the very top of <head>
    (the one-rule black background that has to paint before anything
    network-bound resolves). That block became the first match, it is
    trivially balanced, and so this check started passing instantly without
    ever looking at the real stylesheet below it. A silently-vacuous check
    is worse than no check, because it still reports a pass.
    """
    blocks = re.findall(r"<style>(.*?)</style>", c, re.DOTALL)
    if not blocks:
        return fail("css-braces", "no <style> block found")
    for idx, raw in enumerate(blocks):
        body = re.sub(r"/\*.*?\*/", " ", raw, flags=re.DOTALL)
        where = f"block {idx + 1} of {len(blocks)}"
        opens, closes = body.count("{"), body.count("}")
        if opens != closes:
            fail("css-braces", f"{where}: {opens} open vs {closes} close")
            continue
        depth = 0
        for i, ch in enumerate(body):
            depth += (ch == "{") - (ch == "}")
            if depth < 0:
                line = body[:i].count("\n") + 1
                fail("css-braces",
                     f"{where}: stray closing brace near line {line} of that style block")
                break
    print(f"       [css-braces] checked {len(blocks)} style block(s)")


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


def check_method_coherence(c):
    """
    Every fighter's three per-fighter method rows must sum to his win
    probability, and never above 100%.

    These are mutually exclusive outcomes -- the sums aren't a calibration
    target, they're arithmetic. They were wrong FOUR times for four different
    reasons: unnormalised method-given-win rates; a fix applied to only one of
    two code paths; predict_matchup called with different arguments in each
    path; and the projection path missing fight_history_df while the priced
    path had it. Every version produced plausible-looking numbers, one showing
    a 15-point edge on a submission prop that was pure arithmetic.

    The FIRST version of this check regexed markup that doesn't exist and
    reported "no rows found" -- inert, and worse than absent because it looked
    like a pass. This one is written against the real cell structure:
        <td class="mkt-label">Fighter &mdash; Method</td>
        <td class="mkt-model" data-pct="21.5%" data-odds="+365">21.5%</td>
    """
    pairs = re.findall(
        # [^>]* so the cell may carry attributes. It gained data-pct/data-odds
        # when the Model column became unit-toggleable, and this check failed
        # with "the pattern no longer matches the markup" -- which is the
        # check working, but the pattern was needlessly brittle: it pinned the
        # class to be the LAST attribute, which no markup should have to promise.
        r'<td class="mkt-label">(.*?)</td>\s*<td class="mkt-model"[^>]*>([\d.]+)%</td>',
        c, re.DOTALL)
    if not pairs:
        return fail("method-coherence",
                    "found no market rows at all -- the pattern no longer matches the markup")

    # GROUPED BY FIGHT, not by fighter. The first version summed per fighter
    # and flagged anything over 100% -- which would NOT have caught the
    # original failure, where the two fighters were 73.5% and 53.1%: each
    # under 100, together 126.6%. The invariant lives on the pair, because the
    # six outcomes are exhaustive for the FIGHT.
    blocks = re.split(r'<details class="fight-card', c)[1:]
    if not blocks:
        return warn("method-coherence", "no fight cards found")

    checked, seen_methods = 0, 0
    for block in blocks:
        key = re.search(r'data-fight-key="([^"]+)"', block)
        rows = re.findall(
            # [^>]* so the cell may carry attributes. It gained data-pct/data-odds
        # when the Model column became unit-toggleable, and this check failed
        # with "the pattern no longer matches the markup" -- which is the
        # check working, but the pattern was needlessly brittle: it pinned the
        # class to be the LAST attribute, which no markup should have to promise.
        r'<td class="mkt-label">(.*?)</td>\s*<td class="mkt-model"[^>]*>([\d.]+)%</td>',
            block, re.DOTALL)
        total = 0.0
        n = 0
        for label, prob in rows:
            label = re.sub(r"<[^>]+>", "", label).strip()
            # Per-fighter method rows only. Fight-level rows ("Fight ends
            # by ...") and moneylines have no dash-separated method, and
            # counting them would double the total legitimately.
            if not re.match(r"^.*?\s*[\u2014\u2013-]\s*(KO/TKO|Submission|Decision)$", label):
                continue
            total += float(prob)
            n += 1
        if n == 0:
            continue
        seen_methods += n
        checked += 1
        name = key.group(1) if key else "unknown fight"
        if n == 6 and abs(total - 100.0) > 1.5:
            fail("method-coherence",
                 f"{name}: six method rows sum to {total:.1f}%, not 100%")
        elif total > 101.5:
            fail("method-coherence",
                 f"{name}: {n} method rows already sum to {total:.1f}%")

    if not checked:
        return warn("method-coherence",
                    f"{len(blocks)} fight cards found but none had per-fighter method rows")
    print(f"       [method-coherence] checked {checked} fight(s), {seen_methods} method rows")


def check_headline_matches_table(c):
    """
    The donut headline's method must be the favourite's highest method row.

    Both come from the model; they came from DIFFERENT computations. The
    headline used a hand-weighted divisional blend while the rows used the
    fitted, reconciled grid, so a card could read "Gamrot by Submission" with
    Submission at 13.5% and Decision at 32.0% two inches below.

    That is the fifth instance today of two code paths computing the same
    quantity differently. The others were caught by reading numbers on a
    phone; this check catches it at build time.
    """
    blocks = re.split(r'<details class="fight-card', c)[1:]
    if not blocks:
        return warn("headline-vs-table", "no fight cards found")

    checked = 0
    for block in blocks:
        m = re.search(r'by\s+(KO/TKO|Submission|Decision)', block)
        if not m:
            continue
        headline_method = m.group(1)
        fav = re.search(r'data-fight-key="([^"|]+)', block)
        rows = re.findall(
            # [^>]* so the cell may carry attributes. It gained data-pct/data-odds
        # when the Model column became unit-toggleable, and this check failed
        # with "the pattern no longer matches the markup" -- which is the
        # check working, but the pattern was needlessly brittle: it pinned the
        # class to be the LAST attribute, which no markup should have to promise.
        r'<td class="mkt-label">(.*?)</td>\s*<td class="mkt-model"[^>]*>([\d.]+)%</td>',
            block, re.DOTALL)
        best, best_p, name_of_best = None, -1.0, None
        per_fighter = {}
        for label, prob in rows:
            label = re.sub(r"<[^>]+>", "", label).strip()
            mm = re.match(r"^(.*?)\s*[\u2014\u2013-]\s*(KO/TKO|Submission|Decision)$", label)
            if not mm:
                continue
            per_fighter.setdefault(mm.group(1).strip(), []).append((mm.group(2), float(prob)))
        if not per_fighter:
            continue
        # The favourite is whoever's methods sum highest -- no need to parse
        # the headline name out of prose.
        fav_name = max(per_fighter, key=lambda k: sum(p for _, p in per_fighter[k]))
        top = max(per_fighter[fav_name], key=lambda t: t[1])
        checked += 1
        if top[0] != headline_method:
            fail("headline-vs-table",
                 f"{fav_name}: headline says {headline_method}, table's highest is "
                 f"{top[0]} at {top[1]:.1f}%")

    if checked:
        print(f"       [headline-vs-table] checked {checked} fight(s)")


def check_round_props_monotonic(c):
    """
    Within a fight, P(Under X) must increase with X.

    Pure arithmetic: a fight ending before 0.5 rounds also ended before 1.5.
    It was violated by a dict lookup keyed on the line with a default -- 0.5
    wasn't a key, so it fell through and produced the SAME value as 2.5. A
    card showed "Under 0.5  51.3%" beside "Under 2.5  51.3%", which reads as
    a plausible number and is impossible.
    """
    blocks = re.split(r'<details class="fight-card', c)[1:]
    checked = 0
    for block in blocks:
        rows = re.findall(
            # [^>]* so the cell may carry attributes. It gained data-pct/data-odds
        # when the Model column became unit-toggleable, and this check failed
        # with "the pattern no longer matches the markup" -- which is the
        # check working, but the pattern was needlessly brittle: it pinned the
        # class to be the LAST attribute, which no markup should have to promise.
        r'<td class="mkt-label">(.*?)</td>\s*<td class="mkt-model"[^>]*>([\d.]+)%</td>',
            block, re.DOTALL)
        unders = []
        for label, prob in rows:
            label = re.sub(r"<[^>]+>", "", label).strip()
            m = re.match(r"^Total Rounds Under ([\d.]+)$", label)
            if m:
                unders.append((float(m.group(1)), float(prob)))
        if len(unders) < 2:
            continue
        checked += 1
        unders.sort()
        key = re.search(r'data-fight-key="([^"]+)"', block)
        name = key.group(1) if key else "unknown fight"
        for (l1, p1), (l2, p2) in zip(unders, unders[1:]):
            if p2 < p1 - 0.05:
                fail("round-monotonic",
                     f"{name}: Under {l1} is {p1:.1f}% but Under {l2} is {p2:.1f}% "
                     f"-- a longer line cannot be less likely")
            elif abs(p2 - p1) < 0.05:
                fail("round-monotonic",
                     f"{name}: Under {l1} and Under {l2} are both {p1:.1f}% "
                     f"-- distinct lines cannot share a probability")
    if checked:
        print(f"       [round-monotonic] checked {checked} fight(s)")


def check_distance_vs_rounds(c):
    """
    Over (the widest line) must be at least P(decision), on every fight.

    A decision means the fight reached the final bell, so it is necessarily
    Over the last half-round mark. The gap between them is the finishes that
    land in that final 2:30 -- small, but never negative.

    Nothing guarded this. The two figures come from different code paths (the
    method model and the round-total finder), and every previous disagreement
    on this card started exactly that way: two paths computing related
    quantities without a check tying them together.
    """
    blocks = re.split(r'<details class="fight-card', c)[1:]
    checked = 0
    for block in blocks:
        rows = re.findall(
            # [^>]* so the cell may carry attributes. It gained data-pct/data-odds
        # when the Model column became unit-toggleable, and this check failed
        # with "the pattern no longer matches the markup" -- which is the
        # check working, but the pattern was needlessly brittle: it pinned the
        # class to be the LAST attribute, which no markup should have to promise.
        r'<td class="mkt-label">(.*?)</td>\s*<td class="mkt-model"[^>]*>([\d.]+)%</td>',
            block, re.DOTALL)
        decision, overs = None, []
        for label, prob in rows:
            label = re.sub(r"<[^>]+>", "", label).strip()
            if label == "Fight ends by Decision":
                decision = float(prob)
            m = re.match(r"^Total Rounds Over ([\d.]+)$", label)
            if m:
                overs.append((float(m.group(1)), float(prob)))
        if decision is None or not overs:
            continue
        checked += 1
        widest_line, widest_over = max(overs)
        key = re.search(r'data-fight-key="([^"]+)"', block)
        name = key.group(1) if key else "unknown fight"
        # Tolerance covers rounding only; a real inversion is far larger.
        if widest_over < decision - 1.0:
            fail("distance-vs-rounds",
                 f"{name}: Over {widest_line} is {widest_over:.1f}% but decision is "
                 f"{decision:.1f}% -- a decision is necessarily over the last line")
    if checked:
        print(f"       [distance-vs-rounds] checked {checked} fight(s)")


# Observed frequencies on the 2019+ holdout (research_method_fightlevel.py
# and research_method_given_win.py). Not targets -- reference points.
BASE_RATES = {
    "Fight ends by KO/TKO": 0.312,
    "Fight ends by Submission": 0.165,
    "Fight ends by Decision": 0.524,
}


def check_plausibility(c):
    """
    Warn when a model output sits far outside its historical base rate.

    THE GAP THIS FILLS. Every other check here tests arithmetic -- methods
    summing to 1, Under rising with the line, the headline matching its table.
    Those catch impossible numbers. They do not catch numbers that are merely
    absurd, and today's expensive failures were all in that second category:
    65% submission on a fight (base rate 16.5%), 100% decision, 60.6%
    submission against a market at 15.4%. Each was internally consistent and
    obviously wrong to anyone who knows the sport.

    A machine can't know Makhachev doesn't submit people two thirds of the
    time. It CAN know that 65% is four times the base rate and say so.

    WARNS rather than fails, deliberately. A genuine outlier exists -- some
    fights really are 60% KO -- and a check that blocks the build on an
    unusual-but-correct number would get muted within a week. The job here is
    to put it in front of a human, not to decide.
    """
    blocks = re.split(r'<details class="fight-card', c)[1:]
    flagged = checked = 0
    for block in blocks:
        key = re.search(r'data-fight-key="([^"]+)"', block)
        name = key.group(1) if key else "unknown fight"
        for label, prob in re.findall(
                # [^>]* so the cell may carry attributes. It gained data-pct/data-odds
        # when the Model column became unit-toggleable, and this check failed
        # with "the pattern no longer matches the markup" -- which is the
        # check working, but the pattern was needlessly brittle: it pinned the
        # class to be the LAST attribute, which no markup should have to promise.
        r'<td class="mkt-label">(.*?)</td>\s*<td class="mkt-model"[^>]*>([\d.]+)%</td>',
                block, re.DOTALL):
            label = re.sub(r"<[^>]+>", "", label).strip()
            base = BASE_RATES.get(label)
            if base is None:
                continue
            checked += 1
            p = float(prob) / 100.0
            # TWO tests, because either alone has a blind spot.
            #
            # The RATIO catches a rare outcome inflated out of proportion --
            # 65% submission against a 16.5% base rate is 3.9x and obviously
            # wrong. A flat percentage-point threshold can't do that, since
            # 5pp is enormous at 16.5% and noise at 52%.
            #
            # The CEILING catches what the ratio misses: 100% decision is only
            # 1.9x its 52.4% base rate and sails through, yet no fight is
            # certain. Any method above 85% deserves a look whatever its base
            # rate, and the floor does the same for a near-zero.
            ratio = p / base if base else 0
            reason = None
            if p >= 0.85:
                reason = f"{p:.1%} -- no fight is that certain"
            elif p <= 0.02:
                reason = f"{p:.1%} -- effectively ruled out"
            elif ratio >= 2.5 or ratio <= 0.35:
                reason = f"{p:.1%}, {ratio:.1f}x the {base:.1%} base rate"
            if reason:
                flagged += 1
                warn("plausibility",
                     f"{name}: {label} is {reason} -- worth an eye before betting it")
    if checked:
        print(f"       [plausibility] checked {checked} row(s), {flagged} worth a look")


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
    check_method_coherence(c)
    check_headline_matches_table(c)
    check_round_props_monotonic(c)
    check_distance_vs_rounds(c)
    check_plausibility(c)

    for label, items in (("FAIL", FAILURES), ("WARN", WARNINGS)):
        for check, detail in items:
            print(f"  {label}  [{check}] {detail}")
    if not FAILURES and not WARNINGS:
        print("  all checks passed")
    print(f"\n{len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
