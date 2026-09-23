#!/usr/bin/env python3
"""
Two names, one fighter -- found before a card instead of after one.

This has now happened twice and cost something both times.

  Sean King / "Sean King III"   2026-09-12. card_discovery read the variant's
                                card row as a REPLACEMENT, cancelled the real
                                bout and voided its prediction. He won by KO
                                in round one, so a CORRECT pick left the
                                published record and had to be restored by an
                                owner-authorised section-1 correction.

  Valesca Machado / "Tina Black" 2026-09-23. Same branch, same outcome, three
                                days before the fight. Every book priced her
                                as Valesca Machado, so the prices attached to
                                the row marked cancelled while the row the
                                card displayed had none.

Both were found by accident, while looking at something else. That is not a
process.

THE SIGNAL. A bout is keyed (date, opponent) from one fighter's side. Two
DIFFERENT names cannot hold the same key unless they are the same person --
A's bout against X on date D is A's alone, and X's side of it keys as
(D, A), not (D, X).

The one honest exception is a tournament night: if X fought both A and B on
D, then A and B each legitimately hold (D, X). So a single shared bout proves
nothing and MIN_SHARED exists to swallow it. Two unrelated fighters sharing
three or more identical (date, opponent) pairs is not a bracket, it is one
career written down twice.

Reports; never fails. A name-hygiene check that can freeze a site real money
is staked against would be a worse bug than the one it looks for.

    python3 scripts/check_duplicate_identities.py
"""

import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.names import NAME_ALIASES, canonical_name  # noqa: E402

HISTORY = "data/fight_history.csv"
ROSTER = "data/fighters.csv"
CARDS = ("data/fight_cards.csv", "data/future_cards.csv")

# Three, for the tournament case in the docstring. Both real incidents shared
# far more than this -- Sean King 7 bouts, Valesca Machado all 20 -- so the
# threshold costs nothing against the cases it exists to catch.
MIN_SHARED = 3


def _fold(v) -> str:
    return canonical_name(str(v or "")).strip().lower()


def _bouts():
    """name -> {(date, opponent)}, both folded through the alias table."""
    out = collections.defaultdict(set)
    try:
        with open(HISTORY, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                a, b = _fold(r.get("fighter_a")), _fold(r.get("fighter_b"))
                d = str(r.get("date") or "")[:10]
                if not a or not b or not d:
                    continue
                out[a].add((d, b))
                out[b].add((d, a))
    except (OSError, ValueError) as exc:
        print(f"[dupes] could not read {HISTORY} ({exc}) -- skipping")
        return {}
    return out


def _names_from(path, cols):
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return {_fold(r.get(c)) for r in csv.DictReader(fh) for c in cols} - {""}
    except (OSError, ValueError):
        return set()


def find_duplicates(bouts, min_shared=MIN_SHARED):
    """Name pairs sharing at least `min_shared` identical (date, opponent) bouts."""
    # Inverted index rather than pairwise: the roster is ~400 names and the
    # spine ~12k bouts, so comparing every pair would be 80k set
    # intersections for a check that runs every build.
    claims = collections.defaultdict(set)
    for name, bs in bouts.items():
        for key in bs:
            claims[key].add(name)

    shared = collections.Counter()
    for key, names in claims.items():
        if len(names) < 2:
            continue
        ordered = sorted(names)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                shared[(a, b)] += 1

    return sorted(((a, b, n) for (a, b), n in shared.items() if n >= min_shared),
                  key=lambda t: -t[2])


def main() -> int:
    bouts = _bouts()
    if not bouts:
        return 0
    found = find_duplicates(bouts)

    carded = set()
    for p in CARDS:
        carded |= _names_from(p, ("fighter_a", "fighter_b"))
    roster = _names_from(ROSTER, ("name",))

    rows = []
    for a, b, n in found:
        rows.append({
            "names": [a, b],
            "shared_bouts": n,
            # A pair where BOTH sides are on a card is the urgent shape: that
            # is the one card_discovery turns into a phantom replacement.
            "on_a_card": sorted({x for x in (a, b) if x in carded}),
            "on_roster": sorted({x for x in (a, b) if x in roster}),
        })

    try:
        from src.source_health import record
        record("duplicate_identities", {"ok": not rows, "count": len(rows),
                                        "pairs": rows[:10]})
    except Exception as exc:                      # noqa: BLE001 -- never fatal
        print(f"[dupes] health not recorded ({exc}) -- continuing")

    if not rows:
        print(f"[dupes] clean -- {len(bouts)} fighters, no name shares "
              f"{MIN_SHARED}+ bouts with another")
        return 0

    print(f"[dupes] {len(rows)} possible duplicate identity/identities "
          f"({len(NAME_ALIASES)} alias(es) already applied):")
    for r in rows:
        where = []
        if r["on_a_card"]:
            where.append(f"ON A CARD: {', '.join(r['on_a_card'])}")
        if r["on_roster"]:
            where.append(f"on roster: {', '.join(r['on_roster'])}")
        print(f"  {r['names'][0]!r} == {r['names'][1]!r}  "
              f"({r['shared_bouts']} identical bouts)"
              + (f"  [{' | '.join(where)}]" if where else ""))
    print("  If these are one fighter: add an alias to src/names.py FIRST -- a "
          "data merge without one is undone by the next build -- then merge.")

    # THE STEP SUMMARY, NOT THE SITE. This briefly rendered on the health
    # strip and had to be pulled: every other alert there is transient and
    # self-clearing, while a duplicate identity persists until someone merges
    # it -- so findings sat permanently above the fight card telling readers
    # to edit a source file. This is a maintenance item and belongs where
    # maintenance is read.
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(f"### {len(rows)} possible duplicate identity/identities\n\n")
                fh.write("| names | shared bouts | on a card |\n|---|---|---|\n")
                for r in rows:
                    fh.write(f"| `{r['names'][0]}` = `{r['names'][1]}` "
                             f"| {r['shared_bouts']} "
                             f"| {', '.join(r['on_a_card']) or '-'} |\n")
                fh.write("\nAlias in `src/names.py` first -- a data merge "
                         "without one is undone by the next build.\n")
        except OSError as exc:
            print(f"[dupes] step summary not written ({exc}) -- continuing")
    return 0            # ALWAYS. See the docstring.


if __name__ == "__main__":
    sys.exit(main())
