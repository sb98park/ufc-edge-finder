"""
Shortening a fighter's name to the part a reader recognises.

WHY THIS EXISTS. Five places on the card shortened a name with
`name.split(' ')[-1]` -- the scout pills, the two donut labels, the summary
pick, and the drawer's own copy in JavaScript. That rule is wrong for two
whole classes of name, and it was wrong on the live site:

    Reinier de Ridder        -> "Ridder"     (particle orphaned)
    Khalil Rountree Jr.      -> "Jr."        (suffix mistaken for the name)

38 fighters in ufc_fight_results.csv carry a nobiliary particle and 8 carry a
generational suffix, so this was visible on roughly 1.7% of every name the
site has ever printed, including a main-event fighter.

THE RULE, and why it runs BACKWARDS. Take the last word, then keep absorbing
words to its left for as long as they are particles. Running forwards from the
first particle instead would turn "Tiago dos Santos e Silva" into "dos Santos
e Silva" -- correct in Portuguese, four tokens wide in a pill that has room
for one. Backwards absorption gives the short form every one of these names is
actually listed under, and it stops on its own at the first ordinary word:

    Chris de la Rocha        -> "de la Rocha"    (absorbs "la", then "de")
    Douglas Silva de Andrade -> "de Andrade"     (stops at "Silva")
    Elizeu Zaleski dos Santos-> "dos Santos"     (stops at "Zaleski")

WHAT IS DELIBERATELY NOT A PARTICLE. "bin", "ibn", "al" and "el" are real
particles in Arabic names and are NOT in the set below, because the corpus
contains "Sung Bin Jo" -- where "Bin" is the second syllable of a Korean given
name and the naive answer was already right. Adding "bin" to be thorough would
have broken a name that worked, to fix zero names that were broken. No fighter
in the corpus needs those four, so they stay out until one does.
"""

from __future__ import annotations

# Lowercased and stripped of any trailing period before lookup.
PARTICLES = frozenset({
    "de", "del", "della", "dello", "den", "der", "des", "di", "do", "dos",
    "da", "das", "du", "la", "le", "van", "von", "ten", "ter", "vander",
    "saint", "st",
})

# Kept ATTACHED to the surname rather than stripped, because "Rountree Jr."
# and "Rountree" are two different fighters often enough in this sport that
# dropping it would be its own defect (Lance Gibson Jr. and Lance Gibson both
# have UFC bouts on record).
SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def _norm(token: str) -> str:
    return token.strip().strip(".,").lower()


def surname(full) -> str:
    """
    The recognisable part of a fighter's name. Never returns empty for a
    non-empty input -- a mononym ("Shogun") is its own surname.
    """
    words = str(full or "").split()
    if not words:
        return ""

    # Peel any generational suffix off the end first, so the particle scan
    # sees the real last word, then put it back. Without this, "Marcos de
    # Lima Jr." would look at "Lima" (not a particle) and stop early.
    tail = []
    while len(words) > 1 and _norm(words[-1]) in SUFFIXES:
        tail.insert(0, words.pop())

    i = len(words) - 1
    while i > 1 and _norm(words[i - 1]) in PARTICLES:
        i -= 1

    return " ".join(words[i:] + tail)
