"""
Shared scoring statistics for the validate_* harnesses.

Every harness had grown its own copy of the same paired bootstrap. Two
defects were therefore present in all of them at once, and fixing either one
in a single file would have left the rest quietly disagreeing.

1. CORNER ASSIGNMENT IS NOT RANDOM. In data/fight_history.csv,
   winner == fighter_a on 72.6% of 11,025 rows -- and on 100.0% of rows
   before 2010, because the early data was entered winner-first. Nothing in
   the model reads row order, so this leaks nothing INTO a prediction. What
   it does is destroy the meaning of every accuracy figure: "the model is
   61.9% accurate" was being compared against an unstated floor of 61.0% on
   the same population, and against 72.6% overall. Several such figures are
   quoted in module docstrings across this repo.

   randomize_corner() flips a deterministic half of scored fights, carrying
   p -> 1-p and y -> 1-y together so the pairing between arms is preserved
   exactly. After it, the trivial baseline is 50% and an accuracy number
   means what a reader assumes it means. Brier and log loss are invariant
   under the flip by construction, which is a useful self-check: if they
   move, the flip has been applied to one side only.

2. FIGHTS ON ONE CARD ARE NOT INDEPENDENT. They share a referee, a judging
   panel, a venue and its altitude, a cage size, and often a late-notice
   cascade from one cancellation. Measured on this repo's own data the
   within-card residual correlation is small but real (ICC ~ +0.025,
   permutation p ~ 0.007), which inflates the effective sample size the
   per-fight sign flip assumes.

   paired_signflip() flips a whole card's deltas together when given cluster
   keys. Cards are keyed on fight DATE -- fight_history carries no event id,
   and two same-day events are rare enough that merging them is conservative
   in the right direction (larger clusters, wider intervals).

   MEASURED, THE EFFECT IS SMALLER THAN THAT ICC IMPLIES, and the reason is
   worth stating because it was not obvious in advance. The design effect
   this returns on the durability sweep is 0.87-0.92 in one window and
   1.13-1.15 in the other -- straddling 1.0, not the ~1.4 an ICC of 0.025 on
   raw residuals would suggest.

   The ICC was measured on OUTCOME residuals. This statistic is a paired
   DIFFERENCE between two arms scored on the identical fight, so anything
   shared by a card -- a lenient referee, a slick cage, an altitude the model
   cannot see -- lands on both arms and cancels in the subtraction. Cluster
   dependence in the outcomes is real and simply does not transfer to the
   quantity being tested.

   So this is kept because it is the correct default for a clustered design
   and it costs one argument, not because it rescued anything. A harness that
   reports the design effect beside its p-value lets a reader see that for
   themselves rather than take either assumption on faith.
"""

import hashlib
import math
import random


def corner_flip_key(*parts) -> bool:
    """
    Deterministic per-fight coin flip.

    Hash-based rather than RNG-based so a fight flips the same way on every
    run and across harnesses -- a validation number that moves between runs
    is not a validation number, and two harnesses disagreeing about which
    corner is A would make their populations silently different.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return bool(h[0] & 1)


def randomize_corner(prob_a: float, y: float, *key_parts) -> tuple[float, float]:
    """
    (p, y) with the corners swapped on a deterministic half of fights.

    Both are flipped together, so the fight's information content is
    unchanged and only the LABEL ORDER moves. Apply the same key to every arm
    of a comparison.
    """
    if corner_flip_key(*key_parts):
        return 1.0 - prob_a, 1.0 - y
    return prob_a, y


def trivial_baseline(pairs) -> float:
    """
    Accuracy of always predicting corner A -- the floor an accuracy figure
    must be read against. Should sit near 50% once randomize_corner is in use;
    anything else means it was not applied.
    """
    if not pairs:
        return 0.0
    return sum(1 for _, y in pairs if y == 1.0) / len(pairs)


def score(pairs):
    """(n, accuracy, Brier, log loss)."""
    n = len(pairs)
    if not n:
        return 0, 0.0, 0.0, 0.0
    acc = sum(1 for p, y in pairs if (p >= 0.5) == (y == 1.0)) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
              for p, y in pairs) / n
    return n, acc, brier, ll


def paired_signflip(deltas, clusters=None, n_boot=4000, seed=12345):
    """
    Paired sign-flip bootstrap on per-fight changes in squared error.

    deltas:   one signed value per scored fight (arm minus control).
    clusters: optional same-length sequence of card keys. When given, a whole
              card's deltas flip together, which is the correction for
              within-card dependence. When omitted the behaviour is the
              per-fight flip the harnesses used before.

    Returns (mean_delta, p_two_sided, design_effect). design_effect is the
    ratio of clustered to unclustered variance -- 1.0 means clustering
    changed nothing, above 1.0 means the per-fight version was
    anti-conservative by roughly its square root.
    """
    deltas = list(deltas)
    n = len(deltas)
    if not n:
        return 0.0, 1.0, 1.0
    obs = sum(deltas) / n

    def _run(keys):
        rnd = random.Random(seed)
        hits = 0
        tot = 0.0
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(deltas[i])
        vals = list(groups.values())
        for _ in range(n_boot):
            s = 0.0
            for g in vals:
                sign = 1 if rnd.random() < 0.5 else -1
                s += sign * sum(g)
            m = s / n
            tot += m * m
            if abs(m) >= abs(obs):
                hits += 1
        return hits / n_boot, tot / n_boot

    p_flat, var_flat = _run(range(n))
    if clusters is None:
        return obs, p_flat, 1.0
    p_clu, var_clu = _run(list(clusters))
    deff = (var_clu / var_flat) if var_flat > 0 else 1.0
    return obs, p_clu, deff
