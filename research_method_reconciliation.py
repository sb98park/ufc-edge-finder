"""
Do the PER-FIGHTER method probabilities sum to the FIGHT-LEVEL ones?

WHY THIS MATTERS BEFORE DERIVING ANYTHING. "Fight ends by KO" is exactly
"A wins by KO" OR "B wins by KO", and those are mutually exclusive -- so the
two per-fighter numbers MUST sum to the fight-level one. That identity is
what makes subtraction a valid way to derive a missing leg:

    P(B by KO) = P(fight ends by KO) - P(A by KO)

But the two figures come from DIFFERENT models. Per-fighter probabilities
come from blend_method_probability() in model_preview.py -- a divisional
prior blended with career finish rates. Fight-level probabilities come from
the discrete-time hazard model in method_model.py. Nothing couples them, and
the same mismatch already showed up on screen: KO + SUB + DECISION summed to
103.8% because decision came from a third source again.

If they don't reconcile, subtraction would produce confident nonsense -- and
worse, the site would show a fighter's KO chance that contradicts the
fight-level KO chance sitting two rows above it.

Run: python3 research_method_reconciliation.py
"""

import pandas as pd

from src.elo import EloRatingSystem
from src.power_rating import build_effective_ratings
from src.method_model import method_probabilities
from src.model_preview import build_full_market_projection


def _rate(row, col, denom_col):
    try:
        d = max(int(row.get(denom_col, 0) or 0), 1)
        return float(row.get(col, 0) or 0) / d
    except (TypeError, ValueError):
        return 0.0


def main():
    fighters = pd.read_csv("data/fighters.csv")
    history = pd.read_csv("data/fight_history.csv")
    try:
        cards = pd.read_csv("data/fight_cards.csv")
    except FileNotFoundError:
        print("No fight_cards.csv -- run this where the live card lives.")
        return

    elo = EloRatingSystem()
    elo.build_from_history(history)
    eff = build_effective_ratings(fighters, elo.ratings, history)

    print(f"{'fight':38}{'method':10}{'per-fighter sum':>16}{'fight-level':>13}{'gap':>9}")
    print("-" * 86)
    gaps = {"KO/TKO": [], "SUB": []}
    checked = 0

    for _, c in cards.iterrows():
        a_name, b_name = str(c.get("fighter_a", "")), str(c.get("fighter_b", ""))
        fa = fighters[fighters["name"] == a_name]
        fb = fighters[fighters["name"] == b_name]
        if fa.empty or fb.empty:
            continue
        a, b = fa.iloc[0], fb.iloc[0]

        five = str(c.get("card_position", "")).strip() == "Main Event"
        proj = build_full_market_projection(a_name, b_name, fighters, eff, is_five_round=five)
        if not proj:
            continue

        # Per-fighter legs, summed per method.
        # NORMALISE the label. The projection emits "Method: Submission" while
        # the hazard model calls it "sub" -- an earlier version of this script
        # keyed on the raw uppercased string, so submission never matched and
        # summed to exactly 0.0% on every fight. That looked like a
        # catastrophic model failure and was purely this bug. A diagnostic
        # that silently returns zero is worse than one that crashes.
        CANON = {"ko/tko": "KO/TKO", "ko": "KO/TKO", "tko": "KO/TKO",
                 "submission": "SUB", "sub": "SUB",
                 "decision": "DEC", "dec": "DEC"}
        per = {}
        unmapped = set()
        for r in proj["method_rows"]:
            mk = str(r.get("market", ""))
            raw = (mk.split(":", 1)[1].strip() if ":" in mk else mk).lower()
            key = CANON.get(raw)
            if key is None:
                unmapped.add(raw)
                continue
            per.setdefault(key, 0.0)
            per[key] += float(r.get("model_prob") or 0)
        if unmapped:
            print(f"   [warn] unmapped method labels: {sorted(unmapped)}")

        ko_a, ko_b = _rate(a, "ko_wins", "wins"), _rate(b, "ko_wins", "wins")
        sub_a, sub_b = _rate(a, "sub_wins", "wins"), _rate(b, "sub_wins", "wins")
        kol_a, kol_b = _rate(a, "ko_losses", "losses"), _rate(b, "ko_losses", "losses")
        subl_a, subl_b = _rate(a, "sub_losses", "losses"), _rate(b, "sub_losses", "losses")
        gap_elo = abs(eff.get(a_name, 1500) - eff.get(b_name, 1500)) / 400.0
        fight = method_probabilities(
            ko_press=ko_a * kol_b + ko_b * kol_a,
            sub_press=sub_a * subl_b + sub_b * subl_a,
            ko_rate_sum=ko_a + ko_b, sub_rate_sum=sub_a + sub_b,
            durability=kol_a + kol_b, elo_gap=gap_elo,
            scheduled_rounds=5 if five else 3,
        )
        if not fight:
            continue

        checked += 1
        label = f"{a_name[:16]} vs {b_name[:16]}"
        for method, key in (("KO/TKO", "ko"), ("SUB", "sub")):
            ps = per.get(method, 0.0)
            fl = fight[key]
            gaps[method].append(ps - fl)
            print(f"{label:38}{method:10}{ps:15.1%}{fl:13.1%}{ps - fl:+9.1%}")

    if not checked:
        print("No comparable fights found.")
        return
    print("-" * 86)
    for method, vals in gaps.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        worst = max(vals, key=abs)
        print(f"{method:10} n={len(vals):3}  mean gap {mean:+.1%}  worst {worst:+.1%}")
    print()
    print("A gap near zero means the identity holds and subtraction is safe.")
    print("Anything large means the two models disagree, and a derived leg")
    print("would contradict the fight-level row shown beside it.")


if __name__ == "__main__":
    main()
