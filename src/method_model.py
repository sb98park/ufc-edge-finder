"""
Fight-level method probabilities from the discrete-time hazard model.

WHY THE COEFFICIENTS ARE FROZEN HERE RATHER THAN FITTED AT BUILD TIME.
The model is a multinomial logistic over 8 features and 3 outcomes -- 24
coefficients and 3 intercepts. Training it during every site build would add
scikit-learn to the production dependency chain (generate_site.py and src/
currently import neither sklearn nor scipy, which is why Actions stays
light) and would refit an identical model every few minutes. Exporting the
fit and scoring it with a dot product plus a softmax is exact, instant, and
dependency-free. Refit with research_survival_model.py and paste new numbers
when the training window is extended.

WHAT THIS PREDICTS. Per ROUND: given the fight reached round r, does it end
here, and by KO/TKO or submission? Chaining those hazards across a fight's
scheduled rounds gives P(KO overall), P(SUB overall), and P(decision) as the
probability of surviving every round.

VALIDATED (research_method_calibration.py), frozen 2019+ holdout, n=1743:
    base rates      P(KO) Brier 0.2155   P(SUB) Brier 0.1380
    this model      P(KO) Brier 0.2028   P(SUB) Brier 0.1331

KNOWN LIMITATION, deliberately not corrected. The chained probability is
about 2.4pp OVERCONFIDENT out of sample, and that cannot be fitted away:
isotonic regression made calibration WORSE (0.024 -> 0.031) and a single
shrink parameter selected w=1.00, because the miscalibration is absent from
the training data -- it's drift, not bias. So this is fit for DISPLAY, where
ranking dominates, and explicitly NOT for gating props into Locks at an 82%
floor, where the error runs in exactly the wrong direction.
"""

import math

FEATURES = ["round", "scheduled", "ko_press", "sub_press",
            "ko_rate_sum", "sub_rate_sum", "durability", "elo_gap"]

# Rows are outcomes in order: survive, KO/TKO, submission.
COEF = [
    [0.164828, 0.009001, 0.117059, -1.494370, -0.372273, -0.340968, -0.720587, 0.056450],
    [-0.135041, 0.088418, 0.746369, 0.027571, 0.438113, -0.388517, 0.520133, 0.182444],
    [-0.029786, -0.097418, -0.863428, 1.466799, -0.065840, 0.729486, 0.200455, -0.238894],
]
INTERCEPT = [1.472720, -0.670360, -0.802360]

SURVIVE, KO, SUB = 0, 1, 2


def _round_hazard(feat: dict) -> list[float]:
    """Softmax over the three per-round outcomes."""
    x = [float(feat.get(f, 0.0) or 0.0) for f in FEATURES]
    z = [INTERCEPT[k] + sum(c * v for c, v in zip(COEF[k], x)) for k in range(3)]
    m = max(z)                                  # subtract the max for stability
    e = [math.exp(v - m) for v in z]
    total = sum(e)
    return [v / total for v in e]


def method_probabilities(ko_press: float, sub_press: float, ko_rate_sum: float,
                         sub_rate_sum: float, durability: float, elo_gap: float,
                         scheduled_rounds: int = 3) -> dict | None:
    """
    Chain per-round hazards into fight-level P(KO), P(SUB), P(decision).

    Returns None when an input is missing rather than substituting a default:
    a fabricated feature yields a confident-looking number with nothing
    behind it, and the caller already has a fallback for that case.
    """
    vals = (ko_press, sub_press, ko_rate_sum, sub_rate_sum, durability, elo_gap)
    if any(v is None for v in vals):
        return None
    surv, p_ko, p_sub = 1.0, 0.0, 0.0
    for r in range(1, int(scheduled_rounds) + 1):
        p = _round_hazard({
            "round": r, "scheduled": scheduled_rounds,
            "ko_press": ko_press, "sub_press": sub_press,
            "ko_rate_sum": ko_rate_sum, "sub_rate_sum": sub_rate_sum,
            "durability": durability, "elo_gap": elo_gap,
        })
        p_ko += surv * p[KO]
        p_sub += surv * p[SUB]
        surv *= p[SURVIVE]
    return {"ko": p_ko, "sub": p_sub, "decision": surv}
