"""
One-time backfill of pick_fair_prob / closing_fair_prob in predictions_log.csv.

WHY THIS IS SOUND RATHER THAN AN INVENTION. CLV is now graded on the de-vigged
probability at pick time against the one at closing time, because the American
price stopped being a stable basis the moment real sportsbook prices entered
the pipeline (see _clv_result). Both columns are set from the live edge row, so
they only populate for fights first seen after that change shipped -- and
pick_odds in particular is set once and never rewritten, which left every one
of the rows already on file permanently ungradeable on the new basis.

Those rows are not unknowable, though. Every price in them was captured while
Polymarket was the ONLY source in the pipeline: TheRundown was wired in but
never actually invoked (its date list was built from a field Polymarket does
not have), so no DraftKings or FanDuel price ever reached predictions_log.
Polymarket is peer-to-peer and carries no margin, so for these rows the quoted
price IS the fair line and its implied probability is the fair probability.

RAW IMPLIED ON BOTH SIDES, DELIBERATELY. A two-way de-vig against opponent_odds
would be marginally more precise for pick_fair_prob, but opponent_odds is the
opponent's price at PICK time and there is no stored equivalent at closing
time. Using a sharper estimator on one end of a difference and a blunter one on
the other introduces exactly the systematic gap this whole change exists to
remove. CLV is a difference, so consistency between the two ends matters more
than precision at either.

Refuses to touch a row that already has a value, and refuses settled prices for
the same reason the live path does: a resolved market at 0.9995 is the result
wearing a closing line, and grading against it makes CLV circular.

Idempotent. Run again and it reports zero changes.
"""

import sys

import pandas as pd

sys.path.insert(0, ".")

from src.odds_utils import american_to_implied_prob
from src.track_record import PREDICTIONS_LOG_PATH, _is_settled_price


def _fair_from(price) -> float | None:
    if price in (None, "") or pd.isna(price):
        return None
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if _is_settled_price(price):
        return None
    p = american_to_implied_prob(price)
    return round(p, 4) if 0.0 < p < 1.0 else None


def main(path: str = PREDICTIONS_LOG_PATH, apply: bool = False) -> int:
    df = pd.read_csv(path)
    for col in ("pick_fair_prob", "closing_fair_prob"):
        if col not in df.columns:
            df[col] = ""

    filled = {"pick_fair_prob": 0, "closing_fair_prob": 0}
    for src, dst in (("pick_odds", "pick_fair_prob"),
                     ("closing_odds", "closing_fair_prob")):
        for i in df.index:
            cur = df.at[i, dst]
            if cur not in (None, "") and not pd.isna(cur):
                continue                      # never overwrite a live capture
            val = _fair_from(df.at[i, src])
            if val is None:
                continue
            df.at[i, dst] = val
            filled[dst] += 1

    print(f"rows: {len(df)}")
    for k, v in filled.items():
        print(f"  {k}: filled {v}")
    # notna() rather than a string compare: an unfilled cell reads back as
    # float NaN, and str(NaN) is "nan", which is not the empty string -- so
    # comparing against "" counted every ungraded row as graded.
    def _has(col):
        v = df[col]
        return v.notna() & (v.astype(str).str.strip() != "")
    both = int((_has("pick_fair_prob") & _has("closing_fair_prob")).sum())
    print(f"  rows now gradeable on the fair basis: {both}")

    if apply:
        df.to_csv(path, index=False)
        print(f"wrote {path}")
    else:
        print("dry run -- pass --apply to write")
    return both


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
