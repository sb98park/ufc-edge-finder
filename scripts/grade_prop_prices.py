"""
Grade the recorded derivable-market quotes against results, and report ROI.

WHAT THIS IS FOR. Everything else has been answered no. Moneyline alpha: 21
hypotheses against 16 years of closing lines, nothing survived. The method
market: real 4-point miscalibration, buried under a 21.8% overround. The one
question left open is whether a TWO-WAY quote on the markets the reader
actually bets -- goes-the-distance, Double Chance -- is loose enough to be
worth taking, because every historical price available is a six-cell grid
price and no such quote exists anywhere in the archive.

src/prop_ledger records those quotes as the site sees them. This settles them
once the fights resolve and reports what backing them would have returned.
Until a few hundred accumulate it will report thin samples and say so; that is
the point of starting now rather than concluding now.

THE CLOCK CONVENTION, AND WHY THERE IS A CHECK FOR IT.

Round totals cannot be graded without knowing whether the recorded time counts
UP from the start of the round or DOWN to its end -- a round-2 finish at 0:30
and one at 4:30 fall on opposite sides of an Over 1.5 line. This repo has been
bitten by exactly that before.

data/fight_results.csv counts UP. That is measured, not assumed: joining it to
data/ufc_fight_results.csv (ufcstats, definitively elapsed time) on bout and
round matches on 21 of 21 bouts for count-up and 0 of 21 for count-down.

BUT THE LIVE GRADER IN THE TEMPLATE DISAGREES. It reads ESPN's displayClock
straight from the browser and computes `300 - remain`, i.e. count-DOWN, and
says so in its own comment. Both can be true: results_fetcher takes end_time
from the scoreboard path first and only falls back to the core api's
displayClock, so the file may never contain a displayClock value at all.

That could not be settled from the machine this was written on -- ESPN returns
403 there -- so nothing was changed on a guess. Instead this script CHECKS it:
for every settled fight it grades round totals both ways and reports whether
the two conventions ever disagree on a real stoppage. After one card with a
mid-round finish, the answer stops being a matter of opinion. If they disagree
and the count-up reading is the one matching ufcstats, the template's
`300 - remain` is a live bug and this will say so.

Run:  python3 scripts/grade_prop_prices.py
"""

import random
import re
import sys
import unicodedata

import pandas as pd

sys.path.insert(0, ".")

from src.prop_ledger import load as load_ledger

RESULTS = "data/fight_results.csv"
ROUND_SECONDS = 300
SEED = 17
BOOTSTRAP = 4000

T, F, UNKNOWN = True, False, None


def fold(s) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def to_decimal(american) -> float:
    o = float(american)
    return 1 + (100.0 / -o) if o < 0 else 1 + (o / 100.0)


def clock_seconds(value):
    m = re.fullmatch(r"(\d+):([0-5]\d)", str(value).strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def is_decision(method) -> bool:
    return "dec" in fold(method)


def is_ko(method) -> bool:
    m = fold(method)
    return "ko" in m or "tko" in m


def is_sub(method) -> bool:
    return "sub" in fold(method)


def elapsed_rounds(end_round, end_time, count_up: bool):
    """Total rounds elapsed at the stoppage, under the stated convention."""
    secs = clock_seconds(end_time)
    if secs is None or end_round is None or pd.isna(end_round):
        return None
    within = secs if count_up else (ROUND_SECONDS - secs)
    return (int(end_round) - 1) + within / ROUND_SECONDS


def settle(entry: dict, res: dict, count_up: bool = True):
    """T / F / UNKNOWN for one recorded quote against one result."""
    market = entry.get("market") or ""
    sel = str(entry.get("selection") or "")
    method, winner = res["method"], res["winner"]

    if market == "GoesTheDistance":
        if sel == "Goes The Distance":
            return T if is_decision(method) else F
        if sel == "Ends In Finish":
            return F if is_decision(method) else T
        return UNKNOWN

    if market == "FightMethod":
        if sel == "KO/TKO":
            return T if is_ko(method) else F
        if sel == "Not KO/TKO":
            return F if is_ko(method) else T
        if sel == "SUB":
            return T if is_sub(method) else F
        if sel == "Not SUB":
            return F if is_sub(method) else T
        return UNKNOWN

    if market.startswith("Method"):
        # "<fighter> by <method>" -- both halves must hold.
        want = entry.get("selection_method") or ""
        if fold(sel) != fold(winner):
            return F
        if fold(want) in ("ko/tko", "ko", "tko"):
            return T if is_ko(method) else F
        if fold(want) in ("sub", "submission"):
            return T if is_sub(method) else F
        if fold(want).startswith("dec"):
            return T if is_decision(method) else F
        return UNKNOWN

    if market == "TotalRounds":
        m = re.match(r"(Over|Under)\s+([\d.]+)", sel)
        if not m:
            return UNKNOWN
        side, line = m.group(1).lower(), float(m.group(2))
        if is_decision(method):
            # A decision clears every offered line by construction; the
            # scheduled distance never needs to be known.
            return T if side == "over" else F
        total = elapsed_rounds(res["end_round"], res["end_time"], count_up)
        if total is None:
            return UNKNOWN          # never assume a midpoint stoppage
        return (total > line) if side == "over" else (total < line)

    return UNKNOWN


def clustered_ci(returns, dates, rounds=BOOTSTRAP):
    uniq = list(dict.fromkeys(dates))
    if len(uniq) < 2:
        return None, None
    out = []
    for _ in range(rounds):
        pick = [random.choice(uniq) for _ in uniq]
        vals = [r for r, d in zip(returns, dates) for _ in range(pick.count(d))]
        if vals:
            out.append(sum(vals) / len(vals) * 100)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main():
    random.seed(SEED)
    ledger = load_ledger()
    if not ledger:
        print("no recorded quotes yet -- src/prop_ledger writes on each site build")
        return

    rdf = pd.read_csv(RESULTS)
    results = {}
    for _, r in rdf.iterrows():
        results[frozenset({fold(r.fighter_a), fold(r.fighter_b)})] = {
            "winner": r.winner, "method": r.method,
            "end_round": r.end_round, "end_time": r.end_time}

    graded, pending, clock_conflicts = [], 0, 0
    for e in ledger:
        key = frozenset({fold(e.get("fighter_a")), fold(e.get("fighter_b"))})
        res = results.get(key)
        if not res:
            pending += 1
            continue
        up = settle(e, res, count_up=True)
        if e.get("market") == "TotalRounds":
            down = settle(e, res, count_up=False)
            if up is not None and down is not None and up != down:
                clock_conflicts += 1
        if up is None:
            continue
        graded.append({**e, "won": bool(up)})

    print(f"recorded quotes {len(ledger)}   settled {len(graded)}   "
          f"awaiting results {pending}")
    if clock_conflicts:
        print(f"\n  !! {clock_conflicts} round-total quote(s) grade DIFFERENTLY under "
              f"count-up vs count-down.\n     data/fight_results.csv is count-up "
              f"(21/21 vs ufcstats); the template's live\n     grader assumes "
              f"count-down. One of them is wrong -- see this file's docstring.")
    elif any(g.get("market") == "TotalRounds" for g in graded):
        print("\n  round totals settled identically under both clock conventions "
              "so far\n  (no mid-round stoppage has yet separated them)")

    if not graded:
        print("\nNothing settled yet. This accumulates one card at a time by design.")
        return

    print(f"\n{'market':<18}{'source':<12}{'n':>5}{'hit':>8}{'ROI@first':>11}"
          f"{'ROI@last':>10}{'95% CI (by card)':>22}")
    df = pd.DataFrame(graded)
    for (market, source), sub in df.groupby(["market", "source"]):
        first = [(1.0 if r.won else 0.0) * to_decimal(r.price_first) - 1 for r in sub.itertuples()]
        last = [(1.0 if r.won else 0.0) * to_decimal(r.price_last) - 1 for r in sub.itertuples()]
        lo, hi = clustered_ci(first, list(sub.event_date))
        ci = f"[{lo:+.1f}, {hi:+.1f}]" if lo is not None else "one card only"
        print(f"{market:<18}{str(source):<12}{len(sub):>5}{sub.won.mean()*100:>7.1f}%"
              f"{sum(first)/len(first)*100:>+10.2f}%{sum(last)/len(last)*100:>+9.2f}%{ci:>22}")

    vig_free = df[df.is_vig_free]
    booked = df[~df.is_vig_free]
    print(f"\n  vig-free quotes {len(vig_free)}   book quotes {len(booked)}")
    if booked.empty:
        print("  No book-priced quote has been recorded yet. The open question is "
              "specifically\n  about TWO-WAY BOOK prices, so this cannot answer it "
              "until one appears.")


if __name__ == "__main__":
    main()
