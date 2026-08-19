"""
Every derivable-market price this site sees, written down so it can be graded.

WHY THIS EXISTS. Moneyline alpha was ruled out on 16 years of closing lines,
and the method market was ruled out on 5,470 bouts: finishes are overpriced by
about 4 points, which survives a power de-vig and is therefore real, but the
six-cell grid charges 21.8% against the moneyline's 3.9% and a 4-point edge
cannot pay a 20-point toll (scripts/research_unpriced_markets.py).

One question stayed open, and it is the one the reader actually bets. Every
price in that study was a SIX-CELL GRID price. A book quoting Double Chance or
goes-the-distance as its own TWO-WAY market prices it near a two-way margin --
5-8%, not 20% -- and at 5-8% a 4-point calibration bias is not obviously dead.
data/external_odds.csv contains no such quote anywhere, so the market the
reader bets most is the one that has never been measured.

It cannot be answered from history. It can be answered from here forward, and
only if the prices are recorded as they are seen, with enough context to grade
them once the fight resolves. That is all this file does.

WHAT IS RECORDED, AND WHY EACH PIECE. The temptation is to log the price. That
would be useless in six months:

  market/selection   what the bet actually was, in a form the grader can
                     settle from data/fight_results.csv
  price_american     what was quoted
  source             Polymarket, DraftKings, FanDuel, or manual -- and
  is_vig_free        whether that price carries a margin at all. A vig-free
                     midpoint and a book quote answer different questions and
                     pooling them is the mistake this whole project keeps
                     having to undo
  event_date         so a bet can be tied to a card and intervals clustered by
                     event rather than by bet
  first_seen/last_seen/observations
                     a price drifts all week; the first sighting is the
                     honest one to grade and the last is the closing line

WHAT IS DELIBERATELY NOT RECORDED: the outcome. Grading happens separately, in
scripts/grade_prop_prices.py, against results fetched independently. A render
path that could write an outcome is a render path that could invent one.

FAILURE IS NEVER FATAL. A ledger write that raises must not take down a site
build, for the same reason parlay_ledger does not.
"""

import json
import os
from datetime import datetime, timezone

LEDGER_PATH = "data/prop_price_log.jsonl"

# The markets worth accumulating. Moneyline is excluded on purpose: it is
# already answered, and at ~13 quotes a card it would swamp the file with the
# one market known not to pay.
TRACKED_MARKETS = ("GoesTheDistance", "FightMethod", "Method", "TotalRounds")


def _key(row: dict) -> str:
    return "|".join(str(row.get(k) or "") for k in
                    ("fight_id", "market", "selection", "selection_method", "source"))


def record_prop_prices(rows, event_name=None, event_date=None,
                       path: str = LEDGER_PATH) -> int:
    """
    Merge this render's derivable-market quotes into the ledger.

    `rows` is the upcoming-props record list -- the same shape live_props
    returns, carrying market/selection/odds_american/source/source_is_vig_free.
    Returns the number of distinct quotes on file afterwards.
    """
    try:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        existing = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        prior = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # a torn line must not lose the file
                    if prior.get("key"):
                        existing[prior["key"]] = prior

        seen = 0
        for row in rows or []:
            market = str(row.get("market") or "")
            if not any(market.startswith(m) for m in TRACKED_MARKETS):
                continue
            price = row.get("odds_american")
            if price is None or price != price:
                continue
            key = _key(row)
            prior = existing.get(key)
            seen += 1
            existing[key] = {
                "key": key,
                "event": event_name,
                "event_date": event_date,
                "fight_id": row.get("fight_id"),
                "fighter_a": row.get("fighter_a"),
                "fighter_b": row.get("fighter_b"),
                "market": market,
                "selection": row.get("selection"),
                "selection_method": row.get("selection_method"),
                "source": row.get("source"),
                "is_vig_free": bool(row.get("source_is_vig_free")),
                # FIRST PRICE AND LATEST PRICE, both kept. Grading the first
                # is the honest test of a recommendation made early; grading
                # the last is the closing line. Keeping only one would decide
                # that question now, and it does not need deciding yet.
                "price_first": (prior or {}).get("price_first", float(price)),
                "price_last": float(price),
                "first_seen": (prior or {}).get("first_seen", now),
                "last_seen": now,
                "observations": (prior or {}).get("observations", 0) + 1,
            }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for entry in existing.values():
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        by_source = {}
        for e in existing.values():
            by_source[e.get("source") or "?"] = by_source.get(e.get("source") or "?", 0) + 1
        print(f"[prop_ledger] {seen} quote(s) this render, {len(existing)} on file {by_source}")
        return len(existing)
    except Exception as exc:
        print(f"[prop_ledger] not written ({exc}) -- continuing")
        return 0


def load(path: str = LEDGER_PATH) -> list[dict]:
    """Every recorded quote. A missing file is not an error."""
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []
    return out
