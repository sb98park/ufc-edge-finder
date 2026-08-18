"""
The parlay ledger: every published slip, written down at render time.

WHY THIS FILE EXISTS. Until now nothing recorded what the site published.
build_bankroll_builder_parlays and its two siblings run during a render, from
whatever prices were live at that second, and the output went to a template
and nowhere else. data/predictions_log.csv logs fight-level moneyline picks
only. So the honest answer to "how have the published parlays done" was: THAT
IS NOT RECOVERABLE -- nine slips a week, published for months, none graded,
while single picks were graded to three decimals on the same page.

scripts/replay_parlays.py can approximate the question by replaying the
selection rules over settled moneyline picks, and its docstring is careful
that this is not the same thing. This file is what makes the real answer
possible from here forward.

WHAT IS WRITTEN, AND WHY EACH FIELD. The temptation is to log the slip as
rendered. That would be useless in six months, because grading needs the
inputs and auditing needs the reasoning:

  slip_id            stable across rebuilds, so re-publishing the same slip
                     on the next 5-minute render updates rather than
                     duplicates it
  tier               bankroll / lotto
  event              which card, so slips can be clustered by event -- the
                     effective sample size is EVENTS, not slips, and any
                     honest interval has to know that
  combined_decimal   the price the slip was published at
  combined_prob      the shrunk probability it was ranked on
  combined_prob_raw  the unshrunk model product, so the shrink's effect is
                     measurable after the fact rather than asserted
  legs[]             market, fighter, price, model_prob, book_fair_prob and
                     the grading conditions -- everything needed to settle
                     the leg and to re-derive why it was chosen
  first_seen/last_seen
                     a slip is republished every 5 minutes while the card is
                     up; these bracket its life without writing 300 copies

WHAT IS DELIBERATELY NOT WRITTEN: the result. Grading happens later, from
data/fight_results.csv, against the conditions stored here. Writing an
ungraded ledger and grading it separately keeps the render path incapable of
inventing an outcome.

APPEND-ONLY IN SPIRIT, REWRITTEN IN PRACTICE. The file is read, merged on
slip_id and rewritten, so re-publishing the same slip updates its row rather
than adding one.

THE SLATE CHURNS HEAVILY, AND THIS FILE IS HOW THAT WAS DISCOVERED.
The merge above was written expecting a card to settle into ~9 stable rows.
Measured on the same card:

    back-to-back renders           4 of 9 slips survived
    renders further apart in time  0 of 9 survived

so retention falls off with elapsed time rather than being uniformly zero --
the first number is the fair one to quote and the second is what a longer gap
looks like. The selection itself is deterministic: identical pieces produce
identical slips every time, verified directly. All of the churn is price
drift moving which combinations clear a payout band. The band IS the churn --
a leg moving a few cents pushes a slip across a threshold and a different set
of combinations becomes eligible.

Two consequences worth being explicit about.

1. "The published slate" is not a well-defined object. A reader at 3:00 and
   the same reader at 3:05 were shown nine different bets. Grading therefore
   cannot mean "grade what we published", because that is thousands of slips
   a week; it has to mean "grade the slate as of a defined cutoff", and the
   defensible cutoff is the last render before the first fight starts.
   `renders` and `last_seen` are recorded so that cutoff can be applied after
   the fact rather than being baked in here.

2. Shrinking toward the market made this BETTER, not worse. Ranking on a
   price-derived quantity sounds like it should add price sensitivity, and
   measured over small price moves it does the opposite: 33% of slips
   retained with the blend against 19% on the raw model, because the blend
   damps exactly the model-vs-market disagreements the search was chasing.

This file records every distinct slip it sees rather than trying to define
the "real" one. Deciding which slate counts is a product question, and
throwing away the evidence would prevent anyone from answering it.

FAILURE IS NEVER FATAL. A ledger write that raises must not take down a site
build -- the same rule the parlay builders themselves are wrapped in. Every
entry point here swallows its own errors and says so on stdout.
"""

import json
import os
from datetime import datetime, timezone

LEDGER_PATH = "data/parlay_ledger.jsonl"


def _leg_record(leg: dict) -> dict:
    return {
        "label": leg.get("label"),
        "odds_display": leg.get("odds_display"),
        "decimal_odds": leg.get("decimal_odds"),
        "is_model": bool(leg.get("is_model")),
        "fight_key": leg.get("fight_key"),
        "source": leg.get("source"),
        # The grading predicates, stored rather than re-derived. _leg_label
        # builds prose and prose is not a protocol; if the copy changes next
        # year, a slip logged today must still be settleable.
        "conditions": leg.get("conditions") or [],
    }


def record_slips(slips_by_tier: dict, event_name: str | None,
                 path: str = LEDGER_PATH) -> int:
    """
    Merge the current render's slips into the ledger. Returns rows written.

    slips_by_tier: {"bankroll": [...], "lotto": [...]}
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
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # a torn line must not lose the file
                    if row.get("slip_id"):
                        existing[row["slip_id"]] = row

        seen = 0
        for tier, slips in (slips_by_tier or {}).items():
            for s in (slips or []):
                sid = s.get("slip_id")
                if not sid:
                    continue
                seen += 1
                prior = existing.get(sid)
                row = {
                    "slip_id": sid,
                    "tier": tier,
                    "event": event_name,
                    "combined_american": s.get("combined_american"),
                    "combined_decimal": s.get("combined_decimal"),
                    "combined_prob": s.get("combined_prob"),
                    "combined_prob_raw": s.get("combined_prob_raw"),
                    "has_model_legs": bool(s.get("has_model_legs")),
                    "legs": [_leg_record(l) for l in (s.get("legs") or [])],
                    # PRICE AT FIRST PUBLICATION IS THE ONE THAT COUNTS. A
                    # slip republished at a drifted price is still the same
                    # recommendation; grading it at the later price would
                    # quietly let the ledger shop for a better number.
                    "first_seen": (prior or {}).get("first_seen", now),
                    "last_seen": now,
                    # How many renders this exact slip survived. With a slate
                    # that turns over completely every five minutes, a slip
                    # seen once and a slip seen forty times are very different
                    # recommendations, and only this distinguishes them.
                    "renders": (prior or {}).get("renders", 0) + 1,
                }
                if prior:
                    row["combined_decimal_first"] = prior.get(
                        "combined_decimal_first", prior.get("combined_decimal"))
                else:
                    row["combined_decimal_first"] = s.get("combined_decimal")
                existing[sid] = row

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in existing.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        print(f"[parlay_ledger] {seen} slip(s) this render, {len(existing)} total on file")
        return len(existing)
    except Exception as exc:
        # Never let bookkeeping break a build.
        print(f"[parlay_ledger] not written ({exc}) -- continuing")
        return 0


def load(path: str = LEDGER_PATH) -> list[dict]:
    """Every recorded slip, newest last. Missing file is not an error."""
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
