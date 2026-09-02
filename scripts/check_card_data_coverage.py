"""
What do we actually know about the fighters we are about to price?

WHY THIS EXISTS. Michael Aljarouj was priced for 2026-09-05 off ONE bout.
His own record in fighters.csv says 16. The other fifteen -- including a
2025-04-12 no contest -- are not in fight_history.csv, so the model read his
last fight as 2021-03-18 and charged him 89.3 rating points of ring rust for
a layoff that never happened. Nothing in the pipeline said a word. The gap
was found by a human opening Tapology on a phone, five days before the card,
by chance.

Every input for that alarm was already on disk. `wins + losses` from
fighters.csv is the fighter's own claim about how many times they have
fought; counting their rows in fight_history.csv is what we hold. When the
second is far below the first, every history-derived term -- layoff, recent
form, streaks, Elo -- is computed from a sample we can see is incomplete.

WHY IT IS NOT A GATE. A thin record is a legitimate state: real debutants
exist, and CI gates freeze the entire site on stale data when they trip. This
prints, writes to source_health.json, and exits 0 always. It is an alarm, not
a brake.

SCOPE IS THE POINT. scripts/audit_fighter_data.py already checks missing
fields -- across all 369 roster fighters, on demand, by hand. That is why
nobody reads it. This one looks only at fighters on a card that has not
happened yet, which is the only moment the answer can still change a price.

Run: python3 scripts/check_card_data_coverage.py
Read-only except for its own block in data/source_health.json.
"""

import datetime as dt
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matchup_model import HISTORY_COVERAGE_FLOOR   # noqa: E402
from src.names import _normalize_name                 # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY = "data/fight_history.csv"
CARDS = ("data/fight_cards.csv", "data/future_cards.csv")
HEALTH = "data/source_health.json"

# ONE definition of the floor, shared with the model. This alarm and the
# layoff guard must agree about who is thin, or the site reports a fighter as
# fine while the model is already treating them as unknown -- and vice versa.
COVERAGE_FLOOR = HISTORY_COVERAGE_FLOOR
# A layoff only starts costing rating points after a year (LAYOFF_GRACE_YEARS),
# so a stale last_fight_date is only worth shouting about past that.
STALE_LAST_FIGHT_YEARS = 1.0


def _fold(n) -> str:
    return str(n).strip().lower()


def _split_identities(fighters) -> list:
    """Roster rows that look like one fighter under two spellings.

    Folding cannot catch a middle name -- "jose miguel delgado" and "jose
    delgado" differ by a whole token, and matching on first+last alone would
    merge two different people. So this REPORTS candidates rather than acting
    on them, and a confirmed one gets a line in src/names.NAME_ALIASES.

    Jose Delgado is why. He sat in fighters.csv twice, in fight_history under
    both spellings as two separate Elo nodes, and in ufc_fight_stats under only
    one -- so the card pointed at a fighter with no stats and 14 of his 18
    bouts. Found by the owner noticing an empty scouting drawer four days out,
    which is exactly the discovery route this file exists to replace.
    """
    names = [str(n) for n in fighters["name"].dropna()]
    tok = {n: _normalize_name(n).split() for n in names}
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ta, tb = tok[a], tok[b]
            if not ta or not tb or ta == tb:
                continue
            if ta[0] == tb[0] and ta[-1] == tb[-1] and (set(ta) <= set(tb) or set(tb) <= set(ta)):
                out.append((a, b))
    return out



def _fold(n):
    from src.card_matcher import _normalize_name
    return _normalize_name(str(n))


def _espn_ids(path: str = "data/espn_athlete_ids.csv") -> dict:
    """Folded name -> ESPN id, so a 0-0 finding can say if it is fixable."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "name" not in df.columns or "espn_id" not in df.columns:
        return {}
    return {_fold(r["name"]): str(r["espn_id"]).strip()
            for _, r in df.iterrows() if str(r.get("espn_id") or "").strip()}


def main() -> int:
    try:
        fighters = pd.read_csv(FIGHTERS)
        history = pd.read_csv(HISTORY)
    except FileNotFoundError as exc:
        print(f"[coverage] {exc} -- nothing to check")
        return 0

    today = dt.date.today()
    carded, card_of = set(), {}
    for path in CARDS:
        try:
            d = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if "event_date" not in d.columns:
            continue
        for r in d.to_dict("records"):
            when = str(r.get("event_date") or "")[:10]
            if not when or when < today.isoformat():
                continue        # a card already fought cannot be repriced
            for col in ("fighter_a", "fighter_b"):
                n = r.get(col)
                if pd.isna(n) if n is not None else True:
                    continue
                carded.add(str(n).strip())
                card_of.setdefault(str(n).strip(), (when, r.get("event_name")))

    if not carded:
        print("[coverage] no upcoming card -- nothing to check")
        return 0

    seen = Counter()
    for col in ("fighter_a", "fighter_b"):
        for n in history[col].dropna().astype(str):
            seen[_fold(n)] += 1

    findings = []
    for _, row in fighters.iterrows():
        name = str(row["name"]).strip()
        if name not in carded:
            continue
        when, event = card_of.get(name, ("", ""))
        w, l = row.get("wins"), row.get("losses")
        claimed = (int(w) + int(l)) if pd.notna(w) and pd.notna(l) else None
        held = seen.get(_fold(name), 0)

        # 1. HISTORY COVERAGE -- the one that cost 89 rating points.
        if claimed:
            ratio = held / claimed
            if ratio < COVERAGE_FLOOR:
                findings.append({
                    "severity": "history",
                    "fighter": name, "event": event, "event_date": when,
                    "detail": f"{held} of {claimed} bouts in fight_history "
                              f"({ratio:.0%}) -- layoff, form and Elo all read "
                              f"a partial sample",
                    "last_fight_date": str(row.get("last_fight_date") or ""),
                })
        elif claimed == 0:
            # SAY WHETHER IT IS FIXABLE. This finding fired correctly for
            # Pavel Andrusca and told nobody anything they could act on: his
            # ESPN id was in data/espn_athlete_ids.csv the whole time, so the
            # answer was one lookup away, and the card published him 0-0
            # against a fighter it then held at Medium Confidence for it.
            # fill_from_espn_id_map now repairs that case during the build,
            # which means anything still reported here is either genuinely
            # unsourceable or a lookup that failed -- and those want
            # different responses.
            aid = _espn_ids().get(_fold(name))
            fixable = (f"ESPN id {aid} is on file and the build's id-map pass "
                       f"did not fill it -- check that lookup"
                       if aid else
                       "no ESPN id on file either; add one to "
                       "data/espn_athlete_ids.csv if a profile exists")
            findings.append({
                "severity": "record",
                "fighter": name, "event": event, "event_date": when,
                "detail": f"recorded 0-0 -- either a true debutant or a record "
                          f"we never sourced; power_rating cannot tell them "
                          f"apart. {fixable}",
            })

        # 2. A LAST FIGHT OLDER THAN THE GRACE PERIOD, on a partial history.
        # Stale on its own is fine (real layoffs happen). Stale AND partial is
        # the combination that manufactures a penalty out of nothing.
        lfd = row.get("last_fight_date")
        if claimed and pd.notna(lfd):
            try:
                years = (today - pd.to_datetime(lfd).date()).days / 365.25
            except (ValueError, TypeError):
                years = 0.0
            if years > STALE_LAST_FIGHT_YEARS and held / claimed < COVERAGE_FLOOR:
                findings.append({
                    "severity": "layoff",
                    "fighter": name, "event": event, "event_date": when,
                    "detail": f"last fight {str(lfd)[:10]} ({years:.1f}y) is the "
                              f"newest of only {held} bout{'' if held == 1 else 's'} "
                              f"we hold -- the real one may be far more recent",
                })

        # 3. PHYSICALS. Cheap to state, and reach carries real rating weight.
        gaps = [c for c in ("reach_in", "height_in", "age") if pd.isna(row.get(c))]
        if gaps:
            findings.append({
                "severity": "physicals",
                "fighter": name, "event": event, "event_date": when,
                "detail": "missing " + ", ".join(gaps),
            })

    order = {"history": 0, "layoff": 1, "record": 2, "physicals": 3}
    findings.sort(key=lambda f: (order[f["severity"]], f["event_date"], f["fighter"]))

    counts = Counter(f["severity"] for f in findings)
    print(f"[coverage] {len(carded)} fighter(s) on upcoming cards")

    _split = _split_identities(fighters)
    if _split:
        print(f"\n  SPLIT IDENTITY ({len(_split)}) -- two roster rows that look like one fighter;")
        print( "                    confirm and add to src/names.NAME_ALIASES")
        for a, b in _split:
            print(f"    {a!r} / {b!r}")
    if not findings:
        print("[coverage] every one of them has a full history and complete physicals")
    for sev in ("history", "layoff", "record", "physicals"):
        rows = [f for f in findings if f["severity"] == sev]
        if not rows:
            continue
        print(f"\n  {sev.upper()}  ({len(rows)})")
        for f in rows[:12]:
            print(f"    {f['fighter']:26s} {f['event_date']}  {f['detail']}")
        if len(rows) > 12:
            print(f"    ... and {len(rows) - 12} more")

    payload = {}
    try:
        with open(HEALTH, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        pass
    if not isinstance(payload, dict):
        payload = {}
    payload["card_data_coverage"] = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "carded_fighters": len(carded),
        "counts": dict(counts),
        # Physicals are the long tail and would drown the file; the two that
        # move a price are kept in full.
        "findings": [f for f in findings if f["severity"] in ("history", "layoff", "record")],
    }
    try:
        os.makedirs("data", exist_ok=True)
        with open(HEALTH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError as exc:
        print(f"[coverage] not written ({exc}) -- continuing")
    return 0        # an alarm, never a brake


if __name__ == "__main__":
    sys.exit(main())
