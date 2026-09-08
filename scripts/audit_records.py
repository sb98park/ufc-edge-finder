"""
Find fighters whose stored record disagrees with their career record.

WHY THIS EXISTS. Records were taken from the SCOREBOARD's records[] entry
named "overall", which is a fighter's record WITHIN THAT PROMOTION rather
than their career. A regional fighter debuting in the UFC therefore showed
2-1 against a true career mark of 16-2 -- caught live during a card, on a
fight where the model had 79% confidence in his opponent.

That understates exactly the input the model leans on for debutants, and it
does so silently: a wrong-but-plausible record looks like a real one.

This checks every fighter against ESPN's athlete endpoint, which publishes
the career figure, and reports disagreements without changing anything.

WHY AN AUDIT ISN'T ENOUGH ON ITS OWN. The backfill only refreshes a fighter
when one of their columns is NULL. A wrong-but-present record is never
re-checked -- so fixing the source only helps fighters added afterwards, and
everyone already carrying a promotion-scoped record keeps it forever. Hence
--apply: it overwrites the stored record (and the method splits, which come
from the same ESPN block) with the career figures.

Usage:
    python3 scripts/audit_records.py                 # all fighters, report only
    python3 scripts/audit_records.py --card          # this weekend's card only
    python3 scripts/audit_records.py --card --apply  # and fix what it finds
"""

import csv
import io
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fighter_backfill import _fetch_espn_method_records, BASE_HEADERS, REQUEST_TIMEOUT  # noqa: E402
from src.results_fetcher import ESPN_SCOREBOARD_URL  # noqa: E402
from src.card_matcher import _normalize_name  # noqa: E402

import requests  # noqa: E402


def _athlete_ids_from_scoreboard(dates):
    """
    Resolve athlete ids from the SCOREBOARD, the same source the backfill
    uses -- not ESPN's search endpoint.

    A first version used search and resolved ZERO of 28 fighters: that
    endpoint's response shape doesn't yield athlete ids the way this needed,
    and the failure was silent (every fighter simply counted as
    "unresolved"). The scoreboard already carries athlete.id alongside
    fullName for every competitor on a card, so matching by normalized name
    against it is both reliable and free of a second lookup per fighter.
    """
    ids = {}
    for d in dates:
        try:
            param = str(d).replace("-", "")
            r = requests.get(ESPN_SCOREBOARD_URL, params={"dates": param},
                             headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            for ev in r.json().get("events", []):
                for comp in ev.get("competitions", []):
                    for c in comp.get("competitors", []):
                        ath = c.get("athlete") or {}
                        name, aid = ath.get("fullName"), ath.get("id") or c.get("id")
                        if name and aid:
                            ids[_normalize_name(name)] = str(aid)
        except Exception as e:
            print(f"  [warn] scoreboard fetch failed for {d}: {e}")
    return ids


def _write_corrections(corrections: dict, path: str = "data/fighters.csv") -> None:
    """
    Apply the corrections to the file's own text, touching nothing else.

    NOT fighters.to_csv(). A pandas round-trip rewrites every cell it parsed,
    and this file carries the literal string "nan" in last_fight_method --
    read as NaN, written back as "". Correcting ONE fighter therefore
    produced a 117-line diff across 116 unrelated rows, three separate times,
    each needing hand-cleaning before it could be committed. A data commit
    that restates rows it did not mean to touch is exactly what CLAUDE.md s3
    warns about, and the noise is where a real change hides.

    So the frame is used to DECIDE and the text is used to WRITE. csv round
    trip with lineterminator="\n" -- csv.writer defaults to \r\n and would
    rewrite all 378 lines by itself, which is the same bug wearing different
    clothes.
    """
    if not corrections:
        return
    with open(path, newline="", encoding="utf-8") as fh:
        text = fh.read()
    rows = list(csv.DictReader(io.StringIO(text)))
    fieldnames = csv.DictReader(io.StringIO(text)).fieldnames
    for r in rows:
        fix = corrections.get(r.get("name"))
        if not fix:
            continue
        for col, val in fix.items():
            if col not in r or val is None:
                continue
            # SAME NUMBER, DIFFERENT SPELLING IS NOT A CHANGE. ESPN returns
            # ints and the file stores "16.0", so writing an unchanged split
            # back reformats it and the row's diff claims six corrections
            # where one was made. Compare numerically, write only what moved.
            try:
                if float(r[col] or "nan") == float(val):
                    continue
            except (TypeError, ValueError):
                pass
            r[col] = val
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    apply = "--apply" in sys.argv
    fighters = pd.read_csv("data/fighters.csv")
    names = set(fighters["name"])

    if "--card" in sys.argv:
        try:
            cards = pd.read_csv("data/fight_cards.csv")
            on_card = set(cards["fighter_a"]) | set(cards["fighter_b"])
            names = {n for n in names if n in on_card}
        except (FileNotFoundError, KeyError):
            pass

    # Gather athlete ids once, from the card dates we track.
    dates = []
    for f in ("data/fight_cards.csv", "data/future_cards.csv"):
        try:
            d = pd.read_csv(f)
            if "event_date" in d.columns:
                dates.extend(sorted(set(d["event_date"].dropna().astype(str))))
        except FileNotFoundError:
            pass
    dates = sorted(set(dates))
    print(f"resolving athlete ids from {len(dates)} card date(s)...")
    id_map = _athlete_ids_from_scoreboard(dates)
    print(f"  resolved {len(id_map)} athlete ids\n")

    print(f"checking {len(names)} fighters against ESPN's career record...\n")
    bad, checked, unresolved = [], 0, 0
    corrections: dict = {}
    for name in sorted(names):
        row = fighters[fighters["name"] == name]
        if row.empty or pd.isna(row.iloc[0].get("wins")):
            continue
        stored_w, stored_l = int(row.iloc[0]["wins"]), int(row.iloc[0]["losses"])
        aid = id_map.get(_normalize_name(name))
        if not aid:
            unresolved += 1
            continue
        recs = _fetch_espn_method_records(aid)
        cw, cl = recs.get("_career_w"), recs.get("_career_l")
        time.sleep(0.25)          # be gentle with ESPN
        if cw is None:
            unresolved += 1
            continue
        checked += 1
        if (cw, cl) != (stored_w, stored_l):
            bad.append((name, f"{stored_w}-{stored_l}", f"{cw}-{cl}"))
            print(f"  MISMATCH {name:26} stored {stored_w}-{stored_l:<3} career {cw}-{cl}")
            if apply:
                # Collected as {name: {col: value}} and applied to the FILE's
                # own text below, rather than mutated into the DataFrame and
                # written with to_csv -- see _write_corrections.
                fix = {"wins": cw, "losses": cl}
                # Method splits come from the same statsSummary block, so a
                # wrong overall record almost always means wrong splits too.
                for col in ("ko_wins", "ko_losses", "sub_wins", "sub_losses",
                            "dec_wins", "dec_losses"):
                    if col in recs and col in fighters.columns:
                        fix[col] = recs[col]
                corrections[name] = fix

    print(f"\nchecked {checked} | mismatches {len(bad)} | unresolved {unresolved}")
    if bad and apply:
        _write_corrections(corrections)
        print(f"\nWritten: {len(bad)} record(s) corrected in data/fighters.csv.")
        print("Re-run generate_site.py, then commit data/fighters.csv.")
    elif bad:
        print("\nDRY RUN -- nothing written. Re-run with --apply to fix these.")
    else:
        print("Every stored record matches ESPN's career figure.")


if __name__ == "__main__":
    main()
