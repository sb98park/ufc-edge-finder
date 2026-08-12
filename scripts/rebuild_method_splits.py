"""
Rebuild ko/sub/dec win-loss splits from ESPN's per-fight eventlog.

WHY NOT THE SUMMARY WE USE NOW. Splits currently come from ESPN's
statsSummary endpoint, which is internally inconsistent: for Sumudaerji it
publishes a 19-7 record and method splits totalling 19 wins but only 4 losses.
Our code copies it faithfully; the source contradicts itself.

The per-fight eventlog reconciles exactly -- 13/1/5 wins and 0/6/1 losses
against a 19-7 record -- once it is read properly. It was NOT being read
properly: ESPN paginates at 25 per page and drops the OLDEST fights off the
end, so any fighter with a longer career was truncated to his most recent 25.
That is fixed; this script consumes every page.

THE BIAS THIS CORRECTS IS ONE-DIRECTIONAL. The summary's missing entries fell
on the loss side, so every affected fighter looked harder to finish than he
is. One-directional error does not wash out across a card the way noise does.

THE STORED RECORDS ARE STALE, which is why most fighters fail to reconcile.
Belal Muhammad settled it: his last win in the eventlog is Jul 2024 and he has
three losses across 2025-2026, so he is 24-6 (1 NC) -- while fighters.csv
still says 24-3. The eventlog is not inventing fights; wins/losses were simply
never updated as those fights happened, and the more active the fighter the
larger the gap (Makhachev 26-1 stored vs 28-1 actual, Oliveira 34-10 vs 37-11).

Both the record AND the splits come from ESPN's statsSummary, now shown
unreliable twice: internally inconsistent (Sumudaerji's losses summed to 4 of
7) and stale (Belal). The per-fight eventlog is the source of truth for both.

--refresh-records rewrites wins/losses from the eventlog as well, which is
what makes the splits reconcile. Without it the reconcile gate below compares
correct splits against an out-of-date target and rejects them.

ONLY RECONCILING FIGHTERS ARE WRITTEN, by default.

The first full run proposed changes for 118 fighters, and 93 of them did not
reconcile: the eventlog gives Makhachev 28-1 against a stored 26-1, Oliveira
37-11 against 34-10, Bautista 18-3 against 15-1. Either those stored records
are understated or the eventlog carries bouts that are not in a fighter's
official record -- and that question is not answerable from here.

So the gate is: replace a fighter's splits only when the new splits sum
EXACTLY to the wins and losses already stored. That makes the change purely a
redistribution of a total we already trust, never a revision of the record
itself. Sumudaerji qualifies (0/6/1 = 7 = his stored losses); Makhachev does
not, and is left alone until the record discrepancy is understood.

--all overrides this, deliberately awkward to reach.

Dry run by default.

Usage:
    python3 scripts/rebuild_method_splits.py
    python3 scripts/rebuild_method_splits.py --name "Sumudaerji"
    python3 scripts/rebuild_method_splits.py --apply
"""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT  # noqa: E402

CACHE_DIR = "data/.espn_cache"
EVENTLOG = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/eventlog"

# A result that is neither a win nor a loss by method. These must be counted
# OUT of both sides rather than swept into a bucket -- ESPN reports a No
# Contest with winner=false, so treating the flag alone as "loss" invents one.
# Matched as WHOLE WORDS. As bare substrings this misfired immediately:
# "nc" is inside "punches", so "TKO (Punches)" was classified as a no-contest
# and Sumudaerji lost a KO win. Same too-loose matching that broke the
# UFC.com name comparison earlier in this project.
NON_DECISIVE = (r"no contest", r"\bnc\b", r"\bdraw\b", r"\boverturned\b")


def fold(v):
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def cached(url):
    p = os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fetch(url):
    hit = cached(url)
    if hit is not None:
        return hit
    try:
        r = requests.get(url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
    except (requests.RequestException, ValueError):
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".json"), "w") as f:
            json.dump(data, f)
    except OSError:
        pass
    return data


def classify(method: str) -> str | None:
    """
    KO / SUB / DEC, or None for anything that has no bucket.

    Matched on substrings because ESPN qualifies methods freely: "TKO
    (Punches)", "TKO (Retirement)", "Submission (Arm Lock)", "Decision -
    Split". A DQ or doctor stoppage returns None and is reported rather than
    quietly folded into decisions.
    """
    m = fold(method)
    if not m or any(re.search(x, m) for x in NON_DECISIVE):
        return None
    if "submission" in m:
        return "sub"
    if "decision" in m:
        return "dec"
    if "ko" in m or "tko" in m:
        return "ko"
    return None


def eventlog_items(aid: str) -> list:
    log = fetch(EVENTLOG.format(id=aid))
    if not log:
        return []
    ev = log.get("events") or {}
    items = list(ev.get("items") or [])
    try:
        pages = int(ev.get("pageCount") or 1)
    except (TypeError, ValueError):
        pages = 1
    for page in range(2, pages + 1):
        more = fetch(f"{EVENTLOG.format(id=aid)}?page={page}")
        items += ((more or {}).get("events") or {}).get("items") or []
    return items


def splits_for(aid: str):
    out = {k: 0 for k in ("ko_wins", "sub_wins", "dec_wins",
                          "ko_losses", "sub_losses", "dec_losses")}
    skipped = []
    # The RECORD as the eventlog sees it. A No Contest is excluded from both
    # sides -- ESPN reports one with winner=false, so counting the flag alone
    # would invent a loss. A DQ has no method bucket but IS a real defeat, so
    # it counts here and legitimately leaves splits one short of the record.
    rec = {"wins": 0, "losses": 0}
    for e in eventlog_items(aid):
        if not e.get("played"):
            continue
        cr = (e.get("competitor") or {}).get("$ref")
        if not cr:
            continue
        st = fetch(cr.split("/competitors/")[0].split("?")[0] + "/status")
        res = (st or {}).get("result") or {}
        method = res.get("displayName") or res.get("name") or ""
        bucket = classify(method)
        comp = fetch(cr)
        won = (comp or {}).get("winner")
        m = fold(method)
        non_decisive = any(re.search(x, m) for x in NON_DECISIVE)
        if won is not None and not non_decisive:
            rec["wins" if won else "losses"] += 1
        if bucket is None:
            skipped.append(method or "(no result)")
            continue
        if won is None:
            skipped.append(f"{method} (no winner flag)")
            continue
        out[f"{bucket}_{'wins' if won else 'losses'}"] += 1
    return out, skipped, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--refresh-records", action="store_true",
                    help="also rewrite wins/losses from the eventlog. The stored "
                         "records are STALE for active fighters -- see the header.")
    ap.add_argument("--all", action="store_true",
                    help="also write splits that do NOT reconcile to the record "
                         "(default is to skip those -- see the note in the header)")
    args = ap.parse_args()

    f = pd.read_csv("data/fighters.csv")
    ids = {fold(r["name"]): str(r["espn_id"])
           for _, r in pd.read_csv("data/espn_athlete_ids.csv").iterrows()}

    changed, reconciled, off, written, stale_records, shrunk = 0, 0, [], [], [], []
    for i, r in f.iterrows():
        name = r.get("name")
        if args.name and fold(args.name) != fold(name):
            continue
        aid = ids.get(fold(name))
        if not aid:
            continue
        new, skipped, rec = splits_for(aid)
        if not any(new.values()):
            continue

        def cur(c):
            v = r.get(c)
            try:
                v = float(v)
                return 0 if v != v else int(v)
            except (TypeError, ValueError):
                return 0

        diff = {k: (cur(k), v) for k, v in new.items() if cur(k) != v}
        w = new["ko_wins"] + new["sub_wins"] + new["dec_wins"]
        l = new["ko_losses"] + new["sub_losses"] + new["dec_losses"]
        rec_w, rec_l = cur("wins"), cur("losses")
        stale = (rec["wins"] != rec_w or rec["losses"] != rec_l)

        # NEVER LET A REFRESH SHRINK A RECORD. The eventlog is trusted to add
        # fights that have happened since the stored record was captured --
        # that is the whole staleness fix. It is NOT trusted to prove a fight
        # never happened. Lone'er Kavanagh came back 13-2 -> 10-2, three wins
        # short, alone against 87 fighters who all gained; far likelier ESPN
        # is missing his early bouts than that he never had them. Applying it
        # would delete real history to fix a cosmetic mismatch.
        shrinks = rec["wins"] < rec_w or rec["losses"] < rec_l
        if args.refresh_records and not shrinks:
            rec_w, rec_l = rec["wins"], rec["losses"]
        elif shrinks:
            shrunk.append((name, f"{rec_w}-{rec_l}", f"{rec['wins']}-{rec['losses']}"))
            stale = False

        # A DQ, draw or overturned result is a real outcome with no method
        # bucket, so splits SHOULD fall short of the record by exactly that
        # many. Demanding an exact match rejected ten fighters for having
        # entirely correct data -- Leon Edwards is 22-6 with a DQ loss, so
        # 22-5 in buckets is right, not broken.
        unbucketed_decisive = sum(1 for m in skipped
                                  if not any(re.search(x, fold(m)) for x in NON_DECISIVE))
        ok = (w + l + unbucketed_decisive == rec_w + rec_l) and w <= rec_w and l <= rec_l
        if ok:
            reconciled += 1
        else:
            off.append((name, f"{rec_w}-{rec_l}", f"{w}-{l}", skipped))

        if diff:
            changed += 1
            writable = ok or args.all
            marks = "  ".join(f"{k} {a}->{b}" for k, (a, b) in sorted(diff.items()))
            flag = "" if ok else ("   [does not reconcile -- SKIPPED]" if not args.all
                                  else "   [does not reconcile -- writing anyway (--all)]")
            print(f"  {str(name)[:26]:<27}{marks}{flag}")
            if args.apply and writable:
                for k, v in new.items():
                    f.at[i, k] = v
                if args.refresh_records:
                    f.at[i, "wins"], f.at[i, "losses"] = rec["wins"], rec["losses"]
                written.append(name)
        if stale:
            stale_records.append((name, f"{cur('wins')}-{cur('losses')}",
                                  f"{rec['wins']}-{rec['losses']}"))

    safe = changed - len(off)
    print(f"\n{changed} fighter(s) differ from the eventlog")
    print(f"   {safe} of them reconcile exactly to the stored record -- these are written")
    print(f"   {len(off)} do not -- skipped unless --all")
    if off:
        print(f"\n{len(off)} fighter(s) still do NOT reconcile -- these need a look:")
        for name, rec, got, skipped in off[:15]:
            extra = f"  unbucketed: {', '.join(skipped)}" if skipped else ""
            print(f"   {str(name)[:24]:<25} record {rec:<7} eventlog {got:<7}{extra}")
        print("   (a No Contest or DQ legitimately has no bucket -- those gaps are correct)")

    if shrunk:
        print(f"\n{len(shrunk)} fighter(s) would LOSE fights if refreshed -- record left alone:")
        for n, old, newr in shrunk:
            print(f"   {str(n)[:24]:<25} {old:>8}  ->  {newr}   (eventlog appears incomplete)")

    if stale_records:
        print(f"\n{len(stale_records)} fighter(s) have a STALE record "
              f"(stored vs eventlog){' -- refreshed' if args.refresh_records else ' -- run with --refresh-records to fix'}:")
        for n, old, newr in stale_records[:15]:
            print(f"   {str(n)[:24]:<25} {old:>8}  ->  {newr}")
        if len(stale_records) > 15:
            print(f"   ... and {len(stale_records) - 15} more")

    if args.apply and written:
        f.to_csv("data/fighters.csv", index=False)
        print(f"\nWrote {len(written)} fighter(s) to data/fighters.csv")
    elif changed:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
