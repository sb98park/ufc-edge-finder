"""
Why don't a fighter's method splits add up to his record?

THE HYPOTHESIS THIS TESTS. `wins` and `losses` may be a fighter's FULL
PROFESSIONAL record, while ko/sub/dec splits are derived from UFC-only fight
history. If so the two are counting different populations, every fighter with
pre-UFC or other-promotion bouts shows the same signature, and nothing is
broken -- the model is simply dividing a UFC-only numerator by an all-promotions
denominator, which understates KO and submission vulnerability across the board.

That matters more than a handful of bad rows would: it is a systematic,
one-directional bias rather than noise, and it would make every fighter with a
regional career look more durable than he is.

The alternative is that the splits are genuinely incomplete for these
fighters specifically, which is a data-quality problem with a different fix.
This script tells the two apart by counting the same fighter three ways:

  record        wins+losses from fighters.csv
  splits        ko+sub+dec, wins and losses separately
  our history   rows in fight_history.csv (what the splits are derived from)
  espn cache    played fights in ESPN's eventlog (UFC-only)

If `splits` matches `our history` but both fall short of `record`, the
mismatch is a population difference, not corruption.

Usage:
    python3 scripts/diagnose_split_mismatch.py
    python3 scripts/diagnose_split_mismatch.py --name "Sumudaerji"
"""

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata

import pandas as pd

CACHE_DIR = "data/.espn_cache"
EVENTLOG = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/eventlog"


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


def num(v):
    try:
        f = float(v)
        return 0 if f != f else int(f)
    except (TypeError, ValueError):
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    args = ap.parse_args()

    f = pd.read_csv("data/fighters.csv")
    try:
        hist = pd.read_csv("data/fight_history.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        hist = pd.DataFrame(columns=["fighter_a", "fighter_b"])
    try:
        ids = {fold(r["name"]): str(r["espn_id"])
               for _, r in pd.read_csv("data/espn_athlete_ids.csv").iterrows()}
    except (FileNotFoundError, pd.errors.EmptyDataError):
        ids = {}

    hist_names = {}
    for col in ("fighter_a", "fighter_b"):
        if col in hist.columns:
            for v in hist[col].dropna():
                hist_names[fold(v)] = hist_names.get(fold(v), 0) + 1

    rows = []
    for _, r in f.iterrows():
        name = r.get("name")
        if args.name and fold(args.name) != fold(name):
            continue
        w, l = num(r.get("wins")), num(r.get("losses"))
        kw, sw, dw = num(r.get("ko_wins")), num(r.get("sub_wins")), num(r.get("dec_wins"))
        kl, sl, dl = num(r.get("ko_losses")), num(r.get("sub_losses")), num(r.get("dec_losses"))
        win_split, loss_split = kw + sw + dw, kl + sl + dl
        if not args.name and win_split == w and loss_split == l:
            continue

        n_hist = hist_names.get(fold(name), 0)
        aid = ids.get(fold(name))
        n_espn = None
        if aid:
            log = cached(EVENTLOG.format(id=aid))
            if log:
                ev = log.get("events") or {}
                its = list(ev.get("items") or [])
                try:
                    pgs = int(ev.get("pageCount") or 1)
                except (TypeError, ValueError):
                    pgs = 1
                for pg in range(2, pgs + 1):
                    more = cached(EVENTLOG.format(id=aid) + f"?page={pg}")
                    its += ((more or {}).get("events") or {}).get("items") or []
                n_espn = sum(1 for e in its if e.get("played"))

        rows.append({
            "name": name, "record": f"{w}-{l}", "record_n": w + l,
            "splits_w": win_split, "splits_l": loss_split, "splits_n": win_split + loss_split,
            "history": n_hist, "espn": n_espn,
            "gap_w": w - win_split, "gap_l": l - loss_split,
        })

    if not rows:
        print("No mismatches found.")
        return

    print(f"{'fighter':<26}{'record':>9}{'splits':>9}{'history':>9}{'espn':>7}"
          f"{'W gap':>7}{'L gap':>7}")
    print("-" * 74)
    for r in sorted(rows, key=lambda x: -(abs(x["gap_l"]) + abs(x["gap_w"]))):
        espn = "-" if r["espn"] is None else r["espn"]
        print(f"{str(r['name'])[:25]:<26}{r['record']:>9}{r['splits_n']:>9}"
              f"{r['history']:>9}{str(espn):>7}{r['gap_w']:>+7}{r['gap_l']:>+7}")

    agree = sum(1 for r in rows if r["splits_n"] == r["history"])
    over = [r for r in rows if r["gap_w"] < 0 or r["gap_l"] < 0]
    print(f"\n{len(rows)} fighter(s) with a mismatch")
    print(f"  splits == our history count : {agree}/{len(rows)}")
    print("\nHOW TO READ IT")
    print("  splits == history  <  record   -> POPULATION MISMATCH. The record is the")
    print("     full pro career, the splits only cover fights we hold history for.")
    print("     Nothing is corrupt; the model's denominator is simply the wrong set.")
    print("  splits  <  history <= record   -> derivation is dropping fights it has.")
    print("     A real bug in whatever fills the split columns.")
    if over:
        print(f"  splits > record ({len(over)} fighter(s)) -> impossible either way; that row is")
        print("     wrong at the source and needs a hand correction:")
        for r in over:
            print(f"       {r['name']}: record {r['record']}, splits sum to "
                  f"{r['splits_w']}W/{r['splits_l']}L")


if __name__ == "__main__":
    main()
